"""Shared consumer-group mechanics for the stream workers.

Both workers need the same five things and differ only in what they do with a
payload: create the group idempotently, read a batch, ack on success, leave the
entry pending on failure, and dead-letter one that has failed too often.

Three decisions are worth stating because getting any of them wrong is silent:

**Ack after the work, never before.** An entry acked on receipt is an entry
that vanishes when the process dies three seconds later — invisible,
unreproducible, and indistinguishable from the customer never having written.

**A dead-lettered entry is acked.** It has to be: an entry that fails forever
and is never acked stays pending forever, and the pending list is the thing
every future consumer has to walk. One poisoned message would become a worker
that processes nothing else.

**Attempts are counted from the stream, not from a local variable.** Valkey
tracks a delivery count per pending entry, which survives the worker restarting
— a counter in the process resets on every deploy, and a message that fails at
deploy time would retry forever without ever reaching the limit.
"""

from collections.abc import Awaitable, Callable
from typing import Any

_PAYLOAD = "payload"
#: A dead letter is not a log file.
_TRACEBACK_LIMIT = 4000
_NEW_MESSAGES = ">"
_FROM_START = "0"

Handler = Callable[[str], Awaitable[None]]


class TerminalFailure(Exception):
    """A failure retrying cannot fix.

    Raised by a handler that already knows the answer will not change — an
    unroutable channel, a malformed payload. The consumer buries it on the
    first attempt rather than spending the retry budget discovering what the
    handler could already say.
    """


class StreamConsumer:
    def __init__(
        self,
        *,
        client: Any,
        stream: str,
        group: str,
        consumer: str,
        batch_size: int,
        block_ms: int,
        max_attempts: int,
        dead_letter_stream: str,
        dead_letter_maxlen: int,
    ) -> None:
        self._client = client
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._batch_size = batch_size
        self._block_ms = block_ms
        self._max_attempts = max_attempts
        self._dead_letter = dead_letter_stream
        self._dead_letter_maxlen = dead_letter_maxlen
        self._group_ready = False

    async def _ensure_group(self) -> None:
        """Create the group, tolerating the race and the rerun.

        `mkstream=True` so a worker started before the first message does not
        have to be restarted once traffic arrives — otherwise the first deploy
        of a new channel looks broken for as long as nobody has written yet.
        """
        if self._group_ready:
            return
        try:
            await self._client.xgroup_create(
                self._stream, self._group, id=_FROM_START, mkstream=True
            )
        except Exception as exc:  # noqa: BLE001 - BUSYGROUP is the expected case
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ready = True

    async def run_once(self, handle: Handler, *, block: bool = False) -> int:
        """Process one batch. Returns how many entries were handled successfully.

        Reclaims previously-delivered entries before reading new ones, so a
        worker that took over from a crashed one drains the backlog rather than
        racing ahead of it and leaving the oldest customer waiting longest.
        """
        await self._ensure_group()
        handled = await self._drain(_FROM_START, handle)
        return handled + await self._drain(_NEW_MESSAGES, handle, block=block)

    async def _drain(self, cursor: str, handle: Handler, *, block: bool = False) -> int:
        response = await self._client.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: cursor},
            count=self._batch_size,
            block=self._block_ms if block else None,
        )
        handled = 0
        for _, entries in response or []:
            for entry_id, fields in entries:
                if await self._process(entry_id, fields, handle):
                    handled += 1
        return handled

    async def _process(self, entry_id: str, fields: dict[str, str], handle: Handler) -> bool:
        payload = fields.get(_PAYLOAD)
        if payload is None:
            # Nothing to retry: an entry without a payload will never grow one.
            await self._bury(entry_id, "", "missing payload")
            return False
        try:
            await handle(payload)
        except TerminalFailure as exc:
            # Retrying will not help and the handler knows it. Five attempts at
            # a job for a channel nobody configured is five delays in front of
            # every other tenant's reply, and the outcome is the same row in
            # the same dead-letter stream either way.
            await self._bury(entry_id, payload, repr(exc), exc)
            return False
        except Exception as exc:  # noqa: BLE001 - the handler decides nothing; we retry or bury
            if await self._attempts(entry_id) >= self._max_attempts:
                await self._bury(entry_id, payload, repr(exc), exc)
            # Otherwise: no ack. The entry stays pending and is retried on the
            # next pass, with the delivery count Valkey keeps for us.
            return False
        await self._client.xack(self._stream, self._group, entry_id)
        return True

    async def _attempts(self, entry_id: str) -> int:
        pending = await self._client.xpending_range(
            self._stream, self._group, min=entry_id, max=entry_id, count=1
        )
        return pending[0]["times_delivered"] if pending else 1

    async def _bury(
        self, entry_id: str, payload: str, reason: str, exc: BaseException | None = None
    ) -> None:
        """Move an entry to the dead-letter stream and ack the original.

        The payload travels with the reason so the alert is actionable — a dead
        letter that says only "failed" sends someone hunting through logs for a
        message they cannot identify.

        **And the traceback travels too.** `repr(exc)` names the exception and
        not the line: the first rehearsal produced
        `PermissionError(13, 'Permission denied')` with no path, no frame and no
        clue, on a turn where a customer got silence — and finding it meant
        re-running the whole path by hand. This row is the one place that
        question has to be answerable, and the process whose log would have
        held the frames is by then a container that has restarted.

        Capped, because a dead-letter entry is not a log file and a runaway
        recursion would otherwise write a megabyte per failed message.
        """
        frames = ""
        if exc is not None:
            import traceback

            frames = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-_TRACEBACK_LIMIT:]
        await self._client.xadd(
            self._dead_letter,
            {
                _PAYLOAD: payload,
                "reason": reason,
                "entry_id": entry_id,
                "traceback": frames,
            },
            # Capped like the work streams, and lower. These are read by a
            # person, and five thousand unread failures is not a backlog
            # anyone is working through.
            maxlen=self._dead_letter_maxlen,
            approximate=True,
        )
        await self._client.xack(self._stream, self._group, entry_id)


def consumer_from_config(
    *, client: Any, section: dict[str, Any], consumer: str
) -> StreamConsumer:
    return StreamConsumer(
        client=client,
        stream=section["stream"],
        group=section["group"],
        consumer=consumer,
        batch_size=section["batch_size"],
        block_ms=section["block_ms"],
        max_attempts=section["max_attempts"],
        dead_letter_stream=section["dead_letter_stream"],
        dead_letter_maxlen=section["dead_letter_maxlen"],
    )


__all__ = ["StreamConsumer", "TerminalFailure", "consumer_from_config"]
