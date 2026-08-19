"""The agent worker — design §3.

Consumes the inbound stream, runs one turn, and hands the reply to the outbound
stream. This is where the webhook's promise is kept: the handler returned 200
in milliseconds because this process does the slow part.

Everything the turn needs that the webhook deliberately did not touch happens
here — the tenant session, the conversation row, the ledger writes — inside one
transaction, so a turn that fails leaves no half-written state and its stream
entry stays pending for another attempt.

The reply goes onto a second stream rather than being sent inline. Sending is a
vendor call with its own rate limits and its own failure modes; doing it here
would mean a Twilio 429 fails a turn that was otherwise complete, and the retry
would re-run the model.
"""

from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncEngine

from moc.agent.conversations import ConversationStore
from moc.agent.script_engine import ScriptEngine
from moc.channels.base import InboundMessage, OutboundJob
from moc.channels.valkey import decode
from moc.tenancy.context import tenant_session
from moc.workers.streams import consumer_from_config

_CONSUMER = "inbound-1"


class TurnRunner(Protocol):
    """What this worker needs from an orchestrator, and nothing more."""

    async def handle(self, **kwargs: Any) -> Any: ...


class InboundWorker:
    def __init__(
        self,
        *,
        client: Any,
        engine: AsyncEngine,
        orchestrator: TurnRunner,
        script: str,
        config: dict[str, Any],
        consumer: str = _CONSUMER,
    ) -> None:
        self._client = client
        self._engine = engine
        self._orchestrator = orchestrator
        self._script = script
        self._engine_for_script = ScriptEngine.from_config(script)
        self._outbound_stream = config["outbound"]["stream"]
        self._consumer = consumer_from_config(
            client=client, section=config["inbound"], consumer=consumer
        )

    async def run_once(self, *, block: bool = False) -> int:
        return await self._consumer.run_once(self._handle, block=block)

    async def _handle(self, payload: str) -> None:
        message = decode(payload)
        # One transaction for the whole turn: the conversation row, every
        # ledger row, and the state update commit together or not at all. A
        # turn that half-committed would bill a customer for a reply whose
        # state never advanced, and the retry would bill them again.
        async with tenant_session(self._engine, message.tenant_id) as session:
            store = ConversationStore(session=session, engine=self._engine_for_script)
            state = await store.load(
                channel=str(message.channel), sender_ref=message.sender_ref
            )
            result = await self._orchestrator.handle(
                session=session,
                state=state,
                text=message.text or "",
                channel=str(message.channel),
            )
            await store.save(
                channel=str(message.channel),
                sender_ref=message.sender_ref,
                channel_account_id=message.channel_account_id,
                last_inbound_at=message.received_at,
                state=result.state,
            )
            await session.commit()

        await self._enqueue_reply(message, result.reply)

    async def _enqueue_reply(self, message: InboundMessage, reply: str) -> None:
        """Hand the words to the sender.

        Published *after* the commit, deliberately. Publishing first risks a
        reply going out for a turn whose state was rolled back — the customer
        would be answered from a conversation that, as far as the database is
        concerned, never advanced. The other ordering risks a committed turn
        whose reply is never sent, which the pending-entry retry recovers.
        """
        if not reply:
            return
        job = OutboundJob(
            tenant_id=str(message.tenant_id),
            channel=str(message.channel),
            to=message.sender_ref,
            text=reply,
            # A reply to a message we just received is always inside the
            # 24-hour window (§6.2) — the inbound message is what opened it.
            last_inbound_at=message.received_at.isoformat(),
        )
        await self._client.xadd(self._outbound_stream, {"payload": job.to_json()})


__all__ = ["InboundWorker"]
