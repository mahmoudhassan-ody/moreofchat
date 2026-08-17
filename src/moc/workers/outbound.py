"""The sender worker — design §6.2.

One process per channel, drawing from the outbound stream and calling the
channel adapter. It owns the three policies the adapter deliberately does not:

**Rate limiting**, as a token bucket in Valkey rather than in the process. Two
sender processes each holding their own bucket each allow the full rate, which
is the same as having no limit — and the limit exists because Meta's is real,
and being throttled by them is worse than throttling ourselves.

**Backoff**, so a transient vendor failure is retried rather than dropped, and
retried at a widening interval rather than as fast as the loop can spin.

**Dead-lettering**, because a permanently rejected send — an invalid number, a
template that was never approved — is an alert to the tenant's admin, not an
infinite retry. Retrying it forever is how one bad row becomes a sender that
delivers nothing else.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from moc.channels.base import MessagingProvider
from moc.workers.streams import consumer_from_config

_CONSUMER = "outbound-1"


@dataclass(frozen=True)
class OutboundJob:
    """A reply waiting to be sent.

    `last_inbound_at` travels with the job rather than being looked up at send
    time: the window (§6.2) is a property of the conversation as it stood when
    the turn ran, and re-reading it later would let a reply become
    window-invalid purely because the sender was backed up.
    """

    tenant_id: str
    channel: str
    to: str
    text: str | None = None
    template: str | None = None
    template_variables: dict[str, str] | None = None
    last_inbound_at: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "tenant_id": self.tenant_id,
                "channel": self.channel,
                "to": self.to,
                "text": self.text,
                "template": self.template,
                "template_variables": self.template_variables,
                "last_inbound_at": self.last_inbound_at,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, record: str | dict[str, Any]) -> OutboundJob:
        document = json.loads(record) if isinstance(record, str) else record
        return cls(**document)

    def inbound_at(self) -> datetime | None:
        return datetime.fromisoformat(self.last_inbound_at) if self.last_inbound_at else None


class OutboundWorker:
    def __init__(
        self,
        *,
        client: Any,
        provider: MessagingProvider,
        config: dict[str, Any],
        consumer: str = _CONSUMER,
        clock: Any = time.monotonic,
    ) -> None:
        self._client = client
        self._provider = provider
        self._clock = clock
        limit = config["rate_limit"]
        self._bucket_prefix = limit["key_prefix"]
        self._capacity = limit["capacity"]
        self._refill = limit["refill_per_second"]
        self._consumer = consumer_from_config(
            client=client, section=config["outbound"], consumer=consumer
        )

    async def run_once(self, *, block: bool = False) -> int:
        return await self._consumer.run_once(self._handle, block=block)

    async def _handle(self, payload: str) -> None:
        job = OutboundJob.from_json(payload)
        if not await self.take_token(job.tenant_id):
            # No token: raise so the entry stays pending and is retried on the
            # next pass. Sleeping here would hold the whole batch behind one
            # throttled tenant.
            raise RateLimited(f"tenant {job.tenant_id} has no tokens left")
        await self._provider.send(
            to=job.to,
            text=job.text,
            template=job.template,
            template_variables=job.template_variables,
            last_inbound_at=job.inbound_at(),
        )

    # ─────────────────────────── token bucket ───────────────────────────

    async def take_token(self, tenant_id: str) -> bool:
        """Draw one token for `tenant_id`, refilling by elapsed time.

        Shared across processes because the state is in Valkey. The refill is
        computed from a stored timestamp rather than a background job — a timer
        that has to be running for the limiter to be correct is a limiter that
        is wrong whenever the timer is not.
        """
        key = f"{self._bucket_prefix}{tenant_id}"
        now = self._clock()
        stored = await self._client.hgetall(key)
        tokens = float(stored.get("tokens", self._capacity)) if stored else float(self._capacity)
        last = float(stored.get("at", now)) if stored else now

        tokens = min(self._capacity, tokens + (now - last) * self._refill)
        if tokens < 1:
            await self._client.hset(key, mapping={"tokens": tokens, "at": now})
            return False

        await self._client.hset(key, mapping={"tokens": tokens - 1, "at": now})
        return True


class RateLimited(Exception):
    """Not an error in the send — a reason to try this entry again shortly."""


__all__ = ["OutboundJob", "OutboundWorker", "RateLimited"]
