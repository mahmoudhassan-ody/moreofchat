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

**The adapter is resolved per job, not per process.** Every adapter here is
built from one tenant's credentials — Twilio's own docstring says the sender
comes from that tenant's `channel_accounts` row and is never platform-wide —
and a worker holding one adapter per channel for its whole life sends every
tenant's replies from one number. The customer gets an answer from a business
they never wrote to, correctly worded, on the right channel, under the wrong
name. Nothing raises, at either end.

So `providers` is a registry keyed by (tenant, channel) rather than a mapping
keyed by channel. There is no mapping form accepted here, because the mapping
form is right for exactly one tenant and every process eventually serves two.
"""

import time
from typing import Any, Protocol

from moc.channels.base import MessagingProvider, OutboundJob
from moc.workers.streams import TerminalFailure, consumer_from_config

_CONSUMER = "outbound-1"


class ProviderRegistry(Protocol):
    """(tenant, channel) -> the adapter built from that tenant's credentials.

    Async because the production implementation reads the tenant's channel
    account. Returns None for a pair it cannot serve, which the worker turns
    into a dead letter rather than a guess.
    """

    async def for_job(
        self, *, tenant_id: str, channel: str
    ) -> MessagingProvider | None: ...


class OutboundWorker:
    def __init__(
        self,
        *,
        client: Any,
        providers: ProviderRegistry,
        config: dict[str, Any],
        consumer: str = _CONSUMER,
        clock: Any = time.monotonic,
    ) -> None:
        self._client = client
        self._providers = providers
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
        # **Routed by channel.** One provider was enough while one channel
        # existed, and `job.channel` went unread — so the moment Telegram
        # arrived, whichever sender drew the entry would send it. A Telegram
        # reply handed to Twilio carries a chat id where a phone number
        # belongs: Twilio rejects it, the customer is silently unanswered, and
        # the dead-letter row blames the number.
        provider = await self._providers.for_job(
            tenant_id=job.tenant_id, channel=job.channel
        )
        if provider is None:
            raise Unroutable(
                f"no sender for tenant {job.tenant_id} on channel {job.channel!r} — "
                "the reply cannot be sent, and sending it through another tenant's "
                "account would put their name on somebody else's answer"
            )
        if not await self.take_token(job.tenant_id):
            # No token: raise so the entry stays pending and is retried on the
            # next pass. Sleeping here would hold the whole batch behind one
            # throttled tenant.
            raise RateLimited(f"tenant {job.tenant_id} has no tokens left")
        await provider.send(
            to=job.to,
            text=job.text,
            template=job.template,
            template_variables=job.template_variables,
            last_inbound_at=job.inbound_at(),
            thread_ref=job.thread_ref,
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


class Unroutable(TerminalFailure):
    """No provider for this job's channel.

    Terminal: buried on the first attempt rather than retried. A channel
    nobody configured is a configuration problem for a person to see, and five
    attempts at it are five delays in front of every other tenant's reply.
    """


__all__ = ["OutboundJob", "OutboundWorker", "RateLimited", "Unroutable"]
