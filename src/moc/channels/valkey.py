"""Valkey-backed queue and idempotency store — design §3 and §6.2.

**Streams, not lists.** A consumer group gives at-least-once delivery with an
explicit ack: a worker that dies mid-turn leaves its entry pending, and another
worker claims it. `LPOP` loses the message the instant the process holding it
stops, and the customer is left unanswered with nothing anywhere to show it
ever arrived.

At-least-once makes deduplication someone's job, and that someone is the
webhook — which is why `ValkeyEventLog` lives here beside the queue rather than
in the worker. Both are the same decision seen from two ends.

This module is the only place in `moc.channels` that knows Valkey exists. Same
containment as the provider adapters: one file to change if the queue ever
becomes something else.
"""

import json
from typing import Any

from moc.channels.base import InboundMessage, MediaRef

_PAYLOAD = "payload"
_QUEUES = "workers/queues"


def _encode(message: InboundMessage) -> str:
    """JSON, not pickle.

    A queue entry outlives the process that wrote it and is read by a human
    during an incident. Pickle is neither inspectable nor safe to load from a
    store an attacker might reach, and both properties matter more here than
    the convenience.
    """
    return json.dumps(
        {
            "tenant_id": str(message.tenant_id),
            "channel": str(message.channel),
            "channel_account_id": str(message.channel_account_id),
            "provider_message_id": message.provider_message_id,
            "sender_ref": message.sender_ref,
            "thread_ref": message.thread_ref,
            "text": message.text,
            "received_at": message.received_at.isoformat(),
            "media": [{"url": m.url, "content_type": m.content_type} for m in message.media],
            "raw": message.raw,
        },
        ensure_ascii=False,
    )


def decode(document: str | dict[str, Any]) -> InboundMessage:
    """Rebuild an `InboundMessage` from a stream entry."""
    from datetime import datetime
    from uuid import UUID

    from moc.channels.base import Channel

    record = json.loads(document) if isinstance(document, str) else document
    return InboundMessage(
        tenant_id=UUID(record["tenant_id"]),
        channel=Channel(record["channel"]),
        channel_account_id=UUID(record["channel_account_id"]),
        provider_message_id=record["provider_message_id"],
        sender_ref=record["sender_ref"],
        received_at=datetime.fromisoformat(record["received_at"]),
        thread_ref=record.get("thread_ref"),
        text=record.get("text"),
        media=tuple(
            MediaRef(url=m["url"], content_type=m.get("content_type"))
            for m in record.get("media") or []
        ),
        raw=record.get("raw") or {},
    )


def valkey_client(*, db: int | None = None, config: dict[str, Any] | None = None) -> Any:
    """The one place a Valkey client is constructed.

    **The socket timeout has to outlast the longest blocking read.** A worker
    calling `run_once(block=True)` holds an XREADGROUP open for `block_ms`;
    redis-py 8 defaults `socket_timeout` to 5 seconds, and `block_ms` is 5000.
    A worker started against an idle stream therefore raised `TimeoutError` and
    exited — which is to say, no worker could stay up with nothing to do, which
    is what a worker does most of the time.

    Nothing caught it because nothing ever blocked: every test polls with
    `block=False`. So the coupling lives here, derived from the same config the
    workers read, rather than in whichever process happens to build a client.
    """
    from redis.asyncio import Redis

    from moc.config import settings
    from moc.config_store import load

    queues = config or load(_QUEUES)
    longest = max(
        section["block_ms"]
        for section in queues.values()
        if isinstance(section, dict) and "block_ms" in section
    )
    return Redis.from_url(
        settings.valkey_url(db),
        decode_responses=True,
        socket_timeout=longest / 1000 + queues["client"]["socket_timeout_margin_seconds"],
    )


class ValkeyInboundQueue:
    """`InboundQueue` over a Valkey stream."""

    def __init__(self, *, client: Any, config: dict[str, Any]) -> None:
        self._client = client
        self._stream = config["inbound"]["stream"]

    async def publish(self, message: InboundMessage) -> None:
        await self._client.xadd(self._stream, {_PAYLOAD: _encode(message)})

    async def publish_raw(self, stream: str, payload: str) -> None:
        """Put an already-encoded payload on any stream.

        Used by the inbound worker to hand a reply to the sender, and by tests
        to seed one. Kept explicit rather than overloading `publish`, so a
        caller cannot accidentally publish an outbound job onto the inbound
        stream and have it decode as a customer message.
        """
        await self._client.xadd(stream, {_PAYLOAD: payload})


class ValkeyEventLog:
    """Idempotency for vendor redeliveries (§6.2).

    `SET NX EX` is the whole mechanism: the first writer wins and everyone
    after it is told the id is taken. Atomic, which a `GET` followed by a `SET`
    would not be — two workers racing a redelivery would both see the key
    absent and both enqueue a turn.

    The TTL matters as much as the flag. Vendors retry for hours, so memory has
    to outlast an incident; keeping ids forever turns a correctness mechanism
    into an unbounded key store nobody prunes.
    """

    def __init__(self, *, client: Any, config: dict[str, Any]) -> None:
        self._client = client
        self._prefix = config["idempotency"]["key_prefix"]
        self._ttl = config["idempotency"]["ttl_seconds"]

    def _key(self, provider_message_id: str) -> str:
        return f"{self._prefix}{provider_message_id}"

    async def claim(self, provider_message_id: str) -> bool:
        return bool(
            await self._client.set(self._key(provider_message_id), "1", nx=True, ex=self._ttl)
        )

    async def release(self, provider_message_id: str) -> None:
        """Give the id back after a failed enqueue.

        Without this, the 503 we return asks the vendor to retry and the retry
        is then discarded as a duplicate — the message lost by the mechanism
        meant to protect it.
        """
        await self._client.delete(self._key(provider_message_id))

class ValkeyOutboundPublisher:
    """The agent inbox's reply, onto the same stream the bot's replies use.

    The narrowest possible adapter, and deliberately not a provider. §6.2's one
    send path is the reason: an agent's reply is a message to a customer on a
    messaging platform, so it is subject to the same rate limit and the same
    24-hour window as a bot reply. A second send path means a second token
    bucket, and two buckets each allowing the full rate is the same as no
    limit.

    Had no implementation until Task 39b — the console had never been served,
    so the port had only ever been filled by a fake.
    """

    def __init__(self, *, client: Any, config: dict[str, Any] | None = None) -> None:
        from moc.config_store import load

        self._client = client
        self._stream = (config or load(_QUEUES))["outbound"]["stream"]

    async def publish(self, job: Any) -> None:
        await self._client.xadd(self._stream, {_PAYLOAD: job.to_json()})


class ValkeyInboxEvents:
    """Fan-out for the inbox's live updates, scoped by tenant at the source.

    One pub/sub channel per tenant rather than one channel carrying a tenant
    id that subscribers are trusted to filter on. A filter in the subscriber is
    a filter somebody can forget, and forgetting it here streams one tenant's
    conversations into another tenant's console.
    """

    def __init__(self, *, client: Any, prefix: str = "moc:inbox:") -> None:
        self._client = client
        self._prefix = prefix

    def _channel(self, tenant_id: Any) -> str:
        return f"{self._prefix}{tenant_id}"

    async def publish(self, *, tenant_id: Any, event: dict[str, Any]) -> None:
        await self._client.publish(
            self._channel(tenant_id), json.dumps(event, ensure_ascii=False)
        )

    async def subscribe(self, *, tenant_id: Any) -> Any:
        pubsub = self._client.pubsub()
        await pubsub.subscribe(self._channel(tenant_id))
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(self._channel(tenant_id))
            await pubsub.aclose()



__all__ = [
    "ValkeyEventLog",
    "ValkeyInboundQueue",
    "ValkeyInboxEvents",
    "ValkeyOutboundPublisher",
    "decode",
    "valkey_client",
]