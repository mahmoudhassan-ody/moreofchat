"""The channel port — design §6.1 and §6.2.

Five channels, one normalized inbound shape. Everything downstream of this
package works on `InboundMessage` and never learns which vendor carried the
message, which is what makes §6.2's migration note true: moving WhatsApp from
Twilio to Meta's Graph API directly is a new adapter file, not a refactor of
the agent.

`MessagingProvider` is the outbound half. It is deliberately thin — no
templates are authored here, only referenced (§6.2 says proxy Twilio's rather
than building template management), and no retry or rate-limit policy lives
here either. That belongs to the outbound worker, where a token bucket can be
shared across every message for a tenant instead of per adapter instance.

**`ChannelAccountRegistry` and the bootstrap read.** The design lists
`channel_accounts` as tenant-scoped (§5), and every tenant-scoped table gets
RLS. This lookup is the one that cannot rely on it: it runs *before* a tenant
context exists, because its whole job is to establish which tenant a message
belongs to. Resolving it under RLS with no tenant set returns nothing, and
setting a tenant first requires the answer.

Settled in migration 0007 as a narrowly-privileged `moc_lookup` role with
SELECT on a five-column view and no other privilege in the database — not as a
policy admitting a bootstrap read, which would have widened the base table for
every role rather than creating a separate, smaller door. The implementation
is `channels/accounts.py`; the boundary is asserted in
`tests/tenancy/test_channel_accounts.py`.
"""

import enum
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID


class Channel(enum.StrEnum):
    """§6.1. Separate from `moc.evals.schema.Channel` on purpose: one is a
    runtime routing key, the other is a field in a case file, and coupling them
    would make the eval package a runtime dependency."""

    whatsapp = "whatsapp"
    instagram = "instagram"
    messenger = "messenger"
    telegram = "telegram"
    email = "email"


@dataclass(frozen=True)
class MediaRef:
    url: str
    content_type: str | None = None


@dataclass(frozen=True)
class ChannelAccount:
    """One connected channel for one tenant (§5, `channel_accounts`).

    `secret_ref` *names* the signing secret; it is not the secret. The secret
    is per account and never platform-wide — a shared token means any party
    who has seen one tenant's credentials can forge every other tenant's
    inbound traffic, including that tenant themselves, which is the version
    that ends up in a dispute.

    Holding the reference rather than the value is what lets this object come
    back from the pre-tenant bootstrap read (Task 21). That lookup runs before
    any signature has been verified, so it must not be able to reach the one
    value an attacker would need. `SecretResolver` supplies it separately.
    """

    id: UUID
    tenant_id: UUID
    channel: Channel
    #: The vendor-facing address, vendor prefixes stripped. For WhatsApp this
    #: is the business number in E.164.
    account_ref: str
    secret_ref: str


@dataclass(frozen=True)
class InboundMessage:
    """§6.1's normalized shape, verbatim.

    `raw` is stored and never parsed downstream. It is what a dispute is
    settled from, and it keeps fields this version does not understand —
    including ones the vendor adds after this code was written.
    """

    tenant_id: UUID
    channel: Channel
    channel_account_id: UUID
    provider_message_id: str
    sender_ref: str
    received_at: datetime
    thread_ref: str | None = None
    text: str | None = None
    media: tuple[MediaRef, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutboundReceipt:
    provider_message_id: str
    status: str


@dataclass(frozen=True)
class OutboundJob:
    """A reply waiting to be sent — the shape that crosses to the sender.

    Defined here rather than in the worker because both the orchestrator's
    reply path and the agent inbox construct one, and neither may import the
    workers layer. Putting it in the worker made "one send path" impossible to
    honour without a layer violation, which is a good sign the type belonged
    in the channel vocabulary all along.

    `last_inbound_at` travels with the job rather than being looked up at send
    time: the window (§6.2) is a property of the conversation as it stood when
    the turn ran, and re-reading it later would let a reply become
    window-invalid purely because the sender was backed up.

    `thread_ref` travels for the same reason and is email's alone — §6.1 gives
    it to the inbound message and nothing carried it back out, so every reply
    opened a new thread in the customer's client. Like `template`, it is a
    field one channel reads and the others ignore, which is the price of one
    send path: the alternative is the sender asking which channel it is
    talking to before it builds a call, and that is the branch that eventually
    hands a chat id to a phone network.
    """

    tenant_id: str
    channel: str
    to: str
    text: str | None = None
    template: str | None = None
    template_variables: dict[str, str] | None = None
    last_inbound_at: str | None = None
    thread_ref: str | None = None

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
                "thread_ref": self.thread_ref,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, record: str | dict[str, Any]) -> OutboundJob:
        document = json.loads(record) if isinstance(record, str) else record
        return cls(**document)

    def inbound_at(self) -> datetime | None:
        return datetime.fromisoformat(self.last_inbound_at) if self.last_inbound_at else None


class OutsideServiceWindow(Exception):
    """Freeform text was sent outside the 24-hour window (§6.2).

    Raised rather than silently converted to a template, because the template
    that would have been correct is a business decision with a per-message cost
    attached. Sending anyway is worse: Meta rejects it, and a queue of rejected
    sends reads to the tenant as an outage rather than as a policy they hit.
    """


class SecretResolver(Protocol):
    """`secret_ref` -> the signing secret it names.

    A port rather than a column, because the resolving read runs before any
    signature has been checked and must not be able to reach the secret. See
    `channels/accounts.py` and migration 0007.
    """

    def for_ref(self, secret_ref: str) -> str: ...


class ChannelAccountRegistry(Protocol):
    """Vendor address -> tenant's channel account. See the module note."""

    async def resolve(
        self, *, channel: Channel, account_ref: str
    ) -> ChannelAccount | None: ...


class InboundQueue(Protocol):
    """The Valkey stream the agent worker consumes (§3).

    The seam that keeps the webhook fast: it hands the message over and
    returns, so the ACK never waits on retrieval or a model.
    """

    async def publish(self, message: InboundMessage) -> None: ...


class EventLog(Protocol):
    """Idempotency for provider redeliveries.

    `claim` returns False when this `provider_message_id` has been seen, so a
    retry after a slow ACK does not enqueue a second turn. Both vendors retry
    on anything other than a prompt 2xx, so this is a normal path, not an edge
    case.

    `release` exists because the claim happens *before* the enqueue, and the
    enqueue can fail. Left claimed, the vendor's retry is discarded as a
    duplicate and the message is lost permanently — by the exact mechanism
    that was supposed to protect it. Releasing restores the claim so the retry
    is treated as the first delivery it effectively is.
    """

    async def claim(self, provider_message_id: str) -> bool: ...

    async def release(self, provider_message_id: str) -> None: ...


class MessagingProvider(Protocol):
    """Outbound (§6.2). One implementation per vendor per channel."""

    name: str

    async def send(
        self,
        *,
        to: str,
        text: str | None = None,
        template: str | None = None,
        template_variables: dict[str, str] | None = None,
        last_inbound_at: datetime | None = None,
        thread_ref: str | None = None,
    ) -> OutboundReceipt: ...


__all__ = [
    "Channel",
    "ChannelAccount",
    "ChannelAccountRegistry",
    "EventLog",
    "InboundMessage",
    "InboundQueue",
    "MediaRef",
    "MessagingProvider",
    "OutboundReceipt",
    "OutsideServiceWindow",
]
