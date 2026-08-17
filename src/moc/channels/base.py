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

**A note on `ChannelAccountRegistry`, for whoever wires the table.** The design
lists `channel_accounts` as tenant-scoped (§5), and every tenant-scoped table
gets RLS. This lookup is the one that cannot: it runs *before* a tenant context
exists, because its whole job is to establish which tenant a message belongs
to. Resolving it under RLS with no tenant set returns nothing, and setting a
tenant first requires the answer. The options are a narrowly-privileged lookup
role, or a policy admitting a bootstrap read of only the columns needed to
resolve. That is a decision with security consequences and it deserves its own
review, so this stays a Protocol until it gets one — a seam, not a stub, in the
same shape Task 14 used for retrieval.
"""

import enum
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

    `signing_secret` is per account and never platform-wide. A shared token
    means any party who has seen one tenant's credentials can forge every other
    tenant's inbound traffic — including that tenant themselves, which is the
    version that ends up in a dispute.
    """

    id: UUID
    tenant_id: UUID
    channel: Channel
    #: The vendor-facing address, vendor prefixes stripped. For WhatsApp this
    #: is the business number in E.164.
    account_ref: str
    signing_secret: str


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


class OutsideServiceWindow(Exception):
    """Freeform text was sent outside the 24-hour window (§6.2).

    Raised rather than silently converted to a template, because the template
    that would have been correct is a business decision with a per-message cost
    attached. Sending anyway is worse: Meta rejects it, and a queue of rejected
    sends reads to the tenant as an outage rather than as a policy they hit.
    """


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
