"""Instagram and Messenger — design §6.1, §6.2, demo plan Task 36.

**One integration, two channels.** Meta delivers both down the same webhook and
distinguishes them by the payload's `object` field: `page` is Messenger,
`instagram` is Instagram DM. The message shape is otherwise identical, which is
why this is one adapter rather than two — and why the outbound worker had to
learn to route by channel before this could be written at all.

Three things make it a single integration rather than a simple one, and all
three are silent when they are wrong.

**Echoes.** Every message the page itself sends comes back down the webhook
with `message.is_echo: true`. A bot that does not drop them answers its own
reply, which answers that, and so on — a loop that costs a model call per turn
and reaches the customer as a wall of messages. It is the first thing anyone
integrating Messenger gets wrong, and nothing about it looks like a bug until
the bill arrives.

**Batching.** `entry` is a list and each entry's `messaging` is a list, so one
POST can carry several customers. Every other adapter here parses one message
per request, which is why `parse_inbound` returns a *list*: an adapter that
returned one would drop everyone after the first, silently, on the busiest days.

**Not everything is a message.** Delivery receipts, read receipts, reactions
and postbacks arrive in the same envelope with the same sender. Answering a
read receipt is answering a customer for looking at their phone.

**The app secret is platform-wide, not per tenant.** Twilio signs with a
per-account token; Meta signs with the *app* secret, shared by every page on
the app. Resolution still goes by page id — that is what says which tenant —
but the value that verifies is one secret for everybody, and a leak is a leak
of everyone's inbound integrity at once. Stated here because it is a real
difference from the WhatsApp path and is invisible in the code.

**The access token never leaves this module in an exception**, for the reason
the Telegram adapter learned it: Meta puts it in a query parameter, httpx's
`raise_for_status` renders the URL, and the outbound worker writes `repr(exc)`
into a dead-letter row.
"""

import hashlib
import hmac
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from moc.channels.base import (
    Channel,
    ChannelAccount,
    InboundMessage,
    MediaRef,
    OutboundReceipt,
    OutsideServiceWindow,
)

_CONFIG = "channels/meta"
_SEND_PATH = "/me/messages"


class UnknownObject(ValueError):
    """The payload names a product this adapter does not serve.

    Refused rather than defaulted. Meta ships new objects, and defaulting to
    Messenger would route them into the wrong channel's sender and reply on a
    surface the customer never used.
    """


class MetaRefused(Exception):
    """A send failed, with Meta's reason and without the access token."""


# ─────────────────────────── verification ───────────────────────────


def verify_signature(
    *,
    raw_body: bytes,
    signature: str | None,
    app_secret: str,
    config: dict[str, Any] | None = None,
) -> bool:
    """True only if `signature` is an HMAC-SHA256 of these exact bytes.

    Takes bytes, like the Twilio adapter and for the same reason: a `params:
    dict` parameter would let a caller verify a re-serialization of the request
    rather than the request.

    Fails closed on every missing input, and requires the algorithm prefix.
    Accepting a bare digest would accept a value produced by a future algorithm
    this code does not implement.
    """
    settings = config or _config()
    prefix = settings["signature_prefix"]
    if not signature or not app_secret or not signature.startswith(prefix):
        return False
    expected = hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature.removeprefix(prefix))


def verify_challenge(
    *,
    mode: str | None,
    token: str | None,
    challenge: str | None,
    expected_token: str,
    config: dict[str, Any] | None = None,
) -> str | None:
    """The subscription handshake: echo the challenge, or refuse.

    None rather than the challenge on any mismatch. Echoing regardless would
    let anyone who guessed the URL subscribe their own app to this endpoint.
    """
    settings = (config or _config())["challenge"]
    if not expected_token or not token or not challenge:
        return None
    # compare_digest on a value that is not secret, because the no-`==` rule in
    # an adapter that checks signatures has no exceptions. A whitelist of
    # "comparisons that are fine" is where the next `==` on a digest hides —
    # the same call `passwords.py` makes for the same reason.
    if not hmac.compare_digest(
        (mode or "").encode("utf-8"), str(settings["expected_mode"]).encode("utf-8")
    ):
        return None
    if not hmac.compare_digest(token.encode("utf-8"), expected_token.encode("utf-8")):
        return None
    return challenge


# ─────────────────────────── inbound ───────────────────────────


def parse_inbound(
    *,
    raw_body: bytes,
    account: ChannelAccount,
    received_at: datetime | None = None,
    config: dict[str, Any] | None = None,
) -> list[InboundMessage]:
    """A Meta webhook payload -> §6.1's normalized shape, **as a list**.

    Plural because the vendor batches. Every other adapter here returns one
    message; one that did the same for Meta would drop every customer after the
    first in a busy POST.

    Call only after `verify_signature`.
    """
    settings = config or _config()
    document = json.loads(raw_body.decode("utf-8"))

    channel = _channel_for(document.get("object"), settings)
    return [
        _message(event, account, channel, received_at, document)
        # **Filtered to this account's entries.** Meta batches per app, not per
        # page, so one delivery can carry entries for several pages — and those
        # pages can belong to different tenants. Ignoring `entry.id` would
        # attribute every message in the batch to whichever account the webhook
        # resolved first: a cross-tenant write arriving as ordinary traffic.
        for event in _events(document, account.account_ref)
        if _is_customer_turn(event)
    ]


def account_refs(*, raw_body: bytes) -> list[str]:
    """The page ids in a payload, in order, deduplicated.

    Read from an unverified body because per-page resolution leaves no
    alternative: the webhook cannot know which accounts to look up until it has
    read them. The result is used to select accounts and for nothing else — the
    messages are parsed again from the same bytes once the signature holds,
    which is the rule `webhooks.py` already follows for the Twilio address.
    """
    document = json.loads(raw_body.decode("utf-8"))
    seen: list[str] = []
    for entry in document.get("entry") or []:
        ref = str(entry.get("id", ""))
        if ref and ref not in seen:
            seen.append(ref)
    return seen


def _channel_for(obj: Any, settings: dict[str, Any]) -> Channel:
    name = settings["objects"].get(obj)
    if name is None:
        raise UnknownObject(
            f"payload object {obj!r} is not a channel this adapter serves. "
            "Defaulting to Messenger would reply on a surface the customer "
            "never used."
        )
    return Channel(name)


def _events(document: dict[str, Any], account_ref: str) -> Iterator[dict[str, Any]]:
    for entry in document.get("entry") or []:
        if str(entry.get("id", "")) != account_ref:
            # Another page's messages, in the same POST. Not this tenant's.
            continue
        # `messaging` on Messenger and Instagram both. `changes` carries
        # comments and mentions, which are a different product and not a DM.
        yield from entry.get("messaging") or []


def _is_customer_turn(event: dict[str, Any]) -> bool:
    """A message the customer sent, and not anything else on this webhook.

    Two rejections, and the first is the loop:

    - **an echo** is the page's own message coming back. Answering it answers
      ourselves, forever;
    - **a receipt, reaction or postback** is not a turn. Answering a read
      receipt answers a customer for looking at their phone.
    """
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    return not message.get("is_echo", False)


def _message(
    event: dict[str, Any],
    account: ChannelAccount,
    channel: Channel,
    received_at: datetime | None,
    document: dict[str, Any],
) -> InboundMessage:
    message = event["message"]
    return InboundMessage(
        tenant_id=account.tenant_id,
        channel=channel,
        channel_account_id=account.id,
        # `mid` is Meta's own id and is globally unique, unlike Telegram's
        # per-chat message_id — so it can be the idempotency key as it stands.
        provider_message_id=str(message.get("mid", "")),
        sender_ref=str((event.get("sender") or {}).get("id", "")),
        received_at=received_at or _sent_at(event),
        text=message.get("text") or None,
        media=_media(message),
        # The whole payload, not this one event: §6.1 says `raw` is what a
        # dispute is settled from, and the envelope carries the object and the
        # entry time that say which product delivered it.
        raw=document,
    )


def _sent_at(event: dict[str, Any]) -> datetime:
    """Meta's timestamp is milliseconds. Seconds would put every message in
    1970 and sort an agent's thread backwards."""
    stamp = event.get("timestamp")
    if not isinstance(stamp, int | float):
        return datetime.now(UTC)
    return datetime.fromtimestamp(stamp / 1000, tz=UTC)


def _media(message: dict[str, Any]) -> tuple[MediaRef, ...]:
    return tuple(
        MediaRef(
            url=str((attachment.get("payload") or {}).get("url", "")),
            content_type=attachment.get("type"),
        )
        for attachment in message.get("attachments") or []
        if isinstance(attachment, dict)
    )


# ─────────────────────────── outbound ───────────────────────────


class MetaMessenger:
    """The outbound half, for both Messenger and Instagram.

    One class because the Send API is the same on both surfaces; the page id
    and the token differ per tenant and per channel, which is what the channel
    account carries.
    """

    name = "meta_messenger"

    def __init__(
        self,
        *,
        page_id: str,
        access_token: str,
        config: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = config or _config()
        self._settings = settings
        self._page_id = page_id
        # Held for the query parameter, never logged and never in an exception.
        self._token = access_token
        self._window = timedelta(hours=settings["service_window_hours"])
        http = settings["http"]
        self._client = httpx.AsyncClient(
            base_url=settings["api_base"],
            timeout=httpx.Timeout(
                connect=http["connect_timeout_seconds"],
                read=http["read_timeout_seconds"],
                write=http["read_timeout_seconds"],
                pool=http["connect_timeout_seconds"],
            ),
            transport=transport,
        )

    async def send(
        self,
        *,
        to: str,
        text: str | None = None,
        template: str | None = None,
        template_variables: dict[str, str] | None = None,
        last_inbound_at: datetime | None = None,
        now: datetime | None = None,
    ) -> OutboundReceipt:
        """Send one message, refusing freeform text outside the window (§6.2).

        Refused rather than silently tagged. Outside 24 hours a reply needs a
        message tag; choosing one is a business decision with a per-message
        cost attached, and Instagram allows fewer of them than Messenger does.
        Sending untagged anyway is worse — Meta rejects it, and a queue of
        rejected sends reads as an outage.
        """
        if not self._within_window(last_inbound_at, now):
            raise OutsideServiceWindow(
                "§6.2: outside the "
                f"{self._settings['service_window_hours']}-hour window a Meta "
                "reply needs a message tag. Which tag is a business decision "
                "with a cost, and Instagram allows fewer than Messenger."
            )

        try:
            response = await self._client.post(
                _SEND_PATH,
                params={"access_token": self._token},
                json={
                    "recipient": {"id": to},
                    "message": {"text": text or ""},
                    "messaging_type": self._settings["messaging_type"],
                },
            )
            response.raise_for_status()
            document = response.json()
        except httpx.HTTPStatusError as refused:
            # The URL carries the token in a query parameter, and
            # `raise_for_status` renders the URL. This string ends up in a
            # dead-letter row.
            raise MetaRefused(_reason(refused.response)) from None
        except httpx.HTTPError as failed:
            raise MetaRefused(
                f"meta request failed: {type(failed).__name__}"
            ) from None

        return OutboundReceipt(
            provider_message_id=str(document.get("message_id", "")),
            # Meta answers with ids rather than a delivery status; the receipt
            # arrives later on the webhook.
            status="sent" if document.get("message_id") else "failed",
        )

    def _within_window(
        self, last_inbound_at: datetime | None, now: datetime | None
    ) -> bool:
        """No previous inbound means no window was ever opened.

        First contact is therefore refused, which is Meta's rule and also the
        correct default: an unsolicited message from a business is the thing
        the window exists to prevent.
        """
        if last_inbound_at is None:
            return False
        return (now or datetime.now(UTC)) - last_inbound_at < self._window

    async def aclose(self) -> None:
        await self._client.aclose()


def _reason(response: httpx.Response) -> str:
    """Meta's own `error.message`, or the status code. Never the URL."""
    try:
        described = (response.json().get("error") or {}).get("message")
    except (ValueError, AttributeError):
        described = None
    return f"meta refused: {described or response.status_code}"


def _config() -> dict[str, Any]:
    from moc.config_store import load

    return load(_CONFIG)


__all__ = [
    "MetaMessenger",
    "account_refs",
    "MetaRefused",
    "UnknownObject",
    "parse_inbound",
    "verify_challenge",
    "verify_signature",
]
