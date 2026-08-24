"""Telegram — design §6.1, §6.2, demo plan Task 35.

The cheapest channel to demo: no business verification, no template approval,
no 24-hour window. It is also the one where the security model is weakest, and
two things about it are worth reading before this file is trusted.

**Telegram does not sign the body.** `setWebhook` takes a `secret_token` and
every delivery returns it in a header, so `verify_secret` is a constant-time
comparison of a *bearer token* — it proves the sender knows the secret, not
that the body is unmodified, and it is replayable by anyone who obtains the
header. The Twilio adapter can do better because Twilio signs. Writing
something here called `verify_signature` that verified no signature would be
worse than a comparison that is honest about what it proves.

**The bot token is a secret in a URL.** The Bot API puts it in the path —
`{api_base}/bot<TOKEN>/sendMessage` — which means it appears in any exception
carrying a URL. `httpx.Response.raise_for_status` builds exactly such a
message, `httpx.ConnectError` carries the request, and the outbound worker
writes `repr(exc)` into a dead-letter row. Three pieces of ordinary code, none
of them wrong on its own, and the result is a live bot token sitting in a queue
where anyone who can read the queue can read it.

So **no vendor exception leaves this module**. Everything is caught and
re-raised as `TelegramRefused` carrying the vendor's own reason and no URL,
and a test asserts the token is absent from both `str` and `repr` of what
escapes. The reason survives because a tenant with a misconfigured chat needs
to know it was "chat not found"; only the secret is dropped.

**A Telegram update does not say which bot received it.** There is no bot id in
the payload, so the webhook path carries the account reference — one path per
bot, which is the vendor's own shape. That reference is guessable on purpose:
it selects which secret to check and is not itself a credential, exactly as the
WhatsApp address is.
"""

import hmac
import json
from datetime import UTC, datetime
from typing import Any

import httpx

from moc.channels.base import Channel, ChannelAccount, InboundMessage, MediaRef, OutboundReceipt

_CONFIG = "channels/telegram"

#: The update keys that carry a customer turn. Telegram delivers edits, channel
#: posts, poll answers and callback queries down the same webhook, and treating
#: one as a message would answer a customer's edit of a question they already
#: asked.
_MESSAGE_KEYS = ("message",)

#: Media a customer can send instead of words. A voice note is a turn, not an
#: error — the media is the message.
_MEDIA_KEYS = ("photo", "voice", "audio", "document", "video", "video_note", "sticker")

_SEND_PATH = "/sendMessage"


class NotAMessage(ValueError):
    """The update is not a customer turn. Acknowledged, never processed."""


class TelegramRefused(Exception):
    """A send failed, with the vendor's reason and without the bot token.

    Deliberately not an httpx exception. See the module docstring: every httpx
    error carries the request URL, and the URL is where the token lives.
    """


def verify_secret(*, presented: str | None, expected: str) -> bool:
    """True only if the header matches the secret this bot was registered with.

    Constant time, and failing closed on every missing input. A blank header
    and a blank stored secret are both configuration or attack, never "nothing
    to check" — and without the guard, two empty strings compare equal and an
    unconfigured bot accepts anything.

    Named `verify_secret` rather than `verify_signature` on purpose. It is a
    bearer token: it says the sender knows a secret, and nothing about the body.
    """
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def parse_inbound(
    *,
    raw_body: bytes,
    account: ChannelAccount,
    received_at: datetime | None = None,
    config: dict[str, Any] | None = None,
) -> InboundMessage:
    """A Telegram update -> §6.1's normalized shape.

    Call only after `verify_secret`. Everything here trusts its input, which is
    fine once the secret has established who sent it and not before.
    """
    document = json.loads(raw_body.decode("utf-8"))
    message = next(
        (document[key] for key in _MESSAGE_KEYS if isinstance(document.get(key), dict)),
        None,
    )
    if message is None:
        raise NotAMessage(
            f"update {document.get('update_id')} carries no message — Telegram "
            "delivers edits, channel posts and callback queries down this same "
            "webhook, and none of them is a customer turn"
        )

    chat = message.get("chat") or {}
    # The chat id, not the user id. A reply goes to a chat, and in a group they
    # differ — replying to the user id would send a private message to somebody
    # who asked in public.
    chat_id = str(chat.get("id", ""))

    return InboundMessage(
        tenant_id=account.tenant_id,
        channel=Channel.telegram,
        channel_account_id=account.id,
        # Chat and message together. Telegram's `message_id` restarts per chat,
        # so it alone is not an identity: two customers can hold the same one
        # at the same moment, and the idempotency guard would discard the
        # second as a redelivery of the first — a message lost by the mechanism
        # that exists to protect it.
        provider_message_id=f"tg:{chat_id}:{message.get('message_id', '')}",
        sender_ref=chat_id,
        received_at=received_at or _sent_at(message),
        text=message.get("text") or message.get("caption") or None,
        media=_media(message),
        # Stored whole and never parsed downstream (§6.1).
        raw=document,
    )


def _sent_at(message: dict[str, Any]) -> datetime:
    """Telegram's own timestamp, in UTC. Falls back to now if it is missing —
    a turn with no time is still a turn, and inventing 1970 would put it at the
    top of every thread an agent reads."""
    stamp = message.get("date")
    if not isinstance(stamp, int | float):
        return datetime.now(UTC)
    return datetime.fromtimestamp(stamp, tz=UTC)


def _media(message: dict[str, Any]) -> tuple[MediaRef, ...]:
    """File ids, not URLs.

    Telegram serves media through a second call that embeds the bot token in
    the path, so a URL here would be a token in a database column. The id is
    what a later fetch needs and is useless to anyone without the token.
    """
    found: list[MediaRef] = []
    for key in _MEDIA_KEYS:
        entry = message.get(key)
        if isinstance(entry, list) and entry:
            entry = entry[-1]  # photos arrive as sizes; the last is the largest
        if isinstance(entry, dict) and entry.get("file_id"):
            found.append(MediaRef(url=str(entry["file_id"]), content_type=key))
    return tuple(found)


class TelegramBot:
    """The outbound half.

    Holds no service window, because Telegram has none. The WhatsApp adapter
    refuses freeform text outside 24 hours; a Telegram sender that inherited
    that refusal would reject every reply after a day of quiet, and the refusal
    would read as an outage rather than as a policy.

    Rate limiting, backoff and dead-lettering stay in the outbound worker,
    where one token bucket covers a tenant.
    """

    name = "telegram_bot"

    def __init__(
        self,
        *,
        token: str,
        config: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = config or _config()
        self._settings = settings
        # Held for the path, never logged and never put in an exception.
        self._token = token
        http = settings["http"]
        self._client = httpx.AsyncClient(
            base_url=f"{settings['api_base']}/bot{token}",
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
        thread_ref: str | None = None,
        now: datetime | None = None,
    ) -> OutboundReceipt:
        """Send one message.

        `template` and `last_inbound_at` are accepted and unused: they are part
        of `MessagingProvider`, and Telegram has neither templates nor a
        window. Accepting and ignoring them keeps one send path (§6.2) rather
        than making the worker ask which channel it is talking to before it
        builds a call.
        """
        try:
            response = await self._client.post(
                _SEND_PATH,
                json={
                    "chat_id": to,
                    "text": text or "",
                    # No parse_mode. Composition writes plain sentences, and
                    # asking Telegram to parse markdown would italicise half a
                    # reply from an underscore in an email address — while a
                    # number with punctuation stuck to it becomes invisible to
                    # the grounding gate.
                },
            )
            response.raise_for_status()
            document = response.json()
        except httpx.HTTPStatusError as refused:
            # `raise_for_status` puts the URL — and therefore the token — in
            # its message. Re-raised with the vendor's own reason and nothing
            # else, because this string ends up in a dead-letter row.
            raise TelegramRefused(_reason(refused.response)) from None
        except httpx.HTTPError as failed:
            # ConnectError, ReadTimeout and friends carry the request, which
            # carries the URL. `type(failed).__name__` says what happened
            # without saying where.
            raise TelegramRefused(
                f"telegram request failed: {type(failed).__name__}"
            ) from None

        result = document.get("result") or {}
        return OutboundReceipt(
            provider_message_id=str(result.get("message_id", "")),
            # Telegram answers `ok` rather than a delivery status; there is no
            # queued/sent/delivered ladder to report.
            status="ok" if document.get("ok") else "failed",
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _reason(response: httpx.Response) -> str:
    """The vendor's `description`, or the status code if it sent none.

    Never the URL and never the request. A tenant whose chat id is wrong needs
    to read "chat not found"; nobody needs the token that asked.
    """
    try:
        described = response.json().get("description")
    except (ValueError, AttributeError):
        described = None
    return f"telegram refused: {described or response.status_code}"


def _config() -> dict[str, Any]:
    from moc.config_store import load

    return load(_CONFIG)


__all__ = [
    "NotAMessage",
    "TelegramBot",
    "TelegramRefused",
    "parse_inbound",
    "verify_secret",
]
