"""WhatsApp via Twilio — design §6.2.

**`verify_signature` is flagged for line-by-line human review.** Two properties
carry the weight, and both fail silently when they are wrong:

**Constant-time comparison.** A timing side channel on an HMAC check is
exploitable and behaviourally invisible — `==` passes every functional test
anyone will ever write against this function. The comparison is therefore
isolated in `_matches`, three lines with no equality operator in them, so a
test can assert its shape exactly instead of grepping hopefully. The body-hash
check in `_body_hash_matches` gets the same treatment: guarding one
secret-dependent comparison and not the other leaves the same oracle open one
line further down.

**Verification against the raw bytes.** `verify_signature` takes `bytes` and
has no parameter for a parsed body, deliberately. The classic failure is a
convenient `params: dict` argument — the caller has already parsed the body and
passing it along feels like avoiding duplicate work — after which the check
covers a re-serialization of the request rather than the request. On Twilio's
JSON path that breaks loudly against real traffic; on the form path it can
break quietly, still returning True while no longer asserting anything about
what arrived. The bug is unrepresentable if the bytes are the only input.

Twilio's algorithm is theirs, not a design choice here: for form-encoded posts,
HMAC-SHA1 over the full URL with the parameters appended in sorted order; for
JSON, HMAC over a URL that carries a SHA-256 of the exact body. SHA-1 is their
wire contract — it is a MAC construction, not a collision-sensitive digest, and
we do not get a vote.

No endpoint string appears here (§2.4's guard). The API base is config, which
is also what makes the §6.2 migration to Meta's Graph API a new adapter plus a
config edit rather than a search for everything that assumed Twilio.

**`TwilioTypingIndicator` is a second client on purpose** (§2.5, Task 40). It
talks to a different host, at a different API version, in a different encoding
from everything above — `messaging.twilio.com/v3` taking JSON where
`api.twilio.com/2010-04-01/Accounts/{sid}` takes form data — so folding it into
`TwilioWhatsApp` would give that class two base URLs and two content types and
no way to tell at a glance which call used which.

Two things about it are not technical and are worth reading before it is
switched on:

- **It marks the customer's message read, and Twilio does not separate the
  two.** Every message acknowledged gets a blue tick, including the ones that
  end in a handoff, and a read receipt followed by silence is a worse signal
  than no receipt at all. That is why the config carries `enabled`.
- **Its authentication has never been verified against the real host.** Twilio
  documents an API key/secret pair for this resource; this adapter holds an
  account SID and auth token, and whether those authenticate against
  `messaging.twilio.com/v3` is not stated in the vendor docs and cannot be
  tested without an account. `scripts/preflight.py` answers it on the first
  host that has credentials: a 401 says the scheme is wrong, anything else says
  it is not.
"""

import base64
import binascii
import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlparse

import httpx

from moc.channels.base import (
    Channel,
    ChannelAccount,
    InboundMessage,
    MediaRef,
    OutboundReceipt,
    OutsideServiceWindow,
)

_SID = "MessageSid"
_FROM = "From"
_BODY = "Body"
_NUM_MEDIA = "NumMedia"
_MEDIA_URL = "MediaUrl"
_MEDIA_TYPE = "MediaContentType"


# ─────────────────────────── signature verification ───────────────────────────


def _matches(expected: str, provided: str) -> bool:
    """The only comparison in this module that an attacker can time.

    `hmac.compare_digest`, never `==`. The difference is invisible in every
    test that asserts behaviour, which is exactly why it survives review unless
    someone is looking for it — so it lives alone here, where looking for it is
    three lines of work.
    """
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def _body_hash_matches(url: str, raw_body: bytes, parameter: str) -> bool:
    """Twilio's JSON path: the URL carries a SHA-256 of the exact bytes sent.

    Compared in constant time for the same reason as the signature — the hash
    is derived from content the caller controls and checked against a value
    inside the signed URL, so a fast-exit comparison is an oracle for guessing
    it byte by byte.
    """
    query = parse_qs(urlparse(url).query)
    claimed = query.get(parameter, [""])[0]
    if not claimed:
        return False
    return _matches(hashlib.sha256(raw_body).hexdigest(), claimed)


def verify_signature(
    *,
    url: str,
    raw_body: bytes,
    content_type: str,
    signature: str | None,
    auth_token: str,
    config: dict[str, Any] | None = None,
) -> bool:
    """True only if `signature` was produced by `auth_token` over this request.

    Takes bytes. There is no parameter for a parsed body and there must never
    be one — see the module docstring.

    Fails closed on every missing input. A blank signature header and a blank
    auth token are both configuration or attack, never "nothing to check": the
    tempting `if signature:` guard turns the whole control into something an
    attacker disables by omitting a header, and HMAC with an empty key is a
    perfectly valid HMAC against a key anyone can guess.
    """
    settings = config or _config()
    if not signature or not auth_token:
        return False

    if content_type.split(";")[0].strip() == settings["json_content_type"]:
        # The signed URL already carries the body hash, so the body is covered
        # only if that hash is checked. Skipping it would leave a signature that
        # authenticates a URL and says nothing about the payload.
        if not _body_hash_matches(url, raw_body, settings["body_hash_param"]):
            return False
        canonical = url
    else:
        try:
            params = parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True)
        except UnicodeDecodeError:
            return False
        # Twilio's rule: sorted by key, concatenated onto the URL with no
        # separators. Parsed from the bytes we received, not from anything a
        # framework handed us.
        canonical = url + "".join(key + value for key, value in sorted(params))

    digest = hmac.new(
        auth_token.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha1,  # noqa: S324 - Twilio's wire contract; HMAC, not a bare digest
    ).digest()
    try:
        expected = base64.b64encode(digest).decode("ascii")
    except (binascii.Error, UnicodeDecodeError):  # pragma: no cover - defensive
        return False
    return _matches(expected, signature)


# ─────────────────────────── inbound normalization ───────────────────────────


def parse_inbound(
    *,
    raw_body: bytes,
    account: ChannelAccount,
    received_at: datetime | None = None,
    config: dict[str, Any] | None = None,
) -> InboundMessage:
    """Twilio's form post -> §6.1's normalized shape.

    Call only after `verify_signature`. Everything here trusts its input, which
    is fine once the signature has established who sent it and not before.
    """
    settings = config or _config()
    fields = dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
    prefix = settings["address_prefix"]

    return InboundMessage(
        tenant_id=account.tenant_id,
        channel=Channel.whatsapp,
        channel_account_id=account.id,
        provider_message_id=fields.get(_SID, ""),
        sender_ref=_strip(fields.get(_FROM, ""), prefix),
        received_at=received_at or datetime.now(UTC),
        # A voice note or a sticker carries no Body. That is a turn, not an
        # error — the media is the message.
        text=fields.get(_BODY) or None,
        media=_media(fields),
        # Stored whole and never parsed downstream (§6.1).
        raw=fields,
    )


def _media(fields: dict[str, str]) -> tuple[MediaRef, ...]:
    try:
        count = int(fields.get(_NUM_MEDIA, "0"))
    except ValueError:
        count = 0
    return tuple(
        MediaRef(
            url=fields.get(f"{_MEDIA_URL}{index}", ""),
            content_type=fields.get(f"{_MEDIA_TYPE}{index}"),
        )
        for index in range(count)
    )


def _strip(address: str, prefix: str) -> str:
    """Drop the vendor prefix so nothing downstream learns who carried it."""
    return address[len(prefix) :] if address.startswith(prefix) else address


# ─────────────────────────── outbound ───────────────────────────


class TwilioWhatsApp:
    """The outbound half of §6.2.

    Holds the service-window rule and nothing else. Rate limiting, backoff and
    dead-lettering belong to the outbound worker, where one token bucket covers
    every message for a tenant — per-adapter buckets would each allow the full
    rate, which is the same as having no limit.
    """

    name = "twilio_whatsapp"

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        sender: str,
        config: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = config or _config()
        self._settings = settings
        self._account_sid = account_sid
        # The tenant's own WhatsApp number, from their channel_accounts row —
        # never a platform-wide value. One sender for every tenant would put
        # one tenant's replies under another tenant's brand.
        self._sender = sender
        self._window = timedelta(hours=settings["service_window_hours"])
        self._prefix = settings["address_prefix"]
        http = settings["http"]
        self._client = httpx.AsyncClient(
            base_url=f"{settings['api_base']}/Accounts/{account_sid}",
            auth=(account_sid, auth_token),
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
        """Send one message, refusing freeform text outside the window (§6.2).

        `template` is a Twilio content sid, passed through and never authored
        here — §6.2 says proxy their template management rather than building
        a second one that has to stay in sync with Meta's approvals.
        """
        if template is None and not self._within_window(last_inbound_at, now):
            raise OutsideServiceWindow(
                "§6.2: outside the "
                f"{self._settings['service_window_hours']}-hour customer service "
                "window only approved templates may be sent. Meta rejects freeform "
                "text here, and a queue of rejected sends reads as an outage."
            )

        payload: dict[str, str] = {
            "To": f"{self._prefix}{to}",
            "From": f"{self._prefix}{self._sender}",
        }
        if template is not None:
            payload["ContentSid"] = template
            if template_variables:
                import json

                payload["ContentVariables"] = json.dumps(template_variables)
        else:
            payload["Body"] = text or ""

        response = await self._client.post("/Messages.json", data=payload)
        response.raise_for_status()
        document = response.json()
        return OutboundReceipt(
            provider_message_id=document.get("sid", ""), status=document.get("status", "")
        )

    def _within_window(self, last_inbound_at: datetime | None, now: datetime | None) -> bool:
        """No previous inbound means no window was ever opened.

        First contact is therefore template-only, which is Meta's rule and also
        the correct default: an unsolicited freeform message from a business is
        the thing the window exists to prevent.
        """
        if last_inbound_at is None:
            return False
        return (now or datetime.now(UTC)) - last_inbound_at < self._window

    async def aclose(self) -> None:
        await self._client.aclose()


class TwilioTypingIndicator:
    """"Seen, typing" on the turn path — §2.5, demo plan Task 40.

    Not a message: no content, no service window, and deliberately not on the
    outbound queue. A courtesy that arrives after the reply it was meant to
    precede is worse than none, and the token bucket is exactly what would
    delay it.

    **Nothing here raises.** `typing` returns whether the indicator was
    accepted, and every failure — refusal, timeout, no route — is False. It
    runs on the turn path, so an exception would fail a turn that was otherwise
    fine and the customer would lose the answer to save the hint. The cost of
    that choice is that a permanently broken indicator is silent; the preflight
    is what asks the question out loud.
    """

    name = "twilio_typing"

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        config: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = (config or _config())["typing_indicator"]
        self._settings = settings
        self._enabled = bool(settings["enabled"])
        http = settings["http"]
        self._client = httpx.AsyncClient(
            base_url=settings["api_base"],
            # The SID/token pair the adapter already holds. See the module
            # docstring: the vendor documents an API key/secret for this
            # resource and does not say whether these also work.
            auth=(account_sid, auth_token),
            timeout=httpx.Timeout(
                connect=http["connect_timeout_seconds"],
                read=http["read_timeout_seconds"],
                write=http["read_timeout_seconds"],
                pool=http["connect_timeout_seconds"],
            ),
            transport=transport,
        )

    #: None because one call covers a whole turn — Twilio clears the indicator
    #: on delivery or after 25 seconds, and no measured turn comes near that.
    #: Telegram's clears after five and declares an interval instead.
    resend_every_seconds: float | None = None

    async def typing(self, *, message_id: str, sender_ref: str = "") -> bool:
        """Show the indicator for the turn answering `message_id`.

        One call covers a whole turn: Twilio clears it on delivery or after
        `clears_after_seconds`, and §2.5's p95 turn is well inside that.

        `sender_ref` is accepted and unused. Twilio addresses the message it
        is a reply to; Telegram addresses a chat. The port carries both rather
        than the union of two signatures, because a worker that had to ask
        which channel it was holding before it could call this would be the
        thing this seam exists to avoid.
        """
        if not self._enabled or not message_id:
            return False
        try:
            response = await self._client.post(
                self._settings["path"],
                json={"messageId": message_id, "channel": self._settings["channel"]},
            )
        except httpx.HTTPError:
            return False
        return response.is_success

    async def aclose(self) -> None:
        await self._client.aclose()


def _config() -> dict[str, Any]:
    from moc.config_store import load

    return load("channels/whatsapp")


__all__ = [
    "TwilioTypingIndicator",
    "TwilioWhatsApp",
    "parse_inbound",
    "verify_signature",
]
