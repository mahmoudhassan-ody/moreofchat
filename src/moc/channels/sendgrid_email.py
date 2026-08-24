"""Email via SendGrid — design §6.1, §6.2, demo plan Task 37.

The last channel, and the one least like the others. Four things about it are
worth reading before this file is trusted.

**Nothing signs the body.** Twilio HMACs the request and Meta HMACs the
payload. SendGrid's Inbound Parse posts an unauthenticated multipart form to
whatever URL the receiving hostname is configured with, and the only credential
the vendor offers is HTTP basic auth embedded in that URL. So `verify_secret`
is a constant-time comparison of a *bearer credential*: it proves the sender
knows a secret and says nothing about the body. It is named for what it proves,
exactly as Telegram's is, because a function called `verify_signature` that
verified no signature would be worse than an honest comparison.

**`From:` is not authenticated by SPF, and `From:` is the identity.** SPF
authenticates the envelope sender — the `Return-Path`, which no mail client
displays. Anyone who controls a domain can pass SPF on it while writing
`From: registrar@sinai.edu.eg` in a header. Downstream, `sender_ref` is what
the conversation store threads on, so a forged From does not merely mislabel a
message: it selects a real student's conversation. `authentication` therefore
computes DMARC-style alignment — SPF *or* DKIM passing on a domain that matches
the From domain — and `parse_inbound` refuses anything that fails it.

Either signal, not both, because that is what makes forwarded mail work: a
forwarder rewrites the envelope and SPF fails at the new hop, while the DKIM
signature travels with the message. Requiring SPF would refuse every student
whose university forwards their mail.

**Robots answer robots.** An out-of-office autoresponder replies to our reply,
we answer it, it replies again. This is Meta's echo, except the other end
belongs to somebody else and cannot be fixed from here — so it is guarded from
both sides: RFC 3834's markers are refused on the way in, and every reply
leaves with `Auto-Submitted: auto-generated` so that a correctly-behaved
autoresponder does not answer us.

**The customer's words arrive wrapped in everything they have said before.** A
reply quotes the whole thread. Left in, every turn re-asks all the earlier
questions, the cost grows with the length of the conversation, and figures this
system composed in an earlier reply come back looking like something the
customer stated.

**What this adapter does not keep.** `raw` is the whole payload everywhere else
(§6.1). Here it is not: the HTML alternative is dropped and the text is capped,
because a mail thread has no size bound and this box has 3.3 GB. Attachment
bytes are dropped too and only the filename is kept — Inbound Parse hands them
over once and stores nothing, so what is dropped here is gone. The agent can
still see that a certificate was sent and ask for it again, which is the honest
half of a capability this platform does not yet have.
"""

import hashlib
import hmac
import json
import re
from base64 import b64decode
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parseaddr
from html import unescape
from typing import Any

import httpx

from moc.channels.base import (
    Channel,
    ChannelAccount,
    InboundMessage,
    MediaRef,
    OutboundReceipt,
)

_CONFIG = "channels/email"
_SEND_PATH = "/v3/mail/send"

#: SendGrid's own field names (Inbound Parse). The vendor's wire contract.
_HEADERS = "headers"
_FROM = "from"
_SUBJECT = "subject"
_TEXT = "text"
_HTML = "html"
_ENVELOPE = "envelope"
_SPF = "SPF"
_DKIM = "dkim"
_CHARSETS = "charsets"

#: Dropped from `raw` — see the module docstring.
_NOT_KEPT = (_HTML,)

#: RFC 3834 and the conventions that grew around it. Presence of any of these
#: means a machine sent the message, and answering a machine is a loop.
_MACHINE_HEADERS = ("x-autoreply", "list-id", "list-unsubscribe", "auto-submitted")
_MACHINE_PRECEDENCE = ("bulk", "list", "junk", "auto_reply")
#: RFC 3834's own exemption: ordinary mail may carry `Auto-Submitted: no`.
_NOT_AUTOMATIC = "no"

_ANGLE = re.compile(r"<([^>]+)>")
_TAGS = re.compile(r"<[^>]+>")
_BLOCKS = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAKS = re.compile(r"</(p|div|tr|h[1-6])\s*>|<br\s*/?>", re.IGNORECASE)
#: `{@example.com : pass}{@other.com : fail}` — SendGrid's shape, one per
#: signature, and a message can carry more than one.
_DKIM_RESULT = re.compile(r"\{\s*@([^\s:]+)\s*:\s*(\w+)\s*\}")


class NotACustomerTurn(ValueError):
    """An autoresponder, a mailing list or a bounce. Accepted, never answered."""


class Unauthenticated(ValueError):
    """The From address is not backed by SPF or DKIM on its own domain.

    Refused rather than answered, because `sender_ref` selects a conversation:
    an unauthenticated From is a request to be handed somebody else's thread.
    """


class EmailRefused(Exception):
    """A send failed, with the vendor's reason and no credential in it."""


@dataclass(frozen=True)
class Authentication:
    """What the two vendor-supplied results say about the From address."""

    from_domain: str | None
    spf: str | None
    spf_domain: str | None
    dkim: tuple[tuple[str, str], ...]
    aligned: bool
    reason: str


def verify_secret(
    *, presented: str | None, expected: str, config: dict[str, Any] | None = None
) -> bool:
    """True only if the request carries the basic credential this mailbox was
    registered with.

    `expected` is the whole `user:password` pair, because that is what basic
    auth encodes and splitting it here would mean comparing the two halves
    separately — two comparisons where one will do, and the shorter one leaks
    first.

    Constant time, and failing closed on every missing input. A blank header
    and a blank stored credential are both configuration or attack, never
    "nothing to check" — without the guard, two empty strings compare equal and
    a mailbox whose credential was never set accepts every stranger who omits
    the header.
    """
    settings = config or _config()
    scheme = settings["auth_scheme"]
    if not presented or not expected or not presented.startswith(scheme):
        return False
    encoded = presented[len(scheme) :].strip()
    if not encoded:
        return False
    try:
        offered = b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(offered, expected.encode("utf-8"))


# ─────────────────────── who the message is actually from ───────────────────────


def authentication(
    *,
    from_address: str,
    spf: str | None,
    dkim: str | None,
    envelope_from: str | None,
) -> Authentication:
    """DMARC-style alignment over the two results SendGrid supplies.

    No DNS is consulted: this runs inside the webhook, where a network call is
    a duplicate delivery rather than a latency problem. So this is not DMARC —
    it never learns the domain's published policy, and it applies the check to
    every sender instead of only to those who asked for it. That is the
    stricter direction, which is the right one to be wrong in when the value
    being authenticated is the key a conversation is looked up by.
    """
    from_domain = _domain(from_address)
    spf_domain = _domain(envelope_from or "")
    signatures = tuple(
        (domain.lower(), result.lower()) for domain, result in _DKIM_RESULT.findall(dkim or "")
    )

    if not from_domain:
        return Authentication(None, spf, spf_domain, signatures, False, "no From address")

    if (spf or "").lower() == "pass" and _aligned(spf_domain, from_domain):
        return Authentication(from_domain, spf, spf_domain, signatures, True, "SPF aligned")

    for domain, result in signatures:
        if result == "pass" and _aligned(domain, from_domain):
            return Authentication(from_domain, spf, spf_domain, signatures, True, "DKIM aligned")

    return Authentication(
        from_domain,
        spf,
        spf_domain,
        signatures,
        False,
        f"nothing aligned with {from_domain}: SPF {spf or 'none'} on "
        f"{spf_domain or 'no envelope sender'}, DKIM {signatures or 'none'}",
    )


def _aligned(candidate: str | None, from_domain: str) -> bool:
    """Equal, or one a subdomain of the other, on label boundaries.

    The boundary matters: `notsinai.edu.eg` ends with `sinai.edu.eg` as a
    string and belongs to somebody else. Subdomains are allowed in both
    directions because SendGrid signs from a subdomain of the sending domain,
    so strict alignment would refuse a tenant's own mail coming back.

    Stricter than DMARC's relaxed mode, which aligns anything sharing an
    organisational domain — that needs the public suffix list, and being
    stricter here costs a refusal we can explain rather than an acceptance we
    cannot.
    """
    if not candidate or not from_domain:
        return False
    if candidate == from_domain:
        return True
    return candidate.endswith(f".{from_domain}") or from_domain.endswith(f".{candidate}")


def _domain(address: str) -> str | None:
    _, addr = parseaddr(address or "")
    if "@" not in addr:
        return None
    return addr.rsplit("@", 1)[1].lower().strip(">").strip() or None


# ─────────────────────────── inbound ───────────────────────────


def parse_inbound(
    *,
    raw_body: bytes,
    content_type: str,
    account: ChannelAccount,
    received_at: datetime | None = None,
    config: dict[str, Any] | None = None,
) -> InboundMessage:
    """A SendGrid Inbound Parse POST -> §6.1's normalized shape.

    Raises `NotACustomerTurn` for machine mail and `Unauthenticated` for a From
    address neither SPF nor DKIM stands behind. Both are refusals of a message
    that arrived intact, which is why they are exceptions rather than a `None`
    the caller might forget to check.
    """
    settings = config or _config()
    fields, media = _form(raw_body, content_type)
    charsets = _charsets(fields)
    text_fields = {name: _decode(value, charsets.get(name)) for name, value in fields.items()}

    original = _original_headers(text_fields.get(_HEADERS, ""))
    _refuse_machine_mail(original)

    from_address = original.get("From") or text_fields.get(_FROM, "")
    result = authentication(
        from_address=from_address,
        spf=text_fields.get(_SPF),
        dkim=text_fields.get(_DKIM),
        envelope_from=_envelope_from(text_fields.get(_ENVELOPE)),
    )
    if not result.aligned:
        raise Unauthenticated(
            f"{from_address!r} is not authenticated: {result.reason}. The From "
            "address is what a conversation is looked up by, so an unverified "
            "one is a request to be handed somebody else's thread."
        )

    subject = _words(text_fields.get(_SUBJECT, ""))
    body = _body(text_fields, settings)
    message_id = _bare(original.get("Message-ID")) or _fallback_id(raw_body)

    return InboundMessage(
        tenant_id=account.tenant_id,
        channel=Channel.email,
        channel_account_id=account.id,
        # The RFC Message-ID: globally unique by design, and therefore the
        # idempotency key the other adapters have to construct.
        provider_message_id=message_id,
        sender_ref=_address(from_address),
        # Arrival, not the `Date:` header. This is the one adapter that ignores
        # the timestamp in the payload: Telegram's and Meta's come from the
        # platform, `Date:` comes from the sender's own machine, and a skewed
        # clock would pin a message to the top of the agent's thread forever.
        received_at=received_at or datetime.now(UTC),
        thread_ref=_thread_ref(original, message_id),
        text=body or subject or None,
        media=media,
        # Not the whole payload, unlike every other channel here — see the
        # module docstring for what is dropped and why.
        raw={name: value for name, value in text_fields.items() if name not in _NOT_KEPT},
    )


def _form(raw_body: bytes, content_type: str) -> tuple[dict[str, bytes], tuple[MediaRef, ...]]:
    """The multipart body, as bytes per field.

    Parsed with the standard library's MIME parser rather than a form library
    because the same parser reads the original message's headers a moment
    later, and one parser is one set of folding and encoded-word rules.

    Bytes, not text: SendGrid sends each field in the charset the original
    message used and names it in `charsets`, so decoding here would assume the
    one thing this adapter must not.
    """
    document = BytesParser(policy=policy.compat32).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: "
        + content_type.encode("utf-8", "replace")
        + b"\r\n\r\n"
        + raw_body
    )
    if not document.is_multipart():
        raise ValueError(
            "the body is not multipart/form-data — Inbound Parse posts a form, "
            "so this is either a misconfigured route or not SendGrid at all"
        )

    fields: dict[str, bytes] = {}
    media: list[MediaRef] = []
    for part in _parts(document):
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        if filename:
            # The name and type only. The bytes are dropped here; see the
            # module docstring.
            media.append(MediaRef(url=str(filename), content_type=part.get_content_type()))
            continue
        if not name:
            continue
        fields[str(name)] = part.get_payload(decode=True) or b""
    return fields, tuple(media)


def _parts(document: Any) -> Iterator[Any]:
    payload = document.get_payload()
    return iter(payload) if isinstance(payload, list) else iter(())


def _charsets(fields: dict[str, bytes]) -> dict[str, str]:
    """SendGrid's per-field charset map. Read first, because it says how to
    read everything else — and it is the one field that is always ASCII."""
    try:
        declared = json.loads(fields.get(_CHARSETS, b"{}").decode("ascii", "replace"))
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in declared.items()} if isinstance(declared, dict) else {}


def _decode(value: bytes, charset: str | None) -> str:
    """The declared charset, or UTF-8, and never an exception.

    A message from an older Windows client arrives in windows-1256; decoded as
    UTF-8 it becomes replacement characters, which reach retrieval as a query
    of nothing and come back as a confident answer to it.
    """
    for candidate in (charset, "utf-8"):
        if not candidate:
            continue
        try:
            return value.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return value.decode("utf-8", "replace")


def _original_headers(blob: str) -> dict[str, str]:
    """The original message's headers, unfolded and with encoded words expanded.

    `References` arrives folded across lines and an Arabic `Subject` arrives as
    `=?UTF-8?B?…?=`; both are the standard library's job, not ours.
    """
    parsed = BytesParser(policy=policy.compat32).parsebytes(blob.encode("utf-8", "replace"))
    return {key: _words(value) for key, value in parsed.items()}


def _words(value: str) -> str:
    try:
        return str(make_header(decode_header(value))).strip()
    except (ValueError, UnicodeDecodeError):
        return value.strip()


def _refuse_machine_mail(original: dict[str, str]) -> None:
    """RFC 3834 and the conventions around it.

    Every one of these is a loop waiting to happen: an autoresponder answers
    our reply, we answer it, and neither end stops because neither end is a
    person. `Return-Path: <>` is the null sender — a bounce by definition, and
    replying to it bounces again.
    """
    lowered = {key.lower(): value for key, value in original.items()}

    return_path = lowered.get("return-path")
    if return_path is not None and return_path.strip() in ("<>", ""):
        raise NotACustomerTurn("null return path — this is a bounce, not a question")

    for header in _MACHINE_HEADERS:
        value = lowered.get(header)
        if value is None:
            continue
        if header == "auto-submitted" and value.strip().lower() == _NOT_AUTOMATIC:
            continue
        raise NotACustomerTurn(
            f"{header} present — a machine sent this, and answering it is a loop"
        )

    if lowered.get("precedence", "").strip().lower() in _MACHINE_PRECEDENCE:
        raise NotACustomerTurn("bulk or list precedence — not a customer writing to us")


def _envelope_from(blob: str | None) -> str | None:
    try:
        document = json.loads(blob or "{}")
    except ValueError:
        return None
    return document.get("from") if isinstance(document, dict) else None


def _body(fields: dict[str, str], settings: dict[str, Any]) -> str:
    """What the customer actually wrote, quote-stripped and capped."""
    text = fields.get(_TEXT) or _from_html(fields.get(_HTML, ""))
    stripped = strip_quoted(text, settings)
    cap = settings["max_body_chars"]
    if len(stripped) <= cap:
        return stripped
    return stripped[:cap] + settings["truncation_marker"]


def _from_html(html: str) -> str:
    """A crude HTML-to-text, for the messages that carry no text alternative.

    Outlook sends both; some clients and most marketing tools send only HTML.
    Without this the turn is empty and the customer gets a fallback for a
    question they asked in full.
    """
    if not html:
        return ""
    without_blocks = _BLOCKS.sub(" ", html)
    with_breaks = _BREAKS.sub("\n", without_blocks)
    return unescape(_TAGS.sub("", with_breaks)).strip()


def strip_quoted(body: str, config: dict[str, Any] | None = None) -> str:
    """Cut the thread the customer quoted back at us.

    Deliberately shy. A marker that fires too eagerly truncates a real question
    and the bot answers the half it saw, which nothing downstream can detect —
    so if stripping leaves nothing, the original is kept. A noisy turn is worth
    more than an empty one.
    """
    settings = config or _config()
    markers = [re.compile(pattern, re.MULTILINE) for pattern in settings["quote_markers"]]

    kept: list[str] = []
    for line in (body or "").replace("\r\n", "\n").split("\n"):
        if any(marker.search(line) for marker in markers):
            break
        kept.append(line)

    # Quoting without a marker line above it — some clients do exactly that.
    words = "\n".join(line for line in kept if not line.lstrip().startswith(">")).strip()
    return words or (body or "").strip()


def _thread_ref(original: dict[str, str], message_id: str) -> str:
    """The root of the thread: the first `References` entry.

    First, not last. `References` is the chain from the root forward, and the
    root is the one id every message in a long thread shares — the last is
    whichever message happened to be replied to.
    """
    references = _ANGLE.findall(original.get("References", ""))
    if references:
        return references[0]
    # Outlook frequently sets `In-Reply-To` and no `References` at all.
    in_reply_to = _bare(original.get("In-Reply-To"))
    return in_reply_to or message_id


def _bare(value: str | None) -> str:
    """A Message-ID without its angle brackets. Re-added on the way out."""
    if not value:
        return ""
    found = _ANGLE.search(value)
    return found.group(1) if found else value.strip()


def _fallback_id(raw_body: bytes) -> str:
    """A missing `Message-ID` still needs a unique idempotency key.

    Derived from the bytes, so a genuine redelivery is still recognised as one
    and two different messages are not. A constant would make every
    unidentified message a duplicate of the first, and the guard that exists to
    protect them would discard them all.
    """
    return f"sha256:{hashlib.sha256(raw_body).hexdigest()[:32]}"


def _address(value: str) -> str:
    """The addr-spec, lowercased.

    Not the display name: the same student writes from two clients under two
    names, and this is the key a conversation is looked up by. Case-folded for
    the same reason — mail clients disagree about it and mailboxes do not.
    """
    _, addr = parseaddr(value or "")
    return addr.strip().lower()


# ─────────────────────────── outbound ───────────────────────────


class SendGridEmail:
    """The outbound half.

    Holds no service window, because email has none — inheriting WhatsApp's
    would refuse every reply to a message older than a day. It does hold the
    thread headers and `Auto-Submitted`, because both are properties of the
    message rather than of the queue.

    Rate limiting, backoff and dead-lettering stay in the outbound worker,
    where one token bucket covers a tenant.
    """

    name = "sendgrid_email"

    def __init__(
        self,
        *,
        api_key: str,
        sender: str,
        sender_name: str | None = None,
        config: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = config or _config()
        self._settings = settings
        # The tenant's own address, from their channel_accounts row. One
        # platform sender for every tenant would put Sinai's replies under
        # another tenant's name, on a domain neither of them can authenticate.
        self._sender = sender
        self._sender_name = sender_name
        self._already_a_reply = re.compile(settings["reply_prefix_pattern"], re.IGNORECASE)
        http = settings["http"]
        self._client = httpx.AsyncClient(
            base_url=settings["api_base"],
            headers={"Authorization": f"Bearer {api_key}"},
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
        subject: str | None = None,
        now: datetime | None = None,
    ) -> OutboundReceipt:
        """Send one reply, in the thread it belongs to.

        `template`, `template_variables` and `last_inbound_at` are accepted and
        unused: they are part of `MessagingProvider`, and email has neither
        templates nor a window. Accepting and ignoring them keeps one send path
        (§6.2) rather than making the worker ask which channel it is talking to
        before it builds a call.
        """
        payload = {
            "personalizations": [
                {
                    "to": [{"email": to}],
                    "headers": self._headers(thread_ref),
                }
            ],
            "from": self._from(),
            "subject": self._subject(subject),
            "content": [{"type": "text/plain", "value": text or ""}],
        }

        try:
            response = await self._client.post(_SEND_PATH, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as refused:
            raise EmailRefused(_reason(refused.response)) from None
        except httpx.HTTPError as failed:
            raise EmailRefused(f"sendgrid request failed: {type(failed).__name__}") from None

        return OutboundReceipt(
            # SendGrid's own id, not the RFC Message-ID — it generates that
            # itself and does not tell us, which is why a reply cannot be
            # matched to the customer's next message by id. Threading survives
            # because the chain travels in their client, not in our records.
            provider_message_id=response.headers.get("X-Message-Id", ""),
            status="accepted" if response.status_code == 202 else str(response.status_code),
        )

    def _headers(self, thread_ref: str | None) -> dict[str, str]:
        headers = {
            # RFC 3834, on every reply. The only way a loop with an
            # autoresponder in somebody else's organisation is stopped.
            "Auto-Submitted": self._settings["auto_submitted"],
        }
        if thread_ref:
            # In-Reply-To and References both name the thread root. The leaf
            # would nest one level more precisely in clients that draw a tree;
            # the root is what every message in the thread shares, and it is
            # one value on the shared outbound job rather than two.
            headers["In-Reply-To"] = f"<{thread_ref}>"
            headers["References"] = f"<{thread_ref}>"
        return headers

    def _from(self) -> dict[str, str]:
        sender = {"email": self._sender}
        if self._sender_name:
            sender["name"] = self._sender_name
        return sender

    def _subject(self, subject: str | None) -> str:
        """`Re:` once. Every turn adding another grows a prefix per message,
        and some clients then stop threading on the subject at all."""
        words = (subject or "").strip()
        if not words:
            return self._settings["reply_prefix"].strip()
        if self._already_a_reply.search(words):
            return words
        return f"{self._settings['reply_prefix']}{words}"

    async def aclose(self) -> None:
        await self._client.aclose()


def _reason(response: httpx.Response) -> str:
    """SendGrid's own message, or the status code.

    The API key travels in a header rather than in the URL, so httpx's
    URL-bearing exceptions do not leak it the way Telegram's would. This
    wrapping exists so the dead-letter row carries a reason a tenant's admin
    can act on instead of an httpx repr — and a test still asserts the key is
    absent, because "it cannot leak from here" is a claim with a shelf life.
    """
    try:
        errors = response.json().get("errors") or []
        described = errors[0].get("message") if errors else None
    except (ValueError, AttributeError, IndexError, TypeError):
        described = None
    return f"sendgrid refused: {described or response.status_code}"


def _config() -> dict[str, Any]:
    from moc.config_store import load

    return load(_CONFIG)


__all__ = [
    "Authentication",
    "EmailRefused",
    "NotACustomerTurn",
    "SendGridEmail",
    "Unauthenticated",
    "authentication",
    "parse_inbound",
    "strip_quoted",
    "verify_secret",
]
