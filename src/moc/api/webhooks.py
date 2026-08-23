"""Inbound webhooks — design §6 and the ACK budget in §11.

**Flagged for line-by-line human review.** This is the only endpoint on the
platform that an unauthenticated stranger can reach, and it is the one place
where being slow is a correctness bug rather than a performance one.

The handler does four things, in this order, and nothing else:

    verify -> resolve the tenant -> claim the message id -> enqueue -> 200

**Why it must be fast.** Twilio and Meta retry anything that is not a prompt
2xx, and a retry is a duplicate delivery — the customer asked once and gets
answered twice. So no model call, no retrieval, no database transaction happens
here. That is enforced structurally rather than by a comment: this module
imports neither `moc.agent` nor `moc.llm`, and a test asserts it. The queue is
the seam; the worker owns the turn.

**Why verification comes before everything.** The one unavoidable awkwardness
is that secrets are per channel account (§6.2), so the account has to be
identified from an unverified body in order to select the secret to verify
with. Nothing from that first read survives: the message is parsed again from
the raw bytes only after the signature checks out, and the pre-verification
read is used for exactly one thing — the lookup key. An unknown address is
refused there, before any tenant work is done on a stranger's behalf.

**Why the duplicate check comes after verification.** Claiming a message id
first would let anyone suppress a real inbound message by posting its id ahead
of the vendor, with no signature required. It is a spend of a scarce resource
and it must be authenticated.

**Why a queue failure is a 503, and releases the claim.** A 200 means
"delivered" and stops the vendor retrying, so losing the message and acking it
is the one outcome with no recovery path. But asking for a retry is not enough
on its own: the message id was claimed a moment earlier, and a still-claimed id
makes that retry look like a duplicate. The message would then be dropped by
the guard that exists to protect it. So the claim is released on the way out.
"""

from urllib.parse import parse_qsl

from fastapi import FastAPI, Request, Response

from moc.channels.base import (
    Channel,
    ChannelAccountRegistry,
    EventLog,
    InboundQueue,
    SecretResolver,
)
from moc.channels.telegram import NotAMessage, verify_secret
from moc.channels.telegram import parse_inbound as parse_telegram
from moc.channels.twilio_wa import parse_inbound, verify_signature
from moc.config_store import load

_WHATSAPP = "channels/whatsapp"
_TELEGRAM = "channels/telegram"
_TWILIO_WHATSAPP_PATH = "/webhooks/twilio/whatsapp"
_TO = "To"

_OK = 200
_BAD_REQUEST = 400
_FORBIDDEN = 403
_UNAVAILABLE = 503


def build_app(
    *,
    registry: ChannelAccountRegistry,
    queue: InboundQueue,
    events: EventLog,
    secrets: SecretResolver,
) -> FastAPI:
    """Assemble the webhook app from its collaborators.

    Constructed rather than module-global so a test wires fakes without
    patching, and so it is visible at a glance that the handler has no access
    to a database session, a router, or an orchestrator — it cannot do slow
    work it cannot reach.
    """
    app = FastAPI()
    settings = load(_WHATSAPP)
    prefix = settings["address_prefix"]

    @app.post(_TWILIO_WHATSAPP_PATH)
    async def twilio_whatsapp(request: Request) -> Response:
        # The exact bytes, before anything can normalize them. Every later step
        # works from these; nothing re-serializes and verifies its own output.
        raw_body = await request.body()

        account_ref = _account_ref(raw_body, prefix)
        if account_ref is None:
            return Response(status_code=_BAD_REQUEST)

        # Unknown address: refuse here, before a tenant is resolved, a session
        # opened or a message id claimed. Anyone can POST to this path, and
        # there is no tenant to attribute the work to.
        account = await registry.resolve(channel=Channel.whatsapp, account_ref=account_ref)
        if account is None:
            return Response(status_code=_FORBIDDEN)

        if not verify_signature(
            url=str(request.url),
            raw_body=raw_body,
            content_type=request.headers.get("content-type", ""),
            signature=request.headers.get(settings["signature_header"]),
            # The signing secret is fetched by reference, not carried on the
            # account: resolution runs through `moc_lookup` before anything is
            # verified, and that role must not be able to reach the one value
            # an attacker needs to forge this tenant's traffic (Task 21).
            auth_token=secrets.for_ref(account.secret_ref),
            config=settings,
        ):
            return Response(status_code=_FORBIDDEN)

        message = parse_inbound(raw_body=raw_body, account=account, config=settings)

        # A redelivery after a slow ACK is a normal path, not an edge case.
        # Answer 200 so the vendor stops retrying, and enqueue nothing.
        if not await events.claim(message.provider_message_id):
            return Response(status_code=_OK)

        try:
            await queue.publish(message)
        except Exception:
            # Broad on purpose: whatever went wrong, the message is not on the
            # queue, so the one thing we must not do is tell the vendor it was
            # delivered. Release the claim first — leaving it held would make
            # the retry we are about to ask for look like a duplicate, and the
            # message would be dropped by its own idempotency guard.
            await events.release(message.provider_message_id)
            return Response(status_code=_UNAVAILABLE)
        return Response(status_code=_OK)

    telegram = load(_TELEGRAM)

    @app.post(telegram["webhook_path"])
    async def telegram_update(account_ref: str, request: Request) -> Response:
        """One path per bot, because an update does not say which bot got it.

        There is no bot id anywhere in a Telegram update, so the account
        reference is in the path. It is guessable on purpose: it selects which
        secret to check and is not itself a credential — the same role the
        WhatsApp address plays, and for the same reason (the secret cannot be
        chosen without knowing the account).
        """
        raw_body = await request.body()

        account = await registry.resolve(
            channel=Channel.telegram, account_ref=account_ref
        )
        if account is None:
            return Response(status_code=_FORBIDDEN)

        # A bearer token, not a signature. It proves the sender knows the
        # secret and says nothing about the body — see the adapter's docstring
        # for why this is named for what it is.
        if not verify_secret(
            presented=request.headers.get(telegram["secret_header"]),
            expected=secrets.for_ref(account.secret_ref),
        ):
            return Response(status_code=_FORBIDDEN)

        try:
            message = parse_telegram(raw_body=raw_body, account=account)
        except NotAMessage:
            # An edit, a channel post, a callback query. Acknowledged so
            # Telegram stops retrying, and processed as nothing: answering a
            # customer's edit of a question they already asked is worse than
            # ignoring it.
            return Response(status_code=_OK)
        except ValueError:
            # Malformed JSON from something that had the secret. 400 rather
            # than 500: nothing here is broken, the body is.
            return Response(status_code=_BAD_REQUEST)

        if not await events.claim(message.provider_message_id):
            return Response(status_code=_OK)

        try:
            await queue.publish(message)
        except Exception:
            await events.release(message.provider_message_id)
            return Response(status_code=_UNAVAILABLE)
        return Response(status_code=_OK)

    return app


def _account_ref(raw_body: bytes, prefix: str) -> str | None:
    """The lookup key, and only the lookup key.

    Read from an unverified body because per-account secrets leave no
    alternative: the secret cannot be chosen without knowing the account. The
    result is used to select a secret and for nothing else — the message itself
    is parsed again from the same bytes after the signature holds.
    """
    try:
        fields = dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
    except UnicodeDecodeError:
        return None
    address = fields.get(_TO)
    if not address:
        return None
    return address[len(prefix) :] if address.startswith(prefix) else address


__all__ = ["build_app"]
