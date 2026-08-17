"""Twilio WhatsApp adapter — design §6.1 and §6.2.

`verify_signature` is flagged for line-by-line human review, so these tests are
written as the specification that review is checking. Two of them are the whole
reason the flag exists:

- **Constant time.** A timing side channel on a signature check is exploitable
  and completely invisible in behaviour: `==` passes every functional test ever
  written for it. The guard is structural — the comparison is isolated in one
  three-line function that is asserted to contain no equality operator at all.

- **The raw body.** Twilio's JSON path signs a SHA-256 of the exact bytes it
  sent. Verifying a re-serialized body passes every test written with
  round-tripped fixtures and then fails on real traffic — or, worse, an
  implementation that re-serializes *both* sides keeps passing while no longer
  checking anything about what actually arrived.
"""

import ast
import base64
import hashlib
import hmac
import inspect
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode
from uuid import uuid4

import httpx
import pytest

from moc.channels.base import Channel, ChannelAccount, OutsideServiceWindow
from moc.channels.twilio_wa import (
    TwilioWhatsApp,
    parse_inbound,
    verify_signature,
)
from moc.config_store import load

WHATSAPP = load("channels/whatsapp")
SECRET = "auth-token-for-this-account"
URL = "https://moc.example/webhooks/twilio/whatsapp"

FORM = {
    "MessageSid": "SM1234567890abcdef",
    "AccountSid": "AC0000000000000000",
    "From": "whatsapp:+201012345678",
    "To": "whatsapp:+201555000111",
    "Body": "كام رسوم الساعة؟",
    "NumMedia": "0",
}


def account(**overrides) -> ChannelAccount:
    return ChannelAccount(
        **{
            "id": uuid4(),
            "tenant_id": uuid4(),
            "channel": Channel.whatsapp,
            "account_ref": "+201555000111",
            "signing_secret": SECRET,
            **overrides,
        }
    )


def form_body(**overrides) -> bytes:
    return urlencode({**FORM, **overrides}).encode()


def sign_form(body: bytes, *, url: str = URL, secret: str = SECRET) -> str:
    """Twilio's documented algorithm, written out independently.

    Deliberately not a call into the module under test: a test that signs with
    the same helper it verifies with proves the two agree, and agrees with
    itself when both are wrong.
    """
    from urllib.parse import parse_qsl

    canonical = url + "".join(k + v for k, v in sorted(parse_qsl(body.decode(), True)))
    digest = hmac.new(secret.encode(), canonical.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def sign_json(body: bytes, *, url: str = URL, secret: str = SECRET) -> tuple[str, str]:
    """The JSON path: Twilio hashes the exact bytes into the URL, then signs it."""
    signed_url = f"{url}?{WHATSAPP['body_hash_param']}={hashlib.sha256(body).hexdigest()}"
    digest = hmac.new(secret.encode(), signed_url.encode(), hashlib.sha1).digest()
    return signed_url, base64.b64encode(digest).decode()


def verify(body: bytes, signature: str, *, url: str = URL, content_type: str | None = None):
    return verify_signature(
        url=url,
        raw_body=body,
        content_type=content_type or WHATSAPP["form_content_type"],
        signature=signature,
        auth_token=SECRET,
    )


# ─────────────────────────── constant time ───────────────────────────


def test_signature_verification_uses_constant_time_comparison():
    """hmac.compare_digest, never ==.

    Asserted against the source rather than behaviour, because there is no
    behavioural difference to assert — that is precisely what makes a timing
    leak survive review. The comparison lives alone in `_matches` so this check
    can be exact instead of heuristic.
    """
    from moc.channels.twilio_wa import _matches

    tree = ast.parse(inspect.getsource(_matches).lstrip())
    equality = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops)
    ]
    assert equality == [], "an equality operator in the signature comparison leaks timing"
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "compare_digest" in calls


def test_the_constant_time_guard_would_catch_a_regression():
    """Prove the guard fires rather than trusting that it would."""
    tree = ast.parse("def _matches(a, b):\n    return a == b\n")
    assert [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops)
    ]


def test_the_body_hash_is_compared_in_constant_time_too():
    """The JSON path has a second secret-dependent comparison.

    Checking the signature in constant time and the body hash with `==` leaves
    the same oracle open one line further down.
    """
    from moc.channels.twilio_wa import _body_hash_matches

    tree = ast.parse(inspect.getsource(_body_hash_matches).lstrip())
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops)
    ]


# ─────────────────────────── the raw body ───────────────────────────


def test_verifies_against_the_raw_body_not_a_reserialized_one():
    """The classic bug, demonstrated rather than described.

    Twilio's JSON path signs a hash of the exact bytes. `json.dumps(json.loads(x))`
    is the same *document* and different *bytes* — different spacing, possibly
    different key order — so an implementation that verifies the round-tripped
    form rejects real traffic, and one that round-trips both sides has stopped
    checking anything about what arrived.
    """
    raw = (
        b'{"MessageSid":"SM1","From":"whatsapp:+201012345678",'
        b'"Body":"\\u0623\\u0647\\u0644\\u0627"}'
    )
    signed_url, signature = sign_json(raw)
    reserialized = json.dumps(json.loads(raw)).encode()

    assert reserialized != raw, "the fixture must actually differ in bytes"
    assert verify(
        raw, signature, url=signed_url, content_type=WHATSAPP["json_content_type"]
    )
    assert not verify(
        reserialized, signature, url=signed_url, content_type=WHATSAPP["json_content_type"]
    )


def test_verify_signature_takes_bytes_never_a_parsed_body():
    """A drift guard on the API shape.

    The re-serialization bug arrives as a convenient `params: dict` parameter,
    because the caller has already parsed the body and passing it feels like
    avoiding duplicate work. Taking only bytes makes the bug unrepresentable.
    """
    annotations = inspect.signature(verify_signature).parameters
    assert annotations["raw_body"].annotation is bytes
    assert not {"params", "form", "body", "payload"} & set(annotations)


def test_a_form_body_verifies_against_twilios_documented_algorithm():
    body = form_body()
    assert verify(body, sign_form(body))


def test_form_parameter_order_in_the_body_does_not_change_the_result():
    """Twilio sorts by key; the transmitted order is arbitrary and must not matter."""
    reordered = urlencode(dict(reversed(list(FORM.items())))).encode()
    assert verify(reordered, sign_form(reordered))


def test_a_tampered_body_fails():
    body = form_body()
    signature = sign_form(body)
    assert not verify(form_body(Body="مصاريف مجانية"), signature)


def test_a_tampered_url_fails():
    """The URL is inside the signed string — a replay onto another route fails."""
    body = form_body()
    assert not verify(body, sign_form(body), url=URL + "/other")


# ─────────────────────────── missing and empty inputs ───────────────────────────


@pytest.mark.parametrize("signature", ["", None])
def test_missing_signature_header_is_rejected_not_skipped(signature):
    """Absent must mean rejected, never "nothing to check".

    The tempting shape is `if signature: verify(...)`, which turns the entire
    control into something an attacker disables by omitting one header.
    """
    assert not verify(form_body(), signature)


def test_an_empty_auth_token_rejects_rather_than_signing_with_nothing():
    """A misconfigured account must fail closed.

    HMAC with an empty key is a perfectly valid HMAC, so without this the
    request verifies against a key anyone can guess and the check reports
    success.
    """
    assert not verify_signature(
        url=URL,
        raw_body=form_body(),
        content_type=WHATSAPP["form_content_type"],
        signature=sign_form(form_body(), secret=""),
        auth_token="",
    )


def test_a_signature_that_is_not_valid_base64_is_rejected_not_raised():
    """Garbage in a header is hostile input, not an exception."""
    assert not verify(form_body(), "!!!not base64!!!")


# ─────────────────────────── per-account secrets ───────────────────────────


def test_secret_is_per_channel_account_not_global():
    """One tenant's leaked token must not forge another tenant's traffic.

    A platform-wide auth token makes every tenant's webhook forgeable by
    anyone who has ever seen any tenant's credentials — including the tenant
    themselves.
    """
    body = form_body()
    signed_by_other_tenant = sign_form(body, secret="a-different-tenants-token")
    assert not verify(body, signed_by_other_tenant)
    assert verify(body, sign_form(body, secret=SECRET))


def test_the_account_carries_its_own_secret():
    assert account().signing_secret == SECRET
    assert account(signing_secret="other").signing_secret == "other"


# ─────────────────────────── §6.1 normalization ───────────────────────────


def test_normalizes_to_InboundMessage():
    acct = account()
    message = parse_inbound(raw_body=form_body(), account=acct)

    assert message.tenant_id == acct.tenant_id
    assert message.channel_account_id == acct.id
    assert message.channel is Channel.whatsapp
    assert message.provider_message_id == FORM["MessageSid"]
    assert message.text == FORM["Body"]


def test_the_vendor_address_prefix_is_stripped():
    """Downstream must never learn which vendor carried the message (§6.2)."""
    message = parse_inbound(raw_body=form_body(), account=account())
    assert message.sender_ref == "+201012345678"
    assert WHATSAPP["address_prefix"] not in message.sender_ref


def test_the_raw_payload_is_kept_whole():
    """§6.1: stored, never parsed downstream.

    The raw row is what a dispute is settled from, so it keeps fields this
    version does not understand — including ones Twilio adds later.
    """
    message = parse_inbound(raw_body=form_body(SmsStatus="received"), account=account())
    assert message.raw["SmsStatus"] == "received"
    assert message.raw["AccountSid"] == FORM["AccountSid"]


def test_media_is_collected():
    body = form_body(
        NumMedia="2",
        MediaUrl0="https://api.twilio.com/media/0",
        MediaContentType0="image/jpeg",
        MediaUrl1="https://api.twilio.com/media/1",
        MediaContentType1="application/pdf",
    )
    message = parse_inbound(raw_body=body, account=account())
    assert [m.content_type for m in message.media] == ["image/jpeg", "application/pdf"]


def test_a_message_with_no_text_is_not_an_error():
    """A voice note or a sticker has no Body. It is still a turn."""
    body = urlencode({k: v for k, v in FORM.items() if k != "Body"}).encode()
    message = parse_inbound(raw_body=body, account=account())
    assert message.text is None


# ─────────────────────────── §6.2 the 24-hour window ───────────────────────────


def sender(handler) -> TwilioWhatsApp:
    return TwilioWhatsApp(
        account_sid="AC0",
        auth_token=SECRET,
        sender="+201555000111",
        config=WHATSAPP,
        transport=httpx.MockTransport(handler),
    )


def ok(request: httpx.Request) -> httpx.Response:
    ok.seen = request
    return httpx.Response(201, json={"sid": "SM-out", "status": "queued"})


def test_outbound_respects_the_24_hour_window():
    """Outside it, only approved templates (§6.2).

    Meta rejects freeform text outside the window, so sending anyway produces
    a queue of failures that reads to the tenant as an outage. Refusing here
    turns that into one legible error at the point of the mistake.
    """
    window = timedelta(hours=WHATSAPP["service_window_hours"])
    stale = datetime.now(UTC) - window - timedelta(minutes=1)

    with pytest.raises(OutsideServiceWindow):
        _run(sender(ok).send(to="+201012345678", text="أهلا", last_inbound_at=stale))


def test_inside_the_window_freeform_text_sends():
    fresh = datetime.now(UTC) - timedelta(hours=1)
    result = _run(sender(ok).send(to="+201012345678", text="أهلا", last_inbound_at=fresh))
    assert result.provider_message_id == "SM-out"


def test_outside_the_window_an_approved_template_sends():
    window = timedelta(hours=WHATSAPP["service_window_hours"])
    stale = datetime.now(UTC) - window - timedelta(minutes=1)
    result = _run(
        sender(ok).send(
            to="+201012345678",
            template="HX0123",
            template_variables={"1": "الهندسة"},
            last_inbound_at=stale,
        )
    )
    assert result.provider_message_id == "SM-out"


def test_a_first_contact_with_no_inbound_yet_requires_a_template():
    """No previous inbound means no window was ever opened."""
    with pytest.raises(OutsideServiceWindow):
        _run(sender(ok).send(to="+201012345678", text="أهلا", last_inbound_at=None))


def test_the_window_length_comes_from_config():
    assert WHATSAPP["service_window_hours"] == 24


def test_the_outbound_address_is_re_prefixed_for_the_vendor():
    fresh = datetime.now(UTC) - timedelta(hours=1)
    _run(sender(ok).send(to="+201012345678", text="أهلا", last_inbound_at=fresh))
    sent = dict(x.split("=", 1) for x in ok.seen.content.decode().split("&"))
    assert sent["To"].startswith("whatsapp")
    assert sent["From"].startswith("whatsapp")


def test_the_sending_number_is_the_tenants_own(): 
    """Per channel account, never platform-wide.

    One shared sender would put one tenant's replies under another tenant's
    brand — visible to the customer, and impossible to unsend.
    """
    fresh = datetime.now(UTC) - timedelta(hours=1)
    _run(sender(ok).send(to="+201012345678", text="أهلا", last_inbound_at=fresh))
    sent = dict(x.split("=", 1) for x in ok.seen.content.decode().split("&"))
    assert "201555000111" in sent["From"]
    assert sent["From"] != sent["To"]


def test_no_template_management_is_built():
    """§6.2: proxy Twilio's. A content sid is passed through, never authored."""
    assert not [name for name in dir(TwilioWhatsApp) if "template" in name.lower()]


def _run(coroutine):
    import asyncio

    return asyncio.run(coroutine)
