"""Instagram and Messenger — demo plan Task 36.

One Meta Graph webhook serves both, and the tests are mostly about the three
things that make it a single integration rather than a simple one.

**`test_an_echo_is_not_a_customer_turn` is the one that matters.** Every
message the page sends comes back down the webhook with `is_echo: true`. A bot
that does not drop them answers its own reply, which answers that — a loop that
costs a model call per turn and reaches the customer as a wall of messages.

**Batching.** One POST can carry several customers. Every other adapter here
parses one message per request, and a Meta adapter that did the same would
silently drop everyone after the first.

**Not everything is a message.** Delivery receipts, read receipts and reactions
arrive in the same envelope. Answering a read receipt is answering a customer
for looking at their phone.
"""

import hashlib
import hmac
import json
from uuid import uuid4

import httpx
import pytest

from moc.channels.base import Channel, ChannelAccount
from moc.channels.meta import MetaMessenger, parse_inbound, verify_challenge, verify_signature

APP_SECRET = "meta-app-secret-value"  # noqa: S105 - a test fixture
VERIFY_TOKEN = "our-own-verify-token"  # noqa: S105 - a test fixture
PAGE_ID = "1122334455"
CUSTOMER_ID = "9988776655"

ACCOUNT = ChannelAccount(
    id=uuid4(),
    tenant_id=uuid4(),
    channel=Channel.messenger,
    account_ref=PAGE_ID,
    secret_ref="meta/sinai/app",
)


def envelope(*messaging, obj: str = "page") -> bytes:
    return json.dumps(
        {
            "object": obj,
            "entry": [{"id": PAGE_ID, "time": 1755859200, "messaging": list(messaging)}],
        },
        ensure_ascii=False,
    ).encode()


def turn(text: str = "كام رسوم الساعة؟", **overrides) -> dict:
    message = {"mid": "m_abc123", "text": text}
    message.update(overrides.pop("message", {}))
    return {
        "sender": {"id": CUSTOMER_ID},
        "recipient": {"id": PAGE_ID},
        "timestamp": 1755859200000,
        "message": message,
        **overrides,
    }


def sign(raw: bytes) -> str:
    digest = hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# ─────────────────────────── verification ───────────────────────────


def test_the_verifying_functions_contain_no_equality_comparison():
    """Structural, as in every other adapter here. A timing side channel on an
    HMAC comparison passes every functional test anyone will write.

    Scoped to the two functions that compare secrets, not to the whole module.
    `passwords.py` bans `==` everywhere because it does one thing, so the rule
    costs nothing there; this module also parses payloads, where comparing a
    page id to an account reference is ordinary code and `compare_digest` on it
    would be noise that dilutes the signal.

    The boundary is the reviewable thing: these are the functions where a
    secret is compared, and a new one would have to be added to this list by
    somebody who has just read why the list is short.
    """
    import ast
    import inspect

    from moc.channels import meta

    verifying = {"verify_signature", "verify_challenge"}
    tree = ast.parse(inspect.getsource(meta))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in verifying:
            continue
        compares = [
            inner
            for inner in ast.walk(node)
            if isinstance(inner, ast.Compare)
            and any(isinstance(op, ast.Eq | ast.NotEq) for op in inner.ops)
        ]
        assert compares == [], f"{node.name} compares with == rather than compare_digest"

    found = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in verifying
    }
    assert found == verifying, f"a verifying function was renamed: {found}"
    assert "compare_digest" in inspect.getsource(meta)


def test_a_correctly_signed_body_verifies():
    raw = envelope(turn())
    assert verify_signature(raw_body=raw, signature=sign(raw), app_secret=APP_SECRET) is True


def test_a_tampered_body_is_refused():
    """The property an HMAC buys that Telegram's bearer token does not: the
    body is covered, so a modified payload fails even from a sender who knows
    the secret's header value."""
    raw = envelope(turn())
    # Tamper with something that is actually in the body. The first version of
    # this test replaced b"1400", which the fixture does not contain — so it
    # signed a body, verified the same body, and asserted that tampering was
    # refused while tampering nothing.
    tampered = raw.replace(CUSTOMER_ID.encode(), b"1234567890")
    assert tampered != raw, "the tamper did nothing; the assertion would be vacuous"
    assert (
        verify_signature(
            raw_body=tampered, signature=sign(raw), app_secret=APP_SECRET
        )
        is False
    )


def test_verification_fails_closed_on_every_missing_input():
    raw = envelope(turn())
    assert verify_signature(raw_body=raw, signature=None, app_secret=APP_SECRET) is False
    assert verify_signature(raw_body=raw, signature="", app_secret=APP_SECRET) is False
    assert verify_signature(raw_body=raw, signature=sign(raw), app_secret="") is False
    # No prefix is not a signature. Accepting a bare digest would accept a
    # value from a future algorithm we have not implemented.
    bare = sign(raw).removeprefix("sha256=")
    assert verify_signature(raw_body=raw, signature=bare, app_secret=APP_SECRET) is False


def test_the_subscription_handshake_echoes_the_challenge():
    assert (
        verify_challenge(
            mode="subscribe",
            token=VERIFY_TOKEN,
            challenge="1158201444",
            expected_token=VERIFY_TOKEN,
        )
        == "1158201444"
    )


def test_the_handshake_is_refused_on_a_wrong_token_or_mode():
    """None, not the challenge. Echoing it regardless would subscribe anyone
    who guessed the URL."""
    assert verify_challenge(
        mode="subscribe", token="wrong", challenge="x", expected_token=VERIFY_TOKEN
    ) is None
    assert verify_challenge(
        mode="unsubscribe", token=VERIFY_TOKEN, challenge="x", expected_token=VERIFY_TOKEN
    ) is None


# ─────────────────────────── inbound ───────────────────────────


def test_a_messenger_message_becomes_the_normalized_shape():
    messages = parse_inbound(raw_body=envelope(turn()), account=ACCOUNT)

    assert len(messages) == 1
    message = messages[0]
    assert message.channel is Channel.messenger
    assert message.tenant_id == ACCOUNT.tenant_id
    assert message.text == "كام رسوم الساعة؟"
    assert message.sender_ref == CUSTOMER_ID
    assert message.provider_message_id == "m_abc123"


def test_the_object_field_decides_the_channel():
    """One adapter, two channels. `page` is Messenger and `instagram` is
    Instagram DM, and the message shape is otherwise identical — which is why
    this is one integration and not two."""
    messages = parse_inbound(raw_body=envelope(turn(), obj="instagram"), account=ACCOUNT)

    assert messages[0].channel is Channel.instagram


def test_an_unknown_object_is_refused_rather_than_guessed():
    """Meta ships new product objects. Defaulting to Messenger would route a
    WhatsApp-through-Meta or a threads message into the wrong channel's sender
    and reply on a surface the customer never used."""
    from moc.channels.meta import UnknownObject

    with pytest.raises(UnknownObject):
        parse_inbound(raw_body=envelope(turn(), obj="threads"), account=ACCOUNT)


def test_an_echo_is_not_a_customer_turn():
    """**The one that matters.**

    Every message the page sends comes back with `is_echo: true`. A bot that
    does not drop them answers its own reply, which answers that: a loop that
    costs a model call per turn and reaches the customer as a wall of
    messages. It is the first thing anyone integrating Messenger gets wrong.
    """
    body = envelope(turn(message={"is_echo": True}), turn())
    messages = parse_inbound(raw_body=body, account=ACCOUNT)

    assert len(messages) == 1, "the echo was treated as a customer turn"
    assert messages[0].text == "كام رسوم الساعة؟"


def test_one_post_can_carry_several_customers():
    """`entry` is a list and `messaging` is a list.

    Every other adapter here parses one message per request. One that did the
    same for Meta would drop everyone after the first, silently, on exactly
    the busiest days.
    """
    other = {**turn("سؤال تاني"), "sender": {"id": "5555"}}
    other["message"] = {"mid": "m_second", "text": "سؤال تاني"}
    messages = parse_inbound(raw_body=envelope(turn(), other), account=ACCOUNT)

    assert [m.sender_ref for m in messages] == [CUSTOMER_ID, "5555"]
    assert [m.provider_message_id for m in messages] == ["m_abc123", "m_second"]


def test_delivery_and_read_receipts_are_not_turns():
    """They arrive in the same envelope with the same sender. Answering a read
    receipt answers a customer for looking at their phone."""
    receipts = envelope(
        {"sender": {"id": CUSTOMER_ID}, "recipient": {"id": PAGE_ID},
         "delivery": {"mids": ["m_abc123"], "watermark": 1755859200000}},
        {"sender": {"id": CUSTOMER_ID}, "recipient": {"id": PAGE_ID},
         "read": {"watermark": 1755859200000}},
    )
    assert parse_inbound(raw_body=receipts, account=ACCOUNT) == []


def test_an_attachment_with_no_text_is_still_a_turn():
    """A voice note or an image is a message. The media is the message."""
    body = envelope(
        turn(message={"mid": "m_img", "text": None,
                      "attachments": [{"type": "image",
                                       "payload": {"url": "https://cdn.example/a.jpg"}}]})
    )
    messages = parse_inbound(raw_body=body, account=ACCOUNT)

    assert messages[0].text is None
    assert messages[0].media[0].content_type == "image"


# ─────────────────────────── outbound ───────────────────────────


async def test_a_reply_is_sent_to_the_customer():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message_id": "m_out_1", "recipient_id": CUSTOMER_ID})

    bot = MetaMessenger(
        page_id=PAGE_ID, access_token="tok", transport=httpx.MockTransport(handler)
    )
    receipt = await bot.send(
        to=CUSTOMER_ID, text="الرسوم 1400 جنيه.",
        last_inbound_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    await bot.aclose()

    assert sent[0]["recipient"] == {"id": CUSTOMER_ID}
    assert sent[0]["message"] == {"text": "الرسوم 1400 جنيه."}
    assert sent[0]["messaging_type"] == "RESPONSE"
    assert receipt.provider_message_id == "m_out_1"


async def test_freeform_outside_the_window_is_refused():
    """§6.2, on Meta's own surfaces. Outside 24 hours a reply needs a message
    tag, and an untagged send is rejected — a queue of rejections reads to a
    tenant as an outage rather than as the policy they hit.

    Refused rather than silently tagged: choosing a tag is a business decision
    with a per-message cost, and Instagram allows fewer of them than Messenger.
    """
    from datetime import UTC, datetime, timedelta

    from moc.channels.base import OutsideServiceWindow

    bot = MetaMessenger(
        page_id=PAGE_ID, access_token="tok",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    with pytest.raises(OutsideServiceWindow):
        await bot.send(
            to=CUSTOMER_ID, text="أهلاً",
            last_inbound_at=datetime.now(UTC) - timedelta(hours=25),
        )
    await bot.aclose()


async def test_first_contact_is_refused_too():
    """No previous inbound means no window was ever opened, and an unsolicited
    message from a business is what the window exists to prevent."""
    from moc.channels.base import OutsideServiceWindow

    bot = MetaMessenger(
        page_id=PAGE_ID, access_token="tok",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    with pytest.raises(OutsideServiceWindow):
        await bot.send(to=CUSTOMER_ID, text="أهلاً", last_inbound_at=None)
    await bot.aclose()


async def test_the_access_token_never_appears_in_an_error():
    """The same lesson as Telegram's bot token, learned once.

    Meta puts the token in a query parameter rather than the path, which is the
    same problem: httpx's `raise_for_status` renders the URL, and the outbound
    worker writes `repr(exc)` into a dead-letter row.
    """
    from datetime import UTC, datetime

    from moc.channels.meta import MetaRefused

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Invalid recipient"}})

    bot = MetaMessenger(
        page_id=PAGE_ID, access_token="a-real-page-access-token",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(MetaRefused) as refusal:
        await bot.send(to="nobody", text="مرحبا", last_inbound_at=datetime.now(UTC))
    await bot.aclose()

    rendered = f"{refusal.value!r} {refusal.value}"
    assert "a-real-page-access-token" not in rendered
    assert "Invalid recipient" in rendered


def test_no_endpoint_string_is_hardcoded_in_the_adapter():
    """§2.4."""
    import inspect

    from moc.channels import meta

    assert "graph.facebook.com" not in inspect.getsource(meta)


def test_a_batch_spanning_two_pages_only_yields_this_accounts_messages():
    """**One POST, two tenants.**

    Meta batches per app, not per page, so a delivery can carry entries for
    several pages — and those pages can belong to different tenants. An adapter
    that ignored `entry.id` would attribute every message in the batch to
    whichever account the webhook happened to resolve first: a cross-tenant
    write, arriving as ordinary traffic.
    """
    mine = {
        "id": PAGE_ID,
        "time": 1755859200,
        "messaging": [turn("سؤالي أنا")],
    }
    theirs = {
        "id": "9999999999",
        "time": 1755859200,
        "messaging": [
            {
                "sender": {"id": "7777"},
                "recipient": {"id": "9999999999"},
                "timestamp": 1755859200000,
                "message": {"mid": "m_other_tenant", "text": "سؤال تاني"},
            }
        ],
    }
    body = json.dumps({"object": "page", "entry": [mine, theirs]}).encode()

    messages = parse_inbound(raw_body=body, account=ACCOUNT)

    assert [m.provider_message_id for m in messages] == ["m_abc123"]
    assert all(m.tenant_id == ACCOUNT.tenant_id for m in messages)


def test_the_page_ids_in_a_batch_can_be_read_without_trusting_them():
    """The webhook needs to know which accounts to resolve before it can pick
    a secret — the same pre-verification read the Twilio path makes for the
    `To` address, used to select and for nothing else."""
    from moc.channels.meta import account_refs

    body = json.dumps(
        {"object": "page", "entry": [{"id": PAGE_ID}, {"id": "9999999999"}, {"id": PAGE_ID}]}
    ).encode()

    assert account_refs(raw_body=body) == [PAGE_ID, "9999999999"]
