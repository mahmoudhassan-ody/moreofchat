"""Telegram — demo plan Task 35.

The cheapest channel to demo and the one with the weakest security model, and
the tests are mostly about the second.

**`test_the_bot_token_never_appears_in_an_error` is the one that matters.** The
Bot API puts the token in the URL path, httpx's `raise_for_status` puts the URL
in the exception, and the outbound worker writes `repr(exc)` into a
dead-letter row. That is a live bot token in the database, arriving through
three pieces of ordinary code none of which is wrong on its own.
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from moc.channels.base import Channel, ChannelAccount
from moc.channels.telegram import TelegramBot, parse_inbound, verify_secret

SECRET = "a-long-random-webhook-secret"  # noqa: S105 - a test fixture
TOKEN = "123456:AAHfake-bot-token-value"  # noqa: S105 - a test fixture

ACCOUNT = ChannelAccount(
    id=uuid4(),
    tenant_id=uuid4(),
    channel=Channel.telegram,
    account_ref="sinai_bot",
    secret_ref="telegram/sinai/bot",
)


def update(**overrides) -> bytes:
    document = {
        "update_id": 900001,
        "message": {
            "message_id": 42,
            "date": 1755859200,
            "chat": {"id": 987654321, "type": "private"},
            "from": {"id": 987654321, "first_name": "Mona"},
            "text": "كام رسوم الساعة؟",
            **overrides,
        },
    }
    return json.dumps(document, ensure_ascii=False).encode()


# ─────────────────────────── verification ───────────────────────────


def test_the_secret_token_is_compared_in_constant_time():
    """Structural, like the Twilio adapter's `_matches`.

    A timing side channel on a bearer token is exploitable and passes every
    functional test that will ever be written against it.
    """
    import ast
    import inspect

    from moc.channels import telegram

    tree = ast.parse(inspect.getsource(telegram))
    compares = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops)
    ]
    assert compares == [], "compare secrets with hmac.compare_digest, never =="
    assert "compare_digest" in inspect.getsource(telegram)


def test_a_correct_secret_verifies():
    assert verify_secret(presented=SECRET, expected=SECRET) is True


def test_a_wrong_or_missing_secret_is_refused():
    """Fails closed on every missing input. A blank header and a blank stored
    secret are both configuration or attack, never "nothing to check" — and a
    comparison of two empty strings would otherwise pass."""
    assert verify_secret(presented="wrong", expected=SECRET) is False
    assert verify_secret(presented=None, expected=SECRET) is False
    assert verify_secret(presented="", expected=SECRET) is False
    assert verify_secret(presented=SECRET, expected="") is False
    assert verify_secret(presented="", expected="") is False


# ─────────────────────────── inbound ───────────────────────────


def test_an_update_becomes_the_normalized_shape():
    message = parse_inbound(raw_body=update(), account=ACCOUNT)

    assert message.tenant_id == ACCOUNT.tenant_id
    assert message.channel is Channel.telegram
    assert message.text == "كام رسوم الساعة؟"
    # The chat id, not the user id. A reply goes to a chat, and in a group they
    # are different — replying to the user id would send a private message to
    # somebody who asked in public.
    assert message.sender_ref == "987654321"
    assert message.raw["update_id"] == 900001


def test_the_provider_message_id_is_unique_across_chats():
    """Telegram's `message_id` restarts per chat, so it alone is not an
    identity. Two customers can hold the same one at the same moment, and the
    idempotency guard would discard the second as a redelivery of the first —
    a message silently lost by the mechanism that exists to protect it.
    """
    mine = parse_inbound(raw_body=update(), account=ACCOUNT)
    theirs = parse_inbound(
        raw_body=update(chat={"id": 111222333, "type": "private"}), account=ACCOUNT
    )

    assert mine.provider_message_id != theirs.provider_message_id
    assert "987654321" in mine.provider_message_id


def test_a_message_with_no_text_still_parses():
    """A sticker or a voice note is a turn, not an error."""
    body = json.dumps(
        {
            "update_id": 2,
            "message": {
                "message_id": 7,
                "date": 1755859200,
                "chat": {"id": 5, "type": "private"},
                "sticker": {"file_id": "abc"},
            },
        }
    ).encode()
    message = parse_inbound(raw_body=body, account=ACCOUNT)

    assert message.text is None
    assert message.sender_ref == "5"


def test_an_update_that_is_not_a_message_is_refused():
    """Telegram delivers edits, channel posts, poll answers and callback
    queries down the same webhook. Treating one as a message would answer a
    customer's edit of a question they already asked."""
    from moc.channels.telegram import NotAMessage

    body = json.dumps({"update_id": 3, "edited_message": {"message_id": 7}}).encode()
    with pytest.raises(NotAMessage):
        parse_inbound(raw_body=body, account=ACCOUNT)


# ─────────────────────────── outbound ───────────────────────────


def transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_a_reply_is_sent_to_the_chat():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 55}})

    bot = TelegramBot(token=TOKEN, transport=transport(handler))
    receipt = await bot.send(to="987654321", text="الرسوم 1400 جنيه.")
    await bot.aclose()

    assert sent[0]["chat_id"] == "987654321"
    assert sent[0]["text"] == "الرسوم 1400 جنيه."
    # No parse_mode: composition writes plain sentences, and asking Telegram to
    # parse markdown would make an underscore in an email address italicise
    # half a reply.
    assert "parse_mode" not in sent[0]
    assert receipt.provider_message_id == "55"


async def test_there_is_no_service_window_to_fall_outside_of():
    """Telegram has no 24-hour rule and no templates.

    The WhatsApp adapter refuses freeform text outside the window, and a
    Telegram sender that inherited that refusal would reject every reply after
    a day of quiet — an outage that looks like a policy.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    bot = TelegramBot(token=TOKEN, transport=transport(handler))
    receipt = await bot.send(
        to="1", text="أهلاً", last_inbound_at=datetime(2020, 1, 1, tzinfo=UTC)
    )
    await bot.aclose()

    assert receipt.provider_message_id == "1"


async def test_the_bot_token_never_appears_in_an_error():
    """**The one that matters.**

    The Bot API puts the token in the URL path. httpx's `raise_for_status`
    puts the URL in the exception message. The outbound worker writes
    `repr(exc)` into a dead-letter row. Three pieces of ordinary code, none
    wrong on its own, and the result is a live bot token sitting in the
    database — readable by anyone who can read a queue.
    """
    from moc.channels.telegram import TelegramRefused

    def handler(request: httpx.Request) -> httpx.Response:
        assert TOKEN in str(request.url), "the token has to be in the URL; that is the point"
        return httpx.Response(400, json={"ok": False, "description": "chat not found"})

    bot = TelegramBot(token=TOKEN, transport=transport(handler))
    with pytest.raises(TelegramRefused) as refusal:
        await bot.send(to="nobody", text="مرحبا")
    await bot.aclose()

    rendered = f"{refusal.value!r} {refusal.value}"
    assert TOKEN not in rendered
    assert "chat not found" in rendered, "the reason survives; only the secret does not"


async def test_a_transport_failure_also_hides_the_token():
    """The other way a URL reaches an exception: httpx puts it in
    `ConnectError` too, and that one is not raised by our own code."""
    from moc.channels.telegram import TelegramRefused

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed to connect", request=request)

    bot = TelegramBot(token=TOKEN, transport=transport(handler))
    with pytest.raises(TelegramRefused) as failure:
        await bot.send(to="1", text="مرحبا")
    await bot.aclose()

    assert TOKEN not in f"{failure.value!r} {failure.value}"


def test_no_endpoint_string_is_hardcoded_in_the_adapter():
    """§2.4. The API base is config, which is what makes a proxy or a test
    server a config edit rather than a search through source."""
    import inspect

    from moc.channels import telegram

    assert "api.telegram.org" not in inspect.getsource(telegram)
