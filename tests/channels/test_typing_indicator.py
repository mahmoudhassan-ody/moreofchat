"""The typing indicator — design §2.5, demo plan Task 40.

Perceived latency, not real latency. A p95 turn is about nine seconds and the
design's answer to that was never a faster model: nine seconds of silence reads
worse than nine seconds of "seen, typing".

Three things make this unlike every other call the WhatsApp adapter makes, and
each is a test here.

**It is not a message.** It carries no content, it is not rate limited with
replies, and the 24-hour service window does not apply to it. So it does not go
through the outbound queue — a courtesy that arrives after the reply it was
meant to precede is worse than none, and the token bucket is exactly what would
delay it.

**A different host, a different API version, a different encoding.** Everything
else goes to `api.twilio.com/2010-04-01/Accounts/{sid}` as form data. This goes
to `messaging.twilio.com/v3` as JSON.

**Its failure must be invisible to the customer and harmless to the turn.** It
is a courtesy on the turn path. An indicator that raises would fail a turn that
was otherwise fine, and the customer would lose the answer to save the hint.
"""

import json
from uuid import uuid4

import httpx
import pytest

from moc.channels.base import Channel, ChannelAccount
from moc.channels.twilio_wa import TwilioTypingIndicator
from moc.config_store import load

CONFIG = load("channels/whatsapp")
SETTINGS = CONFIG["typing_indicator"]
AUTH_TOKEN = "twilio-auth-token-value"  # noqa: S105 - a test fixture
ACCOUNT_SID = "AC00000000000000000000000000000000"
INBOUND_SID = "SM0123456789abcdef0123456789abcdef"

ACCOUNT = ChannelAccount(
    id=uuid4(),
    tenant_id=uuid4(),
    channel=Channel.whatsapp,
    account_ref="+201555000111",
    secret_ref="twilio/test/wa",
)


def indicator(handler, **overrides) -> TwilioTypingIndicator:
    return TwilioTypingIndicator(
        account_sid=ACCOUNT_SID,
        auth_token=AUTH_TOKEN,
        transport=httpx.MockTransport(handler),
        **overrides,
    )


def accepts(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"success": True})

    return handler


# ─────────────────────── a different call from every other ───────────────────────


async def test_it_goes_to_the_messaging_host_and_not_the_message_host():
    """`api.twilio.com/2010-04-01/Accounts/{sid}` is where replies go. Sending
    this there is a 404 that reads as the feature not existing."""
    seen: list[httpx.Request] = []
    bot = indicator(accepts(seen))
    await bot.typing(message_id=INBOUND_SID)
    await bot.aclose()

    assert str(seen[0].url).startswith(SETTINGS["api_base"])
    assert ACCOUNT_SID not in str(seen[0].url), "the account is in the auth, not the path"


async def test_it_sends_json_and_not_a_form():
    seen: list[httpx.Request] = []
    bot = indicator(accepts(seen))
    await bot.typing(message_id=INBOUND_SID)
    await bot.aclose()

    assert seen[0].headers["content-type"].startswith("application/json")
    assert json.loads(seen[0].content) == {
        "messageId": INBOUND_SID,
        "channel": SETTINGS["channel"],
    }


async def test_it_carries_the_inbound_message_id_that_the_turn_already_holds():
    """The one input this needed was already on `InboundMessage`: Twilio's
    `MessageSid` becomes `provider_message_id` in §6.1's normalized shape."""
    seen: list[httpx.Request] = []
    bot = indicator(accepts(seen))
    await bot.typing(message_id=INBOUND_SID)
    await bot.aclose()

    assert json.loads(seen[0].content)["messageId"] == INBOUND_SID


async def test_it_authenticates_with_what_the_adapter_actually_holds():
    """Twilio's docs specify an API key/secret pair for this resource; the
    adapter holds an account SID and auth token. Whether SID/token also
    authenticates against `messaging.twilio.com/v3` is not stated anywhere and
    has never been tested against the real host — see the module docstring in
    `twilio_wa.py` and the preflight check that answers it."""
    seen: list[httpx.Request] = []
    bot = indicator(accepts(seen))
    await bot.typing(message_id=INBOUND_SID)
    await bot.aclose()

    assert seen[0].headers.get("authorization", "").startswith("Basic ")


# ─────────────────────── failing quietly, and only here ───────────────────────


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"message": "authenticate"}),
        httpx.Response(400, json={"message": "invalid messageId"}),
        httpx.Response(500, text="upstream"),
    ],
)
async def test_a_refused_indicator_reports_false_rather_than_raising(response):
    """The customer must not lose the answer to save the hint."""
    bot = indicator(lambda request: response)
    assert await bot.typing(message_id=INBOUND_SID) is False
    await bot.aclose()


async def test_a_network_failure_reports_false_rather_than_raising():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    bot = indicator(explode)
    assert await bot.typing(message_id=INBOUND_SID) is False
    await bot.aclose()


async def test_a_successful_indicator_reports_true():
    """The negative control. Without it, an implementation that always returned
    False would pass every test above."""
    bot = indicator(accepts([]))
    assert await bot.typing(message_id=INBOUND_SID) is True
    await bot.aclose()


async def test_nothing_is_sent_when_the_indicator_is_switched_off():
    """`enabled` is a product decision, not a capability question: the call
    marks the customer's message read and that is not separable."""
    seen: list[httpx.Request] = []
    off = {**CONFIG, "typing_indicator": {**SETTINGS, "enabled": False}}
    bot = indicator(accepts(seen), config=off)
    assert await bot.typing(message_id=INBOUND_SID) is False
    await bot.aclose()
    assert seen == [], "a disabled indicator still sent a read receipt"


def test_the_indicator_contains_no_raise_at_all():
    """Structural, and the reason it is structural: `typing` returning False on
    every failure is easy to write and easy to erode. One `raise` added later
    for a case that "should never happen" turns a courtesy into something that
    can fail a turn, and it would pass every behavioural test above.

    Asserted over the syntax tree rather than the text, because the docstrings
    here talk about raising.
    """
    import ast
    import inspect

    from moc.channels.twilio_wa import TwilioTypingIndicator

    tree = ast.parse(inspect.getsource(TwilioTypingIndicator))
    raises = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]
    assert raises == [], (
        "the indicator raises; a courtesy on the turn path must not be able to "
        "fail the turn"
    )
    assert "raise_for_status" not in inspect.getsource(TwilioTypingIndicator)
