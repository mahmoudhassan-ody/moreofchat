"""Live smoke tests — real network, real money, opt-in only.

Marked `live` and excluded from CI. They exist because mocked adapter tests
prove self-consistency and nothing more: every fixture in test_adapters.py was
authored by the same person as the code it checks. These are the only tests
that can catch "the request shape is wrong" or "that model id does not exist".

Run with:  uv run pytest -m live

Cost is a few thousand input tokens per full run. The caching test is the
expensive one, and it is worth it — design §10 calls prompt caching the primary
cost lever, and an unverified cost lever is an assumption on a spreadsheet.
"""

import os

import pytest

from moc import config_store
from moc.arabic.script import detect_language
from moc.llm.anthropic_direct import AnthropicDirect
from moc.llm.base import Message
from moc.llm.openai_direct import OpenAIDirect

pytestmark = pytest.mark.live

ROUTING = config_store.load("llm/routing")
ANSWER = ROUTING["tasks"]["answer_composition"]
EMBEDDING = ROUTING["tasks"]["embedding"]["primary"]


def _key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        pytest.skip(f"{name} not set")
    return value


@pytest.fixture
def anthropic() -> AnthropicDirect:
    return AnthropicDirect(api_key=_key("MOC_ANTHROPIC_API_KEY"), http=ROUTING["http"])


@pytest.fixture
def openai() -> OpenAIDirect:
    return OpenAIDirect(api_key=_key("MOC_OPENAI_API_KEY"), http=ROUTING["http"])


async def test_live_anthropic_returns_arabic(anthropic):
    result = await anthropic.complete(
        model=ANSWER["primary"]["model"],
        messages=[Message(role="user", content="رد بكلمة واحدة بالعربية: تمام")],
        system="أجب بالعربية فقط.",
        cache_blocks=[],
        # The configured budget, not a token or two. Both frontier models spend
        # output budget on reasoning before emitting text, so a small max_tokens
        # returns an empty completion rather than a short one.
        max_tokens=ANSWER["max_tokens"],
    )
    assert result.text.strip()
    assert detect_language(result.text) == "ar"
    assert result.input_tokens > 0
    assert result.output_tokens > 0


async def test_live_openai_returns_arabic(openai):
    """The failover path answers customers too, so it gets the same check."""
    result = await openai.complete(
        model=ANSWER["failover"]["model"],
        messages=[Message(role="user", content="رد بكلمة واحدة بالعربية: تمام")],
        system="أجب بالعربية فقط.",
        cache_blocks=[],
        max_tokens=ANSWER["max_tokens"],
    )
    assert result.text.strip()
    assert detect_language(result.text) == "ar"
    assert result.input_tokens > 0


async def test_live_openai_embedding_has_configured_dimension(openai):
    vectors = await openai.embed(
        model=EMBEDDING["model"],
        texts=["رسوم الساعة المعتمدة لكلية الهندسة"],
        dimensions=EMBEDDING["dimensions"],
    )
    assert [len(v) for v in vectors] == [EMBEDDING["dimensions"]]


async def test_live_embedding_vectors_differ_per_input(openai):
    """A bug returning one vector for every chunk would pass a length check."""
    vectors = await openai.embed(
        model=EMBEDDING["model"],
        texts=["رسوم كلية الهندسة", "شروط القبول بكلية الصيدلة"],
        dimensions=EMBEDDING["dimensions"],
    )
    assert vectors[0] != vectors[1]


async def test_live_anthropic_prompt_caching_reports_a_cache_read(anthropic):
    """§10's primary cost lever, measured rather than assumed.

    Second identical call must read from cache. The block has to clear the
    provider's minimum cacheable size, which is why it is padded rather than
    short — a small block silently caches nothing and the test would pass with
    cached_tokens == 0 if it asserted only >= 0.
    """
    block = (
        "You are the admissions assistant for Sinai University. Answer only "
        "from retrieved passages. Never estimate or round a fee. "
    ) * 120

    async def call():
        return await anthropic.complete(
            model=ANSWER["primary"]["model"],
            messages=[Message(role="user", content="Reply with: ok")],
            system=None,
            cache_blocks=[block],
            max_tokens=8,
        )

    await call()
    second = await call()
    assert second.cached_tokens > 0, "prompt caching is not reaching the provider"


async def test_live_model_ids_in_routing_are_real(anthropic, openai):
    """A 404 here means the config points at a model that does not exist.

    Cheaper than discovering it on a customer's turn, and the failure mode is
    a config error rather than a reason to fail over.
    """
    slots = ROUTING["tasks"]["slot_extraction"]
    for model in {ANSWER["primary"]["model"], slots["primary"]["model"]}:
        result = await anthropic.complete(
            model=model,
            messages=[Message(role="user", content="Reply with: ok")],
            system=None,
            cache_blocks=[],
            max_tokens=8,
        )
        assert result.model

    for model in {ANSWER["failover"]["model"], slots["failover"]["model"]}:
        result = await openai.complete(
            model=model,
            messages=[Message(role="user", content="Reply with: ok")],
            system=None,
            cache_blocks=[],
            max_tokens=64,
        )
        assert result.model
