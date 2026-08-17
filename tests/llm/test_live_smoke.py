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
import time

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


async def test_live_every_configured_candidate_is_callable_as_configured(anthropic, openai):
    """Every (model, reasoning, effort) triple in routing.yaml, against the real API.

    Not just "does this model id exist" — the triple has to be valid together.
    claude-haiku-4-5 rejects `thinking: adaptive` *and* the effort parameter,
    each with its own 400, so a config that pairs it with either is broken in a
    way only a live call reveals. Both of those are ProviderRequestError, which
    means they would never fail over — the task would simply stop working.
    """
    clients = {"anthropic": anthropic, "openai": openai}
    for name, spec in ROUTING["tasks"].items():
        if name == "embedding":
            continue
        for role in ("primary", "failover"):
            candidate = spec.get(role)
            if not candidate:
                continue
            result = await clients[candidate["provider"]].complete(
                model=candidate["model"],
                messages=[Message(role="user", content="Reply with: ok")],
                system=None,
                cache_blocks=[],
                max_tokens=spec["max_tokens"],
                reasoning=candidate["reasoning"],
                effort=candidate.get("effort"),
            )
            assert result.model, f"{name}.{role} ({candidate['model']}) returned no model"


# ─────────────────────────── reasoning + latency (§2.5) ───────────────────────────


async def test_live_answer_composition_reasoning_is_actually_off(anthropic):
    """claude-sonnet-5 thinks by default; config says otherwise for this task.

    Asserted against the provider rather than the request body, because the
    body only proves what we sent — a silently ignored parameter would still
    burn thinking tokens and seconds on every customer turn.
    """
    result = await anthropic.complete(
        model=ANSWER["primary"]["model"],
        messages=[Message(role="user", content="رسوم الساعة 1400 جنيه لعام 2026. رد قصير.")],
        system=None,
        cache_blocks=[],
        max_tokens=ANSWER["max_tokens"],
        reasoning=ANSWER["primary"]["reasoning"],
        effort=ANSWER["primary"].get("effort"),
    )
    assert result.text.strip()
    assert result.stop_reason == "end_turn", "a truncated answer means max_tokens is too low"


def _grounded_turn() -> tuple[str, str]:
    """A realistic answer-composition prompt: cached preamble + retrieved passages."""
    preamble = (
        "أنت مساعد القبول بجامعة سيناء. أجب فقط من المقاطع المسترجعة. "
        "لا تقدّر أي رسوم ولا تقرّبها. اذكر دائماً العام الدراسي مع أي رقم. "
    ) * 40
    passages = (
        "رسوم الساعة المعتمدة لكلية الهندسة للعام الجامعي 2026 هي 1400 جنيه. "
        "عدد الساعات المعتمدة للبرنامج 160 ساعة. الدفع على أربعة أقساط."
    )
    return preamble, passages


async def _timed_answer_composition(provider) -> tuple[float, object]:
    preamble, passages = _grounded_turn()
    started = time.perf_counter()
    result = await provider.complete(
        model=ANSWER["primary"]["model"],
        messages=[Message(role="user", content="المصاريف كام لكلية الهندسة؟")],
        system=passages,
        cache_blocks=[preamble],
        max_tokens=ANSWER["max_tokens"],
        reasoning=ANSWER["primary"]["reasoning"],
        effort=ANSWER["primary"].get("effort"),
    )
    return (time.perf_counter() - started) * 1000, result


async def test_live_answer_composition_latency_is_recorded(anthropic, capsys):
    """§2.5: measure the model-call segment, don't assume it.

    One realistic grounded turn — cached tenant preamble, retrieved passages,
    an Arabic question — timed end to end. The assertion is the *ceiling*, not
    the target: it catches reasoning silently switching back on or a provider
    incident, without flaking on the normal spread. Whether we hit the target
    is the next test's job, and the printed line is the number worth reading.
    """
    elapsed_ms, result = await _timed_answer_composition(anthropic)
    budget = ROUTING["latency_budget"]

    with capsys.disabled():
        print(
            f"\n  answer_composition: {elapsed_ms:.0f} ms"
            f" | target {budget['model_call_p95_ms']} ms"
            f" | turn budget {budget['end_to_end_p95_ms']} ms"
            f" ({elapsed_ms / budget['end_to_end_p95_ms']:.0%} consumed)"
            f" | out={result.output_tokens} in={result.input_tokens}"
            f" cached={result.cached_tokens}"
        )

    assert result.text.strip()
    assert result.stop_reason == "end_turn", "a truncated answer means max_tokens is too low"
    assert elapsed_ms < budget["model_call_ceiling_ms"], (
        f"{elapsed_ms:.0f} ms is past the {budget['model_call_ceiling_ms']} ms ceiling — "
        f"something is wrong, not merely slow"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Measured 2026-08-17 from the Frankfurt VPS: min 2746 / median 4551 / "
        "max 5052 ms for a ~310-token Arabic answer with reasoning off. The "
        "model call alone therefore consumes the whole §2.5 turn budget before "
        "retrieval, guards and the channel hop are counted. Two measured levers: "
        "effort low roughly halved output tokens in probing, and a shorter reply "
        "instruction would too — but both trade against grounding quality, which "
        "is an eval-suite question, not a guess. Encodes the intended end state; "
        "starts passing when one of those lands."
    ),
)
async def test_live_answer_composition_meets_the_model_call_target(anthropic):
    elapsed_ms, _ = await _timed_answer_composition(anthropic)
    assert elapsed_ms < ROUTING["latency_budget"]["model_call_p95_ms"]
