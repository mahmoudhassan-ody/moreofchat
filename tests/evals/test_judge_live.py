"""The judge against real models — opt-in, real money.

Run with:  uv run pytest -m live

Every other judge test feeds the parser a string the same person wrote. That
proves the parser handles what its author expected and nothing else, and this
module's entire premise is an assumption about model behaviour: that a frontier
model handed this prompt returns one JSON object and no prose. Both providers,
both directions, because §5.2 runs both pairings and a prompt that only works
on one family is a prompt that silently halves the suite.

Two cases, deliberately: one reply that should score well and one containing a
figure absent from the passages. A judge that returns valid JSON but grades
everything a 3 passes a smoke test and is worthless, so the second case checks
the judge can actually see the failure it exists to catch.

Cost is four completions per run at eval_grading's settings, which is
reasoning-on Opus and Sol. Not cheap; run it when the prompt or rubric changes.
"""

import os

import pytest

from moc.config_store import load
from moc.evals.judge import Judge
from moc.evals.schema import ExpectedFact, Register
from moc.llm.anthropic_direct import AnthropicDirect
from moc.llm.openai_direct import OpenAIDirect
from moc.llm.router import Router

pytestmark = pytest.mark.live

ROUTING = load("llm/routing")

QUESTION = "كام رسوم الساعة المعتمدة لكلية الهندسة؟"
PASSAGES = [
    "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026/2025.",
    "الحد الأدنى للتسجيل 12 ساعة معتمدة في الفصل الدراسي الواحد.",
]
FACTS = [ExpectedFact(id="f1", claim="The credit hour fee for Engineering is 1400 EGP")]

GROUNDED = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026/2025."
UNGROUNDED = "رسوم الساعة المعتمدة لكلية الهندسة 1750 جنيهًا، والتسجيل يغلق في 15 سبتمبر."


def _key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        pytest.skip(f"{name} not set")
    return value


@pytest.fixture
def judge() -> Judge:
    providers = {
        "anthropic": AnthropicDirect(
            api_key=_key("MOC_ANTHROPIC_API_KEY"), http=ROUTING["http"]
        ),
        "openai": OpenAIDirect(api_key=_key("MOC_OPENAI_API_KEY"), http=ROUTING["http"]),
    }
    return Judge.from_config(router=Router(config=ROUTING, providers=providers))


async def _grade(judge: Judge, reply: str, answer_provider: str):
    return await judge.grade(
        question=QUESTION,
        reply=reply,
        retrieved_passages=PASSAGES,
        expected_facts=FACTS,
        forbidden_claims=["fee_estimate", "deadline"],
        expected_register=Register.msa,
        answer_provider=answer_provider,
    )


@pytest.mark.parametrize("answer_provider", ["anthropic", "openai"])
async def test_live_the_judge_returns_parseable_json(judge, answer_provider, capsys):
    """Both families, on the real prompt.

    `malformed` here means the prompt failed, not the reply — so the assertion
    message says so. Nothing else in the suite can tell those two apart.
    """
    result = await _grade(judge, GROUNDED, answer_provider)

    with capsys.disabled():
        print(
            f"\n  judged by {result.provider}/{result.model}: "
            f"grounding={result.grounding} register={result.register} "
            f"helpfulness={result.helpfulness} | {result.reasoning}"
        )

    assert not result.malformed, (
        f"the judge prompt did not produce JSON on {result.provider}: {result.raw[:300]!r}"
    )
    assert result.provider != answer_provider


@pytest.mark.parametrize("answer_provider", ["anthropic", "openai"])
async def test_live_the_judge_catches_an_ungrounded_figure(judge, answer_provider, capsys):
    """The check that makes the previous test worth running.

    1750 appears in no passage and the September deadline is invented, so §5.3
    puts grounding at 0 — a hard fail. A judge that returns clean JSON and
    grades this a 3 is a judge that would pass every case in the suite.
    """
    result = await _grade(judge, UNGROUNDED, answer_provider)

    with capsys.disabled():
        print(
            f"\n  judged by {result.provider}: grounding={result.grounding} "
            f"forbidden={list(result.forbidden_violated)} | {result.reasoning}"
        )

    assert not result.malformed, f"unparseable: {result.raw[:300]!r}"
    assert result.meets_rubric is False, (
        "an unsupported fee and an invented deadline must not pass the rubric"
    )
    assert result.grounding <= load("evals/judge")["pass_thresholds"]["grounding"] - 1
