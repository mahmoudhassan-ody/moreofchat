"""Stage 2 — the cross-provider judge (eval-harness-spec §5.2, §5.3).

Deterministic checks ran first and are not repeated here. The judge exists for
the dimensions a regex cannot settle: whether a claim traces to the evidence,
whether the register is right, whether the reply moved the customer forward.

Two properties are load-bearing and both are structural rather than advisory:

- **Independence.** A provider never grades its own output. Self-preference
  bias in LLM judges is documented and real, and a judge that quietly failed
  over to the answering provider would produce inflated scores that look
  exactly like good scores.
- **Disagreement escalates.** When the two pairings disagree the case goes to
  a human. Averaging two verdicts produces a number nobody can defend, and
  picking the stricter one is the same decision made silently.

**These tests do not — and the suite must not — support a provider-quality
conclusion.** 22 worked examples prove the machinery works. They do not tell
you which model writes better Egyptian Arabic, and `answer_composition` stays
pinned where it is until the mined corpus can answer that.
"""

import json

import pytest

from moc.config_store import load
from moc.evals.judge import (
    Judge,
    JudgeIndependenceViolation,
    JudgeVerdict,
    reconcile,
)
from moc.evals.schema import ExpectedFact, Register
from moc.llm.fake import FakeProvider
from moc.llm.router import Router

QUESTION = "كام رسوم الساعة المعتمدة لكلية الهندسة؟"
REPLY = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026."
PASSAGE = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026"
FACTS = [ExpectedFact(id="f1", claim="credit hour fee is 1400 EGP")]

GOOD = {
    "fact_coverage": {"f1": "present"},
    "forbidden_violated": [],
    "grounding": 3,
    "register": 3,
    "helpfulness": 3,
    "reasoning": "Fee traced to the passage, MSA register, year stated.",
}


def verdict_json(**overrides) -> str:
    return json.dumps({**GOOD, **overrides}, ensure_ascii=False)


def build(*, anthropic_text: str = "", openai_text: str = "") -> tuple[Judge, dict]:
    providers = {
        "anthropic": FakeProvider("anthropic", text=anthropic_text),
        "openai": FakeProvider("openai", text=openai_text),
    }
    router = Router(config=load("llm/routing"), providers=providers)
    return Judge.from_config(router=router), providers


async def grade(judge: Judge, **overrides) -> JudgeVerdict:
    kwargs = {
        "question": QUESTION,
        "reply": REPLY,
        "retrieved_passages": [PASSAGE],
        "expected_facts": FACTS,
        "forbidden_claims": [],
        "expected_register": Register.msa,
        "answer_provider": "anthropic",
    }
    return await judge.grade(**{**kwargs, **overrides})


# ─────────────────────────── §5.2 independence ───────────────────────────


async def test_judge_provider_differs_from_the_answering_provider():
    """A hard rule, not a preference (§5.2).

    Raising is the whole point: a judge that merely warned would produce a
    green suite with inflated grounding scores, and the inflation is invisible
    because it looks like the model doing well.
    """
    judge, _ = build(anthropic_text=verdict_json())
    with pytest.raises(JudgeIndependenceViolation):
        await grade(judge, answer_provider="anthropic", judge_provider="anthropic")


async def test_the_judge_provider_is_derived_when_it_is_not_named():
    """Deriving it beats trusting the caller to pass the right one."""
    judge, providers = build(openai_text=verdict_json())
    result = await grade(judge, answer_provider="anthropic")
    assert result.provider == "openai"
    assert providers["anthropic"].calls == []


async def test_the_judge_never_fails_over_onto_the_answering_provider():
    """Failover must not launder a violation into a normal-looking verdict.

    Enforced in the router by excluding the provider outright, so there is no
    ordering of candidates in which the answering provider can be reached.
    """
    from moc.llm.base import AllProvidersUnavailable, ProviderUnavailable

    judge, providers = build(anthropic_text=verdict_json())
    providers["openai"].fail_with = ProviderUnavailable("judge provider down")
    with pytest.raises(AllProvidersUnavailable):
        await grade(judge, answer_provider="anthropic")
    assert providers["anthropic"].calls == []


# ─────────────────────────── output contract ───────────────────────────


async def test_returns_json_only_and_parses_it():
    judge, _ = build(openai_text=verdict_json())
    result = await grade(judge)
    assert result.malformed is False
    assert result.reasoning.startswith("Fee traced")


async def test_parses_a_verdict_wrapped_in_a_code_fence():
    """Models fence JSON despite being told not to. Failing the case for that
    would report a grading-harness quirk as a quality regression."""
    judge, _ = build(openai_text=f"```json\n{verdict_json()}\n```")
    result = await grade(judge)
    assert result.malformed is False
    assert result.grounding == 3


@pytest.mark.parametrize(
    "output",
    [
        "I think the reply is quite good overall.",
        "{not json at all",
        json.dumps({"grounding": 3}),
        json.dumps({**GOOD, "grounding": 7}),
        json.dumps({**GOOD, "register": "good"}),
    ],
)
async def test_malformed_judge_output_is_a_failed_case_not_a_crash(output):
    """A judge that returned prose is a failed case, not a broken run.

    Out-of-scale scores are malformed rather than clamped: a 7 means the model
    was not grading against the rubric it was given, and clamping to 3 would
    record a perfect score for a verdict nobody can trust.
    """
    judge, _ = build(openai_text=output)
    result = await grade(judge)
    assert result.malformed is True
    assert result.meets_rubric is False
    assert result.raw == output


async def test_grades_the_three_rubric_dimensions():
    """grounding, register, helpfulness — 0-3 each (§5.3)."""
    judge, _ = build(openai_text=verdict_json(grounding=2, register=3, helpfulness=1))
    result = await grade(judge)
    assert (result.grounding, result.register, result.helpfulness) == (2, 3, 1)
    assert result.meets_rubric is False, "helpfulness 1 is below the configured floor"


async def test_thresholds_come_from_config_not_the_module():
    floors = load("evals/judge")["pass_thresholds"]
    judge, _ = build(openai_text=verdict_json(**dict.fromkeys(floors, floors["grounding"])))
    assert (await grade(judge)).meets_rubric is True


async def test_a_missing_required_fact_fails_the_case():
    judge, _ = build(openai_text=verdict_json(fact_coverage={"f1": "missing"}))
    assert (await grade(judge)).meets_rubric is False


async def test_an_optional_fact_missing_does_not_fail_the_case():
    """§3.1 marks facts required or not so partial credit means something."""
    judge, _ = build(openai_text=verdict_json(fact_coverage={"f2": "missing"}))
    result = await grade(
        judge, expected_facts=[ExpectedFact(id="f2", claim="library hours", required=False)]
    )
    assert result.meets_rubric is True


async def test_a_forbidden_claim_fails_the_case_whatever_the_scores_are():
    judge, _ = build(openai_text=verdict_json(forbidden_violated=["fee_estimate"]))
    assert (await grade(judge)).meets_rubric is False


async def test_grounding_zero_is_a_hard_fail_regardless_of_the_others():
    """§5.3: an unsupported figure is a hard fail whatever else scored."""
    judge, _ = build(openai_text=verdict_json(grounding=0, register=3, helpfulness=3))
    assert (await grade(judge)).meets_rubric is False


# ─────────────────────────── §5.2 disagreement ───────────────────────────


def verdict(**overrides) -> JudgeVerdict:
    defaults = {
        "provider": "openai",
        "model": "m",
        "fact_coverage": {"f1": "present"},
        "forbidden_violated": (),
        "grounding": 3,
        "register": 3,
        "helpfulness": 3,
        "reasoning": "",
        "meets_rubric": True,
    }
    return JudgeVerdict(**{**defaults, **overrides})


def test_disagreement_between_judges_goes_to_the_human_queue():
    """§5.2: escalate, never pick one.

    The two pairings graded the same case and reached opposite conclusions.
    That is precisely the case a human should read — and averaging it produces
    a 1.5 that no rubric row defines.
    """
    outcome = reconcile([verdict(), verdict(provider="anthropic", grounding=0, meets_rubric=False)])
    assert outcome.escalate is True
    assert outcome.verdict is None, "no winner may be picked"


def test_agreement_does_not_escalate():
    outcome = reconcile([verdict(), verdict(provider="anthropic")])
    assert outcome.escalate is False
    assert outcome.verdict is not None


def test_a_score_gap_within_tolerance_is_agreement():
    """One rubric point apart is two graders reading the same reply, not a
    dispute. Escalating on it would send the whole suite to the human queue."""
    gap = load("evals/judge")["disagreement"]["max_score_gap"]
    outcome = reconcile([verdict(), verdict(provider="anthropic", register=3 - gap)])
    assert outcome.escalate is False


def test_a_score_gap_beyond_tolerance_escalates():
    gap = load("evals/judge")["disagreement"]["max_score_gap"]
    outcome = reconcile([verdict(), verdict(provider="anthropic", register=3 - gap - 1)])
    assert outcome.escalate is True


def test_a_malformed_verdict_escalates_rather_than_deferring_to_the_other():
    """One judge failing to answer is not the other judge being right."""
    outcome = reconcile(
        [verdict(), verdict(provider="anthropic", malformed=True, meets_rubric=False)]
    )
    assert outcome.escalate is True


def test_two_verdicts_from_the_same_provider_are_not_independent():
    with pytest.raises(JudgeIndependenceViolation):
        reconcile([verdict(), verdict()])


# ─────────────────────────── §3.1: no golden answer ───────────────────────────


async def test_judge_never_sees_the_expected_answer_text():
    """It grades against passages and expected *facts*, never a model answer.

    Showing it a golden reply turns it into a paraphrase detector, and the
    system's whole premise (§3.1) is that many different sentences are correct
    so long as every figure traces to a source.
    """
    judge, providers = build(openai_text=verdict_json())
    await grade(judge)
    sent = str(providers["openai"].calls)
    assert "credit hour fee is 1400 EGP" in sent, "expected facts are legitimate input"
    assert "gold" not in sent.lower()


def test_grade_has_no_parameter_for_a_reference_reply():
    """A drift guard. The paraphrase-detector failure arrives as a helpful
    new keyword argument, not as someone deciding to break §3.1."""
    import inspect

    banned = {"expected_reply", "golden", "golden_answer", "reference", "reference_reply"}
    assert not banned & set(inspect.signature(Judge.grade).parameters)


# ─────────────────────────── §2.3: version pinning ───────────────────────────


async def test_prompt_version_is_recorded_in_run_metadata():
    judge, _ = build(openai_text=verdict_json())
    binding = judge.task_binding(provider="openai", model="gpt-5.6-sol")
    assert binding.task == "eval_grading"
    assert binding.prompt_version.startswith(load("evals/judge")["prompt"])


def test_the_recorded_prompt_version_changes_when_the_prompt_text_changes():
    """§2.3's gap, closed.

    `config_hash` covers config/ — it does not cover a prompt file under src/.
    Without the digest, editing the judge prompt would change every score in
    the suite while every run still claimed to be comparable to the baseline.
    """
    judge, _ = build()
    before = judge.prompt_version
    path = judge.prompt_path
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(original + "\nAlso be strict about years.\n", encoding="utf-8")
        judge_after, _ = build()
        assert judge_after.prompt_version != before
    finally:
        path.write_text(original, encoding="utf-8")


async def test_the_rubric_reaches_the_prompt_from_config():
    """Rubric text lives in judge.yaml so a threshold moves without moving
    prompt_version — and so it is covered by config_hash instead."""
    judge, providers = build(openai_text=verdict_json())
    await grade(judge)
    sent = str(providers["openai"].calls)
    rubric = load("evals/judge")["rubric"]["grounding"]["3"]
    assert rubric in sent


async def test_the_reasoning_cap_is_stated_in_the_prompt():
    """§5.2 caps it at one sentence, 30 words. An unbounded judge writes an
    essay per case, and 230 cases of essays is a bill nobody approved."""
    judge, providers = build(openai_text=verdict_json())
    await grade(judge)
    assert str(load("evals/judge")["max_reasoning_words"]) in str(providers["openai"].calls)


async def test_the_system_prompt_is_chosen_per_provider_family():
    """Prompt conventions differ between the two, so the system block does."""
    judge, providers = build(openai_text=verdict_json(), anthropic_text=verdict_json())
    await grade(judge, answer_provider="anthropic")
    await grade(judge, answer_provider="openai")
    systems = {
        p.calls[0]["system"] for p in providers.values() if p.calls
    }
    assert len(systems) == 2, "both families received the same system prompt"


# ───────── §3.1 applied to the judge: what the turn was authorised to state ─────────


def test_the_prompt_carries_the_script_s_own_statements():
    """The judge grades against evidence, and `passages` was not all of it.

    A scripted reply is the tenant's sentence, written by a human into
    `replies.yaml`. Graded against the passages retrieved for the customer's
    question — which it was never composed from — it reads as an unsupported
    claim every time. edu-0017 is that failure with both of its expected facts
    marked present: it refuses career advice, offers the faculties and their
    thresholds, and scores grounding 1 because the passages for "which faculty
    gets me a job" do not list thresholds.

    §3.1 already says a figure held in a script node is as legitimate a source
    as a retrieved chunk. This is the same rule for the sentence around it.
    """
    judge, _ = build()
    prompt = judge._render(
        question="q",
        reply="r",
        retrieved_passages=["a passage"],
        expected_facts=[],
        forbidden_claims=[],
        expected_register="masri",
        script_statements=["مش هينفع أرشحلك كلية معينة."],
    )
    assert "مش هينفع أرشحلك كلية معينة." in prompt
    assert "a passage" in prompt


def test_a_turn_with_no_script_statements_says_so():
    """`(none)` rather than an empty section. A blank list under a heading
    reads to a model as "there were some and they are not shown"."""
    judge, _ = build()
    prompt = judge._render(
        question="q",
        reply="r",
        retrieved_passages=["a passage"],
        expected_facts=[],
        forbidden_claims=[],
        expected_register="masri",
        script_statements=[],
    )
    assert "(none)" in prompt


def test_the_script_statements_are_not_folded_into_the_passages():
    """Two sections, never one list.

    Merged, the judge would report a claim as traceable to "the passages" when
    it traces to config — and `grounding` would stop distinguishing a reply
    that used retrieval from one that recited a template. They are both
    legitimate sources and they are not the same source.
    """
    judge, _ = build()
    prompt = judge._render(
        question="q",
        reply="r",
        retrieved_passages=["RETRIEVED"],
        expected_facts=[],
        forbidden_claims=[],
        expected_register="masri",
        script_statements=["SCRIPTED"],
    )
    assert prompt.index("RETRIEVED") < prompt.index("SCRIPTED")
    between = prompt[prompt.index("RETRIEVED") : prompt.index("SCRIPTED")]
    assert "script" in between.lower(), "nothing separates the two sources"
