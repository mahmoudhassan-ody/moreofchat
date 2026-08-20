"""Stage 2 — the cross-provider judge (eval-harness-spec §5.2 and §5.3).

Stage 1 already ran. Every figure has been traced, every action compared, every
slot checked — by code, for free, without flakiness. This module handles only
what code cannot settle: whether a claim traces to the evidence, whether the
register fits the node's policy, whether the reply moved the customer forward.

Two rules here are structural rather than advisory, because both failures are
silent when they happen:

**A provider never grades its own output.** Self-preference bias in LLM judges
is documented and real, and an inflated grounding score looks exactly like a
model doing well. `grade` refuses the pairing, and the router is asked to
exclude the answering provider outright — so there is no ordering of
candidates, and no failover path, that reaches it. If the remaining judge is
down the call raises: a grade from the answering provider is worse than no
grade, because a missing grade is visible and a biased one is not.

**Disagreement escalates; nothing picks a winner.** Averaging two verdicts
produces a score matching no row of the rubric, and taking the stricter one is
the same arbitrary choice made quietly. `reconcile` returns the case to a
human (§5.2).

What the judge is never given is a model answer. It grades against passages and
against atomic expected facts, because §3.1's premise is that many different
sentences are correct so long as every figure traces to a source — and a judge
holding a golden reply grades paraphrase distance instead.

**On reading its output:** 22 worked examples are enough to prove this
machinery runs. They are not enough to conclude which provider writes better
Egyptian Arabic, and `answer_composition` stays pinned where it is until the
mined corpus can answer that question.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moc.config_store import load
from moc.evals.run_metadata import TaskBinding
from moc.evals.schema import ExpectedFact, Register
from moc.llm.base import Message, Role, Task
from moc.llm.router import Router

_JUDGE = "evals/judge"
_PROMPTS = Path(__file__).parent / "prompts"
_PROMPT_SUFFIX = ".md"

_PRESENT = "present"
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

_DIMENSIONS = ("grounding", "register", "helpfulness")


class JudgeIndependenceViolation(Exception):
    """A provider was asked to grade its own output (§5.2).

    Deliberately an exception rather than a warning or a degraded flag. The
    result of the violation is a plausible verdict with inflated scores, which
    no downstream check can distinguish from a genuinely good reply — so it has
    to be stopped where it is detectable, which is here.
    """


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's assessment of one reply.

    `malformed` is a first-class outcome, not an error. A judge that returned
    prose is a failed case: the run continues, the case does not pass, and the
    raw text is kept so someone can see what it actually said rather than
    guessing from a parse error.
    """

    provider: str
    model: str
    fact_coverage: Mapping[str, str] = field(default_factory=dict)
    forbidden_violated: tuple[str, ...] = ()
    grounding: int = 0
    register: int = 0
    helpfulness: int = 0
    reasoning: str = ""
    meets_rubric: bool = False
    malformed: bool = False
    raw: str = ""

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _DIMENSIONS}


@dataclass(frozen=True)
class Reconciliation:
    """The outcome of comparing the two pairings' verdicts on one case.

    `verdict` is None whenever `escalate` is True, and that is enforced rather
    than conventional: a caller that reached for a verdict on an escalated case
    would be picking a winner, which §5.2 forbids.
    """

    escalate: bool
    verdicts: tuple[JudgeVerdict, ...]
    reasons: tuple[str, ...] = ()
    verdict: JudgeVerdict | None = None


class Judge:
    def __init__(self, *, router: Router, config: Mapping[str, Any]) -> None:
        self._router = router
        self._config = config
        self.prompt_path = _PROMPTS / f"{config['prompt']}{_PROMPT_SUFFIX}"
        self._template = self.prompt_path.read_text(encoding="utf-8")

    @classmethod
    def from_config(cls, *, router: Router) -> Judge:
        return cls(router=router, config=load(_JUDGE))

    # ─────────────────────────── §2.3 version pinning ───────────────────────────

    @property
    def prompt_version(self) -> str:
        """Prompt name plus a digest of its text.

        `config_hash` covers `config/`, and this prompt lives under `src/`.
        Without the digest, editing the judge's instructions would change every
        score in the suite while every run still claimed to be comparable to a
        baseline measured under the old wording — the exact failure §2.3 exists
        to prevent, arriving through the one file it does not watch.
        """
        digest = hashlib.sha256(self._template.encode("utf-8")).hexdigest()
        return f"{self._config['prompt']}+{digest[:12]}"

    def task_binding(self, *, provider: str, model: str) -> TaskBinding:
        return TaskBinding(
            task=str(Task.eval_grading),
            prompt_version=self.prompt_version,
            provider=provider,
            model=model,
        )

    # ─────────────────────────── grading ───────────────────────────

    async def grade(
        self,
        *,
        question: str,
        reply: str,
        retrieved_passages: Sequence[str],
        expected_facts: Sequence[ExpectedFact],
        forbidden_claims: Sequence[str],
        expected_register: Register | str,
        answer_provider: str,
        judge_provider: str | None = None,
        script_statements: Sequence[str] = (),
    ) -> JudgeVerdict:
        """Grade one reply with a provider that did not write it.

        There is no parameter for a reference reply, and there must never be
        one: the paraphrase-detector failure (§3.1) arrives as a helpful new
        keyword argument, not as a decision anyone records.
        """
        if judge_provider is not None and judge_provider == answer_provider:
            raise JudgeIndependenceViolation(
                f"§5.2: {answer_provider!r} cannot grade its own output. Self-preference "
                f"bias produces inflated scores that look exactly like good scores."
            )

        prompt = self._render(
            question=question,
            reply=reply,
            retrieved_passages=retrieved_passages,
            expected_facts=expected_facts,
            forbidden_claims=forbidden_claims,
            expected_register=expected_register,
            script_statements=script_statements,
        )
        completion = await self._router.complete(
            task=Task.eval_grading,
            messages=[Message(role=Role.user, content=prompt)],
            system=self._system_for(judge_provider, answer_provider),
            # The structural half of the independence rule. Excluding beats
            # preferring: a preference is satisfied by failover.
            exclude_provider=answer_provider,
        )
        return self._parse(completion.text, completion.provider, completion.model, expected_facts)

    def _system_for(self, judge_provider: str | None, answer_provider: str) -> str:
        """Per provider family (§2.6).

        The same words are not the same instruction on both vendors — they
        differ in how much a system block steers versus the first user turn —
        and sharing one would advantage whichever family it was written for, in
        a comparison whose entire purpose is fairness.
        """
        prompts = self._config["system_prompts"]
        family = judge_provider or next(
            name for name in prompts if name != answer_provider
        )
        return prompts[family]

    # ─────────────────────────── prompt ───────────────────────────

    def _render(
        self,
        *,
        question: str,
        reply: str,
        retrieved_passages: Sequence[str],
        expected_facts: Sequence[ExpectedFact],
        forbidden_claims: Sequence[str],
        expected_register: Register | str,
        script_statements: Sequence[str] = (),
    ) -> str:
        scale = self._config["scale"]
        # str.format is unusable here: the prompt contains a literal JSON
        # example full of braces, and escaping every one of them would make the
        # output contract unreadable in the file where it has to be read.
        filled = self._template
        for key, value in {
            "question": question,
            "reply": reply,
            "passages": _numbered(retrieved_passages),
            "script_statements": _numbered(script_statements),
            "expected_facts": _facts(expected_facts),
            "forbidden_claims": _bulleted(forbidden_claims),
            "expected_register": str(expected_register),
            "rubric": _rubric(self._config["rubric"]),
            "max_reasoning_words": self._config["max_reasoning_words"],
            "scale_min": scale["min"],
            "scale_max": scale["max"],
        }.items():
            filled = filled.replace("{" + key + "}", str(value))
        return filled

    # ─────────────────────────── parsing ───────────────────────────

    def _parse(
        self,
        text: str,
        provider: str,
        model: str,
        expected_facts: Sequence[ExpectedFact],
    ) -> JudgeVerdict:
        # The flag matters as much as the failure. Without it an unparseable
        # verdict lands as a well-formed all-zeros grade, and the report reads
        # "the model wrote a terrible reply" where the truth is "the judge did
        # not answer" — a grading-harness fault charged to the model under test.
        malformed = JudgeVerdict(provider=provider, model=model, raw=text, malformed=True)
        fenced = _FENCE.match(text)
        try:
            document = json.loads(fenced.group(1) if fenced else text)
        except (json.JSONDecodeError, TypeError):
            return malformed
        if not isinstance(document, dict):
            return malformed

        scale = self._config["scale"]
        scores = {}
        for name in _DIMENSIONS:
            value = document.get(name)
            # Out of scale is malformed, never clamped. A 7 means the model was
            # not grading against the rubric it was handed, and clamping it to 3
            # would record a perfect score for a verdict nobody can trust.
            if not isinstance(value, int) or isinstance(value, bool):
                return malformed
            if not scale["min"] <= value <= scale["max"]:
                return malformed
            scores[name] = value

        coverage = document.get("fact_coverage")
        violated = document.get("forbidden_violated")
        if not isinstance(coverage, dict) or not isinstance(violated, list):
            return malformed

        return JudgeVerdict(
            provider=provider,
            model=model,
            fact_coverage=coverage,
            forbidden_violated=tuple(violated),
            reasoning=str(document.get("reasoning", "")),
            meets_rubric=self._meets_rubric(scores, coverage, violated, expected_facts),
            raw=text,
            **scores,
        )

    def _meets_rubric(
        self,
        scores: Mapping[str, int],
        coverage: Mapping[str, str],
        violated: Sequence[str],
        expected_facts: Sequence[ExpectedFact],
    ) -> bool:
        """§5.3's pass definition, minus `expected_action`.

        Action match is a deterministic check (§5.1) and is asserted by code
        before the judge is ever called. Re-deciding it here would give one
        dimension two owners that can disagree.
        """
        if violated:
            return False
        if any(scores[name] == self._config["scale"]["min"] for name in self._config["hard_zero"]):
            return False
        if any(scores[name] < floor for name, floor in self._config["pass_thresholds"].items()):
            return False
        return all(
            coverage.get(fact.id) == _PRESENT for fact in expected_facts if fact.required
        )


# ─────────────────────────── §5.2 reconciliation ───────────────────────────


def reconcile(verdicts: Sequence[JudgeVerdict]) -> Reconciliation:
    """Compare the two pairings' verdicts. Escalate on any disagreement.

    Never averages and never prefers the stricter judge. Both are ways of
    resolving a disagreement that a human has not seen, and the case that two
    competent graders read differently is exactly the case worth reading.
    """
    if len({v.provider for v in verdicts}) < len(verdicts):
        raise JudgeIndependenceViolation(
            "§5.2: two verdicts from the same provider are one opinion sampled twice, "
            "not two graders agreeing"
        )

    gap = load(_JUDGE)["disagreement"]["max_score_gap"]
    reasons: list[str] = []

    malformed = [v.provider for v in verdicts if v.malformed]
    if malformed:
        # One judge failing to answer is not the other judge being right.
        reasons.append(f"malformed verdict from {', '.join(malformed)}")

    if len({v.meets_rubric for v in verdicts}) > 1:
        reasons.append("judges disagree on whether the case passes")

    for name in _DIMENSIONS:
        spread = [v.scores()[name] for v in verdicts]
        if max(spread) - min(spread) > gap:
            reasons.append(
                f"{name} scores {min(spread)} and {max(spread)} differ by more than {gap}"
            )

    if reasons:
        return Reconciliation(escalate=True, verdicts=tuple(verdicts), reasons=tuple(reasons))
    return Reconciliation(escalate=False, verdicts=tuple(verdicts), verdict=verdicts[0])


# ─────────────────────────── rendering helpers ───────────────────────────


def _numbered(items: Sequence[str]) -> str:
    """Numbered so `reasoning` can cite one — "passage 2 says 1400"."""
    return "\n".join(f"[{index}] {item}" for index, item in enumerate(items, start=1)) or "(none)"


def _bulleted(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "(none)"


def _facts(facts: Sequence[ExpectedFact]) -> str:
    """Claims and required flags only.

    `source_chunk` is deliberately withheld: telling the judge which passage a
    fact came from turns coverage into a lookup, and the question is whether
    the reply stated the fact, not whether we can find it ourselves.
    """
    return (
        "\n".join(
            f"- {fact.id} ({'required' if fact.required else 'optional'}): {fact.claim}"
            for fact in facts
        )
        or "(none)"
    )


def _rubric(rubric: Mapping[str, Mapping[str, str]]) -> str:
    blocks = []
    for dimension, rows in rubric.items():
        lines = [f"{dimension.upper()}"]
        lines += [f"  {score} — {text}" for score, text in sorted(rows.items(), reverse=True)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = [
    "Judge",
    "JudgeIndependenceViolation",
    "JudgeVerdict",
    "Reconciliation",
    "reconcile",
]
