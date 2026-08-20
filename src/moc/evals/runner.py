"""The case runner — harness spec §5.

Drives the orchestrator over the case files and produces the numbers. This is
the first point in the project where "is it good?" has an answer other than an
opinion, which makes it the first point where the answer can be quietly wrong.

**Retrieval is real. `gold_chunks` never reach the orchestrator.**

That is the whole design decision, and it is worth restating where someone
will read it while making the runner faster. Feeding gold chunks in measures
generation and skips retrieval — and retrieval is the half most likely to fail
on Arabic. edu-0015 exists precisely to test the retrieval path: a runner that
hands it the right chunk asserts nothing, reports a high number, and the
number looks exactly like quality.

`gold_chunks` are ground truth for `retrieval_recall_at_5` — a measurement of
what fusion returned, never an input to it. This module reads the attribute in
one place, `_recall_for`, and a test enforces that.

Two staging decisions:

**Stage 1 before stage 2.** Deterministic checks are free, fast and non-flaky,
and they cover both hard gates. A case failing them must not cost a judge
call: at 39 cases that is wasteful, at 230 it is the difference between
running the suite and not.

**An error is not a failure.** A provider outage says nothing about quality,
and folding it into the failure rate corrupts the baseline — after which every
comparison against that baseline is measuring the outage too. Errored cases
are counted separately and excluded from the accuracy denominator.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.script_engine import ScriptEngine
from moc.agent.state import ConversationState
from moc.evals.deterministic import (
    CheckResult,
    check_action,
    check_figures,
    check_language,
    check_slots,
)
from moc.evals.judge import JudgeVerdict
from moc.evals.report import CaseResult
from moc.evals.run_metadata import RunMetadata, TaskBinding, capture
from moc.evals.schema import EvalCase, Turn

_RECALL_K = 5


class ChunkSource(Protocol):
    """What retrieval returns *identified*, for the recall measurement.

    Separate from the passages the orchestrator receives. The orchestrator
    needs text; the metric needs chunk ids, and giving the orchestrator ids it
    does not use would put gold-shaped data one refactor away from the prompt.
    """

    async def chunk_ids_for(self, *, query: str) -> tuple[str, ...]: ...


class Judge(Protocol):
    async def grade(self, **kwargs: Any) -> JudgeVerdict: ...


@dataclass(frozen=True)
class TurnOutcome:
    turn_index: int
    reply: str
    action: str
    state: ConversationState
    retrieved_chunk_ids: tuple[str, ...] = ()
    #: What the *runtime* gate saw, which is not what the figure checks
    #: measure. §19.3 discards a composition holding an orphan figure and
    #: sends a scripted reply, so this records an attempt the customer never
    #: received — a signal about composition quality, not a gate.
    grounding: Any = None
    checks: tuple[CheckResult, ...] = ()
    verdict: JudgeVerdict | None = None

    @property
    def passed(self) -> bool:
        stage_one = all(check.passed for check in self.checks)
        stage_two = self.verdict is None or self.verdict.meets_rubric
        return stage_one and stage_two


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    vertical: str
    category: str
    turns: tuple[TurnOutcome, ...] = ()
    errored: bool = False
    error: str = ""
    #: None when the case pins no gold chunks — excluded from recall rather
    #: than scored zero. Many adversarial cases make no retrieval claim, and
    #: scoring them would drag the metric down for cases that never asserted.
    recall_at_5: float | None = None

    @property
    def passed(self) -> bool:
        return not self.errored and bool(self.turns) and all(t.passed for t in self.turns)

    def to_case_result(self) -> CaseResult:
        return CaseResult(
            case_id=self.case_id,
            vertical=self.vertical,
            category=self.category,
            passed=self.passed,
            checks=tuple(check for turn in self.turns for check in turn.checks),
        )


@dataclass(frozen=True)
class RunSummary:
    """Errored cases sit outside the accuracy denominator, deliberately."""

    total: int
    scored: int
    errored: int
    passed: int
    recall_at_5: float | None = None
    recall_cases: int = 0

    @property
    def accuracy(self) -> float:
        return self.passed / self.scored if self.scored else 0.0


def recall_at_k(
    *, retrieved: Sequence[str], gold: Sequence[str], k: int = _RECALL_K
) -> float | None:
    """Fraction of gold chunks present in the first `k` retrieved.

    `None` when the case pins no gold chunks (§3.1) — unmeasured, not failed.
    A case with nothing to find cannot fail to find it, and averaging a zero
    in would make the metric a function of how many adversarial cases the
    suite happens to contain.
    """
    if not gold:
        return None
    top = set(retrieved[:k])
    return sum(1 for chunk_id in gold if chunk_id in top) / len(gold)


class CaseRunner:
    def __init__(
        self,
        *,
        orchestrator: Any,
        retriever: ChunkSource,
        judge: Judge | None = None,
        script: str,
        channel: str = "whatsapp",
    ) -> None:
        self._orchestrator = orchestrator
        self._retriever = retriever
        self._judge = judge
        self._script = script
        self._channel = channel
        self._engine = ScriptEngine.from_config(script)

    # ─────────────────────────── one case ───────────────────────────

    async def run_case(self, case: EvalCase, *, session: AsyncSession) -> CaseOutcome:
        """Run every turn, then measure. Never feeds the case's answers in.

        The orchestrator receives the customer's words, the conversation state
        and the channel. Nothing else from the case is in scope here — the
        expectations are for grading, and grading happens after.
        """
        state = self._engine.start()
        turns: list[TurnOutcome] = []
        retrieved: tuple[str, ...] = ()

        for index, turn in enumerate(case.turns):
            try:
                result = await self._orchestrator.handle(
                    session=session,
                    state=state,
                    text=turn.user,
                    channel=self._channel,
                )
            except Exception as exc:  # noqa: BLE001 - an outage is not a quality signal
                return CaseOutcome(
                    case_id=case.id,
                    vertical=str(case.vertical),
                    category=str(case.category),
                    turns=tuple(turns),
                    errored=True,
                    error=repr(exc)[:500],
                )

            if result.provider_unavailable:
                # Not a failure. No model answered, so the turn says nothing
                # about quality — and counting it as a miss would bake an
                # outage into the baseline every later run is compared to.
                return CaseOutcome(
                    case_id=case.id,
                    vertical=str(case.vertical),
                    category=str(case.category),
                    turns=tuple(turns),
                    errored=True,
                    error="every provider was unavailable",
                )

            state = result.state
            retrieved = await self._retriever.chunk_ids_for(query=turn.user)
            checks = self._stage_one(turn, result)
            verdict = await self._stage_two(turn, result, checks)
            turns.append(
                TurnOutcome(
                    turn_index=index,
                    reply=result.reply,
                    action=str(result.action),
                    state=state,
                    retrieved_chunk_ids=retrieved,
                    grounding=getattr(result, "grounding", None),
                    checks=tuple(checks),
                    verdict=verdict,
                )
            )

        return CaseOutcome(
            case_id=case.id,
            vertical=str(case.vertical),
            category=str(case.category),
            turns=tuple(turns),
            recall_at_5=_recall_for(case, tuple(turns)),
        )

    def _stage_one(self, turn: Turn, result: Any) -> list[CheckResult]:
        """Deterministic checks. Free, fast, non-flaky.

        Covers §2.1's two figure gates — `hallucinated_figure_rate` and
        `hedged_figure_rate`, both at zero in `config/evals/gates.yaml` —
        measured on the reply that was **sent**. A composition the runtime
        gate rejected never reached anyone, and scoring it would fail the
        metric on the turns where the protection worked.

        This docstring previously claimed that coverage while the list below
        held only action, language and slots. Nothing failed: the suite ran,
        the report rendered, and two hard gates simply never appeared in it.
        A gate nothing measures is a gate that cannot fail.
        """
        return [
            check_action(turn.expected_action, result.action),
            check_language(result.reply, turn.expected_lang),
            check_slots(turn.expected_slots, result.state.slots),
            *check_figures(
                result.reply,
                getattr(result, "passages", ()),
                getattr(result, "script_constants", ()),
            ),
        ]

    async def _stage_two(
        self, turn: Turn, result: Any, checks: Sequence[CheckResult]
    ) -> JudgeVerdict | None:
        """The judge, only when stage 1 passed.

        Skipped rather than deferred: a case already failing a hard gate
        cannot be rescued by a good rubric score, so the call would buy
        nothing and cost real money at 230 cases.
        """
        if self._judge is None or not all(check.passed for check in checks):
            return None
        return await self._judge.grade(
            question=turn.user,
            reply=result.reply,
            retrieved_passages=list(result.passages),
            expected_facts=list(turn.expected_facts),
            forbidden_claims=list(turn.forbidden_claims),
            expected_register=turn.expected_register,
            answer_provider=_provider_of(result),
        )

    # ─────────────────────────── a whole run ───────────────────────────

    async def run(
        self, cases: Sequence[EvalCase], *, session: AsyncSession
    ) -> list[CaseOutcome]:
        """Every case, in order.

        Sequential on purpose: the cases share one session and one
        conversation table, and a parallel run would interleave state between
        cases in a way that is invisible in the report and impossible to
        reproduce. The judge is where a full run's cost lives, and batching it
        is a change to the judge rather than to this loop.
        """
        return [await self.run_case(case, session=session) for case in cases]

    def metadata(self, *, git_sha: str) -> RunMetadata:
        """§2.3: what the run was measured under.

        The judge's prompt version travels with it, so a run graded under a
        reworded rubric is not silently compared against one graded under the
        old wording.
        """
        tasks: list[TaskBinding] = []
        if self._judge is not None and hasattr(self._judge, "task_binding"):
            tasks.append(
                self._judge.task_binding(provider="openai", model="gpt-5.6-sol")
            )
        elif self._judge is not None:
            tasks.append(
                TaskBinding(
                    task="eval_grading",
                    prompt_version=getattr(self._judge, "prompt_version", "judge_v1"),
                    provider="unknown",
                    model="unknown",
                )
            )
        return capture(git_sha=git_sha, tasks=tuple(tasks))


def _turns_claiming(case: EvalCase, chunk_id: str) -> tuple[int, ...]:
    """Which turns were supposed to retrieve `chunk_id`.

    `expected_facts[].source_chunk` already records this; nothing else has to
    be authored. A turn claims a chunk when one of its facts traces to it.
    """
    return tuple(
        index
        for index, turn in enumerate(case.turns)
        if any(fact.source_chunk == chunk_id for fact in turn.expected_facts)
    )


def _recall_for(case: EvalCase, turns: Sequence[TurnOutcome]) -> float | None:
    """The only place `gold_chunks` is read.

    Enforced by `test_retrieval_is_real_not_injected`. If this attribute is
    touched anywhere else in the module, gold-shaped data has moved one step
    closer to the orchestrator — and the suite stops testing retrieval while
    still reporting a number.

    **Measured on the turn that claims the chunk, not the last one.** A
    multi-turn conversation ends wherever the customer stopped talking, and in
    Masri that is usually a slot answer — edu-0007 closes on "ثانوية عامة، طب
    أسنان", which carries none of the vocabulary of the question it answers.
    Scoring retrieval there measures the conversation's punctuation rather
    than its question, and reports a miss for a chunk the run found perfectly
    well one turn earlier.

    A gold chunk that no fact traces to is looked for across every turn: the
    case says the conversation should surface it without saying when, and
    picking the last turn would be the same positional guess wearing a
    different justification.
    """
    if not case.gold_chunks:
        return None
    retrieved_by_turn = {turn.turn_index: turn.retrieved_chunk_ids for turn in turns}
    found = 0.0
    for chunk_id in case.gold_chunks:
        claiming = _turns_claiming(case, chunk_id) or tuple(retrieved_by_turn)
        pooled = tuple(
            dict.fromkeys(
                chunk
                for index in claiming
                for chunk in retrieved_by_turn.get(index, ())[:_RECALL_K]
            )
        )
        found += recall_at_k(retrieved=pooled, gold=(chunk_id,), k=len(pooled)) or 0.0
    return found / len(case.gold_chunks)


def _provider_of(result: Any) -> str:
    completions = getattr(result, "completions", ())
    return completions[0].provider if completions else "anthropic"


def gate_directions() -> dict[str, str]:
    """Every gated metric and which way it counts, read from gates.yaml.

    `min` gates report the share of checks that passed, `max` gates the share
    that failed. Reading one as the other is not a rounding error — a 0.0%
    hallucination rate and a 0.0% action accuracy are opposite results that
    print identically.
    """
    from moc.config_store import load

    gates = load("evals/gates")
    return {
        name: spec["direction"]
        for group in ("hard_gates", "soft_gates")
        for name, spec in gates[group].items()
    }


def metrics(outcomes: Sequence[CaseOutcome]) -> dict[str, float | None]:
    """One run's contribution to a spread: every metric, or None where unfed.

    Always names every gate, including from an empty run. A metric missing
    from one run's dict and a metric that run measured as nothing are the same
    fact, but only the second keeps the gate on the report across N runs — and
    a gate that vanishes from a report reads as a gate that passed.
    """
    summary = summarize(outcomes)
    values: dict[str, float | None] = {}

    seen: dict[str, list[bool]] = {}
    for outcome in outcomes:
        for turn in outcome.turns:
            for check in turn.checks:
                if check.skipped:
                    continue
                seen.setdefault(check.metric, []).append(check.passed)

    for name, direction in gate_directions().items():
        results = seen.get(name, [])
        if not results:
            values[name] = None
        elif direction == "min":
            values[name] = sum(results) / len(results)
        else:
            values[name] = sum(not r for r in results) / len(results)

    # Overlaid after the gate loop, not before it. `retrieval_recall_at_5` is
    # a gate name whose value comes from `CaseOutcome.recall_at_5` rather than
    # from any check, so the loop would otherwise write None over it.
    values["overall_accuracy"] = summary.accuracy if summary.scored else None
    values["retrieval_recall_at_5"] = summary.recall_at_5
    # A rate, not a count. Everything here is rendered as a percentage, and a
    # count sent through that renderer prints numbers above 100.
    values["errored_rate"] = summary.errored / summary.total if summary.total else None
    return values


def summarize(outcomes: Sequence[CaseOutcome]) -> RunSummary:
    """Aggregate, keeping errors out of the accuracy denominator."""
    errored = [outcome for outcome in outcomes if outcome.errored]
    scored = [outcome for outcome in outcomes if not outcome.errored]
    measured = [o.recall_at_5 for o in scored if o.recall_at_5 is not None]

    return RunSummary(
        total=len(outcomes),
        scored=len(scored),
        errored=len(errored),
        passed=sum(outcome.passed for outcome in scored),
        recall_at_5=sum(measured) / len(measured) if measured else None,
        recall_cases=len(measured),
    )


__all__ = [
    "CaseOutcome",
    "CaseRunner",
    "RunSummary",
    "TurnOutcome",
    "gate_directions",
    "metrics",
    "recall_at_k",
    "summarize",
]
