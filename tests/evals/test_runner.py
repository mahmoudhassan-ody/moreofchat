"""The case runner — harness spec §5, and the first real number in the project.

The design decision this file exists to protect: **the runner performs real
fusion over the ingested corpus and never hands `gold_chunks` to the
orchestrator.** Feeding them measures generation while skipping retrieval, and
retrieval is the half most likely to fail on Arabic. edu-0015 exists precisely
to test the retrieval path; a runner that hands it the right chunk asserts
nothing at all and reports a number that looks like quality.

`gold_chunks` are ground truth for `retrieval_recall_at_5` — a *measurement of
what fusion returned*, never an input to it.

That shortcut is the one someone takes when the runner is slow, and it leaves
no trace: every case still passes, faster, and the suite quietly stops testing
the thing it was built for. So it is pinned three ways — structurally, by
signature, and behaviourally.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy import text as sql

from moc.agent.guards import check_numeric_grounding
from moc.agent.orchestrator import Orchestrator, Retrieval
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import ConversationState, TurnInput
from moc.config_store import load
from moc.evals import runner as runner_module
from moc.evals.judge import JudgeVerdict
from moc.evals.load import load_cases
from moc.evals.runner import CaseRunner, recall_at_k
from moc.evals.schema import Action, EvalCase, Turn
from moc.llm.base import AllProvidersUnavailable
from moc.llm.fake import FakeProvider
from moc.llm.router import Router
from moc.tenancy.context import tenant_session

CASES = Path(__file__).parents[2] / "evals" / "cases"
SCRIPT = "scripts/education/fees"

GROUNDED = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه."
PASSAGE = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026"


class PerTurnRetriever:
    """Returns a different chunk id per turn, so which turn was measured is
    visible in the recall number rather than inferable."""

    def __init__(self, *, by_turn):
        self._by_turn = list(by_turn)
        self.queries: list[str] = []

    async def search(self, *, query: str) -> Retrieval:
        return Retrieval(passages=(PASSAGE,), confidence=0.9)

    async def chunk_ids_for(self, *, query: str) -> tuple[str, ...]:
        index = min(len(self.queries), len(self._by_turn) - 1)
        self.queries.append(query)
        return tuple(self._by_turn[index])


class RecordingRetriever:
    """Records every query it is asked, and what it returned.

    The recording is the point: it is how "retrieval actually ran" becomes an
    assertion rather than an assumption.
    """

    def __init__(self, *, passages=(PASSAGE,), chunk_ids=("chunk-1",), confidence=0.9):
        self.queries: list[str] = []
        self._passages = passages
        self._chunk_ids = chunk_ids
        self._confidence = confidence

    async def search(self, *, query: str) -> Retrieval:
        self.queries.append(query)
        return Retrieval(
            passages=self._passages,
            confidence=self._confidence,
            script_constants=(),
        )

    async def chunk_ids_for(self, *, query: str) -> tuple[str, ...]:
        return tuple(self._chunk_ids)


class FixedExtractor:
    def __init__(self, turn: TurnInput | None = None):
        self._turn = turn or TurnInput(intent="fees", slots={"faculty": "engineering"})
        self.seen: list[str] = []

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput:
        self.seen.append(text)
        return self._turn


class RecordingJudge:
    def __init__(self, verdict: JudgeVerdict | None = None):
        self.calls: list[dict] = []
        self._verdict = verdict or JudgeVerdict(
            provider="openai", model="m", grounding=3, register=3,
            helpfulness=3, meets_rubric=True, fact_coverage={},
        )

    async def grade(self, **kwargs) -> JudgeVerdict:
        self.calls.append(kwargs)
        return self._verdict


@pytest_asyncio.fixture(loop_scope="session")
async def tenant(engine):
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.tenancy.models import Tenant

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in ("kb_outbox", "kb_chunks", "kb_documents", "usage_ledger",
                      "conversations", "tenants"):
            await s.execute(sql(f"DELETE FROM {table}"))  # noqa: S608
        row = Tenant(slug="runner", name="Runner", vertical="education")
        s.add(row)
        await s.commit()
        return row


def orchestrator(retriever, extractor, *, reply=GROUNDED, fail=None) -> Orchestrator:
    providers = {
        "anthropic": FakeProvider("anthropic", text=reply, fail_with=fail,
                                  fail_kinds=("complete",)),
        "openai": FakeProvider("openai", text=reply, fail_with=fail,
                               fail_kinds=("complete",)),
    }
    return Orchestrator(
        engine=ScriptEngine.from_config(SCRIPT),
        router=Router(config=load("llm/routing"), providers=providers),
        retriever=retriever,
        extractor=extractor,
    )


def build(retriever=None, extractor=None, judge=None, **kwargs) -> tuple:
    retriever = retriever or RecordingRetriever()
    extractor = extractor or FixedExtractor()
    judge = judge or RecordingJudge()
    return (
        CaseRunner(
            orchestrator=orchestrator(retriever, extractor, **kwargs),
            retriever=retriever,
            judge=judge,
            script=SCRIPT,
        ),
        retriever,
        judge,
    )


def a_case(**overrides) -> EvalCase:
    base = {
        "id": "edu-test-1",
        "vertical": "education",
        "source": "synthetic",
        "category": "factual_retrieval",
        "tenant_fixture": "sinai_demo",
        "channel": "whatsapp",
        "input_lang": "masri",
        "turns": [{"user": "كام رسوم الساعة؟", "expected_action": "answer"}],
        "gold_chunks": ["chunk-1"],
    }
    return EvalCase(**{**base, **overrides})


# ─────────────────────────── the design decision ───────────────────────────


def test_retrieval_is_real_not_injected():
    """`gold_chunks` never reach the orchestrator. Pinned structurally.

    The runner may read `gold_chunks` only inside the recall measurement. If
    the attribute is touched anywhere else, the shortcut has been taken —
    silently, because every case would still pass and faster.
    """
    source = Path(inspect.getfile(runner_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    readers = {
        enclosing.name
        for enclosing in ast.walk(tree)
        if isinstance(enclosing, ast.FunctionDef | ast.AsyncFunctionDef)
        for node in ast.walk(enclosing)
        if isinstance(node, ast.Attribute) and node.attr == "gold_chunks"
    }
    assert readers <= {"_recall_for", "recall_at_k"}, (
        f"gold_chunks is read in {readers} — it is ground truth for recall and must "
        f"never reach the orchestrator, or the suite stops testing retrieval"
    )


def test_the_orchestrator_has_no_channel_for_injected_passages():
    """Signature-level. The shortcut arrives as a helpful new argument."""
    parameters = set(inspect.signature(Orchestrator.handle).parameters)
    assert not parameters & {"passages", "gold_chunks", "retrieval", "chunks"}


async def test_a_case_cannot_pass_when_retrieval_finds_nothing(session_tenant):
    """The behavioural half. If gold chunks were secretly injected, an empty
    retriever could still produce a grounded answer — so this failing is what
    proves they are not."""
    session, tenant = session_tenant
    empty = RecordingRetriever(passages=(), chunk_ids=(), confidence=0.0)
    run, retriever, _ = build(retriever=empty)

    outcome = await run.run_case(a_case(), session=session)

    assert retriever.queries, "retrieval was never called"
    assert outcome.turns[0].action != "answer"


async def test_the_retriever_is_asked_the_customers_words(session_tenant):
    session, tenant = session_tenant
    run, retriever, _ = build()
    await run.run_case(a_case(), session=session)
    assert retriever.queries == ["كام رسوم الساعة؟"]


# ─────────────────────────── running cases ───────────────────────────


async def test_runs_a_single_turn_case_end_to_end(session_tenant):
    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(), session=session)

    assert outcome.case_id == "edu-test-1"
    assert len(outcome.turns) == 1
    assert outcome.turns[0].reply == GROUNDED
    assert outcome.errored is False


async def test_multi_turn_case_carries_state_between_turns(session_tenant):
    """edu-0007's shape: state must survive from one turn to the next, or
    every multi-turn case is really three unrelated single-turn cases."""
    session, _ = session_tenant
    run, _, _ = build()
    case = a_case(
        id="edu-multi",
        category="multi_turn_slots",
        turns=[
            {"user": "عايز أعرف المصاريف", "expected_action": "clarify"},
            {"user": "صيدلة", "expected_action": "answer"},
            {"user": "وده بيتدفع على كام قسط؟", "expected_action": "answer"},
        ],
    )
    outcome = await run.run_case(case, session=session)

    assert len(outcome.turns) == 3
    states = [turn.state for turn in outcome.turns]
    assert states[0] is not states[1]
    assert states[-1].slots, "slots did not survive to the last turn"


# ─────────────────────────── recall ───────────────────────────


async def test_recall_is_measured_on_the_turn_that_claims_the_chunk(session_tenant):
    """Not on the last turn, which is only a proxy for it.

    A multi-turn case ends on whatever the customer said last, and in Masri
    that is usually a slot answer — "ثانوية عامة، طب أسنان" — which carries
    none of the vocabulary of the question it is answering. Measuring
    retrieval there scores the conversation's punctuation rather than its
    question.

    `expected_facts[].source_chunk` already says which turn was supposed to
    find what. This asserts the runner reads it: the gold chunk is retrieved
    on turn 2, which is the turn that claims it, and not on turn 3.
    """
    session, _ = session_tenant
    retriever = PerTurnRetriever(by_turn=[["other"], ["chunk-1"], ["unrelated"]])
    run, _, _ = build(retriever=retriever)
    case = a_case(
        id="edu-claims-turn-2",
        category="multi_turn_slots",
        gold_chunks=["chunk-1"],
        turns=[
            {"user": "عايز أعرف المصاريف", "expected_action": "clarify"},
            {
                "user": "صيدلة",
                "expected_action": "answer",
                "expected_facts": [
                    {"id": "f1", "claim": "the pharmacy fee", "source_chunk": "chunk-1"}
                ],
            },
            {"user": "شكرا", "expected_action": "answer"},
        ],
    )
    outcome = await run.run_case(case, session=session)
    assert outcome.recall_at_5 == 1.0, (
        "the gold chunk was retrieved on the turn that claims it; scoring the "
        "last turn instead reports a miss that did not happen"
    )


async def test_a_gold_chunk_no_turn_claims_may_be_found_on_any_turn(session_tenant):
    """When the case names gold but no fact traces to it, the conversation as
    a whole was meant to surface it. Falling back to the last turn would be
    the same positional guess by another name."""
    session, _ = session_tenant
    retriever = PerTurnRetriever(by_turn=[["chunk-1"], ["other"], ["unrelated"]])
    run, _, _ = build(retriever=retriever)
    case = a_case(
        id="edu-claims-nothing",
        category="multi_turn_slots",
        gold_chunks=["chunk-1"],
        turns=[
            {"user": "عايز أعرف المصاريف", "expected_action": "clarify"},
            {"user": "صيدلة", "expected_action": "answer"},
            {"user": "شكرا", "expected_action": "answer"},
        ],
    )
    outcome = await run.run_case(case, session=session)
    assert outcome.recall_at_5 == 1.0


def test_recall_is_measured_against_gold_chunks_not_fed_from_them():
    assert recall_at_k(retrieved=("a", "b", "c"), gold=("b",), k=5) == 1.0
    assert recall_at_k(retrieved=("a", "b"), gold=("b", "z"), k=5) == 0.5
    assert recall_at_k(retrieved=(), gold=("b",), k=5) == 0.0


def test_recall_only_counts_the_first_k():
    assert recall_at_k(retrieved=("x", "x", "x", "x", "x", "gold"), gold=("gold",), k=5) == 0.0


def test_a_case_with_empty_gold_chunks_is_excluded_from_recall():
    """§3.1: not counted as a failure. Many adversarial cases have none, and
    scoring them zero would drag the metric down for cases that never made a
    retrieval claim."""
    assert recall_at_k(retrieved=("a",), gold=(), k=5) is None


async def test_the_outcome_reports_recall_as_none_when_there_is_no_ground_truth(
    session_tenant,
):
    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(gold_chunks=[]), session=session)
    assert outcome.recall_at_5 is None


# ─────────────────────────── staging ───────────────────────────


async def test_deterministic_checks_run_before_the_judge(session_tenant):
    """Stage 1 is free and non-flaky and covers both hard gates. A case that
    fails it must not cost a judge call — at 39 cases that is merely wasteful,
    at 230 it is the difference between running the suite and not."""
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge)

    outcome = await run.run_case(
        a_case(turns=[{"user": "كام؟", "expected_action": "handoff"}]), session=session
    )

    assert outcome.turns[0].checks, "stage 1 did not run"
    assert judge.calls == [], "the judge was called on a case that failed stage 1"


async def test_the_judge_runs_when_stage_one_passes(session_tenant):
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge)
    await run.run_case(a_case(), session=session)
    assert len(judge.calls) == 1


async def test_the_judge_never_receives_the_expected_reply(session_tenant):
    """§3.1 again, one layer up: the runner must not leak a golden answer into
    the grading call any more than into the retrieval call."""
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge)
    await run.run_case(a_case(), session=session)
    assert not {"expected_reply", "golden", "reference"} & set(judge.calls[0])


# ─────────────────────────── errors ───────────────────────────


async def test_a_provider_error_marks_the_case_errored_not_failed(session_tenant):
    """An outage is not a quality signal. Folding it into the failure rate
    corrupts the baseline, and the corruption survives into every later
    comparison against it."""
    session, _ = session_tenant
    run, _, _ = build(fail=AllProvidersUnavailable("both down"))
    outcome = await run.run_case(a_case(), session=session)

    assert outcome.errored is True
    assert outcome.passed is False


async def test_errored_cases_are_excluded_from_the_accuracy_denominator(session_tenant):
    session, _ = session_tenant
    run, _, _ = build()
    broken, _, _ = build(fail=AllProvidersUnavailable("down"))

    good = await run.run_case(a_case(id="ok"), session=session)
    bad = await broken.run_case(a_case(id="err"), session=session)

    summary = runner_module.summarize([good, bad])
    assert summary.scored == 1
    assert summary.errored == 1
    assert summary.accuracy == 1.0, "an outage dragged the accuracy down"


# ─────────────────────────── metadata ───────────────────────────


async def test_run_metadata_records_config_hash_and_prompt_version(session_tenant):
    session, _ = session_tenant
    run, _, judge = build()
    metadata = run.metadata(git_sha="abc123")

    assert metadata.git_sha == "abc123"
    assert metadata.config_hash
    assert any(binding.task == "eval_grading" for binding in metadata.tasks)
    assert any("judge_v1" in binding.prompt_version for binding in metadata.tasks)


async def test_the_runner_produces_case_results_the_report_can_consume(session_tenant):
    from moc.evals.report import build_report

    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(), session=session)
    report = build_report(run.metadata(git_sha="sha"), [outcome.to_case_result()])
    assert report.overall_accuracy in (0.0, 1.0)
    assert report.cases[0].case_id == "edu-test-1"


# ─────────────────────────── the shipped cases ───────────────────────────


def test_every_shipped_case_loads_for_the_runner():
    """Both files, whatever they hold. A case the runner cannot load is a case
    silently absent from the number.

    Asserted as a floor and by id, not as an exact count. Both files are
    append-only, and a literal total turns "a case was added" into a failure
    that reads like a regression — which is how the loader tests broke once
    already. The ids pin the cases the gates are built on, so one deleted or
    renamed fails here rather than quietly leaving the denominator.
    """
    education = load_cases(CASES / "education.yaml")
    realestate = load_cases(CASES / "realestate.yaml")
    ids = {case.id for case in education + realestate}

    assert len(ids) == len(education) + len(realestate), "duplicate case id"
    assert len(ids) >= 37
    assert {"edu-0015", "re-0001", "re-0005", "re-0021", "re-0022", "re-0023"} <= ids


@pytest_asyncio.fixture(loop_scope="session")
async def session_tenant(app_engine, tenant):
    async with tenant_session(app_engine, tenant.id) as s:
        yield s, tenant


def _runner() -> CaseRunner:
    """A runner built only far enough to call `_stage_one`, which is pure."""
    return CaseRunner(orchestrator=None, retriever=None, script=SCRIPT)


def _figure_checks(*, reply: str, passages: list[str], constants=()) -> dict:
    checks = CaseRunner._stage_one(
        _runner(),
        Turn(user="كام؟", expected_action=Action.answer),
        SimpleNamespace(
            reply=reply,
            action=Action.answer,
            state=ConversationState(script_id="s", script_version=1),
            passages=tuple(passages),
            script_constants=tuple(constants),
        ),
    )
    return {c.metric: c for c in checks if c.metric.endswith("figure_rate")}


# ─────────────────── the two figure gates (§2.1) ───────────────────
#
# Both are hard gates at zero in `config/evals/gates.yaml`, and neither was
# fed by a run: `_stage_one` computed action, language and slots, while its
# docstring claimed it covered them. A gate nothing measures is a gate that
# cannot fail, and the docstring made that invisible.


def test_stage_one_feeds_both_figure_gates():
    """Structural, because the omission was invisible for a reason.

    Nothing failed while these were missing — the suite ran, the report
    rendered, and the two metrics simply never appeared in it. So the
    assertion is on the metric names the checks carry, not on any behaviour a
    passing case would exercise.
    """
    checks = _figure_checks(reply="الرسوم 1400 جنيه", passages=["الرسوم 1400 جنيه"])
    metrics = set(checks)
    assert "hallucinated_figure_rate" in metrics
    assert "hedged_figure_rate" in metrics


def test_an_orphan_figure_fails_the_hallucination_gate_only():
    """The two gates stay separate. An orphan means retrieval or the script
    failed to supply the figure; a hedge means generation editorialized over a
    figure it had, and one number covering both says a regression happened
    without saying where."""
    checks = _figure_checks(reply="الرسوم 9999 جنيه", passages=["الرسوم 1400 جنيه"])
    assert checks["hallucinated_figure_rate"].passed is False
    assert checks["hedged_figure_rate"].passed is True


def test_a_hedged_but_grounded_figure_fails_the_hedging_gate_only():
    checks = _figure_checks(reply="الرسوم حوالي 1400 جنيه", passages=["الرسوم 1400 جنيه"])
    assert checks["hallucinated_figure_rate"].passed is True
    assert checks["hedged_figure_rate"].passed is False


def test_a_reply_with_no_figures_is_unmeasured_not_perfect():
    """§5.1's rule about skipped checks, and it matters most here.

    Most education replies state no figure at all. Counting those as passes
    would report `hallucinated_figure_rate` as a flawless zero on a suite that
    never put a number in front of the gate — which is the shape of a metric
    that looks safest exactly when it is testing nothing.
    """
    checks = _figure_checks(reply="ممكن توضّح أكتر؟", passages=["الرسوم 1400 جنيه"])
    assert all(check.skipped for check in checks.values())


def test_a_rejected_composition_does_not_count_against_the_gate():
    """The metric is about what reached the customer, not what was attempted.

    §19.3: when the runtime gate finds an orphan figure it discards the
    composed reply whole and sends a scripted one, so the customer never saw
    the number. Scoring the discarded text would fail the gate on exactly the
    turns where the protection worked — and the harder that gate bit, the
    worse the metric would read.

    The attempted rate is a real signal about composition and is reported
    separately by the live suite; it is not this gate.
    """
    rejected = "الرسوم 9999 جنيه"
    scripted = "معلش، مش لاقي معلومة مؤكدة عن ده."
    checks = CaseRunner._stage_one(
        _runner(),
        Turn(user="كام؟", expected_action=Action.handoff),
        SimpleNamespace(
            reply=scripted,
            action=Action.handoff,
            state=ConversationState(script_id="s", script_version=1),
            passages=("الرسوم 1400 جنيه",),
            script_constants=(),
            # What the runtime gate rejected. Recorded on the turn, and
            # deliberately not what the delivered-figure gate reads.
            grounding=check_numeric_grounding(rejected, ["الرسوم 1400 جنيه"], []),
        ),
    )
    figures = {c.metric: c for c in checks if c.metric.endswith("figure_rate")}
    assert all(check.skipped for check in figures.values()), (
        "the scripted reply states no figure, so the gate has nothing to judge"
    )


def test_a_script_constant_is_a_source():
    """§3.1 lets the script state figures the corpus does not carry. A gate
    that did not know that would fail every scripted fee."""
    checks = _figure_checks(reply="الرسوم 1400 جنيه", passages=[], constants=("1400",))
    assert checks["hallucinated_figure_rate"].passed is True
