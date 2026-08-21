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
from moc.evals.runner import CaseRunner, metrics, recall_at_k
from moc.evals.schema import Action, EvalCase, Turn
from moc.llm.base import AllProvidersUnavailable, Completion
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

    def __init__(
        self, *, passages=(PASSAGE,), chunk_ids=("chunk-1",), confidence=0.9, titles=()
    ):
        self.queries: list[str] = []
        self._passages = passages
        self._chunk_ids = chunk_ids
        self._confidence = confidence
        self._titles = titles

    async def search(self, *, query: str) -> Retrieval:
        self.queries.append(query)
        return Retrieval(
            passages=self._passages,
            confidence=self._confidence,
            script_constants=(),
            titles=self._titles,
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
    """Reports a completion because the real judge does.

    A double that returned a bare verdict would let the runner drop the
    ledger row while every test passed — the same mismatch `FakeExtractor`
    hid until it started making a real call.
    """

    def __init__(self, verdict: JudgeVerdict | None = None):
        self.calls: list[dict] = []
        self._verdict = verdict or JudgeVerdict(
            provider="openai", model="m", grounding=3, register=3,
            helpfulness=3, meets_rubric=True, fact_coverage={},
            completion=Completion(
                text="{}",
                provider="openai",
                model="judge-model",
                input_tokens=1450,
                output_tokens=90,
            ),
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
    at 230 it is the difference between running the suite and not.

    Asserted here on a language failure rather than an action one, because
    action is now the documented exception — see the two tests below.
    """
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge)

    outcome = await run.run_case(
        a_case(turns=[{"user": "كام؟", "expected_action": "answer",
                       "expected_lang": "en"}]),
        session=session,
    )

    assert outcome.turns[0].checks, "stage 1 did not run"
    assert judge.calls == [], "the judge was called on a case that failed stage 1"


async def test_the_judge_grades_a_turn_that_failed_only_on_the_action(session_tenant):
    """The population most likely to carry a bad figure is the one the judge
    could not see.

    A turn that answered where it should have handed off is precisely the turn
    that reached for a nearby number — and until this opened, every such turn
    was dropped before grading. That is why no judge ever saw edu-0012, and
    why `hallucinated_figure_rate` read 0.0% over a population that excluded
    the failure it is named after.

    The turn still fails: the action check is in the list and nothing here
    rescues it. What changes is that its reply gets graded.
    """
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge)

    outcome = await run.run_case(
        a_case(turns=[{"user": "كام؟", "expected_action": "handoff"}]), session=session
    )

    assert len(judge.calls) == 1, "an action-only failure was dropped before grading"
    assert not outcome.turns[0].passed, "the action failure was graded away"


async def test_an_action_failure_alongside_another_still_skips_the_judge(session_tenant):
    """Only action is excused, and only alone.

    A turn that also mirrored the wrong language produced a reply whose
    register and grounding scores describe a reply nobody would have sent.
    Grading it would spend money to add noise to two gates.
    """
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge)

    await run.run_case(
        a_case(turns=[{"user": "كام؟", "expected_action": "handoff",
                       "expected_lang": "en"}]),
        session=session,
    )

    assert judge.calls == []


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


async def test_a_rejected_composition_is_kept_for_the_report(session_tenant):
    """§19.3 discards a composition holding an orphan figure and sends a
    scripted reply. The discarded text is the only thing that says WHY — which
    figure had no source, and whether the passages could ever have supplied
    it — and the harness was throwing it away.

    A gate refusing compositions at 100% recall is either the gate being wrong
    or the passages being unusable, and the two have opposite fixes. Neither
    is visible from a scripted apology and a check name.
    """
    session, _ = session_tenant
    runner, _, _ = build(reply="رسوم التقديم 4500 جنيه.")
    case = a_case(
        turns=[{"user": "رسوم التقديم كام؟", "expected_action": "answer"}]
    )

    outcome = (await runner.run([case], session=session))[0]
    turn = outcome.turns[0]

    assert turn.grounding is not None and not turn.grounding.passed
    assert "4500" in turn.composed, "the text the gate rejected"
    assert turn.composed != turn.reply, "the customer got the scripted reply"
    assert turn.passages, "and what it was checked against"


# ─────────────── judge dimensions that had no gate (2026-08-20) ───────────────


def _verdict(**overrides):
    from moc.evals.judge import JudgeVerdict

    base = {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "grounding": 3,
        "register": 3,
        "helpfulness": 3,
        "meets_rubric": True,
    }
    return JudgeVerdict(**{**base, **overrides})


def test_the_judge_s_register_score_feeds_the_register_gate():
    """`register_accuracy` is a soft gate in gates.yaml and read "not
    measured" on every run the suite has ever produced — while the judge
    scored register on every graded turn and the score was dropped at the
    reporting boundary. §8.2 makes register a product requirement."""
    from moc.evals.runner import checks_from_verdict

    passing = {c.metric: c for c in checks_from_verdict(_verdict(register=3))}
    assert passing["register_accuracy"].passed

    failing = {c.metric: c for c in checks_from_verdict(_verdict(register=1))}
    assert not failing["register_accuracy"].passed
    assert "1" in failing["register_accuracy"].detail


def test_the_register_floor_comes_from_the_judge_config():
    """§19. The bar for "right register" is the rubric's, not a second one
    invented here that could drift from it."""
    from moc.config_store import load
    from moc.evals.runner import checks_from_verdict

    floor = load("evals/judge")["pass_thresholds"]["register"]
    at_floor = {c.metric: c for c in checks_from_verdict(_verdict(register=floor))}
    below = {c.metric: c for c in checks_from_verdict(_verdict(register=floor - 1))}

    assert at_floor["register_accuracy"].passed
    assert not below["register_accuracy"].passed


def test_a_forbidden_claim_the_judge_caught_feeds_its_gate():
    """Zero tolerance, and it read "not measured" while the judge was
    printing violations two lines above it in the same report."""
    from moc.evals.runner import checks_from_verdict

    clean = {c.metric: c for c in checks_from_verdict(_verdict())}
    assert clean["forbidden_claim_violations"].passed

    caught = {
        c.metric: c
        for c in checks_from_verdict(
            _verdict(forbidden_violated=("the Qantara figure",), meets_rubric=False)
        )
    }
    assert not caught["forbidden_claim_violations"].passed
    assert "Qantara" in caught["forbidden_claim_violations"].detail


def test_a_malformed_verdict_grades_nothing():
    """A judge that returned prose has not assessed the register. Counting its
    zero as a register failure would report a judge outage as a quality
    regression."""
    from moc.evals.runner import checks_from_verdict

    checks = {c.metric: c for c in checks_from_verdict(_verdict(malformed=True))}
    assert all(c.skipped for c in checks.values())


def test_a_turn_the_judge_never_saw_contributes_nothing():
    """Stage 2 runs only when stage 1 passed. A turn that failed a
    deterministic check has no verdict, and an absent verdict must not read as
    a register pass."""
    from moc.evals.runner import checks_from_verdict

    assert checks_from_verdict(None) == []


def test_the_judge_catches_a_figure_the_deterministic_check_cannot():
    """The edu-0012 class: a figure lifted from a passage and relabelled.

    `check_numeric_grounding` compares against a SET of numbers, so 500 —
    the track-change fee, and the only 500 in 51 facts — passes as engineering
    tuition. The judge can see the mismatch and its rubric already covers it:
    grounding 0 is "contains an unsupported figure, or contradicts a passage".
    It was scoring that all along and nothing read the score.
    """
    from moc.evals.runner import checks_from_verdict

    checks = {
        c.metric: c
        for c in checks_from_verdict(
            _verdict(grounding=0, meets_rubric=False), reply="الرسوم 500 جنيه"
        )
    }
    assert not checks["hallucinated_figure_rate"].passed
    assert checks["hallucinated_figure_rate"].name == "figure_labelling"


def test_a_grounding_zero_on_a_reply_with_no_figure_is_not_a_figure_failure():
    """Grounding 0 is also "contradicts a passage", which needs no number.
    Counting that under a figure metric would make the rate mean two things.
    """
    from moc.evals.runner import checks_from_verdict

    checks = {
        c.metric: c
        for c in checks_from_verdict(
            _verdict(grounding=0, meets_rubric=False), reply="لا تتوفر لدينا هذه البيانات"
        )
    }
    assert checks["hallucinated_figure_rate"].skipped


def test_a_grounded_reply_with_a_figure_feeds_the_gate_as_a_pass():
    """The denominator has to include the turns that got it right, or the
    rate is a count of failures wearing a percentage sign."""
    from moc.evals.runner import checks_from_verdict

    checks = {
        c.metric: c
        for c in checks_from_verdict(_verdict(grounding=3), reply="الرسوم 2000 جنيه")
    }
    assert checks["hallucinated_figure_rate"].passed
    assert not checks["hallucinated_figure_rate"].skipped


def test_a_partially_grounded_reply_is_not_a_figure_failure():
    """Grounding 1 is an unsupported claim that is NOT a figure — the rubric
    says so. Only 0 reaches this gate."""
    from moc.evals.runner import checks_from_verdict

    checks = {
        c.metric: c
        for c in checks_from_verdict(
            _verdict(grounding=1, meets_rubric=False), reply="الرسوم 2000 جنيه"
        )
    }
    assert checks["hallucinated_figure_rate"].passed


async def test_the_outcome_records_the_titles_the_turn_retrieved(session_tenant):
    """A clarification that names nothing is either a node with no options or
    a retrieval that returned none, and the reply text cannot tell those apart.

    edu-0009 read as the first for two runs while it was the second: `titles`
    was empty in the live path because the arms' payloads were not merged, and
    the report had no field that would have shown it.
    """
    session, _ = session_tenant
    run, _, _ = build(retriever=RecordingRetriever(titles=("مواعيد الباصات؟",)))
    outcome = await run.run_case(a_case(), session=session)
    assert outcome.turns[0].titles == ("مواعيد الباصات؟",)


async def test_the_judge_is_told_what_the_script_was_entitled_to_state(session_tenant):
    """The runner is the only place that knows both, and it passed one.

    A scripted reply is the tenant's own sentence; graded against the passages
    retrieved for the customer's question it reads as an unsupported claim
    every time. edu-0017 is that failure with both expected facts present.
    Script constants are the same rule for figures, and §3.1 has said so since
    it was written.
    """
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge)
    await run.run_case(a_case(), session=session)

    assert "script_statements" in judge.calls[0], (
        "the judge was given the passages and nothing else"
    )


async def test_the_script_statements_carry_the_constants_and_the_authorised_text(
    session_tenant,
):
    session, _ = session_tenant
    judge = RecordingJudge()
    run, _, _ = build(judge=judge, reply=GROUNDED)
    await run.run_case(a_case(), session=session)

    statements = list(judge.calls[0]["script_statements"])
    referral = ScriptEngine.from_config(SCRIPT).referral("ar")
    assert referral in statements, "the configured referral was not offered as a source"


# ───────── §2.5: the latency budget, from the suite ─────────


async def test_every_turn_records_how_long_it_took(session_tenant):
    """`p95_latency_ms` has read "not measured" on every run the suite has ever
    produced. §2.5's 7000 ms budget was set from six hand-run samples, and
    nothing since has checked it against the thing it bounds."""
    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(), session=session)
    assert outcome.turns[0].elapsed_ms is not None
    assert outcome.turns[0].elapsed_ms >= 0


async def test_the_run_reports_a_p95_in_milliseconds(session_tenant):
    """Not a proportion. Everything else in `metrics()` is a share of
    something, and this one is the reason `gates.yaml` declares a unit."""
    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(), session=session)
    value = metrics([outcome])["p95_latency_ms"]
    assert value is not None
    assert value >= 0


def test_a_run_with_no_turns_reports_no_p95():
    """Unmeasured, not zero — and zero is the number a latency gate most wants
    to print when nothing ran."""
    assert metrics([])["p95_latency_ms"] is None


def test_the_p95_is_the_slow_tail_not_the_mean():
    """A budget is about the turn the customer waits through. A mean over
    nineteen turns hides the one that took four seconds, which is the only
    one §2.5 is about."""
    from moc.evals.runner import percentile

    # Ten slow turns in a hundred: the 95th is inside the tail.
    assert percentile([100.0] * 90 + [5000.0] * 10, 95) == 5000.0
    # Exactly five in a hundred puts rank 95 on the boundary, and nearest-rank
    # answers with the boundary rather than interpolating one that nothing
    # took. Pinned because it looks like an off-by-one and is not.
    assert percentile([100.0] * 95 + [5000.0] * 5, 95) == 100.0
    assert percentile([100.0], 95) == 100.0
    # Nineteen turns — the education suite — puts the 95th on the slowest,
    # which is the correct answer to "how long does a customer wait at worst".
    assert percentile([100.0] * 18 + [4000.0], 95) == 4000.0


async def test_the_outcome_carries_the_phase_breakdown(session_tenant):
    """§2.5's p95 is over budget and nothing says where it goes. The runner is
    the only place that sees every turn, so it is where the breakdown has to
    survive to."""
    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(), session=session)
    assert set(outcome.turns[0].timings) >= {
        "intake",
        "intake.extraction",
        "intake.retrieval",
        "total",
    }


def test_the_breakdown_reports_what_no_phase_claimed():
    """The remainder is the finding, not a rounding artifact.

    Composition and the audit were each timed once by hand and together they
    do not explain a whole turn. A breakdown that only listed the phases
    somebody thought to instrument would show a tidy total and hide exactly
    the part nobody has looked at.
    """
    from moc.evals.runner import phase_breakdown

    rows = phase_breakdown(
        [
            {"extraction": 100.0, "retrieval": 200.0, "total": 1000.0},
            {"extraction": 150.0, "retrieval": 250.0, "total": 1200.0},
        ]
    )
    assert rows["unattributed"]["mean"] == 750.0
    assert rows["extraction"]["mean"] == 125.0
    assert rows["total"]["p95"] == 1200.0


def test_a_phase_is_averaged_over_the_turns_that_ran_it():
    """A clarification makes no composition call. Averaging it in as a zero
    would report composition as twice as fast as it is, on a suite where a
    third of turns are scripted."""
    from moc.evals.runner import phase_breakdown

    rows = phase_breakdown(
        [{"composition": 4000.0, "total": 4500.0}, {"total": 300.0}]
    )
    assert rows["composition"]["mean"] == 4000.0
    assert rows["composition"]["turns"] == 1
    assert rows["total"]["turns"] == 2


async def test_the_outcome_records_what_the_judge_was_given(session_tenant):
    """A verdict is only reproducible if the evidence behind it is recorded.

    Re-grading a run with a different judge — to find out whether a cheaper
    tier reaches the same verdicts — has to hand the second judge exactly what
    the first one saw. `passages` was already kept; the script statements were
    assembled inside `_stage_two` and discarded, so a re-grade would have
    silently compared two judges on two different inputs and reported the
    difference as disagreement.
    """
    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(), session=session)

    referral = ScriptEngine.from_config(SCRIPT).referral("ar")
    assert referral in outcome.turns[0].script_statements


async def test_the_judge_call_reaches_the_ledger(session_tenant):
    """The fifth and largest metering gap. Every judge call in a suite run
    lands on the OpenAI failover — §5.2 excludes the answering provider — and
    the ledger showed none of them: a run that spent $0.46 reported $0.31."""
    from sqlalchemy import text as sql

    session, _ = session_tenant
    run, _, _ = build()
    await run.run_case(a_case(), session=session)

    models = [
        r[0]
        for r in (
            await session.execute(
                sql("SELECT model FROM usage_ledger WHERE kind = 'llm_call'")
            )
        ).all()
    ]
    assert "judge-model" in models, "the judge call was never metered"


def test_a_dotted_phase_is_detail_inside_a_counted_one():
    """`intake.extraction` sits within `intake`, so counting both would
    double-charge the turn and drive `unattributed` negative."""
    from moc.evals.runner import phase_breakdown

    rows = phase_breakdown(
        [{"intake": 1000.0, "intake.extraction": 950.0, "intake.retrieval": 300.0,
          "total": 1200.0}]
    )
    assert rows["unattributed"]["mean"] == 200.0
    assert rows["intake.extraction"]["mean"] == 950.0, "the detail is still reported"


async def test_the_outcome_records_which_model_composed(session_tenant):
    """Which model produced the reply, on the turn it produced.

    Written for the haiku-vs-sonnet comparison, where the whole result is a
    claim about one model: an arm that quietly composed on something else
    reports that model's accuracy under the other one's name, and nothing in
    the output would say so.
    """
    session, _ = session_tenant
    run, _, _ = build()
    outcome = await run.run_case(a_case(), session=session)

    primary = load("llm/routing")["tasks"]["answer_composition"]["primary"]["model"]
    assert outcome.turns[0].composition_model == primary


async def test_a_composition_that_failed_over_names_the_failover_model(session_tenant):
    """The contamination case, and the only reason the field is worth having.

    §2.6 fails composition over to OpenAI, silently and by design. During a
    model comparison that is a run measuring the failover while the report
    names the candidate — so the model is recorded per turn rather than
    assumed from the config the run was launched with.
    """
    from moc.llm.base import ProviderUnavailable

    session, _ = session_tenant
    retriever = RecordingRetriever()
    providers = {
        "anthropic": FakeProvider(
            "anthropic", text=GROUNDED,
            fail_with=ProviderUnavailable("down"), fail_kinds=("complete",),
        ),
        "openai": FakeProvider("openai", text=GROUNDED),
    }
    run = CaseRunner(
        orchestrator=Orchestrator(
            engine=ScriptEngine.from_config(SCRIPT),
            router=Router(config=load("llm/routing"), providers=providers),
            retriever=retriever,
            extractor=FixedExtractor(),
        ),
        retriever=retriever,
        judge=RecordingJudge(),
        script=SCRIPT,
    )
    outcome = await run.run_case(a_case(), session=session)

    failover = load("llm/routing")["tasks"]["answer_composition"]["failover"]["model"]
    assert outcome.turns[0].composition_model == failover


def test_a_scripted_turn_names_no_composing_model():
    """None, not the configured primary. A scripted reply is the tenant's
    words — attributing it to the model that never ran would credit a model
    for text it did not write, and inflate its share of a run."""
    from moc.evals.runner import TurnOutcome, composition_models

    rows = composition_models(
        [
            TurnOutcome(turn_index=0, reply="a", action="answer",
                        state=ScriptEngine.from_config(SCRIPT).start(),
                        composition_model="claude-haiku-4-5-20251001"),
            TurnOutcome(turn_index=1, reply="b", action="handoff",
                        state=ScriptEngine.from_config(SCRIPT).start()),
        ]
    )
    assert dict(rows) == {"claude-haiku-4-5-20251001": 1}


def test_the_detail_run_is_the_last_one_that_produced_turns():
    """The report renders per-case detail, phases and composed-by from one run.

    Blindly the last one throws away everything the earlier ones collected when
    a run dies: the haiku comparison lost its register number and its phase
    breakdown to an outage in run 2, having already measured both in run 1.
    """
    from moc.evals.runner import CaseOutcome, TurnOutcome, detail_run

    state = ScriptEngine.from_config(SCRIPT).start()
    good = [CaseOutcome(case_id="a", vertical="education", category="c",
                        turns=(TurnOutcome(turn_index=0, reply="r", action="answer",
                                           state=state),))]
    dead = [CaseOutcome(case_id="a", vertical="education", category="c",
                        errored=True, error="ProviderRequestError")]

    index, runs = detail_run([good, dead, dead])
    assert (index, runs) == (1, good), "run 1 is the last one with anything to show"


def test_the_detail_run_is_the_last_run_when_every_run_has_turns():
    from moc.evals.runner import CaseOutcome, TurnOutcome, detail_run

    state = ScriptEngine.from_config(SCRIPT).start()
    def run(reply):
        return [CaseOutcome(case_id="a", vertical="education", category="c",
                            turns=(TurnOutcome(turn_index=0, reply=reply,
                                               action="answer", state=state),))]

    index, runs = detail_run([run("first"), run("second")])
    assert index == 2
    assert runs[0].turns[0].reply == "second"


def test_a_run_set_with_no_turns_at_all_still_reports_something():
    """Every run errored. The report must still render — a harness that
    raises on an outage reports nothing about the outage."""
    from moc.evals.runner import CaseOutcome, detail_run

    dead = [CaseOutcome(case_id="a", vertical="education", category="c",
                        errored=True, error="boom")]
    index, runs = detail_run([dead, dead])
    assert (index, runs) == (2, dead)


async def test_stage_two_does_not_run_when_grading_is_off(session_tenant):
    """Opt-in per run. Three runs of unchanged code do not need three
    independent verdicts on the same replies — that is 3x for one
    measurement, and the judge is the largest line item in a run."""
    session, _ = session_tenant
    run, _, judge = build()
    outcome = await run.run([a_case()], session=session, grade=False)

    assert judge.calls == []
    assert all(t.verdict is None for o in outcome for t in o.turns)


async def test_an_ungraded_run_reports_stage_one_accuracy_not_overall(session_tenant):
    """The number stage 1 alone produces is a different number, so it gets a
    different name.

    Reported under `overall_accuracy` it would be a free 5-10 points — every
    stage-2 failure simply absent — and it would be compared against judged
    baselines by anyone reading the column.
    """
    session, _ = session_tenant
    run, _, _ = build()
    outcomes = await run.run([a_case()], session=session, grade=False)
    values = metrics(outcomes)

    assert values["overall_accuracy"] is None
    assert values["stage_one_accuracy"] is not None


async def test_an_ungraded_run_measures_no_register(session_tenant):
    """not measured, never 100%. register_accuracy is a judge metric, and a
    suite that switched the judge off while still reporting 98.2% register
    would be reporting a number nothing produced."""
    session, _ = session_tenant
    run, _, _ = build()
    outcomes = await run.run([a_case()], session=session, grade=False)

    assert metrics(outcomes)["register_accuracy"] is None


async def test_a_graded_run_reports_overall_accuracy_and_no_stage_one(session_tenant):
    session, _ = session_tenant
    run, _, _ = build()
    outcomes = await run.run([a_case()], session=session, grade=True)
    values = metrics(outcomes)

    assert values["overall_accuracy"] is not None
    assert values["stage_one_accuracy"] is None, "one accuracy per run, named for what fed it"


async def test_an_ungraded_run_still_meters_the_turn_but_not_the_judge(session_tenant):
    """Stage 1 is free of judge cost, not free. Extraction, composition and
    the figure audit all still make provider calls — the guess that a
    stage-1-only run costs near zero is a guess about the judge's share, not
    about the run."""
    from sqlalchemy import text as sql

    session, _ = session_tenant
    run, _, _ = build()
    await run.run([a_case()], session=session, grade=False)

    models = [
        r[0]
        for r in (
            await session.execute(
                sql("SELECT model FROM usage_ledger WHERE kind = 'llm_call'")
            )
        ).all()
    ]
    assert models, "the turn itself still calls providers"
    assert "judge-model" not in models
