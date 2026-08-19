"""Inventory cases through the runner, and the five gates — P1b Task 26.

**The point of this file is that unmeasured is not zero.**

`arithmetic_in_model_rate`, `type_substitution_rate`, `invented_compound_rate`
and `sold_unit_offered_rate` are all zero-tolerance gates in
`config/evals/gates.yaml`, and every one of them has sat at zero observations
since it was written — not passing, never run. `asof_disclosure_rate` is the
same at 0.98. A gate nothing feeds cannot fail, and a report that renders it
as 0.0% is worse than one that says nothing, because 0.0% is what success
looks like.

So each gate reports its observation count, and a gate with none reports "not
measured" rather than a number.

These cases carry `grounding_mode: inventory`. Routing them through document
fusion would ground a price on a passage, which is the exact failure the
broker fixture is kept out of `kb_chunks` to prevent.
"""

import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.script_engine import ScriptEngine
from moc.config_store import load
from moc.evals.deterministic import InventorySnapshot, ToolCall, check_tool_calls
from moc.evals.inventory_runner import (
    InventoryCaseRunner,
    gate_report,
    snapshot_from_fixture,
)
from moc.evals.load import load_cases
from moc.evals.schema import ExpectedToolCall
from moc.retrieval.inventory import InventoryRepository, load_units
from moc.verticals.realestate.agent import InventoryAgent, SlotExtractor

CASES = Path(__file__).parents[2] / "evals" / "cases" / "realestate.yaml"
FIXTURE = (
    Path(__file__).parents[2] / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)
SCRIPT = "scripts/realestate/search"

#: The gates this task exists to feed. Named here so a gate quietly dropped
#: from the report fails a test rather than disappearing from a summary.
FIVE_GATES = (
    "arithmetic_in_model_rate",
    "type_substitution_rate",
    "invented_compound_rate",
    "sold_unit_offered_rate",
    "asof_disclosure_rate",
)


@pytest_asyncio.fixture(loop_scope="session")
async def stocked(engine, tenant_tables):
    from moc.tenancy.models import Tenant

    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="broker", name="Broker", vertical="realestate"))
        await s.commit()

    from moc.tenancy.context import tenant_session

    async with tenant_session(engine, tenant_id) as session:
        await load_units(session=session, path=FIXTURE)
        await session.commit()

    yield tenant_id

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def runner(app_engine, stocked):
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, stocked) as session:
        repository = InventoryRepository(session=session)
        yield InventoryCaseRunner(
            agent=InventoryAgent(
                repository=repository,
                engine=ScriptEngine.from_config(SCRIPT),
                extractor=SlotExtractor(),
            ),
            snapshot=snapshot_from_fixture(FIXTURE),
            script=SCRIPT,
        ), session


def cases() -> list:
    return load_cases(CASES)


# ─────────────────────── routing ───────────────────────


async def test_inventory_cases_route_to_the_inventory_connector_not_fusion(runner):
    """`grounding_mode: inventory`. Feeding these through document fusion
    would ground a price on a passage — the failure the broker fixture is kept
    out of `kb_chunks` to prevent."""
    case_runner, _ = runner
    case = next(c for c in cases() if c.id == "re-0001")
    outcome = await case_runner.run_case(case)

    assert not outcome.errored, outcome.error
    calls = [call.name for turn in outcome.turns for call in turn.tool_calls]
    assert "inventory_lookup" in calls
    assert all(not turn.passages for turn in outcome.turns), (
        "an inventory turn has no retrieved passages; a price came from a row"
    )


async def test_every_inventory_case_runs_without_erroring(runner):
    """All 22. An errored case says nothing about quality and would sit
    outside every gate's denominator."""
    case_runner, _ = runner
    outcomes = await case_runner.run(cases())
    errored = [(o.case_id, o.error) for o in outcomes if o.errored]
    assert errored == []
    assert len(outcomes) == 22


# ─────────────────────── the checks that feed the gates ───────────────────────


def test_expected_tool_calls_are_asserted_by_argument_subset():
    """The orchestrator may pass more arguments than a case pins; pinning the
    full signature would break every inventory case the first time an
    unrelated parameter is added."""
    expected = [ExpectedToolCall(name="inventory_lookup", args_contain={"city": "New Cairo"})]
    actual = [
        ToolCall(name="inventory_lookup", args={"city": "New Cairo", "property_type": "villa"})
    ]
    assert check_tool_calls(expected, actual).passed

    wrong = [ToolCall(name="inventory_lookup", args={"city": "North Coast"})]
    assert not check_tool_calls(expected, wrong).passed


async def test_expected_computation_string_matches_the_calculator_output(runner):
    """Not "close". 302,344 against 302,343 is a fail — that one-EGP gap is the
    whole demonstration that the model did not do the arithmetic."""
    case_runner, _ = runner
    case = next(c for c in cases() if c.id == "re-0005")
    outcome = await case_runner.run_case(case)

    checks = {c.name: c for turn in outcome.turns for c in turn.checks}
    assert "computation" in checks
    assert checks["computation"].passed, checks["computation"].detail

    reply = outcome.turns[0].reply
    assert "302,343" in reply or "302343" in reply
    assert "302,344" not in reply and "302344" not in reply


async def test_a_reply_quoting_a_divided_figure_fails_the_computation_check(runner):
    """The gate proven load-bearing without sabotaging the calculator: a reply
    holding the number a divider would produce must not pass."""
    case_runner, _ = runner
    case = next(c for c in cases() if c.id == "re-0005")
    outcome = await case_runner.run_case(case)
    schedule = outcome.turns[0].computation

    assert schedule is not None
    result = case_runner.check_computation(
        reply="القسط 302,344 جنيه", computation=schedule
    )
    assert not result.passed
    assert result.metric == "arithmetic_in_model_rate"


async def test_availability_is_cross_checked_against_the_fixture_status(runner):
    """The snapshot is eval ground truth, read from the fixture rather than
    from the connector — the connector cannot show a sold unit, so asking it
    would be asking the thing under test to grade itself."""
    case_runner, _ = runner
    snapshot = case_runner.snapshot

    assert snapshot.unit_status["NOOR-CIT-002-01"] == "sold"
    assert snapshot.unit_status["SOUTHMED-004-02"] == "reserved"
    assert snapshot.unit_status["MADINATY-001-01"] == "available"
    assert len(snapshot.unit_status) == 305

    outcomes = await case_runner.run(cases())
    presented = {
        unit_id
        for outcome in outcomes
        for turn in outcome.turns
        for unit_id in turn.presented_unit_ids
    }
    unavailable = {u for u in presented if snapshot.unit_status.get(u) != "available"}
    assert unavailable == set(), f"sold or reserved units were offered: {unavailable}"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("البيانات محدّثة 2026-08-01.", True),
        ("Inventory current as of 2026-08-01.", True),
        ("البيانات محدثة بتاريخ ٢٠٢٦-٠٨-٠١.", True),
        ("عندنا شقة في مدينتي بـ 5,800,000 جنيه.", False),
    ],
)
def test_asof_disclosure_is_detected_in_arabic_and_english(reply, expected):
    """Arabic-Indic digits included: a reply that discloses the date in the
    numerals the customer reads must not score as undisclosed."""
    from moc.evals.deterministic import check_asof_disclosure

    snapshot = InventorySnapshot(
        fixture="broker_demo_2026_08_01", as_of="2026-08-01", unit_status={}
    )
    assert check_asof_disclosure(reply, snapshot, required=True).passed is expected


# ─────────────────────── the report ───────────────────────


async def test_all_five_gates_report_a_number_not_a_skip(runner):
    """The P1 lesson, as an assertion.

    A gate with no observations must print "not measured". This plan exists
    because five of them printed nothing at all, and a reader would have taken
    silence for compliance.
    """
    case_runner, _ = runner
    outcomes = await case_runner.run(cases())
    report = gate_report(outcomes)

    assert set(FIVE_GATES) <= set(report), "a gate vanished from the report"
    for gate in FIVE_GATES:
        entry = report[gate]
        assert entry.observations > 0, (
            f"{gate} has no observations — it is unmeasured, which is what this "
            f"task exists to stop, and it is not the same as zero"
        )
        assert entry.rate is not None


async def test_the_arithmetic_gate_counts_only_the_computation_check(runner):
    """The split, asserted end to end.

    Before it, `arithmetic_in_model_rate` read 60% over 15 observations with
    14 of them tool calls — a zero-tolerance commercial gate reporting slot
    extraction. A gate whose name does not match what it counts produces
    pressure to weaken the gate rather than to fix the thing failing.
    """
    case_runner, _ = runner
    outcomes = await case_runner.run(cases())
    contributing = {
        check.name
        for outcome in outcomes
        for turn in outcome.turns
        for check in turn.checks
        if check.metric == "arithmetic_in_model_rate" and not check.skipped
    }
    assert contributing == {"computation"}


async def test_extraction_misses_are_tracked_not_gated(runner):
    """They are still reported — skipping them silently would hide a real
    failure — under metrics no threshold acts on."""
    case_runner, _ = runner
    report = gate_report(await case_runner.run(cases()))
    for metric in ("tool_call_accuracy", "unresolved_type_rate"):
        assert metric in report
        assert report[metric].observations > 0

    gates = load("evals/gates")
    for metric in ("tool_call_accuracy", "unresolved_type_rate"):
        assert metric not in gates["hard_gates"]
        assert metric not in gates["soft_gates"]
        assert metric in gates["tracked"]


async def test_a_gate_with_no_observations_says_so_rather_than_zero(runner):
    """The other half. Reporting 0.0% for a gate nothing fed is how five of
    these sat green for a month."""
    report = gate_report([])
    for gate in FIVE_GATES:
        assert report[gate].observations == 0
        assert report[gate].rate is None
        assert "not measured" in report[gate].render().lower()


# ─────────────────────── the three cases with teeth ───────────────────────


async def test_re_0022_does_not_offer_a_chalet(runner):
    """The North Coast is 19 chalets and no studio, so a chalet is the
    ranked-first answer for any naive retriever."""
    case_runner, _ = runner
    case = next(c for c in cases() if c.id == "re-0022")
    outcome = await case_runner.run_case(case)
    reply = outcome.turns[0].reply

    assert "chalet" not in reply.lower()
    assert "شاليه" not in reply
    presented = outcome.turns[0].presented_unit_ids
    for unit_id in presented:
        assert case_runner.snapshot.unit_type[unit_id] == "studio"


async def test_re_0002_and_re_0021_produce_the_two_distinct_shapes(runner):
    """re-0002 names an alternative; re-0021 has none to name and hands off.
    A single shape for both would either invent a villa or refuse to answer a
    question the catalogue can answer."""
    case_runner, _ = runner
    by_id = {c.id: c for c in cases()}

    named = await case_runner.run_case(by_id["re-0002"])
    assert str(named.turns[0].action) == "answer"
    assert named.turns[0].named_compounds, "an alternative compound is named"

    none_anywhere = await case_runner.run_case(by_id["re-0021"])
    assert str(none_anywhere.turns[0].action) == "handoff"
    assert not none_anywhere.turns[0].named_compounds, "nothing to name, so name nothing"
