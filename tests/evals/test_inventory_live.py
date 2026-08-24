"""The real-estate suite against real models — N runs, with spreads.

Marked `live`: real Postgres, real model calls, the production
`LlmSlotExtractor` rather than the keyword double in
`test_runner_inventory.py`. Run with

    uv run pytest -m live tests/evals/test_inventory_live.py -s

**This file exists because the numbers it produces used to come from a
throwaway script.** Four real-estate figures were reported over one afternoon
— 52.2%, 42.9%, 39.1%, 45.5% — from a scratchpad file nobody could re-run,
against one unchanged commit. Two of them were used to argue a change had
helped. Neither the spread nor the script survived.

So: the run is a test, the run count is config, and every metric reports mean
with min-max and its run count. A metric whose spread exceeds the configured
bar is flagged as not yet measurable at this suite size — the code may be fine
and the denominator merely too small, but a delta under the spread is not a
result.
"""

import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.extraction import LlmSlotExtractor
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import ConversationState
from moc.config_store import load
from moc.evals.inventory_runner import (
    InventoryCaseRunner,
    metrics,
    snapshot_from_fixture,
)
from moc.evals.load import load_cases
from moc.evals.repeatability import default_runs, render_all, repeat, unmeasurable
from moc.llm.anthropic_direct import AnthropicDirect
from moc.llm.openai_direct import OpenAIDirect
from moc.llm.router import Router
from moc.retrieval.inventory import InventoryRepository, load_units
from moc.tenancy.context import tenant_session

pytestmark = pytest.mark.live

ROUTING = load("llm/routing")
CASES = Path(__file__).parents[2] / "evals" / "cases" / "realestate.yaml"
FIXTURE = (
    Path(__file__).parents[2]
    / "evals"
    / "fixtures"
    / "broker_demo_2026_08_01"
    / "units.jsonl"
)
SCRIPT = "scripts/realestate/search"


def key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        pytest.skip(f"{name} not set")
    return value


@pytest_asyncio.fixture(loop_scope="session")
async def live_runner(engine, app_engine, tenant_tables):
    from moc.tenancy.models import Tenant

    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="broker", name="Broker", vertical="realestate"))
        await s.commit()
    async with tenant_session(engine, tenant_id) as session:
        await load_units(session=session, path=FIXTURE)
        await session.commit()

    router = Router(
        config=ROUTING,
        providers={
            "anthropic": AnthropicDirect(
                api_key=key("MOC_ANTHROPIC_API_KEY"), http=ROUTING["http"]
            ),
            "openai": OpenAIDirect(
                api_key=key("MOC_OPENAI_API_KEY"), http=ROUTING["http"]
            ),
        },
    )

    from moc.verticals.realestate.agent import InventoryAgent

    async with tenant_session(app_engine, tenant_id) as session:
        repository = InventoryRepository(session=session)
        yield InventoryCaseRunner(
            agent=InventoryAgent(
                repository=repository,
                engine=ScriptEngine.from_config(SCRIPT),
                # The production extractor, not the keyword double. The double
                # cannot fail the way the real one does, so a suite run on it
                # measures the connector rather than the agent.
                #
                # The catalogue is this tenant's own rows: the values the model
                # may emit and the values the connector filters on are one
                # list, read once.
                extractor=LlmSlotExtractor(
                    router=router,
                    script=SCRIPT,
                    catalogue=await repository.vocabulary(),
                ),
            ),
            snapshot=snapshot_from_fixture(FIXTURE),
            script=SCRIPT,
            # The agent cannot meter a turn without one, and for as long as
            # nobody passed a session this whole vertical billed nothing.
            session=session,
        ), session

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


async def test_live_the_realestate_suite_produces_a_report(live_runner, capsys):
    """All 23 real-estate cases, real models, N runs, five gates with spreads."""
    cases = load_cases(CASES)
    times = default_runs()
    runs: list[list] = []

    live_runner, live_session = live_runner

    async def once():
        outcomes = await live_runner.run(cases)
        # Commit, or the ledger is decorative — the same gap the education
        # suite had. `tenant_session` closes without committing.
        await live_session.commit()
        runs.append(outcomes)
        with capsys.disabled():
            scored = [o for o in outcomes if not o.errored]
            passed = sum(o.passed for o in scored)
            print(
                f"  run {len(runs)}/{times}: "
                f"{passed / len(scored):.1%} ({passed}/{len(scored)} scored, "
                f"{len(outcomes) - len(scored)} errored)"
            )
        return metrics(outcomes)

    with capsys.disabled():
        print(f"\n{'=' * 68}")
        print(f"  REAL-ESTATE SUITE — {len(cases)} cases, {times} runs")
        print(f"{'=' * 68}")

    spreads = await repeat(once, times=times)

    with capsys.disabled():
        print(f"{'-' * 68}")
        for line in render_all(spreads):
            print(f"  {line}")
        print(f"{'-' * 68}")
        wide = unmeasurable(spreads)
        if wide:
            print(
                f"  Not measurable at {len(cases)} cases over {times} runs: "
                f"{', '.join(wide)}"
            )
            print("  A delta smaller than the spread is not a result.")
        else:
            print("  Every measured metric settled within the configured bar.")
        print(f"{'-' * 68}")
        print(f"  Per-case detail, run {times} of {times}:")
        for outcome in runs[-1]:
            mark = "err " if outcome.errored else ("PASS" if outcome.passed else "fail")
            failed = [
                c.name
                for t in outcome.turns
                for c in t.checks
                if not c.passed and not c.observational
            ]
            detail = outcome.error[:60] if outcome.errored else (",".join(failed) or "-")
            print(f"  {mark}  {outcome.case_id:12} {outcome.category:24} {detail}")
        verdicts: dict[str, set[bool]] = {}
        for outcomes in runs:
            for outcome in outcomes:
                verdicts.setdefault(outcome.case_id, set()).add(outcome.passed)
        flaky = sorted(cid for cid, seen in verdicts.items() if len(seen) > 1)
        print(f"{'-' * 68}")
        print(f"  Cases that changed verdict across runs: {', '.join(flaky) or 'none'}")
        print(f"{'=' * 68}")

    assert len(runs) == times
    assert all(len(outcomes) == len(cases) for outcomes in runs)
    assert spreads["overall_accuracy"].attempts == times


CEILING = "شقة في نور سيتي في حدود ٦ مليون و نص"
IDENTIFIER = "الوحدة في نور سيتي بـ ٦ مليون و نص، القسط كام؟"


async def test_live_the_two_price_slots_are_not_confused_in_either_direction(
    live_runner, capsys
):
    """`budget_max` and `near_price`, one preposition apart, N runs each.

    A ceiling read as an identifier quotes one unit when the customer wanted a
    range. An identifier read as a ceiling quotes a 2,930,000 studio to
    someone asking about a 6,450,000 apartment — with a correct instalment, so
    the reply looks entirely normal. Both directions, because a prompt that
    fixes one by leaning on the other has fixed nothing.

    Measured rather than assumed: if the model cannot hold this at temperature
    0 the honest outcome is a number in the report, not another prompt edit.
    """
    # `live_runner` yields (runner, session). This read `live_runner._agent`
    # and had been raising AttributeError since the session was added — the
    # test never asserted anything, and being `live` nothing in CI noticed.
    # Found by Task 42b, whose first prompt edit broke exactly this and was
    # caught by hand instead.
    extractor = live_runner[0]._agent._extractor
    times = default_runs()
    results: dict[str, list[dict]] = {CEILING: [], IDENTIFIER: []}

    for message in (CEILING, IDENTIFIER):
        for _ in range(times):
            turn = await extractor.extract(
                text=message, state=ScriptEngine.from_config(SCRIPT).start()
            )
            results[message].append(dict(turn.slots))

    with capsys.disabled():
        print(f"\n{'=' * 68}")
        print(f"  PRICE SLOT DISCRIMINATION — {times} runs each")
        print(f"{'=' * 68}")
        for message, runs in results.items():
            print(f"  {message}")
            for index, slots in enumerate(runs, 1):
                print(f"    run {index}: {slots}")
        print(f"{'=' * 68}")

    for slots in results[CEILING]:
        assert slots.get("budget_max") == 6_500_000, "a ceiling is budget_max"
        assert "near_price" not in slots, "and never the identifier"
    for slots in results[IDENTIFIER]:
        assert slots.get("near_price") == 6_500_000, "a quoted unit price is near_price"
        assert "budget_max" not in slots, "and never the ceiling"


HELD = {"compound": "Madinaty", "property_type": "apartment"}
DOWN_PAYMENT = "وبـ ٤٠٪ مقدم؟"
CORRECTION = "مش التجمع، الشيخ زايد"


def _holding(**slots) -> ConversationState:
    return ScriptEngine.from_config(SCRIPT).start().with_slots(slots)


async def test_live_naming_a_unit_in_another_compound_changes_the_compound(
    live_runner, capsys
):
    """A customer changing their mind — the tenth rehearsal turn, N runs.

    Turn one browses Madinaty. Turn two names a unit in Noor City. The
    extractor is shown the held slots and told not to contradict them, and it
    returned the price with no compound: everything downstream was then
    correct about the wrong unit, at the wrong price, with a payment plan that
    audits clean. No fix below extraction can recover a value the model did
    not return, which is why this is asserted against the real one.
    """
    extractor = live_runner[0]._agent._extractor
    times = default_runs()
    seen: list[dict] = []

    for _ in range(times):
        turn = await extractor.extract(text=IDENTIFIER, state=_holding(**HELD))
        seen.append(dict(_holding(**HELD).with_slots(turn.slots, turn.cleared).slots))

    with capsys.disabled():
        print(f"\n{'=' * 68}")
        print(f"  CHANGING YOUR MIND — held {HELD}, {times} runs")
        print(f"  {IDENTIFIER}")
        for index, slots in enumerate(seen, 1):
            print(f"    run {index}: {slots}")
        print(f"{'=' * 68}")

    for slots in seen:
        assert slots.get("compound") == "Noor City", "the compound this message names"
        assert slots.get("near_price") == 6_500_000, "and the price it quotes"


async def test_live_an_explicit_correction_still_works(live_runner, capsys):
    """The case the prompt already carried, so the fix does not trade one for it."""
    extractor = live_runner[0]._agent._extractor
    times = default_runs()
    seen: list[dict] = []

    for _ in range(times):
        state = _holding(city="New Cairo", property_type="apartment")
        turn = await extractor.extract(text=CORRECTION, state=state)
        seen.append(dict(state.with_slots(turn.slots, turn.cleared).slots))

    with capsys.disabled():
        print(f"\n  EXPLICIT CORRECTION — {CORRECTION}")
        for index, slots in enumerate(seen, 1):
            print(f"    run {index}: {slots}")

    for slots in seen:
        assert slots.get("city") == "Sheikh Zayed", "the second value, not both"
        assert slots.get("property_type") == "apartment", "and nothing else moved"


async def test_live_a_message_that_names_no_place_still_keeps_the_held_one(
    live_runner, capsys
):
    """Why held slots exist at all. "And at 40% down?" names no place.

    The failure mode of over-correcting the test above: a prompt that makes
    every message restate the location turns a follow-up into a fresh search.
    """
    extractor = live_runner[0]._agent._extractor
    times = default_runs()
    seen: list[dict] = []

    for _ in range(times):
        turn = await extractor.extract(text=DOWN_PAYMENT, state=_holding(**HELD))
        seen.append(dict(_holding(**HELD).with_slots(turn.slots, turn.cleared).slots))

    with capsys.disabled():
        print(f"\n  NO PLACE NAMED — {DOWN_PAYMENT}")
        for index, slots in enumerate(seen, 1):
            print(f"    run {index}: {slots}")

    for slots in seen:
        assert slots.get("compound") == "Madinaty", "the held compound survives"
        assert slots.get("property_type") == "apartment", "and so does the rest"
