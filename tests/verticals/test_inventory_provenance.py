"""A broker's figures trace to what produced them — demo plan Task 41b.

The source pane is the demo's centrepiece and it was blank for two of the three
tenants. Not because the evidence was missing — `InventoryTurn` has carried
`presented_unit_ids` and `computation` since P1b — but because it was discarded
at the worker boundary, which is the same failure Task 32 fixed on the document
side one screen over.

Driven through the real agent over the frozen broker fixture, because the claim
is about replies this system actually composes. A unit test over a hand-made
row would prove the tracer and say nothing about whether the turn hands it
anything.
"""

import uuid
from pathlib import Path

import pytest_asyncio

from moc.agent.provenance import CALCULATOR, INVENTORY
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import Action, ConversationState
from moc.retrieval.inventory import InventoryRepository, load_units
from moc.verticals.realestate.agent import InventoryAgent, KeywordSlotExtractor

SCRIPT = "scripts/realestate/search"
FIXTURE = (
    Path(__file__).parents[2] / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)


@pytest_asyncio.fixture(loop_scope="session")
async def agent(app_engine, engine, tenant_tables):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.tenancy.context import tenant_session
    from moc.tenancy.models import Tenant

    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="prov", name="Prov", vertical="realestate"))
        await s.commit()
    async with tenant_session(engine, tenant_id) as session:
        await load_units(session=session, path=FIXTURE)
        await session.commit()

    async with tenant_session(app_engine, tenant_id) as session:
        repository = InventoryRepository(session=session)
        yield InventoryAgent(
            repository=repository,
            engine=ScriptEngine.from_config(SCRIPT),
            extractor=KeywordSlotExtractor(catalogue=await repository.vocabulary()),
        )

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


def fresh() -> ConversationState:
    engine = ScriptEngine.from_config(SCRIPT)
    return ConversationState(script_id=SCRIPT, script_version=engine.version)


async def turn(agent, text: str, state: ConversationState | None = None):
    return await agent.handle(state=state or fresh(), text=text)


async def test_a_quoted_price_traces_to_the_row_it_was_read_from(agent):
    result = await turn(agent, "عايز شقة في مدينتي")
    assert result.action is Action.answer

    figures = result.provenance["figures"]
    assert figures, "a reply quoting a price produced no evidence"
    priced = [f for f in figures if f["source"] == INVENTORY]
    assert priced, f"nothing traced to a row: {figures}"
    assert priced[0]["chunkId"] in result.presented_unit_ids
    assert priced[0]["asOf"], "a price with no date is one the tenant cannot stand behind"


async def test_every_figure_in_an_inventory_reply_has_a_source(agent):
    """The gate, stated as the pane will show it. A figure the tracer cannot
    place is an orphan whether or not the reply was composed from templates —
    which is the point of tracing rather than asserting."""
    result = await turn(agent, "عايز شقة في مدينتي")
    assert result.provenance["gates"]["numeric_grounding"] is True
    assert all(f["grounded"] for f in result.provenance["figures"])


async def test_an_instalment_traces_to_the_calculator_and_names_its_inputs(agent):
    """§19.3: the arithmetic is the tool's and never the model's, so the
    evidence for an instalment is not a sentence anywhere — it is the tool that
    ran and what it ran with."""
    # re-0005's own phrasing. A follow-up needs the unit already held, and a
    # message the extractor routes elsewhere would make this assert nothing.
    result = await turn(agent, "الوحدة في نور سيتي بـ ٦ مليون و نص، القسط كام؟")
    assert result.computation is not None, (
        f"the turn ran no calculator, so there is nothing to trace: {result.reply}"
    )

    computed = [f for f in result.provenance["figures"] if f["source"] == CALCULATOR]
    assert computed, f"nothing traced to the calculator: {result.provenance['figures']}"
    assert "payment_plan_calculator" in computed[0]["excerpt"]
    assert "price=" in computed[0]["excerpt"], (
        "the excerpt names no inputs; a number alone is not checkable"
    )
    assert computed[0]["title"] is None, (
        "the pane would print a tool identifier where a document answer prints "
        "the tenant's own document name"
    )


async def test_a_turn_that_quotes_nothing_carries_no_provenance(agent):
    """A clarification states no figure, so there is nothing to evidence.
    An empty `figures` list would make the pane render an empty panel where a
    reply had nothing to show."""
    result = await turn(agent, "أهلا")
    assert result.action is not Action.answer
    assert result.provenance is None


def test_the_wire_shape_is_the_one_the_pane_already_reads():
    """One renderer. `SourcePane` reads these keys today and only the value of
    `source` is new — a second shape would be a second thing it can fail to
    render."""
    from moc.verticals.realestate.agent import InventoryTurn

    turn = InventoryTurn(
        reply="السعر 1,400 جنيه",
        action=Action.answer,
        register=None,
        state=fresh(),
    )
    assert turn.provenance is None or set(turn.provenance) == {"figures", "gates"}
