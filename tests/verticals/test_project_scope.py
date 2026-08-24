"""A developer asked about someone else's project — demo plan Task 42c.

**The guarantee was never implemented.** Task 38 scoped the extraction
vocabulary to the project, so a developer's extractor was offered one compound
and could not name another. Asked about Noor City it was expected to emit
`Noor City` anyway, fail `_require_known`, and be caught as an out-of-
vocabulary refusal — the right outcome, produced by the model disobeying the
prompt's own "use the exact value from the list" rule.

Task 42b's prompt is followed more closely. The disobedience stopped, the
model returned the only compound it was offered, and the developer's bot
answered *"عندنا apartment في Madinaty بسعر 5,800,000"* to a question about
Noor City. A cross-project substitution — measured 5 of 5, and the one thing
§11.2 exists to prevent.

**Vocabulary scope is not search scope.** The extractor is offered the tenant's
whole sheet so it can name what the customer actually said; the search, the id
lookup and the `compounds()` catalogue stay filtered to the project. A compound
that resolves outside the project is then a fact the turn can act on, compared
against `tenants.project` in code rather than inferred from an exception.

Task 38's objection to an unscoped vocabulary was that "the search then returns
nothing, and the customer is told the developer has no stock". That is exactly
right, and it is why the guard has to exist before the vocabulary widens: the
out-of-project compound never reaches the search.
"""

import uuid
from pathlib import Path

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.script_engine import ScriptEngine
from moc.agent.state import Action, ConversationState
from moc.retrieval.inventory import InventoryRepository, UnitQuery, load_units
from moc.tenancy.context import tenant_session
from moc.tenancy.models import Tenant
from moc.verticals.realestate.agent import InventoryAgent, KeywordSlotExtractor

SCRIPT = "scripts/realestate/search"
FIXTURE = (
    Path(__file__).parents[2] / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)
PROJECT = "Madinaty"
ELSEWHERE = "Noor City"
ASK_ELSEWHERE = "وعندكم إيه في نور سيتي؟"
ASK_OWN = "عايز شقة في مدينتي"


async def _tenant(engine, app_engine, tenant_tables, *, project: str | None):
    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(
            Tenant(
                id=tenant_id,
                slug="scope",
                name="Scope",
                vertical="realestate",
                project=project,
            )
        )
        await s.commit()
    async with tenant_session(engine, tenant_id) as session:
        await load_units(session=session, path=FIXTURE)
        await session.commit()
    return tenant_id


async def _clear(engine, tenant_tables):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def developer(app_engine, engine, tenant_tables):
    tenant_id = await _tenant(engine, app_engine, tenant_tables, project=PROJECT)
    async with tenant_session(app_engine, tenant_id) as session:
        repository = InventoryRepository(session=session)
        yield repository, InventoryAgent(
            repository=repository,
            engine=ScriptEngine.from_config(SCRIPT),
            extractor=KeywordSlotExtractor(catalogue=await repository.vocabulary()),
        )
    await _clear(engine, tenant_tables)


@pytest_asyncio.fixture(loop_scope="session")
async def broker(app_engine, engine, tenant_tables):
    """The negative control. A guard that refuses everyone passes every test
    above it, and a broker refused for naming a second compound is the entire
    product broken."""
    tenant_id = await _tenant(engine, app_engine, tenant_tables, project=None)
    async with tenant_session(app_engine, tenant_id) as session:
        repository = InventoryRepository(session=session)
        yield repository, InventoryAgent(
            repository=repository,
            engine=ScriptEngine.from_config(SCRIPT),
            extractor=KeywordSlotExtractor(catalogue=await repository.vocabulary()),
        )
    await _clear(engine, tenant_tables)


def fresh() -> ConversationState:
    return ConversationState(
        script_id=SCRIPT, script_version=ScriptEngine.from_config(SCRIPT).version
    )


# ─────────────────── vocabulary scope is not search scope ───────────────────


async def test_the_extraction_vocabulary_is_the_whole_sheet(developer):
    """So the model can name what the customer said, instead of substituting
    the only value it was offered."""
    repository, _ = developer
    vocabulary = await repository.vocabulary()
    assert PROJECT in vocabulary["compound"]
    assert ELSEWHERE in vocabulary["compound"], (
        "offered one compound, the extractor returns that compound for every "
        "question — which is a substitution wearing the shape of an answer"
    )


async def test_the_search_is_still_filtered_to_the_project(developer):
    """Widening the vocabulary must not widen what can be quoted."""
    repository, _ = developer
    assert await repository.search(UnitQuery(compound=ELSEWHERE)) == []
    assert await repository.search(UnitQuery(compound=PROJECT)) != []
    assert {u.compound for u in await repository.search(UnitQuery())} == {PROJECT}


async def test_a_unit_outside_the_project_is_still_unreachable_by_id(developer):
    repository, _ = developer
    assert await repository.get("NOOR-CIT-002-02") is None
    assert await repository.get("MADINATY-001-01") is not None


async def test_the_graded_compound_catalogue_stays_scoped(developer):
    """`compounds()` is what `invented_compound_rate` grades against. Unscoped,
    naming the phase next door would read as correct."""
    repository, _ = developer
    assert await repository.compounds() == frozenset({PROJECT})


async def test_the_tenants_project_is_readable_without_reaching_into_privates(
    developer,
):
    repository, _ = developer
    assert await repository.project() == PROJECT


# ─────────────────────────── the refusal ───────────────────────────


async def test_a_compound_outside_the_project_is_refused(developer):
    """The rehearsal's developer turn."""
    _, agent = developer
    turn = await agent.handle(state=fresh(), text=ASK_ELSEWHERE)

    assert turn.action is Action.refuse, "answering this is a cross-project leak"
    assert PROJECT in turn.reply, "the customer is told what this developer does sell"
    assert ELSEWHERE not in turn.reply
    assert not turn.presented_unit_ids, "a refusal quotes no unit"


async def test_the_refusal_states_no_figure(developer):
    """The failure being replaced was a real price for a real unit nobody asked
    about, so a refusal that quotes anything has not fixed it."""
    _, agent = developer
    turn = await agent.handle(state=fresh(), text=ASK_ELSEWHERE)
    assert turn.provenance is None or not turn.provenance["figures"]
    assert not any(character.isdigit() for character in turn.reply)


async def test_the_projects_own_compound_still_answers(developer):
    _, agent = developer
    turn = await agent.handle(state=fresh(), text=ASK_OWN)
    assert turn.action is Action.answer
    assert PROJECT in turn.reply
    assert turn.presented_unit_ids


async def test_a_held_out_of_project_compound_cannot_survive_into_an_answer(developer):
    """The refusal must not leave the compound held, or the next turn — which
    names no place — searches on it and reports the developer has no stock."""
    _, agent = developer
    refused = await agent.handle(state=fresh(), text=ASK_ELSEWHERE)
    following = await agent.handle(state=refused.state, text="عايز شقة")
    assert following.action is Action.answer
    assert PROJECT in following.reply


async def test_a_broker_naming_a_second_compound_is_never_refused(broker):
    """The negative control, and the reason the guard reads `tenants.project`
    rather than comparing against whatever the last turn held."""
    _, agent = broker
    turn = await agent.handle(state=fresh(), text=ASK_ELSEWHERE)
    assert turn.action is not Action.refuse
    assert ELSEWHERE in turn.reply
