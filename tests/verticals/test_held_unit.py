"""Which unit the customer means — found by the Task 42 rehearsal.

`quoted_unit_id` is context for a turn that names nothing: "and at 40% down?"
identifies a unit only by having been preceded by one. It was also outranking
the unit the customer named *in the current message*, and the rehearsal walked
straight into it — browse Madinaty, then ask about a Noor City unit by name and
price, and the bot answers about Madinaty. With a price.

That is the worst class of error this system can make. It is fluent, it is
grounded, every figure traces, and it is about the wrong property. Nothing in
the eval suite caught it because every payment-plan case is a first turn, and a
first turn holds no context to be outranked by.
"""

import uuid
from pathlib import Path

import pytest_asyncio

from moc.agent.script_engine import ScriptEngine
from moc.agent.state import Action, ConversationState
from moc.retrieval.inventory import InventoryRepository, load_units
from moc.verticals.realestate.agent import InventoryAgent, KeywordSlotExtractor

SCRIPT = "scripts/realestate/search"
FIXTURE = (
    Path(__file__).parents[2] / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)


@pytest_asyncio.fixture(loop_scope="session")
async def broker(app_engine, engine, tenant_tables):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.tenancy.context import tenant_session
    from moc.tenancy.models import Tenant

    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="held", name="Held", vertical="realestate"))
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
    return ConversationState(
        script_id=SCRIPT, script_version=ScriptEngine.from_config(SCRIPT).version
    )


async def test_a_unit_named_this_turn_beats_the_one_held_from_the_last(broker):
    """The rehearsal's own sequence."""
    first = await broker.handle(state=fresh(), text="عايز شقة في مدينتي")
    assert first.presented_unit_ids, "the first turn quoted nothing to hold"
    assert first.named_compounds == ("Madinaty",)

    second = await broker.handle(
        state=first.state, text="الوحدة في نور سيتي بـ ٦ مليون و نص، القسط كام؟"
    )
    assert "Madinaty" not in second.reply, (
        "the customer named Noor City and was answered about Madinaty — fluent, "
        "grounded, every figure traced, and about the wrong property"
    )


async def test_a_turn_that_names_nothing_still_uses_the_held_unit(broker):
    """The negative control, and the whole reason the held unit exists. "And at
    40% down?" identifies a unit only by what preceded it, so a fix that made
    the current message always win would break the follow-up this state was
    added for."""
    first = await broker.handle(state=fresh(), text="عايز شقة في مدينتي")
    quoted = first.state.quoted_unit_id
    assert quoted

    second = await broker.handle(state=first.state, text="والتقسيط؟")
    assert second.state.quoted_unit_id == quoted
    assert second.action in (Action.answer, Action.handoff)
    if second.presented_unit_ids:
        assert second.presented_unit_ids == (quoted,)


async def test_naming_the_same_place_again_does_not_switch_units(broker):
    """Repeating the compound is not a new request. Re-resolving from the
    message would silently move the customer onto a different unit in the same
    development — same compound, different price, and no way for them to tell
    which one the last figure described."""
    first = await broker.handle(state=fresh(), text="عايز شقة في مدينتي")
    quoted = first.state.quoted_unit_id

    second = await broker.handle(state=first.state, text="والتقسيط في مدينتي؟")
    if second.presented_unit_ids:
        assert second.presented_unit_ids == (quoted,)


def test_the_search_never_falls_back_to_the_held_unit():
    """Structural, because the behavioural version is unreachable.

    Rule 4 — a named place that resolves to nothing returns nothing — cannot be
    driven from a message here: the extractor resolves compounds against the
    tenant's own catalogue, so a place they do not stock is not extracted as a
    place at all, it is extracted as silence. The customer who names one is
    indistinguishable from the customer who names none, and the held unit is
    then the right answer.

    What can be asserted is the shape: once the message has named a compound,
    no path returns the held unit. That is the line the rehearsal's bug was on,
    and for a project-scoped developer it would be an answer about a project
    they do not sell.
    """
    import ast
    import inspect

    from moc.verticals.realestate.agent import InventoryAgent

    tree = ast.parse(inspect.getsource(InventoryAgent))
    resolve = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_resolve_unit"
    )
    returns = [
        node
        for node in ast.walk(resolve)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "held"
    ]
    assert len(returns) == 2, (
        f"{len(returns)} paths return the held unit; exactly two may — the one "
        "where this message names no place, and the one where it names the place "
        "that unit is already in"
    )


# ─────────── a name this tenant does not know — found by the rehearsal ───────────


class RefusesTheName:
    """An extractor that reads a real place this tenant does not stock.

    Exactly what `LlmSlotExtractor` does for a project-scoped developer: the
    catalogue is scoped to their project, so every other compound is
    out-of-vocabulary — and the model is *right* about what the customer said.
    """

    async def extract(self, *, text, state):
        from moc.agent.extraction import ExtractionFailed

        raise ExtractionFailed(
            "compound='Noor City' is outside the configured vocabulary"
        )


async def test_a_place_this_tenant_does_not_stock_gets_a_reply_not_silence(broker):
    """§2.6 is not "no wrong answer reaches the customer", it is "no error
    does" — and nothing at all is the worst version of one.

    The extractor is right to refuse a value it cannot resolve: one that filters
    nothing reads as absent stock. Refusing by *raising* killed the turn, the
    message dead-lettered, and the customer waited. The Task 42 rehearsal asked
    a developer about the project next door and got silence.
    """
    from moc.verticals.realestate.agent import InventoryAgent

    agent = InventoryAgent(
        repository=broker._repository,
        engine=ScriptEngine.from_config(SCRIPT),
        extractor=RefusesTheName(),
    )
    result = await agent.handle(state=fresh(), text="وعندكم إيه في نور سيتي؟")

    assert result.reply, "the customer got silence"
    assert result.action is Action.handoff


async def test_the_place_comes_from_this_message_and_not_from_the_merge(broker):
    """The engine merges each turn's slots over everything held.

    So a customer who named Madinaty and then asks about somewhere else arrives
    at the resolver with `compound` still reading Madinaty — and comparing the
    *merge* against the held unit says "same place, keep the held one" for a
    message that named a different one. The rehearsal's second question, and
    the reason the first fix was not enough.
    """
    first = await broker.handle(state=fresh(), text="عايز شقة في مدينتي")
    assert first.state.slots.get("compound") == "Madinaty"

    unit = await broker._resolve_unit(
        # What the merge looks like on the next turn: the held compound,
        # because this message left it out of the slots the engine merged.
        {"compound": "Madinaty", "near_price": 6_500_000},
        first.state.quoted_unit_id,
        # What the message actually said.
        {"compound": "Noor City", "near_price": 6_500_000},
    )
    assert unit is not None
    assert unit.compound == "Noor City", (
        "the merge outranked the message; the customer named Noor City and the "
        "resolver returned the unit they were shown a turn earlier"
    )
