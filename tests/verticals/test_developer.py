"""The developer variant — design §11.2, demo plan Task 38.

A tenant flag, not a second product. A broker searches across projects; a
developer answers deeply about one and routes the lead to the right sales
team. Everything else — the availability filter, the no-substitution rule, the
calculator, the `as_of` disclosure — is the same code.

**Why the project scope is not a no-op.** The obvious objection is that a
developer tenant only holds its own project, so scoping to it changes nothing.
That is not how inventory arrives. It arrives as a sheet, and sheets hold more
than they should: a second phase, a sister development, last year's launch,
the row somebody pasted in to compare against. RLS says which tenant; nothing
until now said which project.

**`test_an_alternative_never_comes_from_another_project` is the one that
matters.** The no-substitution rule permits one alternative — the same type
somewhere else — and for a broker that is the whole point. For a developer,
"somewhere else" is a different development, and offering it is either a
competitor's unit or a phase this sales team cannot sell. The rule that saves
a broker leaks a developer, and only the scope closes it.
"""

import ast
import inspect
import json
import uuid
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from moc.retrieval import inventory
from moc.retrieval.inventory import InventoryRepository, UnitQuery, load_units
from moc.tenancy.context import tenant_session
from moc.verticals.realestate.leads import (
    Lead,
    SalesTeam,
    SalesTeamStore,
    open_lead,
    qualify,
    route,
)
from moc.verticals.realestate.replies import NoMatch, find_same_type_elsewhere, route_no_match

MODULE = Path(inventory.__file__)
TREE = ast.parse(MODULE.read_text(encoding="utf-8"))

PROJECT = "Sinai Heights"
OTHER = "Sinai Marina"

#: Anything through which a caller could widen or replace the project scope.
#: Named the way the parameter would be named by whoever adds it in a hurry.
SCOPE_ARGUMENT_NAMES = {
    "project",
    "projects",
    "scope",
    "all_projects",
    "include_other_projects",
    "across_projects",
}

PUBLIC_METHODS = [
    name
    for name, _ in inspect.getmembers(InventoryRepository, inspect.isfunction)
    if not name.startswith("_")
]


def unit(unit_id: str, compound: str, property_type: str, price: int, **extra) -> dict:
    return {
        "unit_id": unit_id,
        "as_of": "2026-08-01",
        "property_type": property_type,
        "compound": compound,
        "city": "New Cairo",
        "price": price,
        "currency": "EGP",
        "availability": "available",
        "bedrooms": extra.get("bedrooms", 3),
        **extra,
    }


#: The developer's own project holds chalets and villas and no studio; the
#: second phase in the same sheet holds the studio. Both belong to this tenant,
#: so RLS separates neither of them.
UNITS = [
    unit("SH-1", PROJECT, "chalet", 6_000_000),
    unit("SH-2", PROJECT, "chalet", 7_500_000),
    unit("SH-3", PROJECT, "villa", 24_000_000),
    unit("SM-1", OTHER, "studio", 3_200_000),
    unit("SM-2", OTHER, "chalet", 5_000_000),
]


@pytest_asyncio.fixture(loop_scope="session")
async def two_kinds_of_tenant(engine, tenant_tables, tmp_path_factory):
    """One developer and one broker, holding identical rows.

    Identical on purpose: every difference below is the flag and nothing else,
    which is what "a tenant flag, not two products" has to mean to be true.
    """
    from moc.tenancy.models import Tenant

    snapshot = tmp_path_factory.mktemp("units") / "units.jsonl"
    snapshot.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in UNITS), encoding="utf-8"
    )

    ids = {"developer": uuid.uuid4(), "broker": uuid.uuid4()}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(
            Tenant(
                id=ids["developer"],
                slug="dev",
                name="Developer",
                vertical="realestate",
                project=PROJECT,
            )
        )
        s.add(Tenant(id=ids["broker"], slug="brk", name="Broker", vertical="realestate"))
        await s.commit()

    for key in ids:
        async with tenant_session(engine, ids[key]) as session:
            await load_units(session=session, path=snapshot)
            await session.commit()

    yield ids

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def developer(app_engine, two_kinds_of_tenant):
    async with tenant_session(app_engine, two_kinds_of_tenant["developer"]) as session:
        yield InventoryRepository(session=session)


@pytest_asyncio.fixture(loop_scope="session")
async def broker(app_engine, two_kinds_of_tenant):
    async with tenant_session(app_engine, two_kinds_of_tenant["broker"]) as session:
        yield InventoryRepository(session=session)


# ─────────────────────── the scope is not a parameter ───────────────────────


def test_no_caller_can_widen_the_project_scope():
    """The same shape the availability filter has, for the same reason.

    A developer whose bot answers about the phase next door is not obviously
    broken from the outside — the units are real, the prices are right, and
    the tenant owns the rows. It surfaces when a customer arrives at a sales
    office asking about a unit that office cannot sell.
    """
    for name in PUBLIC_METHODS:
        parameters = set(inspect.signature(getattr(InventoryRepository, name)).parameters)
        assert not parameters & SCOPE_ARGUMENT_NAMES, (
            f"{name} lets the caller name a project; the scope stops being a "
            f"guarantee the moment it is an argument"
        )

    query_fields = set(inspect.signature(UnitQuery).parameters)
    assert not query_fields & SCOPE_ARGUMENT_NAMES, (
        "UnitQuery exposes a project argument; it is what a caller fills in"
    )

    constructor = set(inspect.signature(InventoryRepository.__init__).parameters)
    assert not constructor & SCOPE_ARGUMENT_NAMES, (
        "the repository takes the project from its caller — which is a caller "
        "that can pass the wrong one, or forget it and get broker behaviour"
    )


def test_the_project_predicate_is_written_once_and_actually_wired_in():
    """The same five-part shape the availability filter gets, because the
    failure is identical: a clause that is defined, described in a docstring,
    and never reaches a statement leaves every query unfiltered while the
    structure of the guard is intact.
    """
    source = MODULE.read_text(encoding="utf-8")

    # Written in one place. A second occurrence is a second thing to keep
    # correct, and one of the two eventually is not.
    assert source.count("cast(:project as text) IS NULL") == 1

    builder = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "_where"
    )
    returns = [node for node in ast.walk(builder) if isinstance(node, ast.Return)]
    assert len(returns) == 1, (
        "more than one return from the predicate builder — each is a path that "
        "has to be checked separately, and one of them eventually is not"
    )
    referenced = {node.id for node in ast.walk(builder) if isinstance(node, ast.Name)}
    assert "_PROJECT" in referenced, (
        "the project predicate is defined but never used by the builder; every "
        "query would run unscoped while this test still passed"
    )


def test_the_builder_binds_the_scope_on_every_call_including_a_brokers():
    """Bound unconditionally, NULL for a broker.

    A clause that appeared only when a project was set would be a clause a
    statement could be written without — and the statement that is written
    without it is the one nobody re-reads.
    """
    scoped, params = inventory._where(UnitQuery(), PROJECT)
    assert "compound" in scoped
    assert params["project"] == PROJECT

    unscoped, params = inventory._where(UnitQuery(), None)
    assert unscoped == scoped, "the predicate differs between a broker and a developer"
    assert params["project"] is None


def test_every_statement_that_reads_inventory_is_project_filtered():
    """Including `vocabulary`, which feeds the extractor.

    An unscoped vocabulary is worse than an unscoped search: the extractor
    resolves a compound the sales team does not sell, the search then returns
    nothing, and the customer is told the developer has no stock — about a
    project they never asked about.

    Read from source segments rather than bare constants: the queries are
    f-strings, and a constant-only scan sees `... FROM inventory_units WHERE `
    and calls it unfiltered.
    """
    source = MODULE.read_text(encoding="utf-8")
    nested = {
        id(child)
        for node in ast.walk(TREE)
        if isinstance(node, ast.JoinedStr)
        for child in ast.walk(node)
        if child is not node
    }
    statements = [
        segment
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant | ast.JoinedStr)
        and id(node) not in nested
        and (segment := ast.get_source_segment(source, node) or "")
        and "inventory_units" in segment
        and "SELECT" in segment
    ]
    assert statements, "no SQL found — this assertion would pass vacuously"
    for statement in statements:
        assert "{where}" in statement, (
            f"a read reaches inventory_units without the shared predicate, so it "
            f"is scoped to no project:\n{statement}"
        )


# ─────────────────────────── the scope, in behaviour ───────────────────────────


async def test_a_developer_tenant_scopes_every_search_to_its_own_project(developer):
    units = await developer.search(UnitQuery())
    assert {row.compound for row in units} == {PROJECT}
    assert {row.unit_id for row in units} == {"SH-1", "SH-2", "SH-3"}


async def test_a_broker_tenant_searches_across_projects(broker):
    """The negative control. Without it, a scope that matched nothing would
    pass every test above."""
    units = await broker.search(UnitQuery())
    assert {row.compound for row in units} == {PROJECT, OTHER}


async def test_a_developer_asked_about_another_project_finds_nothing(developer):
    """Not an error and not the nearest thing. They do not sell it."""
    assert await developer.search(UnitQuery(compound=OTHER)) == []


async def test_a_unit_in_another_project_is_unreachable_even_by_id(developer):
    """The id path is what a caller reaches for when search says no — the same
    reason it carries the availability filter."""
    assert await developer.get("SM-1") is None
    assert (await developer.get("SH-1")) is not None


async def test_the_extraction_vocabulary_holds_only_the_projects_own_names(developer):
    vocabulary = await developer.vocabulary()
    assert vocabulary["compound"] == frozenset({PROJECT})


async def test_the_compound_catalogue_is_scoped_too(developer):
    """`compounds()` is what `invented_compound_rate` grades against. Unscoped,
    naming the phase next door would read as correct."""
    assert await developer.compounds() == frozenset({PROJECT})


# ─────────────────── no substitution, inside one project ───────────────────


async def test_the_no_substitution_rule_still_holds_within_one_project(developer):
    """A chalet is not a studio, and a single-project catalogue does not change
    that. The project holds chalets and villas; a studio request must not be
    answered with the closest-priced thing that is neither."""
    units = await developer.search(UnitQuery(property_type="studio"))
    assert units == []

    alternative = await find_same_type_elsewhere(developer, property_type="studio")
    assert alternative is None


async def test_an_alternative_never_comes_from_another_project(developer):
    """The rule that saves a broker leaks a developer.

    `find_same_type_elsewhere` is the one permitted alternative: same type,
    different place. For a broker that is the right answer. Here the only
    studio this tenant holds is in a different development, and offering it
    sends a buyer to a sales office that cannot sell it.

    Nothing in the reply rules had to change — the scope is in the repository,
    so the alternative search cannot leave the project.
    """
    alternative = await find_same_type_elsewhere(
        developer, property_type="studio", exclude_compound=PROJECT
    )
    assert alternative is None

    no_match = NoMatch(requested_type="studio", asked_about=PROJECT, alternative=alternative)
    from moc.agent.state import Action

    assert route_no_match(no_match) is Action.handoff


async def test_a_broker_is_still_offered_the_same_type_elsewhere(broker):
    """The negative control, and the behaviour the rule was written for."""
    alternative = await find_same_type_elsewhere(
        broker, property_type="studio", exclude_compound=PROJECT
    )
    assert alternative is not None
    assert alternative.compound == OTHER
    assert alternative.property_type == "studio"


# ─────────────────────────── qualification ───────────────────────────


def test_a_lead_is_qualified_by_what_a_salesperson_needs_to_act():
    lead = qualify({"property_type": "villa", "budget_max": 25_000_000})
    assert lead.qualified is True
    assert lead.missing == ()


def test_a_lead_missing_a_budget_is_not_qualified_and_says_what_is_missing():
    lead = qualify({"property_type": "villa"})
    assert lead.qualified is False
    assert "budget" in " ".join(lead.missing)


def test_qualification_reads_slots_and_never_the_customers_enthusiasm():
    """Structural. Scoring intent from wording is exactly the guess this
    platform refuses everywhere else, and a lead score is a number a sales
    manager will act on."""
    source = inspect.getsource(qualify)
    for word in ("text", "message", "sentiment", "eager", "keen"):
        assert word not in source, f"qualification looks at {word!r} rather than at slots"


def test_the_score_counts_known_signals_and_nothing_it_invented():
    known = qualify({"property_type": "villa", "budget_max": 25_000_000, "city": "New Cairo"})
    fewer = qualify({"property_type": "villa", "budget_max": 25_000_000})
    assert known.score > fewer.score
    assert qualify({}).score == 0


# ─────────────────────────── routing ───────────────────────────


TEAMS = (
    SalesTeam(
        team_key="villas",
        name="Villas",
        contact="villas@dev.example",
        property_type="villa",
    ),
    SalesTeam(team_key="main", name="Sales", contact="sales@dev.example", property_type=None),
)


def test_routing_sends_a_qualified_lead_to_the_configured_team():
    lead = qualify({"property_type": "villa", "budget_max": 25_000_000})
    assert route(lead, TEAMS).team_key == "villas"


def test_a_type_with_no_team_of_its_own_goes_to_the_fallback():
    lead = qualify({"property_type": "chalet", "budget_max": 7_000_000})
    assert route(lead, TEAMS).team_key == "main"


def test_a_lead_that_named_no_type_is_not_handed_to_the_only_specialist():
    """With one specialist configured, the naive router returns it for
    everything. A lead that named no type is not a villa lead because villas is
    the only team on the list.

    Asserted against specialists alone, deliberately. With a fallback present
    the naive router happens to be right — `None == None` matches the fallback
    row, whose rule column is also NULL — so the version of this test that
    included one passed under a sabotage that reintroduced the bug.
    """
    villas_only = (TEAMS[0],)
    assert route(qualify({}), villas_only) is None
    assert route(qualify({}), TEAMS).team_key == "main"


def test_a_tenant_with_no_fallback_team_routes_nowhere_rather_than_somewhere():
    only_specialists = (TEAMS[0],)
    lead = qualify({"property_type": "chalet", "budget_max": 7_000_000})
    assert route(lead, only_specialists) is None


# ─────────────────────── the teams, and the handoff ───────────────────────


@pytest_asyncio.fixture(loop_scope="session")
async def developer_session(app_engine, two_kinds_of_tenant):
    async with tenant_session(app_engine, two_kinds_of_tenant["developer"]) as session:
        yield session


async def test_two_teams_cannot_claim_the_same_property_type(developer_session):
    """A rule that names a team is the team's own row, so a rule can never name
    a team that does not exist — and two rows claiming villas is refused by the
    database rather than by whichever router read them first."""
    store = SalesTeamStore(session=developer_session)
    await store.add(team_key="villas", name="Villas", contact="a@x", property_type="villa")
    with pytest.raises(IntegrityError):
        await store.add(
            team_key="villas2", name="Villas 2", contact="b@x", property_type="villa"
        )
    await developer_session.rollback()


async def test_a_tenant_cannot_have_two_fallback_teams(developer_session):
    store = SalesTeamStore(session=developer_session)
    await store.add(team_key="main", name="Sales", contact="a@x")
    with pytest.raises(IntegrityError):
        await store.add(team_key="main2", name="Sales 2", contact="b@x")
    await developer_session.rollback()


async def test_a_qualified_lead_reaches_the_inbox_carrying_its_team(developer_session):
    """End of the routing path: a handoff row an agent can filter by team."""
    store = SalesTeamStore(session=developer_session)
    await store.add(team_key="villas", name="Villas", contact="v@x", property_type="villa")
    await store.add(team_key="main", name="Sales", contact="s@x")

    conversation_id = await _a_conversation(developer_session)
    handoff, lead = await open_lead(
        developer_session,
        conversation_id=conversation_id,
        reason="lead",
        slots={"property_type": "villa", "budget_max": 25_000_000},
        resume_state={},
    )
    assert lead.qualified is True
    assert handoff.team == "villas"
    assert handoff.lead_qualified is True
    await developer_session.rollback()


async def test_a_lead_that_routes_nowhere_still_reaches_the_inbox(developer_session):
    """An unroutable lead is not a dropped lead. A developer who has configured
    no fallback team has a configuration problem; the customer who asked has a
    question, and it must still be in front of somebody."""
    conversation_id = await _a_conversation(developer_session)
    handoff, _ = await open_lead(
        developer_session,
        conversation_id=conversation_id,
        reason="lead",
        slots={"property_type": "villa", "budget_max": 25_000_000},
        resume_state={},
    )
    assert handoff.team is None
    assert handoff.id is not None
    await developer_session.rollback()


async def test_sales_teams_are_tenant_scoped(app_engine, two_kinds_of_tenant):
    """Under `moc_app`, never the owner — owners bypass RLS."""
    async with tenant_session(app_engine, two_kinds_of_tenant["developer"]) as session:
        await SalesTeamStore(session=session).add(
            team_key="villas", name="Villas", contact="v@x", property_type="villa"
        )
        await session.commit()

    async with tenant_session(app_engine, two_kinds_of_tenant["broker"]) as session:
        assert await SalesTeamStore(session=session).all() == ()


async def _a_conversation(session) -> uuid.UUID:
    conversation_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO conversations (id, tenant_id, channel, sender_ref) VALUES "
            "(:id, nullif(current_setting('moc.tenant_id', true), '')::uuid, "
            "'whatsapp', :ref)"
        ),
        {"id": conversation_id, "ref": f"+2010{uuid.uuid4().int % 10**8:08d}"},
    )
    return conversation_id


def test_the_lead_type_is_frozen():
    """A mutable lead is one a later step can upgrade to qualified."""
    lead = Lead(qualified=False, score=0, property_type=None, budget_max=None, missing=())
    with pytest.raises(FrozenInstanceError):
        lead.qualified = True  # type: ignore[misc]
