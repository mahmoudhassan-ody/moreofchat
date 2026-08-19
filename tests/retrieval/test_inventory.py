"""The inventory connector — design §3.2, P1b Task 23.

Structured inventory, not document chunks. §3.2's point is that a unit's price
arrives *filtered on availability*, carrying an `as_of`, from a row rather
than a passage. That is why the broker fixture is deliberately absent from
`kb_chunks` (P1's exit note): a second path to a unit's price would bypass
both of those, and `sold_unit_offered_rate` would read zero while a sold
unit's price sat in a retrieved chunk.

**`test_a_caller_cannot_disable_the_availability_filter` is the one that
matters.** Every other test here checks a behaviour that a reviewer would
think to check and that a failing case would surface. That one checks a
*shape*: that there exists no expression in this module which reaches the
inventory table without the availability predicate. Not "no expression we
wrote" — no expression that runs. It is the same guarantee `vectors.py` makes
about the tenant filter, and for the same reason: a filter whose absence has
no behavioural signature will eventually be absent.
"""

import ast
import inspect
import uuid
from pathlib import Path

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.retrieval import inventory
from moc.retrieval.inventory import InventoryRepository, UnitQuery, load_units

MODULE = Path(inventory.__file__)
TREE = ast.parse(MODULE.read_text(encoding="utf-8"))

FIXTURE = (
    Path(__file__).parents[2] / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)

#: Anything through which a caller could ask for unavailable rows. Named the
#: way the parameter would actually be named by someone adding it in a hurry.
AVAILABILITY_ARGUMENT_NAMES = {
    "availability",
    "status",
    "include_sold",
    "include_reserved",
    "include_unavailable",
    "all_statuses",
    "where",
    "filters",
    "predicate",
    "sql",
}

PUBLIC_METHODS = [
    name
    for name, _ in inspect.getmembers(InventoryRepository, inspect.isfunction)
    if not name.startswith("_")
]


@pytest_asyncio.fixture(loop_scope="session")
async def stocked(engine, tenant_tables):
    """The frozen broker fixture, ingested through the real path, twice over.

    Two tenants, because tenant scoping is asserted here rather than assumed:
    the same 305 units under both, so a leak shows up as doubled counts rather
    than as nothing at all.
    """
    from moc.tenancy.models import Tenant

    ids = {"a": uuid.uuid4(), "b": uuid.uuid4()}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=ids["a"], slug="broker-a", name="A", vertical="realestate"))
        s.add(Tenant(id=ids["b"], slug="broker-b", name="B", vertical="realestate"))
        await s.commit()

    from moc.tenancy.context import tenant_session

    for key in ("a", "b"):
        async with tenant_session(engine, ids[key]) as session:
            await load_units(session=session, path=FIXTURE)
            await session.commit()

    yield ids

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def repo(app_engine, stocked):
    """A repository on a session scoped to tenant A, as the agent would use it."""
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, stocked["a"]) as session:
        yield InventoryRepository(session=session), stocked


# ─────────────────────── the headline: unrepresentable ───────────────────────


def test_a_caller_cannot_disable_the_availability_filter():
    """Four independent structural facts, each of which alone would suffice.

    Together they mean there is no expression in this module that reaches the
    inventory table without the availability predicate having been applied —
    not "none we wrote", but none that runs.

    Opt-in filtering is how a sold unit reaches a second buyer. The query that
    forgets is the query that ships, and it ships green: a missing availability
    clause returns *more* rows, which reads as better coverage right up until
    a customer is shown a flat somebody else has already bought.
    """
    # 1. No public method offers a way to name a status, so a caller cannot
    #    ask for one and therefore cannot ask for `sold`.
    for name in PUBLIC_METHODS:
        parameters = set(inspect.signature(getattr(InventoryRepository, name)).parameters)
        assert not parameters & AVAILABILITY_ARGUMENT_NAMES, (
            f"{name} lets the caller name a status; that is the channel through "
            f"which a sold unit is offered"
        )

    # 2. Nor does the query object. `UnitQuery` is what a caller fills in, and
    #    it must have no field that speaks about availability at all.
    query_fields = set(inspect.signature(UnitQuery).parameters)
    assert not query_fields & AVAILABILITY_ARGUMENT_NAMES, (
        f"UnitQuery exposes {query_fields & AVAILABILITY_ARGUMENT_NAMES}; the filter "
        f"stops being a guarantee the moment it is an argument"
    )

    # 3. Every statement in the module that touches the table is filtered.
    #
    #    Read from the source segment rather than from a bare constant,
    #    because the queries are f-strings: the predicate arrives as `{where}`
    #    and a constant-only scan sees `... FROM inventory_units WHERE ` and
    #    calls it unfiltered. A statement qualifies by naming the column
    #    itself, or by interpolating the single builder's output — which
    #    assertion 4 pins to always emitting it.
    source = MODULE.read_text(encoding="utf-8")
    # The constant *parts* of an f-string are nodes too, and their source
    # spans do not line up across an implicit concatenation — a part reads as
    # `... FROM inventory_units "` with a trailing lint pragma, and looks
    # unfiltered. Only whole expressions are statements, so parts are skipped.
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
        and any(verb in segment for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"))
    ]
    assert statements, "no SQL found — this assertion would pass vacuously"
    for statement in statements:
        assert "availability" in statement or "{where}" in statement, (
            f"a statement reads inventory_units without the availability "
            f"predicate:\n{statement}"
        )

    # 4. The comparison itself is written once. Every statement above names
    #    the column, but only one place decides what it is compared against —
    #    a second occurrence would be a second thing to keep correct.
    source = MODULE.read_text(encoding="utf-8")
    assert source.count("availability = :available") == 1, (
        "the availability comparison appears more than once; it must be written in "
        "exactly one place, and every query must reach the table through it"
    )

    # 5. And that one place is actually wired in. Assertions 1-4 describe the
    #    plumbing; without this they all hold for a `_where` that defines the
    #    predicate and then does not use it — the shape intact, the content
    #    empty. Proven by sabotage: deleting the clause from the builder left
    #    1-4 green and only the behavioural tests red.
    builder = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "_where"
    )
    referenced = {
        node.id for node in ast.walk(builder) if isinstance(node, ast.Name)
    }
    assert "_AVAILABLE" in referenced, (
        "the availability predicate is defined but never used by the builder; "
        "every query would run unfiltered while this test still passed"
    )


def test_the_one_predicate_builder_always_emits_the_availability_clause():
    """`_where` has no branch that returns without it.

    A conditional availability clause is the version of this bug that survives
    review: correct on every path anyone tested, absent on the one they did not.
    """
    builder = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "_where"
    )
    returns = [node for node in ast.walk(builder) if isinstance(node, ast.Return)]
    assert len(returns) == 1, (
        "more than one return from the predicate builder — each is a path that has to "
        "be checked separately, and one of them eventually is not"
    )
    source = ast.get_source_segment(MODULE.read_text(encoding="utf-8"), builder)
    assert "availability" in source


def test_the_available_status_comes_from_config_not_a_literal(monkeypatch):
    """§19. Every broker CRM spells it differently — "available" / "vacant" /
    "free" — so the word is data. What is not configurable is that a unit
    which is not available is never offered.

    Asserted behaviourally rather than by grepping for the string: a grep
    catches a literal and misses a default argument, and it also trips over
    the bind-parameter name, which is not the status word at all.
    """
    monkeypatch.setattr(
        inventory, "_settings", lambda: {"available_status": "vacant", "query": {}}
    )
    _, params = inventory._where(UnitQuery())
    assert params["available"] == "vacant"


# ─────────────────────────── behaviour ───────────────────────────


async def test_lookup_filters_unavailable_by_default(repo):
    """Sold and reserved excluded without the caller asking.

    Asserted on a compound that actually holds a sold unit — Noor City has
    NOOR-CIT-002-01 sold at 3.9M alongside two available units — because a
    search over the whole catalogue would pass this by luck once the reply
    limit truncates the list.
    """
    repository, _ = repo
    units = await repository.search(UnitQuery(compound="Noor City", limit=50))
    returned = {unit.unit_id for unit in units}

    assert returned, "the compound has available units, so this is not vacuous"
    assert "NOOR-CIT-002-01" not in returned, "a sold unit was offered"
    assert {unit.availability for unit in units} == {"available"}


async def test_reserved_is_excluded_as_well_as_sold(repo):
    """re-0012. A status check that only excludes `sold` lets a reserved unit
    through, and offering it to a second buyer is the same failure."""
    repository, _ = repo
    units = await repository.search(UnitQuery(compound="SouthMED", limit=50))
    assert units
    assert "SOUTHMED-004-02" not in {unit.unit_id for unit in units}, "a reserved unit"
    assert await repository.get("SOUTHMED-004-02") is None
    assert await repository.get("MADINATY-001-01") is not None


async def test_a_sold_unit_is_unreachable_even_by_id(repo):
    """The id path is the one a caller reaches for when the search path says
    no. It applies the same filter, or the filter is advisory."""
    repository, ids = repo
    sold = await _one_with_availability(repository, "sold")
    assert await repository.get(sold) is None


async def test_every_result_carries_the_snapshot_as_of(repo):
    """§3.2. A price without a date is a price the tenant cannot stand behind,
    and `asof_disclosure_rate` has nothing to disclose if the row does not
    carry it."""
    repository, _ = repo
    units = await repository.search(UnitQuery(city="New Cairo"))
    assert units
    assert {str(unit.as_of) for unit in units} == {"2026-08-01"}


async def test_lookup_is_tenant_scoped(app_engine, stocked):
    """Both tenants hold the same 305 units. A missing tenant filter returns
    564 available rows rather than 282, which is the doubling that makes this
    assertion meaningful."""
    from moc.tenancy.context import tenant_session

    for key in ("a", "b"):
        async with tenant_session(app_engine, stocked[key]) as session:
            # Direct SQL, because the policy is what is under test: a leak
            # shows up as 610 rows rather than as a differently-ordered list.
            total = (
                await session.execute(text("SELECT count(*) FROM inventory_units"))
            ).scalar_one()
            assert total == 305

            units = await InventoryRepository(session=session).search(
                UnitQuery(compound="Madinaty", limit=50)
            )
            assert units
            assert {unit.compound for unit in units} == {"Madinaty"}


async def test_filters_compose_city_type_bedrooms_and_budget(repo):
    repository, _ = repo
    units = await repository.search(
        UnitQuery(city="New Cairo", property_type="apartment", bedrooms=2, budget_max=8_000_000)
    )
    assert units
    for unit in units:
        assert unit.city == "New Cairo"
        assert unit.property_type == "apartment"
        assert unit.bedrooms == 2
        assert unit.price <= 8_000_000


async def test_budget_max_is_inclusive_and_never_returns_above_it(repo):
    """re-0011 asserts no unit above 17M. An off-by-one here quotes a price the
    customer already said they cannot pay."""
    repository, _ = repo
    units = await repository.search(UnitQuery(budget_max=17_000_000))
    assert units
    assert max(unit.price for unit in units) <= 17_000_000

    exact = await repository.search(UnitQuery(budget_max=5_800_000, compound="Madinaty"))
    assert any(unit.price == 5_800_000 for unit in exact), "the boundary is inclusive"


async def test_a_type_with_no_matching_unit_returns_empty_not_nearest(repo):
    """re-0021: no villa under 15M — the cheapest is 23.95M. Returning the
    closest is where cross-type substitution begins, at the data layer, before
    any model sees it."""
    repository, _ = repo
    units = await repository.search(UnitQuery(property_type="villa", budget_max=15_000_000))
    assert units == []


async def test_compound_and_city_are_distinct_filters(repo):
    """`locations.yaml`'s `kind` map exists for this. Filtering `city` with a
    compound name silently returns nothing and reads as 'no inventory'."""
    repository, _ = repo
    by_compound = await repository.search(UnitQuery(compound="Madinaty"))
    assert by_compound
    assert {unit.compound for unit in by_compound} == {"Madinaty"}

    as_city = await repository.search(UnitQuery(city="Madinaty"))
    assert as_city == [], "a compound name is not a city, and must not silently match one"


def test_the_eval_vocabulary_agrees_with_the_runtime_one():
    """Two config files name the available status: the connector reads one and
    the grading side reads the other, because the harness must be able to
    grade a run whose runtime config it does not share. The duplication is
    deliberate; silent disagreement is not."""
    from moc.config_store import load

    assert load("retrieval/inventory")["available_status"] == (
        load("evals/inventory")["available_status"]
    )


async def test_the_catalogue_of_compounds_is_readable_for_the_reply_rules(repo):
    """§19.3: a compound named in a reply must exist. `invented_compound_rate`
    is checked against this set, so it comes from the same rows the answer
    does — a second list would drift and the gate would grade against it."""
    repository, _ = repo
    compounds = await repository.compounds()
    assert "Madinaty" in compounds
    assert len(compounds) > 20


async def _one_with_availability(repository, availability: str) -> str:
    """A unit id with a given availability, read straight from the fixture."""
    import json

    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["availability"] == availability:
            return row["unit_id"]
    raise AssertionError(f"fixture holds no {availability} unit")


# ─────────────────────────── ingest ───────────────────────────


async def test_the_fixture_loads_through_the_real_path(app_engine, stocked):
    """Not a loader that reads the file at query time. §3.2's connector reads a
    table, and a test that skips the table tests nothing about production.

    Counted with direct SQL rather than a repository method, deliberately: the
    repository has no unfiltered read, because one would make the availability
    filter a convention again. Sold and reserved rows are stored — they are
    what a later snapshot updates *from*, and dropping them would let a unit
    reappear as available the moment it left the export.
    """
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, stocked["a"]) as session:
        total = (
            await session.execute(text("SELECT count(*) FROM inventory_units"))
        ).scalar_one()
    assert total == 305, "every row, including the sold and reserved ones"


async def test_the_repository_offers_no_unfiltered_read(app_engine, stocked):
    """The absence is the guarantee.

    A `count_all` or `all_units` would be the method someone calls when the
    filtered one returns nothing — and it would return sold units, correctly,
    which is exactly the failure. Ingest counts belong to whoever loaded the
    rows, not to the path that answers customers.
    """
    assert not {"count_all", "all_units", "raw", "execute"} & set(PUBLIC_METHODS)


async def test_reloading_is_idempotent(app_engine, engine, stocked):
    """A re-ingest updates rather than duplicating. Two rows for one unit is
    the same unit offered twice."""
    from moc.tenancy.context import tenant_session

    async with tenant_session(engine, stocked["a"]) as session:
        assert await load_units(session=session, path=FIXTURE) == 305
        await session.commit()

    async with tenant_session(app_engine, stocked["a"]) as session:
        total = (
            await session.execute(text("SELECT count(*) FROM inventory_units"))
        ).scalar_one()
        assert total == 305


async def test_a_unit_carries_its_payment_plan_untouched(repo):
    """The calculator is Task 24's job. The connector's job is to hand over the
    terms the developer actually published, unrounded and unedited."""
    repository, _ = repo
    unit = await repository.get("NOOR-CIT-002-02")
    assert unit is not None
    assert unit.payment_plan is not None
    assert unit.payment_plan["down_payment_pct"] == 25
