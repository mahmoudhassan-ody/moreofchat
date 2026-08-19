"""The real-estate script and the no-substitution reply — P1b Task 25.

Where the owner decision of 2026-08-17 becomes runtime behaviour: **a request
for one property type is never answered with another.** A chalet is not a
studio, an office is not retail, and the closest-priced unit of the wrong kind
is the wrong answer wearing the right price — the customer finds out at the
viewing.

**re-0022 is the case where the rule is under real pressure.** The North Coast
holds 33 units: 19 chalets, 11 townhouses, 3 villas, and no studio at all. A
customer asking for a studio there is asking for something that does not
exist, and every naive ranking puts a chalet first — it matches the city, the
budget, the language, and it is the single most common thing on that coast.
The rule has to survive the case where breaking it looks helpful.

re-0002 is the same shape at compound level, and re-0021 is the case with no
alternative at all: all five villas are 23.9M and up, so a 15M villa budget
has no compound to name. Saying nothing and handing off is the answer there;
naming a villa above budget would be answering a different question.
"""

import ast
import inspect
import json
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.state import Action, Register
from moc.config_store import load
from moc.retrieval.inventory import InventoryRepository, UnitQuery, load_units
from moc.verticals.realestate import replies
from moc.verticals.realestate.replies import (
    NoMatch,
    TypeSubstitution,
    find_same_type_elsewhere,
    render_no_match,
)

SCRIPT = "scripts/realestate/search"
FIXTURE = (
    Path(__file__).parents[2] / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)
MODULE = Path(replies.__file__)

#: Every residential pair, not just the one the case names. A rule tested on
#: one pair is a rule that holds for one pair.
RESIDENTIAL = ["studio", "apartment", "chalet", "townhouse", "villa", "duplex", "penthouse"]


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
async def repo(app_engine, stocked):
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, stocked) as session:
        yield InventoryRepository(session=session)


# ─────────────────────── the hard case: re-0022 ───────────────────────


async def test_a_studio_request_on_the_coast_is_never_answered_with_a_chalet(repo):
    """re-0022, and the reason the studio units were added to the fixture.

    Cross-type substitution was untestable before 2026-08-18: with one
    residential type in the catalogue, no request could be answered with the
    wrong one. Now the North Coast holds 19 chalets and zero studios, so the
    tempting answer is available and the rule can actually fail.
    """
    alternative = await find_same_type_elsewhere(
        repo, property_type="studio", exclude_city="North Coast"
    )
    assert alternative is not None, "studios exist inland, so there is one to name"
    assert alternative.property_type == "studio"
    assert alternative.city != "North Coast"

    reply = render_no_match(
        NoMatch(
            requested_type="studio",
            asked_about="North Coast",
            alternative=alternative,
        ),
        register=Register.masri,
        as_of="2026-08-01",
    )
    assert "chalet" not in reply.lower()
    assert alternative.compound in reply


async def test_the_coast_really_does_rank_a_chalet_first(repo):
    """The pressure, made explicit.

    If this ever returns nothing, the test above stops proving anything: it
    would be asserting that we avoid a substitution nothing was offering.
    """
    coast = await repo.search(UnitQuery(city="North Coast", limit=50))
    assert coast, "the coast has inventory"
    assert coast[0].property_type != "studio"
    assert not [unit for unit in coast if unit.property_type == "studio"]


@pytest.mark.parametrize("requested", RESIDENTIAL)
async def test_no_type_falls_back_to_another_when_it_has_no_match(repo, requested):
    """The sweep with the pressure applied to every type at once.

    At a 3M budget only the studio has any inventory, so seven of these eight
    requests find nothing — which is exactly when "be helpful, offer the
    nearest thing" gets added. Without the budget every type has a match and
    the fallback is never reached, so the sweep passes while the rule is
    broken. Proven: adding an any-type fallback fails this for seven types.
    """
    alternative = await find_same_type_elsewhere(
        repo, property_type=requested, budget_max=3_000_000
    )
    assert alternative is None or alternative.property_type == requested


@pytest.mark.parametrize("requested", RESIDENTIAL)
async def test_a_different_type_is_never_offered(repo, requested):
    """Every pair, not just villa/apartment.

    The alternative search pins the requested type into the query, so a unit
    of another kind cannot come back — there is no ranking step at which the
    wrong type could win.
    """
    alternative = await find_same_type_elsewhere(
        repo, property_type=requested, exclude_city="North Coast"
    )
    if alternative is not None:
        assert alternative.property_type == requested


def test_the_renderer_refuses_a_mismatched_alternative():
    """Defence in depth. The search cannot produce one, so this is the case
    where someone later builds a `NoMatch` by hand — and it must fail loudly
    rather than compose the substitution the rule forbids."""

    class FakeUnit:
        property_type = "chalet"
        compound = "Marassi"
        city = "North Coast"
        price = 9_000_000
        currency = "EGP"

    with pytest.raises(TypeSubstitution, match="studio"):
        render_no_match(
            NoMatch(requested_type="studio", asked_about="North Coast", alternative=FakeUnit()),
            register=Register.masri,
            as_of="2026-08-01",
        )


# ─────────────────────── naming both, from the catalogue ───────────────────────


async def test_no_match_names_both_compounds(repo):
    """re-0002: "no villa in <compound asked about> — we have a villa in
    <compound>". Both named, both from the catalogue."""
    alternative = await find_same_type_elsewhere(
        repo, property_type="villa", exclude_city="New Cairo"
    )
    assert alternative is not None
    reply = render_no_match(
        NoMatch(requested_type="villa", asked_about="Mivida", alternative=alternative),
        register=Register.masri,
        as_of="2026-08-01",
    )
    assert "Mivida" in reply, "the compound the customer asked about"
    assert alternative.compound in reply, "the compound that actually has one"


async def test_a_compound_named_in_a_reply_exists_in_the_catalogue(repo):
    """An invented compound is the same failure class as an invented price,
    and more convincing — a plausible Egyptian name reads as local knowledge
    until a customer drives to somewhere that does not exist."""
    catalogue = await repo.compounds()
    for requested in RESIDENTIAL:
        alternative = await find_same_type_elsewhere(repo, property_type=requested)
        if alternative is not None:
            assert alternative.compound in catalogue


async def test_no_match_anywhere_hands_off_without_an_alternative(repo):
    """re-0021: all five villas are 23.9M+, so a 15M villa budget has nothing
    to name. Unlike re-0002 there is no compound to offer, and naming one
    above budget would answer a different question."""
    alternative = await find_same_type_elsewhere(
        repo, property_type="villa", budget_max=15_000_000
    )
    assert alternative is None

    decision = replies.route_no_match(
        NoMatch(requested_type="villa", asked_about="New Cairo", alternative=None)
    )
    assert decision is Action.handoff


async def test_a_findable_alternative_is_answered_not_handed_off(repo):
    """The other half: a handoff for every no-match would be a bot that never
    answers the question re-0002 exists to test."""
    alternative = await find_same_type_elsewhere(
        repo, property_type="villa", exclude_city="New Cairo"
    )
    assert replies.route_no_match(
        NoMatch(requested_type="villa", asked_about="Mivida", alternative=alternative)
    ) is Action.answer


# ─────────────────────── the template is not free to vary ───────────────────────


def test_the_template_names_the_same_type_twice():
    """The rule lives in code; the wording lives in config. A template free to
    vary the type between its two halves would express the substitution the
    rule forbids — it would read "no studio here, we have a chalet there" and
    every check in this file would still pass."""
    templates = load(SCRIPT)["replies"]["no_match_same_type"]
    for register, template in templates.items():
        assert template.count("{type}") == 2, (
            f"the {register} template does not name the type twice; one slot means "
            f"the two halves can disagree"
        )
        assert "{asked_about}" in template
        assert "{compound}" in template


def test_the_template_cannot_introduce_a_second_type_slot():
    """`{alternative_type}` is the shape of the bug this rule exists to stop:
    a well-meant edit that makes the reply "more helpful"."""
    templates = load(SCRIPT)["replies"]
    rendered = json.dumps(templates, ensure_ascii=False)
    for forbidden in ("{alternative_type}", "{other_type}", "{nearest_type}"):
        assert forbidden not in rendered


def test_the_type_is_substituted_from_one_value():
    """Structural: the renderer formats `type` once, from the requested type.

    Two `.format` calls, or a second variable named `type`, is how the halves
    come apart later.
    """
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.endswith("type")
    }
    assert assignments <= {"requested_type"}, (
        f"more than one type value is in play in the renderer: {assignments}"
    )


# ─────────────────────── disclosure and routing ───────────────────────


async def test_the_asof_is_disclosed_on_every_inventory_reply(repo):
    """§3.2. A price without a date is a price the tenant cannot stand behind,
    and inventory moves faster than a snapshot."""
    alternative = await find_same_type_elsewhere(repo, property_type="studio")
    reply = render_no_match(
        NoMatch(requested_type="studio", asked_about="North Coast", alternative=alternative),
        register=Register.masri,
        as_of="2026-08-01",
    )
    assert "2026-08-01" in reply


def test_negotiation_intent_routes_to_handoff():
    """re-0013, the highest commercial risk in the suite. The bot has no
    authority to concede anything, and whatever it concedes the broker must
    honour or publicly deny."""
    node = load(SCRIPT)["nodes"]["negotiation"]
    assert node["action"] == Action.handoff


def test_investment_projection_is_refused():
    """re-0015. Every number would be invented, and it attracts regulatory
    attention besides."""
    node = load(SCRIPT)["nodes"]["investment_projection"]
    assert node["action"] == Action.refuse


def test_the_script_loads_and_pins_its_version():
    """§5: a conversation pins the version, so the file is immutable once
    published."""
    script = load(SCRIPT)
    assert script["script_id"] == "realestate_search"
    assert script["version"] == 1
    assert script["entry"] in script["nodes"]


def test_a_refusal_has_its_own_reply_rather_than_the_clarify_text():
    """re-0015 must not be answered with "could you tell me more?" — the
    customer asked a clear question and the answer is that we will not."""
    assert "refuse" in load("agent/replies")["replies"]


def test_the_search_node_requires_a_property_type():
    """Without one there is no rule to enforce: "what have you got?" answered
    with anything is not a substitution, and asking is the correct turn."""
    node = load(SCRIPT)["nodes"]["inventory_lookup"]
    assert "property_type" in node["requires_slots"]


def test_find_same_type_elsewhere_offers_no_way_to_change_the_type():
    """Unrepresentable rather than forbidden, the shape used for the tenant
    and availability filters: there is no parameter through which a different
    type could be requested, and the caller cannot widen the search."""
    parameters = set(inspect.signature(find_same_type_elsewhere).parameters)
    assert "property_type" in parameters
    assert not parameters & {
        "any_type",
        "fallback_type",
        "alternative_type",
        "types",
        "include_other_types",
        "relax",
    }
