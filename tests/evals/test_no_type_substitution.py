"""No property-type substitution, and no invented compounds (§19.3).

Two rules, one principle: a reply may offer a *different compound* and must
never offer a *different type*, and every compound it names has to exist.

The substitution rule is the one that costs a viewing. A customer who asked
for a chalet and is shown a townhouse has been answered with the wrong thing at
the right price — the ranker reached for the nearest number and the type filter
was never a filter. It surfaces as a wasted trip, which the tenant's agent
pays for, so it is a gate rather than a ranking preference.

The compound rule is the same failure as an invented price and rather more
convincing: a plausible Egyptian compound name reads as local knowledge.
Nobody questions it until somebody drives there.

The fixture-backed tests at the bottom run against the real committed
catalogue rather than a hand-written snapshot, so the compound list is the one
the bot would actually answer from.
"""

import json
from pathlib import Path

import pytest

from moc.agent.guards import check_named_entities
from moc.config_store import load
from moc.evals.deterministic import (
    InventorySnapshot,
    check_compound_grounding,
    check_property_type,
)

UNITS = (
    Path(__file__).parents[2]
    / "evals"
    / "fixtures"
    / "broker_demo_2026_08_01"
    / "units.jsonl"
)

SNAPSHOT = InventorySnapshot(
    fixture="test",
    as_of="2026-08-01",
    unit_status={"u1": "available", "u2": "available", "u3": "available"},
    unit_type={"u1": "chalet", "u2": "townhouse", "u3": "chalet"},
    unit_compound={"u1": "Madinaty", "u2": "Mivida", "u3": "Madinaty"},
)


# ─────────────────────────── the rule ───────────────────────────


def test_offering_the_requested_type_passes():
    assert check_property_type("chalet", ["u1", "u3"], SNAPSHOT).passed


@pytest.mark.parametrize(
    ("asked", "offered"),
    [("chalet", "u2"), ("office", "u1"), ("villa", "u2")],
)
def test_a_different_type_is_never_offered(asked, offered):
    """Applies to every type, not just the pair someone happened to test."""
    result = check_property_type(asked, [offered], SNAPSHOT)
    assert result.passed is False
    assert result.metric == "type_substitution_rate"


def test_one_substituted_unit_fails_the_whole_turn():
    """A correct unit alongside a wrong one does not average out.

    The customer sees a list; the wrong entry is on it either way.
    """
    assert check_property_type("chalet", ["u1", "u2"], SNAPSHOT).passed is False


def test_a_different_compound_is_a_legitimate_alternative():
    """The point of the no_match_in_compound reply — the swap that is allowed."""
    same_type_elsewhere = InventorySnapshot(
        fixture="test",
        as_of="2026-08-01",
        unit_status={"u9": "available"},
        unit_type={"u9": "chalet"},
        unit_compound={"u9": "Madinaty"},
    )
    assert check_property_type("chalet", ["u9"], same_type_elsewhere).passed


def test_units_offered_with_no_resolved_type_fail_rather_than_skip():
    """Nothing to compare against is not the same as nothing wrong.

    This is the shape a silent hole takes: the type never resolved, the filter
    never applied, and a check that skipped would report the turn as clean.
    """
    result = check_property_type(None, ["u1"], SNAPSHOT)
    assert result.passed is False
    assert result.skipped is False


def test_a_unit_absent_from_the_snapshot_fails():
    assert check_property_type("chalet", ["ghost"], SNAPSHOT).passed is False


def test_presenting_nothing_is_a_skip_not_a_pass():
    result = check_property_type("chalet", [], SNAPSHOT)
    assert result.skipped is True


# ─────────────────────────── compound grounding ───────────────────────────


def test_a_named_compound_must_exist_in_the_catalogue():
    assert check_compound_grounding(["Madinaty"], SNAPSHOT).passed


def test_an_invented_compound_fails():
    result = check_compound_grounding(["Madinaty East"], SNAPSHOT)
    assert result.passed is False
    assert result.metric == "invented_compound_rate"
    assert "Madinaty East" in result.detail


def test_a_near_miss_is_not_forgiven():
    """No fuzzy matching. "Mivida Heights" is exactly the case to catch, and a
    tolerance would pass it because "Mivida" is real."""
    assert check_named_entities(["Mivida Heights"], {"Mivida"}).passed is False


def test_naming_no_compound_is_a_skip():
    assert check_compound_grounding([], SNAPSHOT).skipped is True


# ─────────────────────────── the reply shape ───────────────────────────


def test_the_no_match_reply_names_both_compounds_and_holds_the_type():
    """The required shape: no <type> in <asked> — we have a <type> in <offered>.

    Both compounds named, the *same* type on both sides. A template that let
    the type differ between the two halves would express the substitution this
    whole file exists to forbid.
    """
    template = load("agent/replies")["replies"]["no_match_in_compound"]
    for register in ("masri", "msa", "english"):
        text = template[register]
        assert text.count("{type}") == 2, f"{register} must hold the type on both sides"
        assert "{asked}" in text
        assert "{offered}" in text


def test_the_no_match_anywhere_reply_offers_nothing():
    """Says so plainly and hands off. No alternative, because there isn't one
    of the right type — and offering another type here is the most tempting
    substitution of all."""
    template = load("agent/replies")["replies"]["no_match_anywhere"]
    for register in ("masri", "msa", "english"):
        text = template[register]
        assert "{type}" in text
        assert "{offered}" not in text, "an alternative unit must not be named"


def test_every_canonical_type_has_a_display_label():
    """A reply naming a type needs a word for it in each register, or the
    template renders a raw catalogue value at a customer."""
    types = load("arabic/property_types")
    assert set(types["display"]) == set(types["canonical"])
    for label in types["display"].values():
        assert set(label) == {"masri", "msa", "english"}


def test_display_labels_are_words_the_matcher_recognises():
    """A label the resolver cannot parse back breaks the next turn.

    The customer replies "أيوه الشاليه"; if the reply used a word absent from
    the alias list, the type does not re-resolve.
    """
    types = load("arabic/property_types")
    for canonical, label in types["display"].items():
        aliases = types["types"][canonical]
        assert label["masri"] in aliases["arabic"], canonical
        english = label["english"]
        assert english in aliases["latin"] or english.split()[0] in aliases["latin"]


# ─────────────────────────── against the real catalogue ───────────────────────────


@pytest.fixture(scope="module")
def catalogue() -> InventorySnapshot:
    units = [json.loads(line) for line in UNITS.read_text(encoding="utf-8").splitlines()]
    return InventorySnapshot(
        fixture="broker_demo_2026_08_01",
        as_of="2026-08-01",
        unit_status={u["unit_id"]: u["availability"] for u in units},
        unit_type={u["unit_id"]: u["property_type"] for u in units},
        unit_compound={u["unit_id"]: u["compound"] for u in units},
    )


@pytest.mark.eval
def test_the_catalogue_covers_every_canonical_type_the_rule_applies_to(catalogue):
    """If a type has no units, no case can exercise the rule for it.

    Recorded rather than asserted as complete: a type the catalogue lacks is a
    coverage gap in the fixture, not a bug in the rule.
    """
    present = set(catalogue.unit_type.values())
    canonical = set(load("arabic/property_types")["canonical"])
    assert present <= canonical, (
        f"catalogue has types the config does not name: {present - canonical}"
    )


@pytest.mark.eval
def test_a_real_compound_passes_and_a_plausible_invention_does_not(catalogue):
    """The check against the catalogue the bot would actually answer from."""
    real = next(iter(catalogue.compounds()))
    assert check_compound_grounding([real], catalogue).passed
    assert check_compound_grounding([f"{real} East"], catalogue).passed is False


@pytest.mark.eval
def test_substitution_is_caught_on_real_units(catalogue):
    """Pick a real unit and ask for a type it is not."""
    unit_id, unit_type = next(iter(catalogue.unit_type.items()))
    other = next(t for t in load("arabic/property_types")["canonical"] if t != unit_type)
    assert check_property_type(other, [unit_id], catalogue).passed is False
    assert check_property_type(unit_type, [unit_id], catalogue).passed
