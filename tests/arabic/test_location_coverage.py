"""`locations.yaml` is an alias overlay, not a second catalogue.

**It used to be both, and it declared ten of the catalogue's ninety-four
compounds.** The gap did not present as a missing alias. Asked about
`كريك تاون` — Creek Town, three units in the fixture — the model picked the
nearest value it was allowed to emit and answered with a Jefaira townhouse.
A real compound, the wrong one, and `invented_compound_rate` stayed at 0.0%
because Jefaira exists.

So the vocabulary now comes from the catalogue (`InventoryRepository.
vocabulary`), and this file holds only what the catalogue cannot supply: the
Arabic and franco names customers type. Which makes the invariant here
narrow and checkable — every key names a real catalogue value, spelled the
way the column spells it.
"""

import json
from pathlib import Path

from moc.config_store import load

FIXTURE = (
    Path(__file__).parents[2]
    / "evals"
    / "fixtures"
    / "broker_demo_2026_08_01"
    / "units.jsonl"
)
LOCATIONS = "arabic/locations"


def catalogue() -> dict[str, set[str]]:
    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "city": {r["city"] for r in rows if r.get("city")},
        "compound": {r["compound"] for r in rows if r.get("compound")},
    }


def aliases() -> dict[str, dict]:
    return load(LOCATIONS)["aliases"]


def test_every_alias_key_is_a_value_the_catalogue_actually_holds():
    """The drift alarm. A key that is not a catalogue value resolves to a
    filter that matches nothing, and the reply reads as absent stock."""
    known = catalogue()["city"] | catalogue()["compound"]
    unknown = sorted(set(aliases()) - known)
    assert unknown == [], f"aliases for values not in the catalogue: {unknown}"


def test_the_file_declares_no_vocabulary_of_its_own():
    """The `kind:` map is gone. Which column a value sits in is its kind, and
    a second copy of that is a second thing that can disagree with the rows."""
    assert "kind" not in load(LOCATIONS)


def test_every_city_is_reachable_in_arabic():
    """Thirteen cities, all of them named in Arabic constantly. Compounds are
    Latin in the catalogue and mostly typed that way; cities are not."""
    covered = {name for name, forms in aliases().items() if forms.get("arabic")}
    missing = sorted(catalogue()["city"] - covered)
    assert missing == [], f"no Arabic surface form: {missing}"


def test_every_alias_block_carries_at_least_one_surface_form():
    empty = sorted(
        name
        for name, forms in aliases().items()
        if not forms.get("arabic") and not forms.get("latin")
    )
    assert empty == [], f"alias blocks with nothing in them: {empty}"


def test_new_cairo_carries_the_names_customers_actually_use():
    """Both cases that errored named New Cairo, neither by its catalogue
    name. `التجمع الخامس` and `tagamo3 el 5ames` are what people type."""
    forms = aliases()["New Cairo"]
    assert "التجمع الخامس" in forms["arabic"]
    latin = {value.lower() for value in forms["latin"]}
    assert "tagamoa el khames" in latin
    assert "tagamo3 el 5ames" in latin, "the digit-substituted spelling is the common one"


def test_creek_town_is_reachable_in_arabic():
    """The compound the model substituted `Jefaira` for."""
    assert "كريك تاون" in aliases()["Creek Town"]["arabic"]


def test_no_alias_is_claimed_by_two_catalogue_values():
    """A duplicate resolves to whichever the matcher reaches first, and the
    two may filter different columns."""
    seen: dict[str, str] = {}
    for canonical, forms in aliases().items():
        for script in ("arabic", "latin"):
            for alias in forms.get(script, []):
                key = alias.strip().lower()
                assert key not in seen, (
                    f"{alias!r} is claimed by both {seen.get(key)!r} and {canonical!r}"
                )
                seen[key] = canonical


def test_longest_alias_wins_when_one_contains_another():
    """`alias in text` matching means a short alias fires inside a longer one.
    `_aliases()` sorts longest-first so the longer wins; this pins that
    ordering, so an alias that breaks the assumption fails here rather than
    in a suite run."""
    from moc.verticals.realestate.agent import _aliases

    ordered = list(_aliases())
    for index, alias in enumerate(ordered):
        for longer in ordered[:index]:
            assert len(longer) >= len(alias), (
                f"{longer!r} sorts before {alias!r} but is shorter"
            )
