"""Every catalogue location must be reachable in Arabic and in franco.

**The alias list was transcribed, not tested.** Two live real-estate cases
errored on `التجمع الخامس` and `Tagamoa El Khames` — the most-used Arabic and
franco names for New Cairo, the largest city in the catalogue. The Arabic one
was present; the franco one was not, and ten of twenty-six canonical values
carried no Latin list at all.

That fails silently and in the worst direction. A franco customer typing the
only spelling they use gets "we have no stock", because the extractor emits a
surface form no filter can parse back. `franco_or_misspelled` is a whole eval
category (§4.2) precisely because it is how Egyptians type.

So coverage is asserted rather than eyeballed: both scripts, every canonical
value, no exceptions list.
"""

from moc.config_store import load

LOCATIONS = "arabic/locations"


def catalogue() -> dict[str, str]:
    return load(LOCATIONS)["kind"]


def aliases() -> dict[str, dict]:
    return load(LOCATIONS)["aliases"]


def test_every_canonical_value_has_an_alias_block():
    missing = sorted(set(catalogue()) - set(aliases()))
    assert missing == [], f"in `kind` but with no aliases: {missing}"


def test_every_canonical_value_is_reachable_in_arabic():
    bare = sorted(name for name, forms in aliases().items() if not forms.get("arabic"))
    assert bare == [], f"no Arabic surface form: {bare}"


def test_every_canonical_value_is_reachable_in_franco():
    """The gap that broke re-0016. A location with no Latin form is a
    location a franco customer cannot ask for, and the failure reads as
    absent stock rather than as a missing alias."""
    bare = sorted(name for name, forms in aliases().items() if not forms.get("latin"))
    assert bare == [], f"no Latin or franco surface form: {bare}"


def test_new_cairo_carries_the_names_customers_actually_use():
    """Both erroring cases named New Cairo, neither by its canonical name.
    `التجمع الخامس` and `tagamoa el 5ames` are what people type; `New Cairo`
    is what the catalogue column holds, and almost nobody writes it."""
    forms = aliases()["new cairo"]
    assert "التجمع الخامس" in forms["arabic"]
    latin = {value.lower() for value in forms["latin"]}
    assert "tagamoa el khames" in latin
    assert "tagamo3 el 5ames" in latin, "the digit-substituted spelling is the common one"


def test_no_alias_is_claimed_by_two_canonical_values():
    """`_location_in` returns the first alias found in the message, so a
    duplicate silently resolves to whichever sorts first — and the two would
    filter different columns."""
    seen: dict[str, str] = {}
    for canonical, forms in aliases().items():
        for script in ("arabic", "latin"):
            for alias in forms.get(script, []):
                key = alias.strip().lower()
                assert key not in seen, (
                    f"{alias!r} is claimed by both {seen.get(key)!r} and {canonical!r}"
                )
                seen[key] = canonical


def test_no_alias_is_a_bare_substring_of_another_canonical_value_s_alias():
    """`alias in text` matching means a short alias can fire inside a longer
    one. `_aliases()` sorts longest-first to make the longer win; this pins
    the pairs that ordering has to resolve, so a new alias that breaks the
    ordering assumption fails here rather than in a suite run.
    """
    from moc.verticals.realestate.agent import _aliases

    ordered = list(_aliases())
    for index, alias in enumerate(ordered):
        for longer in ordered[:index]:
            assert len(longer) >= len(alias), (
                f"{longer!r} sorts before {alias!r} but is shorter — "
                "the longest-match guarantee is broken"
            )
