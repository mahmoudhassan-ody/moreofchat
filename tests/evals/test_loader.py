from pathlib import Path

import pytest

from moc.evals.load import load_cases
from moc.evals.schema import (
    CATEGORIES_BY_VERTICAL,
    Action,
    Category,
    GroundingMode,
    InputLang,
    Register,
    Source,
    Vertical,
)

REPO_ROOT = Path(__file__).parents[2]
EDUCATION = REPO_ROOT / "evals" / "cases" / "education.yaml"
REALESTATE = REPO_ROOT / "evals" / "cases" / "realestate.yaml"
FIXTURE = (
    REPO_ROOT / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)

MINIMAL = (
    "- {{id: {id}, vertical: education, source: synthetic, category: ambiguous,\n"
    "   tenant_fixture: f, channel: whatsapp, input_lang: masri,\n"
    '   turns: [{{user: "hi", expected_action: clarify}}]}}\n'
)

NO_TURNS = (
    "- {id: x-5, vertical: education, source: synthetic, category: ambiguous,\n"
    "  tenant_fixture: f, channel: whatsapp, input_lang: masri, turns: []}\n"
)


def by_id(path: Path, case_id: str):
    """Fetch a case by id.

    Never by index. Both files are append-only and unsorted — `education.yaml`
    currently runs edu-0001, edu-0012, edu-0013, edu-0002 — so a positional
    lookup asserts against whichever case happens to sit there today. That is
    how the previous version of this file ended up raising IndexError when the
    cases were rewritten: it was reaching for a position, not for a case.
    """
    cases = load_cases(path)
    return next(c for c in cases if c.id == case_id)


# Floors, not exact counts. The suite grows toward §4's 150/80 targets and an
# equality assert would fail on every legitimate append, which trains people to
# edit the number without reading it. A floor still catches the failure worth
# catching — cases silently disappearing — and the named ids below catch a
# specific case being dropped while others are added.
EDUCATION_CASES = 17
# Lowered from 21 on 2026-08-20: re-0002/re-0003 merged into re-0001 and
# re-0006 into re-0005, as turns of the conversations they were always part of.
# Three cases fewer, no coverage lost — each was asserting a slot stated by a
# turn it had been separated from.
REALESTATE_CASES = 20


@pytest.mark.eval
def test_loads_education_worked_examples():
    cases = load_cases(EDUCATION)
    assert len(cases) >= EDUCATION_CASES
    assert {c.vertical for c in cases} == {Vertical.education}
    assert all(c.id.startswith("edu-") for c in cases)


@pytest.mark.eval
def test_loads_realestate_worked_examples():
    cases = load_cases(REALESTATE)
    assert len(cases) >= REALESTATE_CASES
    assert {c.vertical for c in cases} == {Vertical.realestate}
    assert all(c.id.startswith("re-") for c in cases)


@pytest.mark.eval
@pytest.mark.parametrize(
    ("path", "case_id"),
    [
        (EDUCATION, "edu-0001"),
        (EDUCATION, "edu-0006"),
        (EDUCATION, "edu-0007"),
        (REALESTATE, "re-0005"),
        (REALESTATE, "re-0018"),
    ],
)
def test_the_cases_these_tests_pin_are_still_present(path, case_id):
    """A named case being deleted must fail here, not further down as a
    StopIteration inside another test's setup."""
    assert by_id(path, case_id).id == case_id


@pytest.mark.eval
def test_parses_education_case_fully():
    """edu-0006: the richest education shape — facts, a source chunk, a slot.

    Chosen for what it contains rather than where it sits. Franco/misspelled
    input ("القنطره" without the hamza) still has to resolve to the Qantara
    branch, so the slot assertion is load-bearing rather than incidental.
    """
    case = by_id(EDUCATION, "edu-0006")
    assert case.vertical is Vertical.education
    assert case.source is Source.synthetic
    assert case.category is Category.franco_or_misspelled
    assert case.tenant_fixture == "sinai_demo"
    assert case.input_lang is InputLang.masri
    assert case.gold_chunks == ["sinai_housing_availability_ar"]

    turn = case.turns[0]
    assert turn.expected_action is Action.answer
    assert turn.expected_register is Register.masri
    assert turn.expected_lang == "ar"
    assert [f.id for f in turn.expected_facts] == ["f1", "f2"]
    assert turn.expected_facts[0].required is True
    assert turn.expected_facts[1].required is False, (
        "required and optional facts must both survive the parse, or partial "
        "credit stops meaning anything (§3.1)"
    )
    assert turn.expected_facts[0].source_chunk == "sinai_housing_availability_ar"
    assert turn.expected_slots == {"branch": "qantara"}


@pytest.mark.eval
def test_documents_is_the_default_grounding_mode():
    """Education grounds against chunks and never states grounding_mode."""
    assert by_id(EDUCATION, "edu-0006").grounding_mode is GroundingMode.documents


@pytest.mark.eval
def test_parses_structured_inventory_fields():
    """Spec §3.2: the three real-estate-only failure modes.

    re-0005 is the payment-plan case — the one where the model must not do
    arithmetic. It carries all three fields at once, which is why it is the
    case worth pinning.
    """
    case = by_id(REALESTATE, "re-0005")
    assert case.grounding_mode is GroundingMode.inventory
    assert case.inventory_fixture == "broker_demo_2026_08_01"
    assert case.category is Category.payment_plan_math

    turn = case.turns[0]
    assert turn.expected_asof_disclosure is True
    assert [tc.name for tc in turn.expected_tool_calls] == ["payment_plan_calculator"]
    assert turn.expected_tool_calls[0].args_contain == {"unit_id": "NOOR-CIT-002-02"}
    assert turn.expected_computation is not None
    assert turn.expected_computation.tool == "payment_plan_calculator"
    assert turn.expected_computation.inputs["unit_id"] == "NOOR-CIT-002-02"
    assert turn.expected_computation.must_match_fixture is True


def test_asof_disclosure_defaults_to_false():
    """Only cases that opt in are graded on it; absence must not read as required."""
    assert by_id(EDUCATION, "edu-0006").turns[0].expected_asof_disclosure is False


@pytest.mark.eval
def test_multi_turn_case_keeps_turn_order():
    """edu-0007: two clarifications then an answer, slots accumulating.

    F5 lives here — a slot captured in turn 2 must still be held in turn 3.
    Asserting the order matters because a loader that sorted or deduplicated
    turns would still parse cleanly and quietly destroy the assertion.
    """
    case = by_id(EDUCATION, "edu-0007")
    assert [t.expected_action for t in case.turns] == [
        Action.clarify,
        Action.clarify,
        Action.answer,
    ]
    assert case.turns[1].expected_slots == {"branch": "arish"}
    assert case.turns[2].expected_slots == {
        "branch": "arish",
        "certificate": "general_secondary",
        "faculty": "dentistry",
    }


@pytest.mark.eval
def test_slot_value_may_be_a_list():
    """re-0018 holds two cities at once; a scalar-only type would silently coerce."""
    case = by_id(REALESTATE, "re-0018")
    # Catalogue spelling since 2026-08-19 — see
    # `test_location_slots_use_the_catalogue_spelling` for why the files
    # settled on one form. The shape under test is the list, not the casing.
    assert case.turns[2].expected_slots["city"] == ["Sheikh Zayed", "October"]


def test_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(MINIMAL.format(id="x-1") + MINIMAL.format(id="x-1"))
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(p)


def test_rejects_golden_answer_strings(tmp_path):
    """Design rule §3.1: cases grade facts, never wording."""
    p = tmp_path / "golden.yaml"
    p.write_text(
        "- {id: x-2, vertical: education, source: synthetic, category: ambiguous,\n"
        "   tenant_fixture: f, channel: whatsapp, input_lang: masri,\n"
        '   turns: [{user: "hi", expected_action: answer, expected_reply: "مرحبا"}]}\n'
    )
    with pytest.raises(ValueError, match="expected_reply"):
        load_cases(p)


def test_rejects_case_with_no_turns(tmp_path):
    """A turn-less case is unrunnable but would load clean.

    It then counts toward the 150/80 targets in spec §4, so the suite reads as
    further along than it is — the one error a progress metric must not make.
    """
    p = tmp_path / "noturns.yaml"
    p.write_text(NO_TURNS)
    with pytest.raises(ValueError, match="turns"):
        load_cases(p)


def test_rejects_case_with_turns_omitted(tmp_path):
    p = tmp_path / "missing.yaml"
    p.write_text(NO_TURNS.replace(", turns: []", ""))
    with pytest.raises(ValueError, match="turns"):
        load_cases(p)


def test_rejects_unknown_field(tmp_path):
    """extra='forbid': a typo must be a load error, not a skipped assertion."""
    p = tmp_path / "typo.yaml"
    p.write_text(MINIMAL.format(id="x-3").rstrip()[:-1] + ", expected_slot: {a: b}}\n")
    with pytest.raises(ValueError, match="expected_slot"):
        load_cases(p)


def test_rejects_unknown_enum_value(tmp_path):
    p = tmp_path / "badenum.yaml"
    p.write_text(MINIMAL.format(id="x-4").replace("category: ambiguous", "category: vibes"))
    with pytest.raises(ValueError, match="category"):
        load_cases(p)


def test_error_names_the_offending_case(tmp_path):
    """A 150-case file needs the id in the message, not just a field path."""
    p = tmp_path / "many.yaml"
    p.write_text(MINIMAL.format(id="x-good") + MINIMAL.format(id="x-bad").replace(
        "channel: whatsapp", "channel: carrier_pigeon"
    ))
    with pytest.raises(ValueError, match="x-bad"):
        load_cases(p)


def test_rejects_realestate_category_on_an_education_case(tmp_path):
    """Categories are per-vertical (§4.1, §4.2). Education has no payment plans."""
    p = tmp_path / "wrongcat.yaml"
    p.write_text(
        MINIMAL.format(id="x-6").replace("category: ambiguous", "category: payment_plan_math")
    )
    with pytest.raises(ValueError, match="payment_plan_math"):
        load_cases(p)


def test_rejects_education_category_on_a_realestate_case(tmp_path):
    p = tmp_path / "wrongcat2.yaml"
    p.write_text(MINIMAL.format(id="x-7").replace("vertical: education", "vertical: realestate"))
    with pytest.raises(ValueError, match="ambiguous"):
        load_cases(p)


def test_error_names_the_vertical_and_the_allowed_categories(tmp_path):
    p = tmp_path / "wrongcat3.yaml"
    p.write_text(MINIMAL.format(id="x-8").replace("category: ambiguous", "category: staleness"))
    with pytest.raises(ValueError) as exc:
        load_cases(p)
    assert "education" in str(exc.value)
    assert "factual_retrieval" in str(exc.value), "message must list what IS allowed"


@pytest.mark.parametrize("vertical", ["education", "realestate"])
@pytest.mark.parametrize(
    "category",
    ["adversarial_figures", "franco_or_misspelled", "multi_turn_slots", "out_of_scope"],
)
def test_shared_categories_are_allowed_in_both_verticals(tmp_path, vertical, category):
    p = tmp_path / f"{vertical}_{category}.yaml"
    p.write_text(
        MINIMAL.format(id="x-9")
        .replace("vertical: education", f"vertical: {vertical}")
        .replace("category: ambiguous", f"category: {category}")
    )
    assert len(load_cases(p)) == 1


def test_every_category_belongs_to_some_vertical():
    """A new Category member must be assigned, not left silently unreachable."""
    assigned = set().union(*CATEGORIES_BY_VERTICAL.values())
    assert assigned == set(Category)
    assert set(CATEGORIES_BY_VERTICAL) == set(Vertical)


@pytest.mark.eval
def test_shipped_case_files_respect_the_mapping():
    for case in load_cases(EDUCATION) + load_cases(REALESTATE):
        assert case.category in CATEGORIES_BY_VERTICAL[case.vertical], case.id


@pytest.mark.eval
def test_ids_are_unique_across_the_whole_suite():
    """Ids are append-only and never reused — including between verticals."""
    ids = [c.id for c in load_cases(EDUCATION) + load_cases(REALESTATE)]
    assert len(ids) == len(set(ids))


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Real estate has no conversation corpus yet, so all 11 worked examples are "
        "synthetic — 12/22 overall. Spec §4.2: launch the real-estate suite at 80 "
        "cases mined from pilot broker traffic, which is what brings this under 30%. "
        "Encodes the intended end state, and starts passing as real cases land."
    ),
)
def test_synthetic_share_under_30_percent():
    cases = load_cases(EDUCATION) + load_cases(REALESTATE)
    synthetic = sum(1 for c in cases if c.source is Source.synthetic)
    assert synthetic / len(cases) <= 0.30


def test_location_slots_use_the_catalogue_spelling():
    """One form across every case, and it is the one everything else uses.

    The files carried both `new_cairo` and `"New Cairo"` for the same slot,
    while `args_contain` always used the second. No extractor can satisfy
    both, so `slot_retention_accuracy` had a ceiling below 100% that no
    prompt work could lift — and the cause looked like a model failure.

    The catalogue's own spelling is the form chosen, because it is what
    `inventory_units.city` and `.compound` hold, what the extractor's
    vocabulary is now read from, and what the tool-call assertions already
    pinned. The other two would each have needed a translation step
    somewhere, and a translation step is where the drift comes back.

    Checked against the fixture rather than against config: config no longer
    declares the values, which is the point — see
    tests/arabic/test_location_coverage.py.
    """
    import json

    from moc.agent.extraction import slot_vocabulary

    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vocabulary = slot_vocabulary(
        "scripts/realestate/search",
        catalogue={
            "city": {r["city"] for r in rows if r.get("city")},
            "compound": {r["compound"] for r in rows if r.get("compound")},
        },
    )
    cases = load_cases(REALESTATE)
    seen = 0
    for case in cases:
        for turn in case.turns:
            for slot in ("city", "compound"):
                value = (turn.expected_slots or {}).get(slot)
                for one in value if isinstance(value, list) else [value] if value else []:
                    seen += 1
                    assert one in vocabulary[slot], (
                        f"{case.id} pins {slot}={one!r}, which the extractor cannot "
                        f"emit and the connector cannot filter on"
                    )
    assert seen >= 8, "no location slots checked — this assertion would pass vacuously"
