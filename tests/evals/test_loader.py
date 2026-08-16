from pathlib import Path

import pytest

from moc.evals.load import load_cases
from moc.evals.schema import (
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

MINIMAL = (
    "- {{id: {id}, vertical: education, source: synthetic, category: ambiguous,\n"
    "   tenant_fixture: f, channel: whatsapp, input_lang: masri, turns: []}}\n"
)


def test_loads_education_worked_examples():
    cases = load_cases(EDUCATION)
    assert len(cases) == 11
    assert cases[0].id == "edu-0001"


def test_loads_realestate_worked_examples():
    cases = load_cases(REALESTATE)
    assert len(cases) == 11
    assert cases[0].id == "re-0001"


def test_parses_education_case_fully():
    case = load_cases(EDUCATION)[0]
    assert case.vertical is Vertical.education
    assert case.source is Source.real_conversation
    assert case.category is Category.factual_retrieval
    assert case.tenant_fixture == "sinai_demo"
    assert case.input_lang is InputLang.masri
    assert case.gold_chunks == ["chunk_eng_fees_2026"]

    turn = case.turns[0]
    assert turn.expected_action is Action.answer
    assert turn.expected_register is Register.msa
    assert turn.expected_lang == "ar"
    assert [f.id for f in turn.expected_facts] == ["f1", "f2"]
    assert turn.expected_facts[0].required is True
    assert turn.expected_facts[0].source_chunk == "chunk_eng_fees_2026"
    assert turn.expected_slots == {"faculty": "engineering"}


def test_documents_is_the_default_grounding_mode():
    """Education grounds against chunks and never states grounding_mode."""
    assert load_cases(EDUCATION)[0].grounding_mode is GroundingMode.documents


def test_parses_structured_inventory_fields():
    """Spec §3.2: the three real-estate-only failure modes."""
    case = next(c for c in load_cases(REALESTATE) if c.id == "re-0002")
    assert case.grounding_mode is GroundingMode.inventory
    assert case.inventory_fixture == "broker_demo_2026_08_01"

    turn = case.turns[0]
    assert turn.expected_asof_disclosure is True
    assert [tc.name for tc in turn.expected_tool_calls] == ["payment_plan_calculator"]
    assert turn.expected_tool_calls[0].args_contain == {
        "unit_id": "u_1042",
        "down_payment_pct": 10,
        "years": 8,
    }
    assert turn.expected_computation is not None
    assert turn.expected_computation.tool == "payment_plan_calculator"
    assert turn.expected_computation.inputs["years"] == 8
    assert turn.expected_computation.must_match_fixture is True


def test_asof_disclosure_defaults_to_false():
    """Only cases that opt in are graded on it; absence must not read as required."""
    assert load_cases(EDUCATION)[0].turns[0].expected_asof_disclosure is False


def test_multi_turn_case_keeps_turn_order():
    case = next(c for c in load_cases(EDUCATION) if c.id == "edu-0004")
    assert [t.expected_action for t in case.turns] == [
        Action.clarify,
        Action.answer,
        Action.answer,
    ]
    assert case.turns[2].expected_slots == {"faculty": "pharmacy"}


def test_slot_value_may_be_a_list():
    """re-0009 holds two areas at once; a scalar-only type would silently coerce."""
    case = next(c for c in load_cases(REALESTATE) if c.id == "re-0009")
    assert case.turns[2].expected_slots["area"] == ["el_shorouk", "mostakbal_city"]


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
