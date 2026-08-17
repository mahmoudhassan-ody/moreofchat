"""Stage 1 checks beyond numeric grounding — harness spec §5.1 and §3.2."""

import pytest

from moc.evals.deterministic import (
    InventorySnapshot,
    ToolCall,
    check_action,
    check_asof_disclosure,
    check_availability,
    check_language,
    check_slots,
    check_tool_calls,
)
from moc.evals.schema import Action, ExpectedToolCall

# ── The P1 integration seam ──────────────────────────────────────────────
# Real snapshots load from evals/fixtures/broker_demo_2026_08_01/ once the
# fixture loader lands in P1 (spec §8.1). Until then the checks take an
# InventorySnapshot the caller builds, and this is the only place one is built.
# Nothing in src/ fabricates a snapshot, so a missing fixture is an obvious
# TypeError at the call site rather than a check that silently passes.
BROKER_DEMO = InventorySnapshot(
    fixture="broker_demo_2026_08_01",
    as_of="2026-08-01",
    unit_status={"u_1042": "available", "u_2001": "sold", "u_3005": "reserved"},
)


# ─────────────────────────── action ───────────────────────────


def test_action_match_passes():
    assert check_action(Action.handoff, Action.handoff).passed is True


def test_action_mismatch_names_both_sides():
    result = check_action(Action.handoff, Action.answer)
    assert result.passed is False
    assert "handoff" in result.detail and "answer" in result.detail


def test_action_check_feeds_the_action_metric():
    assert check_action(Action.answer, Action.answer).metric == "expected_action_accuracy"


def test_answering_when_handoff_was_expected_is_the_f2_failure():
    """Confident answer where handoff was correct — the case that destroys trust."""
    assert check_action(Action.handoff, Action.answer).passed is False


# ─────────────────────────── language ───────────────────────────


def test_language_match_passes():
    assert check_language("رسوم الساعة 1400 جنيه", "ar").passed is True


def test_language_mismatch_fails():
    assert check_language("The fee is 1400 EGP", "ar").passed is False


def test_language_check_skips_when_the_case_pins_nothing():
    """expected_lang is optional in the schema; absence must not fail a case."""
    result = check_language("أي كلام", None)
    assert result.passed is True
    assert result.skipped is True


def test_undetectable_reply_does_not_silently_pass():
    result = check_language("1400 2026 !!!", "ar")
    assert result.passed is False


# ─────────────────────────── slots ───────────────────────────


def test_slots_exact_match_passes():
    assert check_slots({"faculty": "pharmacy"}, {"faculty": "pharmacy"}).passed is True


def test_missing_slot_fails():
    assert check_slots({"faculty": "pharmacy"}, {}).passed is False


def test_extra_slot_fails():
    """expected_slots is the whole expected state, so an unexpected slot is a miss."""
    result = check_slots({"faculty": "pharmacy"}, {"faculty": "pharmacy", "area": "x"})
    assert result.passed is False


def test_empty_expected_slots_requires_an_empty_state():
    """edu-0004 turn 1 asserts nothing has been captured yet — that must be checkable."""
    assert check_slots({}, {}).passed is True
    assert check_slots({}, {"faculty": "pharmacy"}).passed is False


def test_list_valued_slot_matches_regardless_of_order():
    """re-0009 holds two areas at once; ordering is not part of the assertion."""
    result = check_slots(
        {"budget_max": 4000000, "area": ["el_shorouk", "mostakbal_city"]},
        {"budget_max": 4000000, "area": ["mostakbal_city", "el_shorouk"]},
    )
    assert result.passed is True


def test_list_valued_slot_fails_when_one_area_was_dropped():
    """The actual re-0009 assertion: the second area must not overwrite the first.

    A scalar comparison would compare "el_shorouk" against a one-item list and
    could pass, hiding exactly the bug the case was written to catch.
    """
    result = check_slots(
        {"area": ["el_shorouk", "mostakbal_city"]},
        {"area": ["mostakbal_city"]},
    )
    assert result.passed is False
    assert "area" in result.detail


def test_scalar_and_single_item_list_are_the_same_slot_state():
    assert check_slots({"area": "el_shorouk"}, {"area": ["el_shorouk"]}).passed is True


def test_slot_check_feeds_the_retention_metric():
    assert check_slots({}, {}).metric == "slot_retention_accuracy"


# ─────────────────────────── tool calls (§3.2) ───────────────────────────


def test_tool_call_matches_on_a_subset_of_arguments():
    """args_contain is a subset check: extra orchestrator arguments are fine."""
    expected = [ExpectedToolCall(name="inventory_lookup", args_contain={"area": "new cairo"})]
    actual = [ToolCall(name="inventory_lookup", args={"area": "new cairo", "limit": 10})]
    assert check_tool_calls(expected, actual).passed is True


def test_tool_call_fails_on_a_wrong_argument_value():
    expected = [ExpectedToolCall(name="inventory_lookup", args_contain={"area": "new cairo"})]
    actual = [ToolCall(name="inventory_lookup", args={"area": "sheikh zayed"})]
    assert check_tool_calls(expected, actual).passed is False


def test_tool_call_fails_when_the_tool_was_never_called():
    expected = [ExpectedToolCall(name="payment_plan_calculator", args_contain={"years": 8})]
    result = check_tool_calls(expected, [])
    assert result.passed is False
    assert "payment_plan_calculator" in result.detail


def test_tool_call_check_skips_when_the_case_expects_none():
    result = check_tool_calls([], [ToolCall(name="inventory_lookup", args={})])
    assert result.passed is True
    assert result.skipped is True


def test_missing_expected_argument_key_fails():
    expected = [ExpectedToolCall(name="payment_plan_calculator", args_contain={"years": 8})]
    actual = [ToolCall(name="payment_plan_calculator", args={"unit_id": "u_1042"})]
    assert check_tool_calls(expected, actual).passed is False


# ─────────────────────────── availability (§5.1) ───────────────────────────


def test_offering_an_available_unit_passes():
    assert check_availability(["u_1042"], BROKER_DEMO).passed is True


@pytest.mark.parametrize("unit", ["u_2001", "u_3005"])
def test_offering_a_sold_or_reserved_unit_is_a_hard_fail(unit):
    result = check_availability([unit], BROKER_DEMO)
    assert result.passed is False
    assert unit in result.detail
    assert result.metric == "sold_unit_offered_rate"


def test_unknown_unit_id_fails_rather_than_passing():
    """A unit absent from the frozen snapshot cannot be asserted available."""
    result = check_availability(["u_9999"], BROKER_DEMO)
    assert result.passed is False


def test_availability_check_skips_when_no_unit_was_presented():
    result = check_availability([], BROKER_DEMO)
    assert result.passed is True
    assert result.skipped is True


# ─────────────────────────── as_of disclosure (§3.2) ───────────────────────────


def test_iso_date_in_the_reply_discloses_asof():
    assert check_asof_disclosure("الأسعار بتاريخ 2026-08-01", BROKER_DEMO, True).passed is True


def test_arabic_indic_date_discloses_asof():
    """The reply will render the date in Arabic-Indic digits more often than not."""
    assert check_asof_disclosure("الأسعار حتى ٢٠٢٦-٠٨-٠١", BROKER_DEMO, True).passed is True


def test_temporal_qualifier_without_a_literal_date_discloses_asof():
    """§5.1 accepts an equivalent temporal qualifier, not only the date."""
    assert check_asof_disclosure("الأسعار حتى تاريخه قابلة للتغيير", BROKER_DEMO, True).passed


def test_price_without_any_temporal_qualifier_fails():
    result = check_asof_disclosure("سعر الوحدة 6000000 جنيه", BROKER_DEMO, True)
    assert result.passed is False
    assert result.metric == "asof_disclosure_rate"


def test_asof_check_skips_when_the_case_does_not_require_it():
    result = check_asof_disclosure("المصاريف 1400 جنيه", BROKER_DEMO, False)
    assert result.passed is True
    assert result.skipped is True
