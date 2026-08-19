"""The payment plan calculator — design §3.2, P1b Task 24.

A deterministic tool. The model never divides.

**The rounding is the entire demonstration.** NOOR-CIT-002-02 is 6,450,000 at
25% down over 16 quarters: 4,837,500 remaining, which is 302,343.75 per
quarter. A model asked to divide produces 302,344 — a natural, defensible,
*wrong* number. The fixture says 302,343 fifteen times and 302,355 once,
because the remainder is carried to the end rather than spread. One pound
either way looks like a formatting preference and is the whole proof that the
figure came from a tool rather than from a language model's arithmetic.

`arithmetic_in_model_rate` string-matches reply figures against this output,
so "close" is a failure here in the same way it is at the gate. Every one of
the 189 planned units must reconcile exactly, not on average.
"""

import json
from pathlib import Path

import pytest

from moc.verticals.realestate.calculator import (
    PaymentPlanUnavailable,
    UnofferedTerms,
    payment_schedule,
)

FIXTURE = (
    Path(__file__).parents[2] / "evals" / "fixtures" / "broker_demo_2026_08_01" / "units.jsonl"
)


def units() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def unit(unit_id: str) -> dict:
    return next(row for row in units() if row["unit_id"] == unit_id)


PLANNED = [row for row in units() if row["payment_plan"]]


# ─────────────────────────── the fixture, exactly ───────────────────────────


def test_reproduces_the_fixture_schedule_exactly():
    """NOOR-CIT-002-02, the case `arithmetic_in_model_rate` is built on.

    6,450,000 at 25% over 16 quarters -> down 1,612,500, regular 302,343 x15,
    final 302,355. Every figure matches the fixture byte for byte, because the
    eval gate string-matches reply figures against this output — and because
    302,344 is exactly what a model that divided would say.
    """
    schedule = payment_schedule(unit("NOOR-CIT-002-02"))

    assert schedule.unit_id == "NOOR-CIT-002-02"
    assert schedule.down_payment == 1_612_500
    assert schedule.installment_amount == 302_343
    assert schedule.installment_count == 16
    assert schedule.final_installment_amount == 302_355
    assert schedule.frequency == "quarterly"
    assert schedule.years == 4
    assert schedule.total == 6_450_000


def test_the_regular_instalment_is_floored_not_rounded():
    """4,837,500 / 16 is 302,343.75. Rounding gives 302,344 and the schedule
    then overshoots the price by twelve pounds; flooring gives 302,343 and the
    remainder lands on the final instalment where the developer put it."""
    schedule = payment_schedule(unit("NOOR-CIT-002-02"))
    assert schedule.installment_amount == 302_343
    assert schedule.installment_amount != round(4_837_500 / 16)


@pytest.mark.parametrize("row", PLANNED, ids=lambda row: row["unit_id"])
def test_every_planned_unit_reproduces_its_fixture_plan(row):
    """All 189, one assertion each.

    Parametrized rather than looped so a single disagreeing unit names itself.
    A loop reports "one of 189 differs" and leaves the reader to find which.
    """
    plan = row["payment_plan"]
    schedule = payment_schedule(row)

    assert schedule.down_payment == plan["down_payment"]
    assert schedule.installment_amount == plan["installment_amount"]
    assert schedule.installment_count == plan["installment_count"]
    assert schedule.final_installment_amount == plan["final_installment_amount"]
    assert schedule.total == plan["total"]


def test_the_schedule_sums_exactly_to_the_price():
    """Across all 189. A schedule that does not reconcile is one a customer
    can dispute, and the gate would then be checking a wrong total."""
    for row in PLANNED:
        schedule = payment_schedule(row)
        paid = (
            schedule.down_payment
            + schedule.installment_amount * (schedule.installment_count - 1)
            + schedule.final_installment_amount
        )
        assert paid == row["price"], f"{row['unit_id']} does not reconcile"


def test_the_final_instalment_absorbs_the_remainder():
    """Not spread across the schedule, and not dropped.

    Spreading changes fifteen published figures to fix one; dropping leaves
    the developer short by up to `installment_count - 1` pounds per unit.
    """
    schedule = payment_schedule(unit("NOOR-CIT-002-02"))
    remainder = schedule.final_installment_amount - schedule.installment_amount
    assert remainder == 12
    assert schedule.installments[-1] == schedule.final_installment_amount
    assert set(schedule.installments[:-1]) == {schedule.installment_amount}


def test_the_installment_list_is_the_whole_schedule():
    """One entry per payment, so a reply quoting "the third instalment" has a
    figure to quote rather than an inference to make."""
    schedule = payment_schedule(unit("NOOR-CIT-002-02"))
    assert len(schedule.installments) == 16
    assert sum(schedule.installments) == 6_450_000 - 1_612_500


# ─────────────────────────── refusing ───────────────────────────


def test_a_cash_only_unit_returns_no_schedule():
    """re-0007: MADINATY-001-01 is Completed, so cash only in the fixture.
    Returning an invented plan is pattern-completion over data — every other
    unit has one, so the shape of an answer is available even when the facts
    are not."""
    with pytest.raises(PaymentPlanUnavailable, match="MADINATY-001-01"):
        payment_schedule(unit("MADINATY-001-01"))


def test_an_unoffered_term_is_refused_not_computed():
    """re-0006: the customer proposes 40% down; the fixture offers 25%.

    The calculator would happily compute it, and the arithmetic would be
    correct — which is what makes this dangerous. A computed schedule for
    terms the developer never published creates an expectation they must
    honour or deny, and the customer heard it from us.
    """
    with pytest.raises(UnofferedTerms, match="40"):
        payment_schedule(unit("NOOR-CIT-002-02"), down_payment_pct=40)


def test_the_offered_term_is_accepted_when_named():
    """Naming the terms the unit actually offers is not a different request —
    a customer confirming 25% must not be refused for being specific."""
    schedule = payment_schedule(unit("NOOR-CIT-002-02"), down_payment_pct=25)
    assert schedule.down_payment == 1_612_500


def test_the_refusal_names_what_is_offered():
    """An agent has to say what *is* available, and a bare refusal makes them
    look it up again."""
    with pytest.raises(UnofferedTerms) as raised:
        payment_schedule(unit("NOOR-CIT-002-02"), down_payment_pct=40)
    assert "25" in str(raised.value)


# ─────────────────────────── the shape of the numbers ───────────────────────────


def test_output_is_integer_egp_with_no_floats():
    """A float in a price is a rounding difference waiting to disagree with
    the gate — and with the developer's own published figure."""
    for row in PLANNED[:40]:
        schedule = payment_schedule(row)
        for value in (
            schedule.down_payment,
            schedule.installment_amount,
            schedule.final_installment_amount,
            schedule.total,
            *schedule.installments,
        ):
            assert isinstance(value, int) and not isinstance(value, bool)


def test_zero_interest_is_explicit_not_implied():
    """The fixture states zero on all 189. Stated rather than assumed, so a
    plan that ever carries interest cannot be silently computed as if it did
    not — the schedule would be wrong and nothing would say so."""
    schedule = payment_schedule(unit("NOOR-CIT-002-02"))
    assert schedule.interest_rate == 0

    charged = dict(unit("NOOR-CIT-002-02"))
    charged["payment_plan"] = {**charged["payment_plan"], "interest_rate": 5}
    with pytest.raises(UnofferedTerms, match="interest"):
        payment_schedule(charged)


def test_every_figure_carries_the_unit_it_came_from():
    """`expected_computation.must_match_fixture` is checked per unit, so a
    schedule that cannot say which unit it describes cannot be checked."""
    schedule = payment_schedule(unit("NOOR-CIT-002-02"))
    assert schedule.unit_id == "NOOR-CIT-002-02"
    assert schedule.as_of == "2026-08-01"


def test_the_tool_output_is_json_serialisable_for_the_grounding_check():
    """§3.2: every figure in a reply must string-match a value in the tool's
    output, so the output has to be something the harness can hold."""
    payload = payment_schedule(unit("NOOR-CIT-002-02")).to_json()
    assert json.loads(json.dumps(payload))["installment_amount"] == 302_343
    assert "302343" in json.dumps(payload)


def test_a_unit_with_no_plan_and_no_status_still_refuses():
    """Defensive: absence of a plan is refusal regardless of why. A unit whose
    export dropped the plan column is not a unit with flexible terms."""
    bare = {"unit_id": "X-1", "price": 1_000_000, "payment_plan": None}
    with pytest.raises(PaymentPlanUnavailable):
        payment_schedule(bare)
