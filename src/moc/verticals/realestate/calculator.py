"""The payment plan calculator — design §3.2 and §19.3.

A deterministic tool. **The model never divides.**

§3.1's rule is that the LLM composes words, never figures; §3.2 makes that
enforceable for instalments by producing them here and string-matching the
reply against this output. So the arithmetic has to be right in the way the
developer's own published schedule is right, not merely defensible.

**Why flooring, and why it matters more than it looks.**

NOOR-CIT-002-02 is 6,450,000 with 25% down over 16 quarters. That leaves
4,837,500, which is 302,343.75 a quarter. A model asked to divide answers
302,344 — natural, defensible, and wrong: fifteen of those plus a sixteenth
overshoots the price. The developer floors each instalment to 302,343 and
carries the accumulated remainder onto the last one, 302,355.

Twelve pounds. It reads as a formatting preference and it is the entire
demonstration that a tool produced the number, which is why the rule is stated
here rather than left to `round()` to decide by accident. Verified against all
189 planned units in the fixture, each reconciling to its price exactly.

**What this refuses to do.** Terms the developer did not publish. The
calculator would compute 40% down perfectly well, and the arithmetic would be
correct — which is what makes it dangerous. A schedule for unoffered terms is
an expectation the developer must then honour or deny, and the customer heard
it from us. Same for a unit with no plan at all: a completed project is cash
only, and inventing a plan because every other unit has one is
pattern-completion over data.
"""

from dataclasses import dataclass
from typing import Any

#: Payments per year, by the word the fixture uses. A frequency absent from
#: here is refused rather than guessed — a wrong divisor produces a schedule
#: that reconciles to the price and bears no relation to what was published.
PAYMENTS_PER_YEAR = {
    "monthly": 12,
    "quarterly": 4,
    "semi-annually": 2,
    "annually": 1,
    "yearly": 1,
}


class PaymentPlanUnavailable(Exception):
    """This unit has no published plan. Cash only, and the reply must say so."""


class UnofferedTerms(Exception):
    """The terms asked for are not the terms published for this unit."""


@dataclass(frozen=True)
class PaymentSchedule:
    """One unit's schedule, in whole EGP.

    Integers throughout. A float in a price is a rounding difference waiting
    to disagree with the developer's published figure and with the gate that
    string-matches against it.

    `unit_id` and `as_of` travel with the numbers because
    `expected_computation.must_match_fixture` is checked per unit, and a
    schedule that cannot say what it describes cannot be checked at all.
    """

    unit_id: str
    as_of: str | None
    total: int
    down_payment: int
    down_payment_pct: int
    installment_amount: int
    installment_count: int
    final_installment_amount: int
    installments: tuple[int, ...]
    frequency: str
    years: int
    interest_rate: int = 0

    def to_json(self) -> dict[str, Any]:
        """The tool output the orchestrator grounds a reply against."""
        return {
            "unit_id": self.unit_id,
            "as_of": self.as_of,
            "total": self.total,
            "down_payment": self.down_payment,
            "down_payment_pct": self.down_payment_pct,
            "installment_amount": self.installment_amount,
            "installment_count": self.installment_count,
            "final_installment_amount": self.final_installment_amount,
            "installments": list(self.installments),
            "frequency": self.frequency,
            "years": self.years,
            "interest_rate": self.interest_rate,
        }


def payment_schedule(
    unit: dict[str, Any], *, down_payment_pct: int | None = None
) -> PaymentSchedule:
    """The published schedule for `unit`, or a refusal.

    `down_payment_pct` is not a knob. It is there so a customer naming the
    terms can be answered — and so a customer naming *different* terms is
    refused rather than quietly computed. Passing the offered percentage is
    the same request as passing nothing.
    """
    plan = unit.get("payment_plan")
    if not plan:
        raise PaymentPlanUnavailable(
            f"{unit.get('unit_id')} has no published payment plan — cash only"
        )

    offered = int(plan["down_payment_pct"])
    if down_payment_pct is not None and int(down_payment_pct) != offered:
        raise UnofferedTerms(
            f"{unit.get('unit_id')} is offered at {offered}% down, not "
            f"{down_payment_pct}% — the developer publishes one plan per unit"
        )

    # Stated, not assumed. The fixture is zero on all 189, and a plan that
    # ever carries interest must not be computed as if it did not: the
    # schedule would be wrong and nothing in the output would say so.
    interest_rate = int(plan.get("interest_rate", 0))
    if interest_rate != 0:
        raise UnofferedTerms(
            f"{unit.get('unit_id')} carries interest at {interest_rate}%, which this "
            f"calculator does not compute — it produces zero-interest schedules only"
        )

    frequency = plan["frequency"]
    if frequency not in PAYMENTS_PER_YEAR:
        raise UnofferedTerms(
            f"{unit.get('unit_id')} uses an unknown instalment frequency {frequency!r}"
        )

    price = int(unit["price"])
    # Half-up on the exact integer product rather than `round()`, which is
    # banker's and would differ on a tie. Unexercised by this fixture — every
    # one of the 189 down payments divides exactly — so it is written to be
    # defensible rather than to match an observed case.
    down_payment = (price * offered + 50) // 100

    count = int(plan["installment_count"])
    expected_count = int(plan["years"]) * PAYMENTS_PER_YEAR[frequency]
    if count != expected_count:
        raise UnofferedTerms(
            f"{unit.get('unit_id')} states {count} instalments but {plan['years']} years "
            f"at {frequency} is {expected_count} — the published plan is inconsistent"
        )

    remaining = price - down_payment
    # Floor, then carry the accumulated remainder onto the last instalment.
    # Rounding instead overshoots the price; spreading the remainder changes
    # every published figure to fix one; dropping it leaves the developer
    # short by up to `count - 1`.
    installment = remaining // count
    final = remaining - installment * (count - 1)
    installments = (installment,) * (count - 1) + (final,)

    return PaymentSchedule(
        unit_id=unit["unit_id"],
        as_of=unit.get("as_of"),
        total=price,
        down_payment=down_payment,
        down_payment_pct=offered,
        installment_amount=installment,
        installment_count=count,
        final_installment_amount=final,
        installments=installments,
        frequency=frequency,
        years=int(plan["years"]),
        interest_rate=interest_rate,
    )


__all__ = [
    "PAYMENTS_PER_YEAR",
    "PaymentPlanUnavailable",
    "PaymentSchedule",
    "UnofferedTerms",
    "payment_schedule",
]
