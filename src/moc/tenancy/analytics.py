"""What a buyer asks — demo plan Task 34.

How many did the bot answer, how many needed a person, what did it cost, and
why did it give up. Three of those had no answer before the ledger; the fourth
is the handoff reason, which is the most actionable thing on the screen —
"three clarifications" fifty times is a script that cannot route a question the
corpus can answer.

**Containment is reported and never gated.** It is the most tempting number in
the product to put a target on, and the moment it has one every handoff is a
regression — the way to move it is to answer where the honest behaviour is to
hand off. §19.3 exists to stop the bot guessing; a containment gate would pay
it to. `gates.yaml` lists it under `tracked`, and a test asserts it stays
there.

**An unpriced model shows unknown, never zero.** A model with no rate in the
price table contributes NULL to the sum, and `sum()` over NULLs is a smaller
number that looks complete. So the report carries what it could not price, and
`cost_is_complete` is False the moment one row is unpriced — the screen says
"at least" rather than a total it cannot stand behind. This is the same rule
`priced_models()` follows in the price table itself.

**No query here names a tenant.** Every read opens a `tenant_session` and RLS
does the scoping, because a `WHERE tenant_id = ...` is a filter somebody can
forget and a policy is not. A test asserts the string is absent from this file.
"""

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from moc.tenancy.context import tenant_session

_ZERO = Decimal("0")


@dataclass(frozen=True)
class ModelCost:
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    #: None when nothing in the price table covers this model. Not zero — see
    #: the module docstring.
    cost_usd: Decimal | None


@dataclass(frozen=True)
class Report:
    conversations: int
    handed_off: int
    messages: int
    #: None when there were no conversations. Zero conversations is not 100%
    #: containment: a rate over an empty denominator is a number nobody
    #: measured, and on this metric it is the most flattering one available.
    containment_rate: float | None
    cost_usd: Decimal
    cost_per_conversation: Decimal | None
    unpriced_calls: int
    unpriced_models: tuple[str, ...] = ()
    by_model: tuple[ModelCost, ...] = ()
    handoff_reasons: list[tuple[str, int]] = field(default_factory=list)

    @property
    def cost_is_complete(self) -> bool:
        """False the moment one call could not be priced.

        The screen reads this to decide between "cost" and "at least". A total
        presented as complete while some of its rows were unpriced is wrong in
        the direction nobody checks.
        """
        return self.unpriced_calls == 0


class AnalyticsStore:
    def __init__(self, *, engine: Any) -> None:
        self._engine = engine

    async def report(self, *, tenant_id: uuid.UUID) -> Report:
        async with tenant_session(self._engine, tenant_id) as session:
            counts = (
                await session.execute(
                    text(
                        "SELECT "
                        "(SELECT count(*) FROM conversations) AS conversations, "
                        # DISTINCT: a conversation handed off twice is one
                        # conversation that needed a person, not two.
                        "(SELECT count(DISTINCT conversation_id) FROM handoffs) AS handed_off, "
                        "(SELECT count(*) FROM messages) AS messages"
                    )
                )
            ).one()

            spend = (
                await session.execute(
                    text(
                        "SELECT model, count(*) AS calls, "
                        "sum(input_tokens) AS input_tokens, "
                        "sum(output_tokens) AS output_tokens, "
                        # NULL if ANY row for this model is unpriced, rather
                        # than a partial sum: half a model's spend presented as
                        # its spend is the failure this column exists to avoid.
                        "CASE WHEN count(*) FILTER (WHERE provider_cost_usd IS NULL) > 0 "
                        "     THEN NULL ELSE sum(provider_cost_usd) END AS cost_usd, "
                        "count(*) FILTER (WHERE provider_cost_usd IS NULL) AS unpriced "
                        "FROM usage_ledger WHERE model IS NOT NULL "
                        "GROUP BY model ORDER BY model"
                    )
                )
            ).all()

            reasons = (
                await session.execute(
                    text(
                        "SELECT reason, count(*) AS times FROM handoffs "
                        "GROUP BY reason ORDER BY times DESC, reason"
                    )
                )
            ).all()

        by_model = tuple(
            ModelCost(
                model=row.model,
                calls=row.calls,
                input_tokens=row.input_tokens or 0,
                output_tokens=row.output_tokens or 0,
                cost_usd=row.cost_usd,
            )
            for row in spend
        )
        priced = sum((row.cost_usd for row in by_model if row.cost_usd is not None), _ZERO)
        unpriced_calls = sum(row.unpriced for row in spend)

        return Report(
            conversations=counts.conversations,
            handed_off=counts.handed_off,
            messages=counts.messages,
            containment_rate=(
                (counts.conversations - counts.handed_off) / counts.conversations
                if counts.conversations
                else None
            ),
            cost_usd=priced,
            cost_per_conversation=(
                priced / counts.conversations if counts.conversations else None
            ),
            unpriced_calls=unpriced_calls,
            unpriced_models=tuple(row.model for row in spend if row.unpriced),
            by_model=by_model,
            handoff_reasons=[(row.reason, row.times) for row in reasons],
        )


__all__ = ["AnalyticsStore", "ModelCost", "Report"]
