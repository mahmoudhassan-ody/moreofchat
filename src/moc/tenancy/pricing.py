"""What a provider call cost — design §19, and the column nothing filled.

`usage_ledger.provider_cost_usd` has existed since migration 0004 and no caller
ever passed it, so "what did that eval run cost" was reconstructed from code
paths and token estimates rather than read from the ledger. This module is what
makes it a query.

**An unpriced model is None, never zero.** A missing rate is a rate nobody has
confirmed, and writing 0.0 for it produces a run total that looks complete and
understates the bill. That is the same failure `GateResult.evaluated` exists to
prevent one layer up: unmeasured and clean print identically unless something
insists on the difference. Here the difference is NULL in the column and a
count of unpriced rows beside the total.

The rates live in `config/billing/pricing.yaml` because they are values that
vary and change without notice, which is §19's whole test for what belongs in
config.

**In `moc.tenancy`, not `moc.llm`, and the import linter is why.** Tenancy is
the bottom layer and the ledger lives here, so a price table above it could
only be reached by having each caller price its own row — four call sites,
four chances to forget, and the one that forgot invisible. That is how the
column came to be null on every row it has ever held. What a call cost is a
billing fact; the model it names is incidental to that.

Filling a rate in is a config edit with an audit trail, and it moves
`config_hash` — correctly, because a run priced under different rates is not
comparable to one priced under these.

**Decimal throughout.** The column is NUMERIC(14,6) and these are fractions of
a cent per call. Float arithmetic produces a total that disagrees with the
provider's invoice by an amount nobody can account for, and the discrepancy
grows with the number of calls rather than staying negligible.
"""

from decimal import Decimal
from functools import lru_cache
from typing import Any

from moc.config_store import load

_PRICING = "billing/pricing"
_PER_MILLION = Decimal(1_000_000)


def cost_usd(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> Decimal | None:
    """Cost of one call, or None when the model has no confirmed rate.

    `cached_tokens` bills at the cache-read rate and is charged once, not on
    top of `input_tokens` — a cached token is a read, not a fresh input token,
    and charging both would overstate every turn that hits the prompt cache.
    On this workload that is most of them: the retrieved passages are the
    stable prefix and travel as cache blocks.

    A model with no `cached_input` rate falls back to its base input rate,
    which is what a provider without a cache tier charges anyway.
    """
    rates = _rates().get(model)
    if rates is None:
        return None
    unit_in = _rate(rates, "input")
    unit_out = _rate(rates, "output")
    if unit_in is None or unit_out is None:
        return None
    unit_cached = _rate(rates, "cached_input")
    if unit_cached is None:
        unit_cached = unit_in

    total = (
        Decimal(input_tokens) * unit_in
        + Decimal(output_tokens) * unit_out
        + Decimal(cached_tokens) * unit_cached
    ) / _PER_MILLION
    # Six places, matching the column. Quantizing here rather than at the
    # insert keeps the rounding in one place, where it is visible.
    return total.quantize(Decimal("0.000001")).normalize()


def priced_models() -> frozenset[str]:
    """Every model this table can cost.

    Exposed so a report can say how many rows it could not price, and for
    which models, rather than presenting a partial sum as a total.
    """
    return frozenset(
        name
        for name, rates in _rates().items()
        if _rate(rates, "input") is not None and _rate(rates, "output") is not None
    )


def _rate(rates: dict[str, Any], key: str) -> Decimal | None:
    value = rates.get(key)
    # `str` first: Decimal(0.13) is 0.13000000000000000444089209850062616169452667236328125,
    # and a rate that is not the number in the config file is a rate nobody
    # can reconcile against an invoice.
    return None if value is None else Decimal(str(value))


@lru_cache(maxsize=1)
def _rates() -> dict[str, dict[str, Any]]:
    return dict(load(_PRICING)["per_million_tokens"])


__all__ = ["cost_usd", "priced_models"]
