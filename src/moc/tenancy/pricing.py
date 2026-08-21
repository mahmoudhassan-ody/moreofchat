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

**Four rates, not two.** A cache *read* is a tenth of base input on Anthropic
and a cache *write* is a quarter more than it — they move in opposite
directions, so one field priced at one rate is wrong both ways depending on the
turn. `Completion.cached_tokens` is reads only on both adapters;
`cache_write_tokens` carries Anthropic's `cache_creation_input_tokens` and is
always 0 for OpenAI, whose field name is not documented on any page reachable
from here. That understates an OpenAI turn that fills the cache, and the config
says so rather than pretending otherwise.

**Two tiers, on the prompt.** OpenAI bills 2x input and 1.5x output for a
request whose prompt exceeds a threshold, and it applies to the whole request
rather than the excess. The prompt is `input + cached + cache_write` — tokens
served from cache are still tokens you sent, and `input_tokens` is net of cached
on both adapters, so a threshold read off it alone would bill a long
conversation short precisely because most of it was a cache hit. Output does not
count toward it.

The tier is returned as well as the price, because it cannot be recovered from
the token counts later: the threshold is a vendor policy that moves, and a row
priced under one is not comparable to a row priced under another.

**Decimal throughout.** The column is NUMERIC(14,6) and these are fractions of
a cent per call. Float arithmetic produces a total that disagrees with the
provider's invoice by an amount nobody can account for, and the discrepancy
grows with the number of calls rather than staying negligible.
"""

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any

from moc.config_store import load

_PRICING = "billing/pricing"
_PER_MILLION = Decimal(1_000_000)


#: Which rate block priced a call. Recorded on the ledger row rather than
#: inferred later — the threshold is a vendor policy that moves, and once it
#: has, the token counts no longer say which side of it a past row fell.
SHORT = "short"
LONG = "long"
_LONG = "long_context"
_THRESHOLD = "threshold_input_tokens"


@dataclass(frozen=True)
class Priced:
    """What a call cost, and under which rate block.

    `usd` is None for a model with no confirmed rate. `tier` is reported
    regardless, so an unpriced row still records what it would have been.
    """

    usd: Decimal | None
    tier: str


def price(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Priced:
    """Cost of one call and the tier it fell in.

    A model with no `long_context` block is priced flat at any size — which is
    a different fact from a model whose long rates nobody has looked up, and
    the config distinguishes them.
    """
    rates = _rates().get(model)
    if rates is None:
        return Priced(None, SHORT)

    prompt = input_tokens + cached_tokens + cache_write_tokens
    long = rates.get(_LONG) or {}
    threshold = long.get(_THRESHOLD)
    over = threshold is not None and prompt > int(threshold)
    tier = LONG if over else SHORT
    book = long if over else rates

    unit_in = _rate(book, "input")
    unit_out = _rate(book, "output")
    if unit_in is None or unit_out is None:
        return Priced(None, tier)
    # A cache rate the vendor does not publish falls back to base input rather
    # than to zero: charging nothing for a token the invoice charges for is the
    # one direction this table must never round.
    unit_cached = _rate(book, "cached_input")
    unit_write = _rate(book, "cache_write")
    if unit_cached is None and cached_tokens:
        return Priced(None, tier)
    if unit_write is None and cache_write_tokens:
        return Priced(None, tier)

    total = (
        Decimal(input_tokens) * unit_in
        + Decimal(output_tokens) * unit_out
        + Decimal(cached_tokens) * (unit_cached or Decimal(0))
        + Decimal(cache_write_tokens) * (unit_write or Decimal(0))
    ) / _PER_MILLION
    # Six places, matching the column. Quantized here rather than at the insert
    # so the rounding lives in one place, where it is visible.
    return Priced(total.quantize(Decimal("0.000001")).normalize(), tier)


def cost_usd(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal | None:
    """`price(...).usd`, for callers that do not record the tier."""
    return price(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    ).usd

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


__all__ = ["LONG", "SHORT", "Priced", "cost_usd", "price", "priced_models"]
