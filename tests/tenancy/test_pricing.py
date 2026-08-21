"""Provider pricing — the table that turns `provider_cost_usd` into a query.

The column has existed since migration 0004 and nothing populated it, so "what
did that eval run cost" was reconstructed from code paths and token estimates
rather than read from the ledger. These tests fix the two properties that
matter: the arithmetic is exact, and an unpriced model is unpriced rather than
free.
"""

from decimal import Decimal

import pytest

from moc.tenancy.pricing import cost_usd, priced_models


def test_a_priced_model_costs_what_the_table_says():
    """Sonnet at $3.00 in and $15.00 out per million."""
    assert cost_usd(
        model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=0
    ) == Decimal("3.00")
    assert cost_usd(
        model="claude-sonnet-5", input_tokens=0, output_tokens=1_000_000
    ) == Decimal("15.00")


def test_cached_input_is_billed_at_the_cache_rate_and_not_twice():
    """A cached token is a read, not a fresh input token. Charging it at both
    rates would overstate every turn that hits the prompt cache, which on this
    workload is most of them — the passages are the stable prefix."""
    both = cost_usd(
        model="claude-sonnet-5",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_tokens=1_000_000,
    )
    assert both == Decimal("3.30"), "3.00 for input plus 0.30 for the cache read"


def test_decimal_all_the_way_through():
    """The column is NUMERIC(14,6) and these are fractions of a cent. Float
    arithmetic here produces a total that disagrees with the provider's
    invoice by an amount nobody can account for."""
    value = cost_usd(model="claude-haiku-4-5-20251001", input_tokens=281, output_tokens=82)
    assert isinstance(value, Decimal)
    # 281 in at $1/M plus 82 out at $5/M — the measured shape of one figure audit.
    assert value == Decimal("0.000691")


def test_an_unpriced_model_is_none_not_zero():
    """The whole reason this file exists rather than a dict of best guesses.

    Nothing here carries a confirmed rate for the OpenAI chat models, and every
    judge call in the suite lands on one of them. Writing 0.0 would produce a
    run total that looks complete and understates the largest line item —
    exactly the failure `GateResult.evaluated` exists to prevent one layer up.
    """
    assert cost_usd(model="gpt-5.6-sol", input_tokens=10_000, output_tokens=500) is None


def test_a_model_absent_from_the_table_is_also_none():
    """A model nobody has priced and a model nobody has heard of are the same
    fact to a billing column."""
    assert cost_usd(model="some-model-shipped-tomorrow", input_tokens=100) is None


def test_priced_models_names_what_can_be_costed():
    """So a report can say how many rows it could not price, and for which
    models, rather than presenting a partial sum as a total."""
    priced = priced_models()
    assert "claude-sonnet-5" in priced
    assert "text-embedding-3-large" in priced
    assert "gpt-5.6-sol" not in priced


def test_an_embedding_model_costs_its_input_only():
    """Embeddings have no output tokens, and a table row that quietly priced
    them would be charging for something the provider does not return."""
    assert cost_usd(
        model="text-embedding-3-large", input_tokens=1_000_000, output_tokens=0
    ) == Decimal("0.13")


def test_zero_tokens_costs_zero_rather_than_none():
    """Distinct from unpriced. A call that used no tokens is priced and free;
    a call on an unpriced model is neither."""
    assert cost_usd(model="claude-sonnet-5", input_tokens=0, output_tokens=0) == Decimal("0")


@pytest.mark.parametrize(
    "model", ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"]
)
def test_every_task_s_anthropic_primary_is_priced(model):
    """A routing table that names a model the price table does not is how the
    ledger comes to be full of NULLs nobody notices."""
    assert model in priced_models()


def test_the_routing_table_and_the_price_table_are_compared_somewhere():
    """Every model any task can route to, primary or failover, listed against
    what can be priced. Not an assertion that all are priced — the OpenAI chat
    rates are knowingly absent — but the gap has to be visible and named.
    """
    from moc.config_store import load

    tasks = load("llm/routing")["tasks"]
    routed = {
        entry[role]["model"]
        for entry in tasks.values()
        for role in ("primary", "failover")
        if entry.get(role)
    }
    unpriced = sorted(routed - priced_models())
    assert unpriced == ["gpt-5.6-luna", "gpt-5.6-sol"], (
        f"the set of unpriced routed models changed: {unpriced}. Either a rate "
        f"was filled in, or a new model is routing to an unpriced row."
    )
