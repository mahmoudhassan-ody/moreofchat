"""Provider pricing — the table that turns `provider_cost_usd` into a query.

The column has existed since migration 0004 and nothing populated it, so "what
did that eval run cost" was reconstructed from code paths and token estimates
rather than read from the ledger. These tests fix the two properties that
matter: the arithmetic is exact, and an unpriced model is unpriced rather than
free.
"""

from decimal import Decimal

import pytest

from moc.tenancy.pricing import SHORT, cost_usd, price, priced_models


def test_a_priced_model_costs_what_the_table_says():
    """Sonnet at the introductory $2.00 in and $10.00 out per million.

    These are promotional rates running to 2026-08-31. When they lapse this
    test fails, which is the point — the alternative is every composition row
    understating by a third with nothing to notice it.
    """
    assert cost_usd(
        model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=0
    ) == Decimal("2.00")
    assert cost_usd(
        model="claude-sonnet-5", input_tokens=0, output_tokens=1_000_000
    ) == Decimal("10.00")


def test_cached_input_is_billed_at_the_cache_rate_and_not_twice():
    """A cached token is a read, not a fresh input token. Charging it at the
    input rate overstates a cache hit tenfold on Anthropic, and this workload
    caches the tenant script and policy preamble on every composed turn — so
    the overstatement was on almost every row."""
    both = cost_usd(
        model="claude-sonnet-5",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_tokens=1_000_000,
    )
    assert both == Decimal("2.20"), "2.00 for input plus 0.20 for the cache read"


def test_a_cache_write_costs_more_than_input_not_less():
    """The direction that makes conflating the two rates dangerous. A read is a
    tenth of input; a write is a quarter more than it. One field priced at one
    rate would be wrong both ways depending on the turn."""
    assert cost_usd(
        model="claude-sonnet-5", cache_write_tokens=1_000_000
    ) == Decimal("2.50")
    assert cost_usd(model="claude-sonnet-5", cached_tokens=1_000_000) == Decimal("0.20")


# ─────────────── OpenAI's long-context tier ───────────────


def test_a_prompt_under_the_threshold_is_billed_short():
    """272,000 is the line, so a million-token prompt is emphatically not
    under it — the first version of this test asserted the short rate on a
    long prompt and failed for the right reason."""
    priced = price(model="gpt-5.6-sol", input_tokens=200_000, output_tokens=0)
    assert priced.tier == SHORT
    assert priced.usd == Decimal("0.8"), "200k at 4.00/M"


def test_a_prompt_over_the_threshold_is_billed_long_for_the_whole_request():
    """">272K input tokens are priced at 2x input and 1.5x output for the full
    request" — the excess is not billed separately, the request is."""
    priced = price(model="gpt-5.6-sol", input_tokens=300_000, output_tokens=1_000_000)
    assert priced.tier == "long"
    # 300k at 8.00/M, plus 1M output at the long rate of 30.00/M.
    assert priced.usd == Decimal("2.4") + Decimal("30.00")


def test_the_threshold_counts_the_whole_prompt_including_cached_tokens():
    """Tokens served from cache are still tokens you sent. `input_tokens` is
    net of cached on both adapters, so a threshold read off it alone would bill
    a 300K prompt at the short rate whenever most of it was a cache hit — the
    exact shape of a long conversation on this system."""
    priced = price(model="gpt-5.6-sol", input_tokens=100_000, cached_tokens=200_000)
    assert priced.tier == "long"


def test_output_tokens_do_not_push_a_request_into_the_long_tier():
    """The threshold is on the prompt. Counting the reply would move a short
    request into the expensive tier for producing a long answer."""
    priced = price(model="gpt-5.6-sol", input_tokens=1000, output_tokens=500_000)
    assert priced.tier == SHORT


def test_the_long_rates_are_the_multiples_the_vendor_states():
    """2x input and 1.5x output, checked against the published long rows rather
    than computed from them. If the table and the sentence ever disagree, this
    is where it surfaces — and the sentence is what the threshold rests on."""
    from moc.config_store import load

    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        rates = load("billing/pricing")["per_million_tokens"][model]
        long = rates["long_context"]
        assert Decimal(str(long["input"])) == Decimal(str(rates["input"])) * 2
        assert Decimal(str(long["output"])) == Decimal(str(rates["output"])) * Decimal("1.5")
        assert Decimal(str(long["cached_input"])) == Decimal(str(rates["cached_input"])) * 2


def test_a_model_with_no_long_tier_is_one_rate_at_any_size():
    """Anthropic declares none on these models. Absent is not the same as
    unlooked-up: a model with no block is priced flat, a model whose rates are
    null is not priced at all."""
    small = price(model="claude-opus-5", input_tokens=1000)
    large = price(model="claude-opus-5", input_tokens=900_000)
    assert small.tier == SHORT and large.tier == SHORT
    assert large.usd == Decimal("4.5")


def test_the_tier_is_reported_even_when_the_model_is_unpriced():
    """So a NULL cost row still records which tier it would have been. The
    reason to record the tier is that it cannot be recovered later from the
    token counts alone once the threshold moves."""
    priced = price(model="some-model-shipped-tomorrow", input_tokens=1000)
    assert priced.usd is None
    assert priced.tier == SHORT


def test_the_read_date_is_recorded():
    """A rate table with no read date cannot be audited against an invoice."""
    from moc.config_store import load

    assert str(load("billing/pricing")["read_on"]) == "2026-08-21"


def test_decimal_all_the_way_through():
    """The column is NUMERIC(14,6) and these are fractions of a cent. Float
    arithmetic here produces a total that disagrees with the provider's
    invoice by an amount nobody can account for."""
    value = cost_usd(model="claude-haiku-4-5-20251001", input_tokens=281, output_tokens=82)
    assert isinstance(value, Decimal)
    # 281 in at $1/M plus 82 out at $5/M — the measured shape of one figure audit.
    assert value == Decimal("0.000691")


def test_a_priced_model_now_needs_all_four_rates():
    """`priced_models` gates on input and output. A model carrying those and no
    cache rates would price a cache hit at the input rate silently, which is
    the overstatement this schema change exists to end."""
    from moc.config_store import load

    for name, rates in load("billing/pricing")["per_million_tokens"].items():
        assert set(rates) >= {"input", "cached_input", "cache_write", "output"}, name


def test_an_unpriced_model_is_none_not_zero():
    """The whole reason this file exists rather than a dict of best guesses.

    Every rate routing can reach is now confirmed, so this is asserted against
    a model that carries an explicit null — the shape has to keep working, or
    the next unlooked-up model prices as free.
    """
    assert cost_usd(model="text-embedding-3-large", cached_tokens=10_000) is None, (
        "the vendor lists no cache tier for embeddings, and null must not be 0"
    )


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
    assert "gpt-5.6-sol" in priced, "sourced from the vendor page 2026-08-21"


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
    assert unpriced == [], (
        f"these models route somewhere and have no confirmed rate: {unpriced}. "
        f"Every one of their rows prices as NULL, and the run total that omits "
        f"them reads as complete."
    )
