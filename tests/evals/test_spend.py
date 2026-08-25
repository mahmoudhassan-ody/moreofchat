"""Recording what a run cost, somewhere it survives the run.

`usage_ledger` has never seen an eval run. They execute against `moc_test`,
which `tests/conftest.py` drops and recreates every session, and the fixtures
truncate the ledger between tests on top of that. So the one instrument built
to answer "what does this cost" was blind to the activity that does most of
the spending — roughly seven dollars over three weeks, none of it recorded,
ending in an exhausted account on 2026-08-25.
"""

import pytest

from moc.evals.spend import Spend, summarise

ROWS = [
    # (model, provider, input, output, cached, cache_write, usd)
    ("claude-haiku-4-5-20251001", "anthropic", 1763, 57, 0, 0, 0.002048),
    ("claude-sonnet-5", "anthropic", 1265, 93, 0, 0, 0.003460),
    ("claude-sonnet-5", "anthropic", 1171, 87, 0, 1607, 0.007230),
    ("text-embedding-3-large", "openai", 29, 0, 0, 0, 0.000004),
]


def test_a_run_is_summarised_by_model():
    spend = summarise(ROWS)
    assert isinstance(spend, Spend)
    assert spend.total_usd == pytest.approx(0.012742)
    assert spend.by_model["claude-sonnet-5"]["calls"] == 2
    assert spend.by_model["claude-sonnet-5"]["usd"] == pytest.approx(0.010690)
    assert spend.by_model["claude-sonnet-5"]["cache_write"] == 1607


def test_the_summary_keeps_tokens_as_well_as_money():
    """Money moves when a price list changes; tokens do not. A run recorded
    only in dollars cannot be repriced after 2026-08-31, when composition goes
    from introductory to list rates and every stored figure understates."""
    spend = summarise(ROWS)
    haiku = spend.by_model["claude-haiku-4-5-20251001"]
    assert haiku["input"] == 1763
    assert haiku["output"] == 57


def test_an_empty_run_is_zero_and_says_it_measured_nothing():
    """Distinct from an unmeasured run. This one ran and cost nothing, which
    happens when every case errors before a provider is reached — the shape of
    the 2026-08-25 failure."""
    spend = summarise([])
    assert spend.total_usd == 0.0
    assert spend.by_model == {}
    assert spend.calls == 0


def test_a_run_names_the_models_that_were_not_primaries():
    """The row has to say this itself.

    The first version set `substituted` from the headroom check, which only
    runs when grading — so a stage-1 run on failover recorded `NULL` while
    `by_model` plainly showed OpenAI. A durable row that needs a second column
    read to be understood will be quoted without it.
    """
    from moc.evals.spend import not_primary

    routing = {
        "tasks": {
            "slot_extraction": {
                "primary": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
            },
            "answer_composition": {
                "primary": {"provider": "anthropic", "model": "claude-sonnet-5"}
            },
            "embedding": {"primary": {"provider": "openai", "model": "text-embedding-3-large"}},
        }
    }
    spend = summarise(
        [
            ("gpt-5.6-sol", "openai", 10, 1, 0, 0, 0.1),
            ("claude-haiku-4-5-20251001", "anthropic", 10, 1, 0, 0, 0.01),
            ("text-embedding-3-large", "openai", 10, 0, 0, 0, 0.0),
        ]
    )
    assert not_primary(spend, routing=routing) == ["gpt-5.6-sol"]


def test_a_run_entirely_on_primaries_names_nothing():
    from moc.evals.spend import not_primary

    routing = {
        "tasks": {
            "answer_composition": {
                "primary": {"provider": "anthropic", "model": "claude-sonnet-5"}
            }
        }
    }
    spend = summarise([("claude-sonnet-5", "anthropic", 10, 1, 0, 0, 0.1)])
    assert not_primary(spend, routing=routing) == []
