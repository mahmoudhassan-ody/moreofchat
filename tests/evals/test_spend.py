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


# ───────────── §5.2's judge is not a substitution ─────────────
#
# `eval_grading` routes to Opus first and never reaches it: §5.2 excludes
# whichever provider composed, so with composition on Anthropic every judge
# call in every graded run lands on `gpt-5.6-sol`. Comparing what ran against
# the set of *primaries* therefore flagged the one correct graded baseline —
# eval_runs f9b1d26b, 2026-08-25 — with "ran on gpt-5.6-sol, which no task
# names as primary". True of the primaries table, false about the run, and the
# warning tells the next reader not to quote the only number worth quoting.
#
# The same conflation as `check_primaries` had, in a second place. That one was
# green about a model the run would not call; this one is red about the model
# the run is required to call.

JUDGED = {
    "tasks": {
        "answer_composition": {
            "primary": {"provider": "anthropic", "model": "claude-sonnet-5"},
            "failover": {"provider": "openai", "model": "gpt-5.6-sol"},
        },
        "eval_grading": {
            "primary": {"provider": "anthropic", "model": "claude-opus-5"},
            "failover": {"provider": "openai", "model": "gpt-5.6-sol"},
        },
    }
}


def test_the_judge_5_2_forces_is_not_flagged_as_a_substitution():
    from moc.evals.spend import not_primary

    spend = summarise(
        [
            ("claude-sonnet-5", "anthropic", 10, 1, 0, 0, 0.18),
            ("gpt-5.6-sol", "openai", 10, 1, 0, 0, 0.61),
        ]
    )
    assert not_primary(spend, routing=JUDGED, graded=True) == []


def test_the_same_model_on_an_ungraded_run_still_is():
    """No judge ran, so `gpt-5.6-sol` can only be composition on failover —
    which is exactly what the first recorded run was, and it was right to say
    so. The exemption is the judge's, not the model's."""
    from moc.evals.spend import not_primary

    spend = summarise([("gpt-5.6-sol", "openai", 10, 1, 0, 0, 0.16)])
    assert not_primary(spend, routing=JUDGED, graded=False) == ["gpt-5.6-sol"]


def test_a_graded_run_still_names_a_model_that_is_neither_primary_nor_judge():
    """The exemption is one model wide. Extraction on `gpt-5.6-luna` is a
    substituted baseline whether or not the judge was also running."""
    from moc.evals.spend import not_primary

    spend = summarise(
        [
            ("gpt-5.6-sol", "openai", 10, 1, 0, 0, 0.61),
            ("gpt-5.6-luna", "openai", 10, 1, 0, 0, 0.02),
        ]
    )
    assert not_primary(spend, routing=JUDGED, graded=True) == ["gpt-5.6-luna"]
