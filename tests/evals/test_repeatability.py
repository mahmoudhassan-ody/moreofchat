"""Repeated runs and spreads — harness spec §2.4.

A single run of a 17- or 23-case suite against a non-deterministic model is a
sample, not a measurement. Four runs of the real-estate suite over one
unchanged commit read 52.2%, 42.9%, 39.1% and 45.5%; any one of them reported
alone is a point estimate the code cannot support.

These tests pin the two things that stop that happening again: a metric is
carried as every value it took, and it renders with its spread and its run
count attached so a reader cannot mistake one sample for a number.
"""

import pytest

from moc.evals.repeatability import MetricSpread, aggregate, default_runs, repeat


def spread(*values: float | None, metric: str = "overall_accuracy") -> MetricSpread:
    return MetricSpread(metric=metric, values=tuple(values))


# ───────────────────────── what a spread reports ─────────────────────────


def test_a_metric_renders_with_its_spread_and_run_count():
    """`45.5% (39.1–52.2, n=4)` — never the bare 45.5%."""
    rendered = spread(0.522, 0.429, 0.391, 0.455).render()

    assert "44.9%" in rendered  # the mean, not the last run
    assert "39.1" in rendered
    assert "52.2" in rendered
    assert "n=4" in rendered


def test_the_mean_is_the_mean_not_the_last_run():
    assert spread(0.4, 0.6).mean == pytest.approx(0.5)
    assert spread(0.4, 0.6).minimum == pytest.approx(0.4)
    assert spread(0.4, 0.6).maximum == pytest.approx(0.6)


def test_spread_is_reported_in_percentage_points():
    """The threshold the user set is in points, so the spread must be too."""
    assert spread(0.391, 0.522).spread_pp == pytest.approx(13.1)


# ───────────────────── what makes a metric measurable ─────────────────────


def test_a_spread_wider_than_the_threshold_is_not_measurable():
    wide = spread(0.391, 0.522)
    assert wide.spread_pp > 10.0
    assert not wide.measurable
    assert "not measurable at this suite size" in wide.render()


def test_a_tight_spread_is_measurable():
    tight = spread(0.450, 0.470, 0.460)
    assert tight.measurable
    assert "not measurable" not in tight.render()


def test_a_single_run_is_never_measurable():
    """One sample has zero spread, and zero spread is exactly what a
    measurable metric looks like. Reading n=1 as measurable is the false
    confidence this module exists to prevent."""
    one = spread(0.455)
    assert one.spread_pp == 0.0
    assert not one.measurable
    assert "n=1" in one.render()


# ────────────────────────── unmeasured is not zero ──────────────────────────


def test_a_metric_no_run_measured_says_so_rather_than_zero():
    """Same rule as `GateResult.rate`: 0.0% is what a passing gate shows."""
    absent = spread(None, None, None)
    assert absent.runs == 0
    assert absent.mean is None
    assert not absent.measurable
    assert "not measured" in absent.render()
    assert "0.0%" not in absent.render()


def test_a_metric_measured_in_some_runs_reports_only_those_runs():
    """A gate that fed on two of three runs is n=2, and the report must say
    n=2 rather than dividing by three or hiding the difference."""
    partial = spread(0.5, None, 0.7)
    assert partial.runs == 2
    assert partial.attempts == 3
    assert partial.mean == pytest.approx(0.6)
    assert "n=2 of 3 runs" in partial.render()


# ──────────────────────────── aggregation ────────────────────────────


def test_aggregate_carries_every_metric_from_every_run():
    spreads = aggregate(
        [
            {"overall_accuracy": 0.5, "asof_disclosure_rate": 0.8},
            {"overall_accuracy": 0.6, "asof_disclosure_rate": None},
        ]
    )

    assert set(spreads) == {"overall_accuracy", "asof_disclosure_rate"}
    assert spreads["overall_accuracy"].values == (0.5, 0.6)
    assert spreads["asof_disclosure_rate"].runs == 1


def test_a_metric_missing_from_one_run_is_a_gap_not_an_omission():
    """A key absent from a run's dict and a key present as None mean the same
    thing — that run did not measure it — and both must keep the run in the
    denominator of `attempts`."""
    spreads = aggregate([{"a": 0.5}, {}])

    assert spreads["a"].attempts == 2
    assert spreads["a"].runs == 1


def test_aggregate_of_no_runs_is_empty_rather_than_an_error():
    assert aggregate([]) == {}


# ──────────────────────────── running N times ────────────────────────────


async def test_repeat_runs_the_suite_n_times_and_collects_each():
    seen = []

    async def once():
        seen.append(len(seen))
        return {"overall_accuracy": 0.1 * len(seen)}

    spreads = await repeat(once, times=3)

    assert seen == [0, 1, 2]
    assert spreads["overall_accuracy"].values == (0.1, 0.2, 0.30000000000000004)


async def test_repeat_refuses_fewer_than_two_runs():
    """`times=1` produces a point estimate wearing a spread's clothes."""

    async def once():
        return {}

    with pytest.raises(ValueError, match="at least 2"):
        await repeat(once, times=1)


def test_the_run_count_and_the_threshold_come_from_config():
    """§19: nothing that varies is a literal in code. Raising N to steady a
    number must be a config edit with an audit trail — and it moves
    config_hash, which correctly invalidates the baseline it was compared to."""
    from moc.config_store import load

    config = load("evals/repeat")
    assert default_runs() == config["runs"]
    assert config["runs"] >= 2
    assert MetricSpread.measurable_spread_pp() == config["measurable_spread_pp"]


def test_no_threshold_is_a_literal_in_the_module():
    """The spread bar is config, like every other threshold in the harness."""
    import inspect

    from moc.evals import repeatability

    source = inspect.getsource(repeatability)
    assert "10.0" not in source
    assert "10)" not in source


# ─────────────────── what one run of each suite contributes ───────────────────


def _outcome(case_id, *, passed=True, errored=False, recall=None, checks=()):
    from moc.evals.runner import CaseOutcome, TurnOutcome

    return CaseOutcome(
        case_id=case_id,
        vertical="education",
        category="factual_retrieval",
        turns=() if errored else (TurnOutcome(turn_index=0, reply="r", action="answer",
                                              state=None, checks=tuple(checks)),),
        errored=errored,
        recall_at_5=recall,
    )


def _check(metric, passed, *, skipped=False):
    from moc.evals.deterministic import CheckResult

    return CheckResult(name=metric, metric=metric, passed=passed, skipped=skipped)


def test_a_document_run_contributes_accuracy_recall_and_every_gate():
    from moc.evals.runner import metrics

    got = metrics(
        [
            _outcome("a", checks=(_check("hallucinated_figure_rate", True),), recall=1.0),
            _outcome("b", checks=(_check("hallucinated_figure_rate", False),), recall=0.0),
        ]
    )

    assert got["overall_accuracy"] == pytest.approx(0.5)
    assert got["retrieval_recall_at_5"] == pytest.approx(0.5)
    # direction `max` in gates.yaml: the metric is the share that FAILED.
    assert got["hallucinated_figure_rate"] == pytest.approx(0.5)


def test_a_metric_direction_comes_from_gates_config_not_from_this_module():
    """`expected_action_accuracy` is a `min` gate — the share that passed —
    and `hallucinated_figure_rate` a `max` one. Reading the same check both
    ways is how a 0% hallucination rate and a 0% action accuracy would print
    identically."""
    from moc.evals.runner import metrics

    got = metrics([_outcome("a", checks=(_check("expected_action_accuracy", True),))])
    assert got["expected_action_accuracy"] == pytest.approx(1.0)


def test_a_metric_no_turn_fed_is_none_not_zero():
    from moc.evals.runner import metrics

    got = metrics([_outcome("a", checks=(_check("hedged_figure_rate", True, skipped=True),))])
    assert got["hedged_figure_rate"] is None
    assert got["retrieval_recall_at_5"] is None


def test_an_empty_run_still_names_every_gate():
    """A metric absent from a run's dict looks the same as a run that measured
    nothing, but only one of them keeps the gate visible across the spread."""
    from moc.evals.runner import metrics

    got = metrics([])
    assert "hallucinated_figure_rate" in got
    assert got["hallucinated_figure_rate"] is None
    assert got["overall_accuracy"] is None


def _inv_outcome(case_id, *, checks=(), errored=False):
    from moc.evals.inventory_runner import InventoryCaseOutcome, InventoryTurnOutcome

    return InventoryCaseOutcome(
        case_id=case_id,
        vertical="realestate",
        category="inventory",
        turns=()
        if errored
        else (
            InventoryTurnOutcome(
                turn_index=0, reply="r", action="answer", state=None, checks=tuple(checks)
            ),
        ),
        errored=errored,
    )


def test_an_inventory_run_contributes_all_five_gates_and_both_tracked():
    from moc.evals.inventory_runner import GATES, TRACKED, metrics

    got = metrics([_inv_outcome("re-0001")])

    for name in (*GATES, *TRACKED):
        assert name in got, name
        assert got[name] is None, f"{name} fed nothing and must not read 0.0"
    assert got["overall_accuracy"] == pytest.approx(1.0)


def test_an_inventory_failure_rate_gate_counts_failures_not_passes():
    """`sold_unit_offered_rate` is `direction: max`. One failure in two
    observations is 50% — not 50% the other way round."""
    from moc.evals.inventory_runner import metrics

    got = metrics(
        [
            _inv_outcome("a", checks=(_check("sold_unit_offered_rate", True),)),
            _inv_outcome("b", checks=(_check("sold_unit_offered_rate", False),)),
        ]
    )
    assert got["sold_unit_offered_rate"] == pytest.approx(0.5)


def test_asof_is_the_one_inventory_gate_that_counts_successes():
    from moc.evals.inventory_runner import metrics

    got = metrics(
        [
            _inv_outcome("a", checks=(_check("asof_disclosure_rate", True),)),
            _inv_outcome("b", checks=(_check("asof_disclosure_rate", True),)),
            _inv_outcome("c", checks=(_check("asof_disclosure_rate", False),)),
        ]
    )
    assert got["asof_disclosure_rate"] == pytest.approx(2 / 3)


def test_an_errored_inventory_case_stays_out_of_the_accuracy_denominator():
    """Same rule the document runner already applies: an exception is an
    outage, not a quality signal, and folding it into accuracy makes a flaky
    network read as a regression."""
    from moc.evals.inventory_runner import metrics

    got = metrics([_inv_outcome("a"), _inv_outcome("b", errored=True)])
    assert got["overall_accuracy"] == pytest.approx(1.0)
    assert got["errored_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "extract",
    ["moc.evals.runner:metrics", "moc.evals.inventory_runner:metrics"],
)
def test_every_metric_a_run_contributes_is_a_proportion(extract):
    """`MetricSpread.render` multiplies by 100. A count sent through it reads
    as a percentage — one errored case in a three-run spread printed
    `errored_cases 133.3% (100.0-200.0, n=3)`, which is not a thing. Every
    value here is a rate or it is None.
    """
    import importlib

    module, name = extract.split(":")
    fn = getattr(importlib.import_module(module), name)
    outcomes = (
        [_outcome("a"), _outcome("b", errored=True)]
        if "inventory" not in module
        else [_inv_outcome("a"), _inv_outcome("b", errored=True)]
    )
    for metric, value in fn(outcomes).items():
        assert value is None or 0.0 <= value <= 1.0, f"{metric} is not a proportion"
