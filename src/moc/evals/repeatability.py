"""Repeated runs and spreads — harness spec §2.4.

**A suite run is a sample. This module is what turns samples into a number.**

The education and real-estate suites are 17 and 23 cases, graded partly by a
model, driven by a model. Four consecutive runs of real-estate over one
unchanged commit read 52.2%, 42.9%, 39.1% and 45.5%. Each was reported as
*the* figure at the time, and two of them were used to reason about whether a
change had helped. Neither conclusion was supportable: the spread between them
is wider than any change measured that day.

So every metric here is carried as the tuple of values it took, and it renders
as `44.9% (39.1–52.2, n=4)`. The run count is not decoration — a mean without
one is a point estimate with extra steps.

Two rules the types enforce rather than document:

- **A single run is never measurable.** One sample has zero spread, and zero
  spread is exactly what a settled metric looks like. `n=1` is the shape of
  the mistake, so `measurable` refuses it outright.
- **Unmeasured is not zero.** A metric no run fed reports "not measured", the
  same distinction `GateResult.rate` draws for the same reason: 0.0% is what
  four of the five commercial gates print when they pass.
"""

import statistics
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from moc.config_store import load

_REPEAT = "evals/repeat"
_PERCENT = 100.0
_MIN_RUNS = 2

#: One suite run's metrics: name -> value, or None where that run measured
#: nothing. A key absent and a key present as None mean the same thing.
RunMetrics = Mapping[str, float | None]


def default_runs() -> int:
    return int(load(_REPEAT)["runs"])


@dataclass(frozen=True)
class MetricSpread:
    """Every value one metric took across N runs, and what that permits saying."""

    metric: str
    values: tuple[float | None, ...]

    @staticmethod
    def measurable_spread_pp() -> float:
        """The bar above which a metric is reported as not yet measurable."""
        return float(load(_REPEAT)["measurable_spread_pp"])

    @property
    def measured(self) -> tuple[float, ...]:
        return tuple(v for v in self.values if v is not None)

    @property
    def runs(self) -> int:
        """Runs that actually measured this metric."""
        return len(self.measured)

    @property
    def attempts(self) -> int:
        """Runs that could have. Kept distinct so a gate that fed twice in
        three runs reports n=2 of 3 rather than quietly becoming n=2."""
        return len(self.values)

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.measured) if self.measured else None

    @property
    def minimum(self) -> float | None:
        return min(self.measured) if self.measured else None

    @property
    def maximum(self) -> float | None:
        return max(self.measured) if self.measured else None

    @property
    def spread_pp(self) -> float:
        """Min-max in percentage points. Zero when fewer than two runs fed it,
        which is why `measurable` cannot be derived from this alone."""
        if self.runs < _MIN_RUNS:
            return 0.0
        return (self.maximum - self.minimum) * _PERCENT

    @property
    def measurable(self) -> bool:
        return self.runs >= _MIN_RUNS and self.spread_pp <= self.measurable_spread_pp()

    def render(self) -> str:
        if not self.measured:
            return f"{self.metric:26} not measured  (0 of {self.attempts} runs)"
        body = (
            f"{self.mean * _PERCENT:.1f}% "
            f"({self.minimum * _PERCENT:.1f}–{self.maximum * _PERCENT:.1f}, {self._n()})"
        )
        flag = "" if self.measurable else "  ← not measurable at this suite size"
        return f"{self.metric:26} {body}{flag}"

    def _n(self) -> str:
        if self.runs == self.attempts:
            return f"n={self.runs}"
        return f"n={self.runs} of {self.attempts} runs"

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "values": list(self.values),
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "spread_pp": self.spread_pp,
            "runs": self.runs,
            "attempts": self.attempts,
            "measurable": self.measurable,
        }


def aggregate(runs: Sequence[RunMetrics]) -> dict[str, MetricSpread]:
    """Fold N runs' metrics into one spread each.

    A metric any run reported appears for every run: a key missing from one
    run's dict is that run failing to measure it, not the metric ceasing to
    exist, and dropping the run from the denominator would hide exactly the
    inconsistency worth seeing.
    """
    names = sorted({name for run in runs for name in run})
    return {
        name: MetricSpread(metric=name, values=tuple(run.get(name) for run in runs))
        for name in names
    }


async def repeat(
    run_once: Callable[[], Awaitable[RunMetrics]], *, times: int | None = None
) -> dict[str, MetricSpread]:
    """Run one suite `times` times and aggregate. Defaults to config's `runs`.

    Refuses a single run rather than returning a spread of one. A caller that
    genuinely wants one sample should call `run_once` itself and report it as
    a sample — the refusal is here so that "just run it once for now" cannot
    arrive dressed as a measurement.
    """
    times = default_runs() if times is None else times
    if times < _MIN_RUNS:
        raise ValueError(f"a spread needs at least {_MIN_RUNS} runs, got {times}")
    return aggregate([await run_once() for _ in range(times)])


def render_all(spreads: Mapping[str, MetricSpread]) -> list[str]:
    """Every metric, one per line, in name order."""
    return [spreads[name].render() for name in sorted(spreads)]


def unmeasurable(spreads: Mapping[str, MetricSpread]) -> tuple[str, ...]:
    """Metrics too wide to compare against. The list a reader needs before
    treating any delta in the report as signal."""
    return tuple(
        name for name, s in sorted(spreads.items()) if s.measured and not s.measurable
    )


__all__ = [
    "MetricSpread",
    "RunMetrics",
    "aggregate",
    "default_runs",
    "render_all",
    "repeat",
    "unmeasurable",
]
