"""Eval report writer — harness spec §6.

Two artifacts from one run:

- **JSON** — the machine record. It carries the run metadata, so the next run
  can decide whether a comparison against it is even valid (§2.3), and it is
  what lands in `evals/baselines/<sha>.json`.
- **Markdown** — the CI job page. Grouped by category, because "accuracy fell
  two points" is not actionable and "adversarial_figures fell from 19/20 to
  12/20" is.

Every threshold is read from `config/evals/gates.yaml`, never written here
(design doc §19). Tightening a gate is then a config edit with an audit trail,
and it moves `config_hash`, which correctly invalidates comparison against a
baseline measured under the old bar.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moc.config_store import load
from moc.evals.deterministic import CheckResult
from moc.evals.run_metadata import RunMetadata

_GATES = "evals/gates"

#: Which gates the judge feeds, and therefore which ones see fewer turns than
#: the suite has. Stated in the artifact rather than in a docstring: a reader
#: comparing a register rate against overall accuracy is comparing two
#: different populations, and nothing in the table says so.
_STAGE_TWO_GATES = ("register_accuracy", "forbidden_claim_violations")

_STAGE_TWO_NOTE = (
    "> **Coverage:** `"
    + "` and `".join(_STAGE_TWO_GATES)
    + "` are graded by the judge, which runs only on turns that passed stage 1."
    " Their observation counts are therefore smaller than the suite's, and a"
    " turn that failed a deterministic check contributes nothing to them"
    " rather than a pass.",
    "",
)
_PERCENT = 100.0


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    vertical: str
    category: str
    passed: bool
    checks: tuple[CheckResult, ...] = ()


@dataclass(frozen=True)
class Tally:
    total: int
    passed: int

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class GateResult:
    name: str
    kind: str
    direction: str
    threshold: float
    value: float | None
    observations: int
    passed: bool
    description: str = ""

    @property
    def evaluated(self) -> bool:
        """False when no case fed this metric.

        Reported distinctly from a pass. A gate nothing exercised is unmeasured,
        and printing it as green is how a suite claims coverage it does not have.
        """
        return self.observations > 0


@dataclass(frozen=True)
class BaselineComparison:
    """§2.1's regression gate, subject to §2.3's comparability rule."""

    sha: str | None
    comparable: bool
    tolerance_pp: float
    delta_pp: float | None = None
    baseline_accuracy: float | None = None
    regressed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Report:
    run: RunMetadata
    overall_accuracy: float
    cases: tuple[CaseResult, ...]
    by_category: dict[str, Tally]
    by_vertical: dict[str, Tally]
    hard_gates: tuple[GateResult, ...]
    soft_gates: tuple[GateResult, ...]
    tracked: tuple[str, ...]
    baseline: BaselineComparison | None = None
    failed_cases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Hard gates and the baseline regression block a merge. Soft gates warn."""
        if any(not gate.passed for gate in self.hard_gates):
            return False
        return not (self.baseline and self.baseline.regressed)

    def to_json(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "overall_accuracy": self.overall_accuracy,
            "passed": self.passed,
            "cases": [
                {
                    "case_id": c.case_id,
                    "vertical": c.vertical,
                    "category": c.category,
                    "passed": c.passed,
                    "checks": [
                        {
                            "name": k.name,
                            "metric": k.metric,
                            "passed": k.passed,
                            "skipped": k.skipped,
                            "detail": k.detail,
                        }
                        for k in c.checks
                    ],
                }
                for c in self.cases
            ],
            "by_category": {k: _tally_json(v) for k, v in self.by_category.items()},
            "by_vertical": {k: _tally_json(v) for k, v in self.by_vertical.items()},
            "hard_gates": [_gate_json(g) for g in self.hard_gates],
            "soft_gates": [_gate_json(g) for g in self.soft_gates],
            "tracked": list(self.tracked),
            "baseline": _baseline_json(self.baseline),
        }

    def to_markdown(self) -> str:
        return "\n".join(self._markdown_lines())

    def _markdown_lines(self) -> list[str]:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Eval run {self.run.git_sha} — {verdict}",
            "",
            f"- Overall accuracy: **{_pct(self.overall_accuracy)}** "
            f"({sum(t.passed for t in self.by_category.values())}/{len(self.cases)} cases)",
            f"- config_hash: `{self.run.config_hash}` · lexicon v{self.run.lexicon_version}",
            f"- Tasks: {_task_summary(self.run)}",
            "",
            *self._baseline_lines(),
            "## Hard gates",
            "",
            *_gate_table(self.hard_gates),
            "",
            "## Soft gates",
            "",
            *_gate_table(self.soft_gates),
            "",
            f"Tracked, never gated: {', '.join(self.tracked)}",
            "",
            *_STAGE_TWO_NOTE,
            "## By category",
            "",
            "| Category | Passed | Total | Accuracy |",
            "|---|---|---|---|",
            *(
                f"| {name} | {t.passed} | {t.total} | {_pct(t.accuracy)} |"
                for name, t in sorted(self.by_category.items())
            ),
            "",
            "## By vertical",
            "",
            "| Vertical | Passed | Total | Accuracy |",
            "|---|---|---|---|",
            *(
                f"| {name} | {t.passed} | {t.total} | {_pct(t.accuracy)} |"
                for name, t in sorted(self.by_vertical.items())
            ),
            "",
        ]
        if self.failed_cases:
            lines += ["## Failed cases", "", *(f"- `{c}`" for c in self.failed_cases), ""]
        return lines

    def _baseline_lines(self) -> list[str]:
        if self.baseline is None:
            return ["## Baseline", "", "No baseline for this run.", ""]
        if not self.baseline.comparable:
            # §2.3: state why, and print no delta at all. A number with a caveat
            # beside it gets read as a number.
            return ["## Baseline", "", f"Comparison refused — {self.baseline.error}", ""]
        verdict = "REGRESSION" if self.baseline.regressed else "within tolerance"
        return [
            "## Baseline",
            "",
            f"- Baseline accuracy: {_pct(self.baseline.baseline_accuracy)}",
            f"- Delta: {self.baseline.delta_pp:+.2f} pp "
            f"(tolerance {self.baseline.tolerance_pp} pp) — {verdict}",
            "",
        ]


# ─────────────────────────────── building ───────────────────────────────


def build_report(
    run: RunMetadata,
    cases: Sequence[CaseResult],
    baseline: dict[str, Any] | None = None,
) -> Report:
    """Aggregate case results into the run's two artifacts."""
    cases = tuple(cases)
    gates = load(_GATES)
    observations = _observations(cases)

    passed_count = sum(case.passed for case in cases)
    overall = passed_count / len(cases) if cases else 0.0

    return Report(
        run=run,
        overall_accuracy=overall,
        cases=cases,
        by_category=_group(cases, lambda c: c.category),
        by_vertical=_group(cases, lambda c: c.vertical),
        hard_gates=_gates(gates["hard_gates"], "hard", observations),
        soft_gates=_gates(gates["soft_gates"], "soft", observations),
        tracked=tuple(gates["tracked"]),
        baseline=_compare(run, overall, baseline, gates["baseline"]),
        failed_cases=tuple(c.case_id for c in cases if not c.passed),
    )


def _observations(cases: Sequence[CaseResult]) -> dict[str, list[bool]]:
    """Per metric, the pass/fail of every check that fed it.

    Skipped checks are excluded, not counted as passes — that is what keeps an
    unexercised gate reading as unmeasured rather than perfect.
    """
    seen: dict[str, list[bool]] = {}
    for case in cases:
        for check in case.checks:
            if check.skipped:
                continue
            seen.setdefault(check.metric, []).append(check.passed)
    return seen


def _group(cases: Sequence[CaseResult], key) -> dict[str, Tally]:
    grouped: dict[str, list[CaseResult]] = {}
    for case in cases:
        grouped.setdefault(key(case), []).append(case)
    return {
        name: Tally(total=len(group), passed=sum(c.passed for c in group))
        for name, group in grouped.items()
    }


def _gates(spec: dict[str, Any], kind: str, observations: dict[str, list[bool]]) -> tuple:
    return tuple(
        _gate(name, config, kind, observations.get(name, [])) for name, config in spec.items()
    )


def _gate(name: str, config: dict[str, Any], kind: str, results: list[bool]) -> GateResult:
    """Compute one metric and test it against its configured threshold.

    `direction` decides what the metric even means: a `min` gate measures the
    share of checks that passed, a `max` gate the share that failed. Encoding
    it in config rather than in a lookup here means adding a metric is a config
    edit, not a code change plus a config edit.
    """
    direction, threshold = config["direction"], config["threshold"]
    if not results:
        return GateResult(
            name=name,
            kind=kind,
            direction=direction,
            threshold=threshold,
            value=None,
            observations=0,
            passed=True,
            description=config.get("description", ""),
        )
    if direction == "min":
        value = sum(results) / len(results)
        passed = value >= threshold
    else:
        value = sum(not r for r in results) / len(results)
        passed = value <= threshold
    return GateResult(
        name=name,
        kind=kind,
        direction=direction,
        threshold=threshold,
        value=value,
        observations=len(results),
        passed=passed,
        description=config.get("description", ""),
    )


def _compare(
    run: RunMetadata,
    accuracy: float,
    baseline: dict[str, Any] | None,
    config: dict[str, Any],
) -> BaselineComparison | None:
    """§2.1's regression gate, gated in turn by §2.3's comparability rule."""
    if baseline is None:
        return None
    tolerance = config["max_accuracy_regression_pp"]
    previous = RunMetadata.from_dict(baseline["run"])

    error = run.comparability_error(previous)
    if error is not None:
        return BaselineComparison(
            sha=previous.git_sha, comparable=False, tolerance_pp=tolerance, error=error
        )

    baseline_accuracy = baseline["overall_accuracy"]
    delta_pp = (accuracy - baseline_accuracy) * _PERCENT
    return BaselineComparison(
        sha=previous.git_sha,
        comparable=True,
        tolerance_pp=tolerance,
        delta_pp=delta_pp,
        baseline_accuracy=baseline_accuracy,
        regressed=delta_pp < -tolerance,
    )


# ─────────────────────────────── artifacts ───────────────────────────────


def load_baseline(git_sha: str, *, directory: Path | str) -> dict[str, Any] | None:
    """Read `evals/baselines/<sha>.json`, or None when there is no baseline yet."""
    path = Path(directory) / f"{git_sha}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(report: Report, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_markdown(report: Report, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return path


# ─────────────────────────────── formatting ───────────────────────────────


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * _PERCENT:.1f}%"


def _task_summary(run: RunMetadata) -> str:
    """§2.3 records provider and model per task, so the report shows them per task."""
    return ", ".join(
        f"{t.task}={t.provider}/{t.model}@{t.prompt_version}" for t in run.tasks
    )


def _gate_table(gates: Sequence[GateResult]) -> list[str]:
    rows = ["| Metric | Value | Threshold | Status |", "|---|---|---|---|"]
    for gate in sorted(gates, key=lambda g: g.name):
        if not gate.evaluated:
            status, value = "not measured", "n/a"
        else:
            status = "pass" if gate.passed else "FAIL"
            value = _pct(gate.value)
        rows.append(f"| `{gate.name}` | {value} | {gate.direction} {gate.threshold} | {status} |")
    return rows


def _tally_json(tally: Tally) -> dict[str, Any]:
    return {"total": tally.total, "passed": tally.passed, "accuracy": tally.accuracy}


def _gate_json(gate: GateResult) -> dict[str, Any]:
    return {
        "name": gate.name,
        "kind": gate.kind,
        "direction": gate.direction,
        "threshold": gate.threshold,
        "value": gate.value,
        "observations": gate.observations,
        "evaluated": gate.evaluated,
        "passed": gate.passed,
        "description": gate.description,
    }


def _baseline_json(baseline: BaselineComparison | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    return {
        "sha": baseline.sha,
        "comparable": baseline.comparable,
        "tolerance_pp": baseline.tolerance_pp,
        "delta_pp": baseline.delta_pp,
        "baseline_accuracy": baseline.baseline_accuracy,
        "regressed": baseline.regressed,
        "error": baseline.error,
    }
