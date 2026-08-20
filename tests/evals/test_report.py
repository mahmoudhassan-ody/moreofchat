"""Report writer — harness spec §6, gates from §2.1-2.2, config pinning from §2.3."""

import json

import pytest

from moc.evals.deterministic import CheckResult
from moc.evals.report import (
    CaseResult,
    build_report,
    load_baseline,
    write_json,
    write_markdown,
)
from moc.evals.run_metadata import RunMetadata, TaskBinding

TASKS = (
    TaskBinding(task="answer", prompt_version="p7", provider="bedrock", model="claude-opus-5"),
)


def run(config_hash: str = "cfg1", git_sha: str = "abc1234") -> RunMetadata:
    return RunMetadata(
        git_sha=git_sha, config_hash=config_hash, lexicon_version=1, tasks=TASKS
    )


def case(
    case_id: str,
    *,
    category: str = "factual_retrieval",
    vertical: str = "education",
    passed: bool = True,
    checks: tuple[CheckResult, ...] = (),
) -> CaseResult:
    return CaseResult(
        case_id=case_id, vertical=vertical, category=category, passed=passed, checks=checks
    )


def check(metric: str, passed: bool) -> CheckResult:
    return CheckResult(name=metric, metric=metric, passed=passed)


# ─────────────────────────── accuracy and grouping ───────────────────────────


def test_overall_accuracy_is_cases_passed_over_cases_run():
    report = build_report(run(), [case("a"), case("b"), case("c", passed=False)])
    assert report.overall_accuracy == pytest.approx(2 / 3)


def test_groups_by_category_so_a_regression_is_attributable():
    report = build_report(
        run(),
        [
            case("a", category="factual_retrieval"),
            case("b", category="adversarial_figures", passed=False),
            case("c", category="adversarial_figures", passed=False),
        ],
    )
    assert report.by_category["adversarial_figures"].total == 2
    assert report.by_category["adversarial_figures"].passed == 0
    assert report.by_category["factual_retrieval"].accuracy == pytest.approx(1.0)


def test_groups_by_vertical_too():
    report = build_report(run(), [case("a"), case("b", vertical="realestate", passed=False)])
    assert report.by_vertical["realestate"].accuracy == pytest.approx(0.0)


def test_empty_run_does_not_divide_by_zero():
    assert build_report(run(), []).overall_accuracy == pytest.approx(0.0)


# ─────────────────────────── gates ───────────────────────────


def test_hard_and_soft_gates_are_reported_separately():
    report = build_report(run(), [case("a", checks=(check("expected_action_accuracy", True),))])
    assert "expected_action_accuracy" in {g.name for g in report.hard_gates}
    assert "slot_retention_accuracy" in {g.name for g in report.soft_gates}
    assert {g.name for g in report.hard_gates} & {g.name for g in report.soft_gates} == set()


def test_thresholds_come_from_config_not_from_literals():
    from moc import config_store

    gates = config_store.load("evals/gates")
    report = build_report(run(), [case("a")])
    by_name = {g.name: g for g in report.hard_gates}
    assert by_name["expected_action_accuracy"].threshold == (
        gates["hard_gates"]["expected_action_accuracy"]["threshold"]
    )


def test_min_direction_gate_fails_below_threshold():
    checks = tuple(check("expected_action_accuracy", passed) for passed in (True, False))
    report = build_report(run(), [case("a", checks=checks)])
    gate = next(g for g in report.hard_gates if g.name == "expected_action_accuracy")
    assert gate.value == pytest.approx(0.5)
    assert gate.passed is False


def test_max_direction_gate_fails_above_zero():
    report = build_report(
        run(), [case("a", checks=(check("hallucinated_figure_rate", False),))]
    )
    gate = next(g for g in report.hard_gates if g.name == "hallucinated_figure_rate")
    assert gate.value == pytest.approx(1.0)
    assert gate.passed is False


def test_hedged_and_hallucinated_gates_are_both_present_and_separate():
    report = build_report(run(), [case("a")])
    names = {g.name for g in report.hard_gates}
    assert {"hallucinated_figure_rate", "hedged_figure_rate"} <= names


def test_gate_with_no_observations_is_not_evaluated():
    """A metric nothing fed must not read as a perfect score."""
    report = build_report(run(), [case("a")])
    gate = next(g for g in report.hard_gates if g.name == "retrieval_recall_at_5")
    assert gate.evaluated is False
    assert gate.passed is True


def test_tracked_metrics_are_never_gated():
    """§2.2: gating containment creates pressure to answer instead of hand off."""
    report = build_report(run(), [case("a")])
    gated = {g.name for g in report.hard_gates} | {g.name for g in report.soft_gates}
    assert "containment_rate" in report.tracked
    assert "containment_rate" not in gated


def test_passed_reflects_hard_gates_only():
    """A soft gate warns; it does not block a merge."""
    soft_failure = tuple(check("slot_retention_accuracy", False) for _ in range(1))
    report = build_report(run(), [case("a", checks=soft_failure)])
    assert next(g for g in report.soft_gates if g.name == "slot_retention_accuracy").passed is False
    assert report.passed is True


def test_a_failing_hard_gate_fails_the_report():
    report = build_report(run(), [case("a", checks=(check("hallucinated_figure_rate", False),))])
    assert report.passed is False


# ─────────────────────────── baseline comparison ───────────────────────────


def test_no_baseline_means_no_comparison():
    report = build_report(run(), [case("a")])
    assert report.baseline is None


def test_accuracy_delta_against_a_comparable_baseline():
    baseline = build_report(run(), [case("a"), case("b")]).to_json()
    report = build_report(run(), [case("a"), case("b", passed=False)], baseline=baseline)
    assert report.baseline.delta_pp == pytest.approx(-50.0)
    assert report.baseline.regressed is True


def test_regression_within_tolerance_is_not_flagged():
    passing = [case(str(i)) for i in range(100)]
    baseline = build_report(run(), passing).to_json()
    slightly_worse = [case(str(i), passed=i > 0) for i in range(100)]
    report = build_report(run(), slightly_worse, baseline=baseline)
    assert report.baseline.delta_pp == pytest.approx(-1.0)
    assert report.baseline.regressed is False


def test_regression_tolerance_comes_from_config():
    from moc import config_store

    tolerance = config_store.load("evals/gates")["baseline"]["max_accuracy_regression_pp"]
    baseline = build_report(run(), [case("a")]).to_json()
    report = build_report(run(), [case("a")], baseline=baseline)
    assert report.baseline.tolerance_pp == tolerance


def test_config_change_refuses_the_comparison_instead_of_showing_a_delta():
    """§2.3: a delta across a config change is not a measurement."""
    baseline = build_report(run(config_hash="old"), [case("a"), case("b")]).to_json()
    report = build_report(run(config_hash="new"), [case("a", passed=False)], baseline=baseline)
    assert report.baseline.comparable is False
    assert report.baseline.delta_pp is None
    assert "config" in report.baseline.error


def test_incomparable_baseline_never_reports_a_regression():
    baseline = build_report(run(config_hash="old"), [case("a")]).to_json()
    report = build_report(run(config_hash="new"), [case("a", passed=False)], baseline=baseline)
    assert report.baseline.regressed is False


def test_load_baseline_reads_by_git_sha(tmp_path):
    payload = build_report(run(git_sha="deadbee"), [case("a")]).to_json()
    (tmp_path / "deadbee.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_baseline("deadbee", directory=tmp_path) == payload


def test_load_baseline_returns_none_when_absent(tmp_path):
    assert load_baseline("nosuchsha", directory=tmp_path) is None


# ─────────────────────────── artifacts ───────────────────────────


def test_json_artifact_carries_run_metadata_for_the_next_comparison():
    payload = build_report(run(), [case("a")]).to_json()
    assert payload["run"]["config_hash"] == "cfg1"
    assert payload["run"]["lexicon_version"] == 1
    assert payload["run"]["tasks"][0]["model"] == "claude-opus-5"


def test_json_artifact_round_trips_through_a_file(tmp_path):
    report = build_report(run(), [case("a"), case("b", passed=False)])
    path = write_json(report, tmp_path / "out.json")
    assert json.loads(path.read_text(encoding="utf-8"))["overall_accuracy"] == pytest.approx(0.5)


def test_markdown_groups_by_category():
    report = build_report(
        run(), [case("a", category="adversarial_figures", passed=False), case("b")]
    )
    markdown = report.to_markdown()
    assert "adversarial_figures" in markdown
    assert "factual_retrieval" in markdown


def test_markdown_separates_hard_from_soft_gates():
    markdown = build_report(run(), [case("a")]).to_markdown()
    assert "Hard gates" in markdown
    assert "Soft gates" in markdown


def test_markdown_lists_failing_cases_so_ci_is_actionable():
    report = build_report(run(), [case("edu-0005", passed=False), case("edu-0001")])
    markdown = report.to_markdown()
    assert "edu-0005" in markdown
    assert "edu-0001" not in markdown.split("Failed cases")[-1]


def test_markdown_states_the_comparability_error_and_no_delta():
    baseline = build_report(run(config_hash="old"), [case("a")]).to_json()
    report = build_report(run(config_hash="new"), [case("a", passed=False)], baseline=baseline)
    markdown = report.to_markdown()
    assert "config" in markdown
    assert "%" not in markdown.split("Baseline")[-1].split("\n\n")[0]


def test_markdown_records_the_config_hash_so_a_reader_can_check_pinning():
    markdown = build_report(run(), [case("a")]).to_markdown()
    assert "cfg1" in markdown


def test_write_markdown_creates_the_file(tmp_path):
    path = write_markdown(build_report(run(), [case("a")]), tmp_path / "out.md")
    assert path.read_text(encoding="utf-8").startswith("#")


def test_the_stage_two_gates_carry_their_coverage_limit_in_the_report():
    """`register_accuracy` and `forbidden_claim_violations` are graded by the
    judge, and the judge runs only on turns that passed stage 1.

    Their denominators are therefore smaller than the suite's, and a reader
    comparing 100% register against 49% accuracy is comparing two different
    populations. Stated in the artifact rather than in a docstring nobody
    reading the report will open.
    """
    from moc.evals.deterministic import CheckResult

    report = build_report(
        run(),
        [
            case("edu-1", checks=(CheckResult("register", "register_accuracy", True),)),
            case("edu-2", passed=False),
        ],
    )
    markdown = report.to_markdown()

    assert "stage 1" in markdown
    assert "register_accuracy" in markdown
    assert "forbidden_claim_violations" in markdown
