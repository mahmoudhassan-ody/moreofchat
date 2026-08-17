"""The CI workflow declares the job matrix from eval-harness-spec §6.1.

A workflow is only validated by pushing it, which makes a YAML typo or a
silently-deleted job expensive to find. These assertions are cheap and run with
every other test.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yaml"

REQUIRED_JOBS = {
    "lint",
    "unit",
    "migration-drift",
    "eval-smoke",
    "eval-full",
    "eval-judge",
}


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def triggers(workflow) -> dict:
    # PyYAML resolves the bare key `on` to the boolean True (YAML 1.1).
    return workflow.get("on", workflow.get(True))


def test_workflow_parses(workflow):
    assert workflow["name"]


def test_runs_on_every_push(triggers):
    assert "push" in triggers


def test_runs_on_pull_requests_to_main(triggers):
    assert "main" in triggers["pull_request"]["branches"]


def test_has_a_nightly_schedule(triggers):
    """§6.1: nightly on main, full suite, trend report."""
    assert triggers["schedule"]


def test_declares_every_required_job(workflow):
    assert REQUIRED_JOBS <= set(workflow["jobs"])


def test_migration_drift_job_runs_the_guard(workflow):
    """Task 6's finding: a generated migration must never drop the billing ledger."""
    steps = yaml.dump(workflow["jobs"]["migration-drift"])
    assert "autogenerate" in steps
    assert "check_migration_drift.py" in steps


def _ci_sources(*, strip_comments: bool = False) -> str:
    """The workflow plus the composite actions it calls — the whole CI surface.

    `strip_comments` for checks about what CI *declares*: a comment explaining
    why an image is not restated here would otherwise read as restating it.
    """
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / ".github").rglob("*.yaml"))
    )
    if not strip_comments:
        return text
    lines = (line for line in text.splitlines() if not line.strip().startswith("#"))
    return "\n".join(line.split("#", maxsplit=1)[0] for line in lines)


def test_infra_comes_from_compose_not_a_second_service_definition(workflow):
    """§6.1 services must mirror compose.yaml — the cheapest mirror is compose itself.

    A `services:` block duplicating postgres/valkey/qdrant/meilisearch would
    drift from compose.yaml the first time either is edited alone.
    """
    assert "docker compose" in _ci_sources()
    for job in workflow["jobs"].values():
        assert "services" not in job, "infra is declared once, in compose.yaml"


def test_compose_images_are_not_restated_in_ci():
    declared = _ci_sources(strip_comments=True)
    for image in ("postgres:18", "valkey/valkey:9", "qdrant/qdrant", "getmeili/meilisearch"):
        assert image not in declared, f"{image} is pinned in compose.yaml; do not restate it"


def test_skipped_security_tests_are_reported_not_silent(workflow):
    """MOC_PUBLIC_IP is unset in CI, so the external-exposure tests skip.

    A security test that skips unnoticed is worse than no test: it reads as
    coverage. The suite must run with -rs so skips are printed, and a dedicated
    step must state the gap.
    """
    body = yaml.dump(workflow["jobs"])
    assert "-rs" in body
    assert "MOC_PUBLIC_IP" in body


def test_judge_job_is_gated_so_week_two_is_additive(workflow):
    """The judge needs provider credentials that do not exist yet."""
    judge = workflow["jobs"]["eval-judge"]
    assert "if" in judge, "the judge job must be conditional until credentials exist"


def test_nothing_depends_on_the_judge_job(workflow):
    """Adding the judge must be a new job, not a restructure of the existing ones."""
    for name, job in workflow["jobs"].items():
        needs = job.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        assert "eval-judge" not in needs, f"{name} would block on the week-2 judge"


def test_eval_full_does_baseline_comparison(workflow):
    body = yaml.dump(workflow["jobs"]["eval-full"])
    assert "baseline" in body.lower()


def test_sensitive_paths_from_the_spec_trigger_the_full_suite(workflow):
    """§6.1: a config or lexicon edit can regress retrieval as badly as a prompt edit."""
    body = yaml.dump(workflow["jobs"])
    for path in ("agent/prompts", "retrieval", "arabic", "config"):
        assert path in body


def test_git_sha_is_supplied_by_ci(workflow):
    """Task 8 kept subprocess out of the measurement path; CI provides the sha.

    Workflow-level so every job sees it — a per-job copy is one more place for
    an eval job to be added without it.
    """
    assert workflow["env"]["MOC_GIT_SHA"] == "${{ github.sha }}"
