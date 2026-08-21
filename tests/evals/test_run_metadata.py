"""What every eval run records — eval-harness-spec §2.3."""

import pytest

from moc import config_store
from moc.evals.run_metadata import RunMetadata, TaskBinding, capture

TASKS = (
    TaskBinding(task="answer", prompt_version="p7", provider="bedrock", model="claude-opus-5"),
    TaskBinding(task="judge", prompt_version="j3", provider="openai", model="gpt-5"),
)


def test_capture_records_the_live_config_hash():
    run = capture(git_sha="abc1234", tasks=TASKS)
    assert run.config_hash == config_store.config_hash()


def test_capture_records_the_lexicon_version():
    run = capture(git_sha="abc1234", tasks=TASKS)
    assert run.lexicon_version == config_store.load("arabic/lexicon")["version"]


def test_records_every_field_the_spec_names():
    record = capture(git_sha="abc1234", tasks=TASKS).to_dict()
    assert set(record) == {
        "git_sha",
        "config_hash",
        "lexicon_version",
        "tasks",
        # Not in §2.3's list, because §2.3 was written when the judge always
        # ran. Whether it ran is a condition of the run like any other.
        "graded",
    }
    assert record["tasks"][0] == {
        "task": "answer",
        "prompt_version": "p7",
        "provider": "bedrock",
        "model": "claude-opus-5",
    }


def test_capture_reads_a_pinned_config_directory(tmp_path):
    """§2.3: the suite pins config, so a tenant editing synonyms cannot move CI."""
    (tmp_path / "arabic").mkdir()
    (tmp_path / "arabic" / "lexicon.yaml").write_text("version: 99\n", encoding="utf-8")
    config_store.clear_cache()
    run = capture(git_sha="abc1234", tasks=TASKS, config_root=tmp_path)
    assert run.lexicon_version == 99
    assert run.config_hash != config_store.config_hash()


def _run(config_hash: str, git_sha: str = "abc1234") -> RunMetadata:
    return RunMetadata(
        git_sha=git_sha, config_hash=config_hash, lexicon_version=1, tasks=TASKS
    )


def test_same_config_hash_is_comparable():
    assert _run("h1").is_comparable_to(_run("h1", git_sha="def5678")) is True


def test_different_config_hash_is_not_comparable():
    assert _run("h1").is_comparable_to(_run("h2")) is False


def test_incomparable_runs_explain_why_instead_of_showing_a_delta():
    """A misleading delta is worse than a refusal — that is the whole rule."""
    reason = _run("h1").comparability_error(_run("h2"))
    assert reason is not None
    assert "config" in reason
    assert "h1" in reason and "h2" in reason


def test_comparable_runs_have_no_error():
    assert _run("h1").comparability_error(_run("h1")) is None


def test_metadata_is_frozen():
    """A run record edited after the fact is not a record."""
    with pytest.raises(Exception):  # noqa: B017
        _run("h1").config_hash = "h2"


def test_the_prompts_that_live_under_src_are_recorded():
    """§2.3. `config_hash` covers `config/`; the extraction and composition
    prompts live under `src/`, so without their digests a prompt edit leaves a
    run claiming comparability it does not have.

    The judge's version travelled from the start. The other two did not, and
    the composition prompt was rewritten from nothing on 2026-08-20 — exactly
    the edit that must invalidate a baseline.
    """
    from moc.evals.runner import CaseRunner

    runner = CaseRunner(orchestrator=None, retriever=None, script="scripts/education/fees")
    run = runner.metadata(git_sha="abc1234")
    versions = {t.task: t.prompt_version for t in run.tasks}

    assert versions["answer_composition"].startswith("composition_v1+")
    assert versions["slot_extraction"].startswith("extraction_v1+")


def test_a_stage_one_run_is_not_comparable_to_a_graded_baseline():
    """§2.3's rule, applied to the judge being optional.

    A stage-1-only run's accuracy is higher than a judged one by exactly the
    failures stage 1 cannot see — register misses, ungrounded claims,
    forbidden claims. Subtracting it from a judged baseline reports the
    judge's absence as an improvement, and the PR gate would pass on it.
    """
    from moc.evals.run_metadata import RunMetadata

    graded = RunMetadata(git_sha="a", config_hash="h", lexicon_version=1, tasks=())
    cheap = RunMetadata(
        git_sha="b", config_hash="h", lexicon_version=1, tasks=(), graded=False
    )

    error = cheap.comparability_error(graded)
    assert error is not None
    assert "stage 2" in error


def test_two_stage_one_runs_are_comparable_to_each_other():
    from moc.evals.run_metadata import RunMetadata

    one = RunMetadata(
        git_sha="a", config_hash="h", lexicon_version=1, tasks=(), graded=False
    )
    two = RunMetadata(
        git_sha="b", config_hash="h", lexicon_version=1, tasks=(), graded=False
    )
    assert two.comparability_error(one) is None


def test_a_baseline_written_before_the_field_existed_reads_as_graded():
    """Every run persisted before stage 2 became optional ran the judge, so
    True is the honest default rather than a convenient one."""
    from moc.evals.run_metadata import RunMetadata

    restored = RunMetadata.from_dict(
        {"git_sha": "a", "config_hash": "h", "lexicon_version": 1, "tasks": []}
    )
    assert restored.graded is True
