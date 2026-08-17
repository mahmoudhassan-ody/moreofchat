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
    assert set(record) == {"git_sha", "config_hash", "lexicon_version", "tasks"}
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
