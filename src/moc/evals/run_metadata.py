"""What every eval run records about the conditions it ran under (spec §2.3).

Config is not code, so it is a variable in every measurement. A regression
measured against a different Arabic lexicon than the baseline is not a
regression — it is two different experiments subtracted from each other.

This module is the seam: the report writer (Task 9) persists `to_dict()` to
`eval_runs`, and asks `comparability_error()` before printing any delta against
a baseline. `git_sha` is passed in rather than shelled out for, because CI
already knows it and a subprocess in the measurement path is a failure mode
nobody debugs at 2am.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moc import config_store

_LEXICON = "arabic/lexicon"


@dataclass(frozen=True)
class TaskBinding:
    """Spec §2.3 records prompt version, provider and model *per task*.

    One run answers with one model and judges with another (§5.2's judge
    independence rule), so a single provider field per run would erase exactly
    the distinction the cross-provider policy depends on.
    """

    task: str
    prompt_version: str
    provider: str
    model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "task": self.task,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
        }


@dataclass(frozen=True)
class RunMetadata:
    git_sha: str
    config_hash: str
    lexicon_version: Any
    tasks: tuple[TaskBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "lexicon_version": self.lexicon_version,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> RunMetadata:
        """Rebuild from a persisted artifact, so a baseline can be compared against."""
        return cls(
            git_sha=record["git_sha"],
            config_hash=record["config_hash"],
            lexicon_version=record["lexicon_version"],
            tasks=tuple(TaskBinding(**task) for task in record["tasks"]),
        )

    def is_comparable_to(self, baseline: RunMetadata) -> bool:
        """A baseline is comparable only under an identical config hash (§2.3)."""
        return self.config_hash == baseline.config_hash

    def comparability_error(self, baseline: RunMetadata) -> str | None:
        """Why a delta against `baseline` would be meaningless, or None if it holds.

        The report states this instead of the delta. Showing a number and a
        caveat means someone reads the number; refusing the number means they
        read the caveat.
        """
        if self.is_comparable_to(baseline):
            return None
        return (
            f"config changed since the baseline: run config_hash {self.config_hash} "
            f"!= baseline {baseline.config_hash}. A delta across a config change is "
            f"not a measurement — re-run the baseline on this config first (§2.3)."
        )


def capture(
    *,
    git_sha: str,
    tasks: tuple[TaskBinding, ...],
    config_root: Path | None = None,
) -> RunMetadata:
    """Snapshot the conditions of a run.

    `config_root` pins the suite to a fixture config tree rather than live
    tenant config (§2.3) — otherwise a tenant editing their area synonyms
    changes CI results for everyone.
    """
    return RunMetadata(
        git_sha=git_sha,
        config_hash=config_store.config_hash(root=config_root),
        lexicon_version=config_store.load(_LEXICON, root=config_root)["version"],
        tasks=tuple(tasks),
    )
