"""Parse and validate eval case files.

Every failure here raises `ValueError` naming the offending case id. A case file
grows to 150 entries; a pydantic error path like `12.turns.0.expected_action` is
not enough to find the case in a review.
"""

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from moc.evals.schema import EvalCase

# Spec §3.1: never store a golden answer string. Grading wording turns the suite
# into a paraphrase detector and blocks legitimate improvements — so this is a
# load error, caught by name rather than left to extra="forbid" to report as a
# generic unknown field.
GOLDEN_ANSWER_KEYS = frozenset(
    {"expected_reply", "expected_response", "expected_text", "golden_answer", "golden_reply"}
)


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load, validate and return the cases in one YAML file, in file order."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of cases, got {type(raw).__name__}")

    cases = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}[{index}]: expected a case mapping, got {type(entry).__name__}"
            )
        label = entry.get("id") or f"[{index}]"
        _reject_golden_answers(entry, path, label)
        try:
            cases.append(EvalCase.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(f"{path}: case {label} is invalid — {exc}") from exc

    _reject_duplicate_ids(cases, path)
    return cases


def _reject_golden_answers(node: Any, path: Path, label: str) -> None:
    """Walk the whole case; a golden string can hide at any nesting depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in GOLDEN_ANSWER_KEYS:
                raise ValueError(
                    f"{path}: case {label} sets {key!r}. Cases grade facts and behaviour, "
                    f"never wording (spec §3.1) — express it as an expected_fact instead."
                )
            _reject_golden_answers(value, path, label)
    elif isinstance(node, list):
        for item in node:
            _reject_golden_answers(item, path, label)


def _reject_duplicate_ids(cases: list[EvalCase], path: Path) -> None:
    """Ids are stable and never reused (§3) — a duplicate breaks trend comparability."""
    dupes = sorted(id_ for id_, n in Counter(c.id for c in cases).items() if n > 1)
    if dupes:
        raise ValueError(f"{path}: duplicate case id(s): {', '.join(dupes)}")
