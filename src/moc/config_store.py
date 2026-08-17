"""The one place platform config is read (design doc §19).

Callers ask for a logical name — `config_store.load("arabic/lexicon")` — and
never learn that YAML is involved. §19 commits to YAML (P0-P1) -> database table
(P3) -> admin console (P4), with each step a one-file change; that only holds
while this module is the sole reader. Two tests in tests/test_config_store.py
enforce it against the whole of src/moc.

`config_hash()` identifies the config a result was produced under. The eval
suite pins it (§19.4): a regression measured against a different lexicon is not
a measurement, and "the bot got worse yesterday" needs an answer that survives
someone having edited a synonym list.
"""

import hashlib
import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

_SUFFIX = ".yaml"
_ENV_ROOT = "MOC_CONFIG_DIR"


def default_root() -> Path:
    """Where platform config lives. Overridable so tests never touch the real tree."""
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "config"


def load(name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Return one config document by logical name, e.g. "arabic/lexicon".

    The result is cached and shared. Treat it as read-only — mutating it would
    change what every other caller sees, and in P3 there will be no local dict
    to mutate anyway.
    """
    return _load_cached(name, root or default_root())


def config_hash(*, root: Path | None = None) -> str:
    """Stable digest of the whole config tree.

    Derived from relative paths and file bytes only — no mtimes, no absolute
    paths — so the same content hashes identically on a developer's machine, in
    CI, and on the VPS. Two eval runs sharing this hash saw the same lexicon.
    """
    return _hash_cached(root or default_root())


def clear_cache() -> None:
    _load_cached.cache_clear()
    _hash_cached.cache_clear()


@cache
def _load_cached(name: str, root: Path) -> dict[str, Any]:
    path = root / f"{name}{_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"no config document named {name!r} (looked in {path})")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"config {name!r} must be a mapping, got {type(document).__name__}")
    return document


@cache
def _hash_cached(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob(f"*{_SUFFIX}"), key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
