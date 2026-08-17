"""The config surface from design doc §19.

Callers never open YAML. The P3 move to a database table has to be a change to
config_store and nothing else, which only holds if nothing else reads the files.
"""

from pathlib import Path

import pytest

from moc import config_store

REPO_ROOT = Path(__file__).parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def test_loads_the_arabic_lexicon():
    lexicon = config_store.load("arabic/lexicon")
    assert isinstance(lexicon, dict)
    assert lexicon["version"]


def test_lexicon_has_every_required_section():
    """§19.2 names these. A missing section means a literal went back into code."""
    lexicon = config_store.load("arabic/lexicon")
    for section in (
        "digits",
        "separators",
        "unit_multipliers",
        "fractions",
        "conjunctions",
        "ordinal_markers",
        "floor_markers",
        "place_markers",
        "approximation_markers",
        "currency_markers",
        "percent_markers",
        "count_markers",
        "year",
        "franco_transliteration",
    ):
        assert section in lexicon, f"lexicon.yaml is missing {section!r}"


def test_unknown_config_name_raises():
    with pytest.raises(FileNotFoundError, match="arabic/nope"):
        config_store.load("arabic/nope")


def test_result_is_cached():
    assert config_store.load("arabic/lexicon") is config_store.load("arabic/lexicon")


def test_clear_cache_forces_a_reload():
    first = config_store.load("arabic/lexicon")
    config_store.clear_cache()
    assert config_store.load("arabic/lexicon") is not first


def test_config_hash_is_stable_across_calls():
    assert config_store.config_hash() == config_store.config_hash()


def test_config_hash_changes_when_a_value_changes(tmp_path):
    (tmp_path / "a.yaml").write_text("version: 1\n", encoding="utf-8")
    before = config_store.config_hash(root=tmp_path)
    (tmp_path / "a.yaml").write_text("version: 2\n", encoding="utf-8")
    config_store.clear_cache()
    assert config_store.config_hash(root=tmp_path) != before


def test_config_hash_ignores_file_order_and_location(tmp_path):
    """Same content must hash the same on any machine — no absolute paths, no mtime."""
    one, two = tmp_path / "one", tmp_path / "two"
    for root in (one, two):
        (root / "arabic").mkdir(parents=True)
        (root / "arabic" / "lexicon.yaml").write_text("version: 1\n", encoding="utf-8")
        (root / "b.yaml").write_text("k: v\n", encoding="utf-8")
    config_store.clear_cache()
    first = config_store.config_hash(root=one)
    config_store.clear_cache()
    assert config_store.config_hash(root=two) == first


# evals/load.py parses case *data* supplied by the caller, not platform config.
# Case files are inputs to the harness; they do not move to a DB table in P3.
YAML_PARSING_ALLOWED_IN = frozenset({"config_store.py", "load.py"})


def test_only_config_store_parses_platform_config():
    """§19: swapping YAML for a DB table in P3 must be a one-module change."""
    offenders = []
    for path in (REPO_ROOT / "src" / "moc").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        parses_yaml = "yaml.safe_load" in source or "yaml.load" in source
        if parses_yaml and path.name not in YAML_PARSING_ALLOWED_IN:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"these parse YAML outside config_store: {offenders}"


def test_config_store_is_the_only_reader_of_the_config_directory():
    offenders = []
    for path in (REPO_ROOT / "src" / "moc").rglob("*.py"):
        if path.name == "config_store.py":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("#", maxsplit=1)[0]
            if "config/" in code or 'Path("config' in code:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {line.strip()}")
    assert offenders == [], f"these reach into config/ directly: {offenders}"
