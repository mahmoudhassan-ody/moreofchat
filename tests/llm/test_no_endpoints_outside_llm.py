"""Design §2.4: a region change must be a new adapter, never a refactor.

That migration table is only true while every endpoint lives in one package.
The moment a `base_url` assumption leaks into the router or a call site, moving
Claude to Bedrock `eu-central-1` stops being "write an adapter" and becomes
"find every place that assumed the US endpoint" — which is exactly the audit
nobody has time for when a university's procurement is waiting on the answer.

Same guard shape as the §19 Arabic-literal check: AST-precise, docstrings
exempt, executable string constants checked.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SRC = REPO_ROOT / "src" / "moc"
LLM_PACKAGE = SRC / "llm"

# Substrings that mean a module has an opinion about where a provider lives.
ENDPOINT_MARKERS = ("api.anthropic.com", "api.openai.com", "https://", "http://", "base_url")


def _docstring_ids(tree: ast.AST) -> set[int]:
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, holders) and ast.get_docstring(node, clean=False) is not None:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _label(path: Path) -> str:
    """Repo-relative where possible; tmp_path fixtures live outside the repo."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _endpoint_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_ids(tree)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        if any(marker in node.value for marker in ENDPOINT_MARKERS):
            hits.append(f"{_label(path)}:{node.lineno}: {node.value!r}")
    return hits


def test_no_endpoint_strings_outside_the_llm_package():
    offenders = []
    for path in SRC.rglob("*.py"):
        if LLM_PACKAGE in path.parents:
            continue
        offenders.extend(_endpoint_strings(path))
    assert offenders == [], (
        "design §2.4: endpoints live only in src/moc/llm/, so a region change "
        f"stays one new adapter file. Found: {offenders}"
    )


def test_keyword_arguments_named_base_url_are_caught_too():
    """A leak is as likely to be `client(base_url=...)` as a literal URL."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if LLM_PACKAGE in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "base_url":
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.value.lineno}")
    assert offenders == [], f"base_url passed outside src/moc/llm/: {offenders}"


def test_the_guard_detects_a_planted_violation(tmp_path):
    """The guard is only worth having if it fires. Prove it on a synthetic module."""
    planted = tmp_path / "leaky.py"
    planted.write_text('CLIENT = connect("https://api.anthropic.com/v1")\n', encoding="utf-8")
    assert _endpoint_strings(planted) != []


@pytest.mark.parametrize("marker", ENDPOINT_MARKERS)
def test_every_marker_is_detected(marker, tmp_path):
    planted = tmp_path / "leaky.py"
    planted.write_text(f'X = "prefix {marker} suffix"\n', encoding="utf-8")
    assert _endpoint_strings(planted) != [], f"{marker} slipped through"


def test_docstrings_may_discuss_endpoints(tmp_path):
    """Prose explaining the rule must not trip the rule enforcing it."""
    documented = tmp_path / "documented.py"
    documented.write_text('"""We call https://api.openai.com from the adapter."""\n', "utf-8")
    assert _endpoint_strings(documented) == []


def test_the_llm_package_is_the_only_exemption():
    """If the allowlist ever widens, this test should be the thing that objects."""
    assert LLM_PACKAGE.is_dir()
    exempt = [p for p in SRC.rglob("*.py") if LLM_PACKAGE in p.parents]
    assert exempt, "the llm package should contain the adapters that hold endpoints"
