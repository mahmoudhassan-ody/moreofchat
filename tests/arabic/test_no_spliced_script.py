"""No Latin letters spliced inside Arabic words.

The corruption that prompted this: `العLMين` — Latin `LM` sitting inside
`العلمين` where `لم` belongs. It came from the running bot's `locations.ts`
and was corrected when the aliases were transcribed into
`config/arabic/locations.yaml`.

It is worth a permanent check because of how it fails. Nothing errors. The
alias list looks fine at a glance, `العلمين` simply never matches, and the
symptom reaching the team is "the bot doesn't cover Alamein" — a coverage
complaint pointing at retrieval, three layers from the two wrong characters
causing it. The same splice in a compound name would silently exclude every
unit in it.

**The test is the boundary, not the mixing.** Arabic and Latin legitimately
share a string all over this corpus: `(IGCSE)`, `dorms.a@su.edu.eg`,
`Pharm D`, `(Mass Communication)`. Every one of those has a separator — a
space, a parenthesis, a colon. Splicing has none, which makes adjacency the
signal and mere co-occurrence noise. A scan for co-occurrence returns 61 hits
here and nothing actionable in any of them.
"""

import csv
import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
CONFIG = REPO_ROOT / "config"
FIXTURES = REPO_ROOT / "evals" / "fixtures"
CASES = REPO_ROOT / "evals" / "cases"

ARABIC = r"؀-ۿ"
SPLICE = re.compile(f"[{ARABIC}][A-Za-z]|[A-Za-z][{ARABIC}]")


def spliced(value: object) -> list[str]:
    return SPLICE.findall(value) if isinstance(value, str) else []


def strings(node: object, path: str = "") -> list[tuple[str, str]]:
    if isinstance(node, dict):
        return [
            pair
            for key, value in node.items()
            for pair in [*strings(key, path), *strings(value, f"{path}.{key}")]
        ]
    if isinstance(node, list):
        return [pair for i, v in enumerate(node) for pair in strings(v, f"{path}[{i}]")]
    return [(path, node)] if isinstance(node, str) else []


def report(hits: list[tuple[str, str, list[str]]]) -> str:
    return "\n".join(
        f"  {where}: {value[:120]!r} (spliced at {found})" for where, value, found in hits
    )


def test_no_spliced_script_in_config():
    """The alias and lexicon files — where a splice costs the most.

    A corrupt alias here does not degrade matching, it removes it: the surface
    form is unreachable and every query using it misses completely.
    """
    hits = [
        (f"{path.relative_to(REPO_ROOT)}{where}", value, found)
        for path in sorted(CONFIG.rglob("*.yaml"))
        for where, value in strings(yaml.safe_load(path.read_text(encoding="utf-8")))
        if (found := spliced(value))
    ]
    assert hits == [], f"Latin spliced into Arabic in config:\n{report(hits)}"


def test_no_spliced_script_in_the_fixture_artifacts():
    """The corpus the bot answers from. A splice in a compound name would
    silently exclude every unit in that compound."""
    hits = [
        (f"{path.relative_to(REPO_ROOT)}:{number} [{key}]", value, found)
        for path in sorted(FIXTURES.rglob("*.jsonl"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for key, value in json.loads(line).items()
        if (found := spliced(value))
    ]
    assert hits == [], f"Latin spliced into Arabic in a fixture:\n{report(hits)}"


def test_no_spliced_script_in_the_committed_sources():
    """Catch it at the source, so a rebuild cannot reintroduce it."""
    hits = []
    for path in sorted(FIXTURES.rglob("source/*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle), 1):
                hits += [
                    (f"{path.relative_to(REPO_ROOT)}:{number} [{key}]", value, found)
                    for key, value in row.items()
                    if (found := spliced(value))
                ]
    assert hits == [], f"Latin spliced into Arabic in a source CSV:\n{report(hits)}"


def test_no_spliced_script_in_case_expectations():
    """Everything a case *asserts* — but not what a customer typed.

    `turns[].user` is exempt and must be: edu-0014 sends
    "الapplication methods", which is real Egyptian code-switching and the
    entire point of the `code_switching` category. Asserting cleanliness on
    customer text would delete the cases that test messy input.
    """
    hits = [
        (f"{path.relative_to(REPO_ROOT)}{where}", value, found)
        for path in sorted(CASES.glob("*.yaml"))
        for where, value in strings(yaml.safe_load(path.read_text(encoding="utf-8")))
        if not where.endswith(".user") and (found := spliced(value))
    ]
    assert hits == [], f"Latin spliced into Arabic in a case expectation:\n{report(hits)}"


def test_the_check_catches_the_original_corruption():
    """Prove it fires on the exact string that prompted it."""
    assert spliced("العLMين")
    assert spliced("العLMين الجديدة")


@pytest.mark.parametrize(
    "legitimate",
    [
        "ما الأوراق المطلوبة لطلاب الثانوية الإنجليزية (IGCSE)؟",
        "العريش: dorms.a@su.edu.eg",
        "كلية الصيدلة\nتخصص: Pharm D",
        "الإعلام (Mass Communication)، الهندسة",
        "السكن (Dorms)",
    ],
)
def test_bilingual_text_with_a_separator_is_not_flagged(legitimate):
    """The false positives that would get this check deleted.

    All five are real strings from the shipped corpus. A check that fired on
    them would be switched off within a week, and then it would catch nothing.
    """
    assert spliced(legitimate) == []


def test_the_corrected_alamein_aliases_are_present_and_clean():
    """The specific fix, pinned so a re-transcription cannot undo it."""
    aliases = yaml.safe_load((CONFIG / "arabic" / "locations.yaml").read_text("utf-8"))
    arabic = aliases["aliases"]["new alamein"]["arabic"]
    assert "العلمين" in arabic
    assert "العلمين الجديدة" in arabic
    assert all(not spliced(form) for form in arabic)
