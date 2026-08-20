"""Guards — design §7.3, §11.2 and §19.3.

`guards.py` is flagged for line-by-line human review, so these tests are
written to be read as the specification of what that review is checking:

1. **Redaction happens, in both digit scripts.** An Egyptian national ID
   written ٢٩٨٠١٢٣٤٥٦٧٨٩٠ is the same identifier as 29801234567890, and a
   Latin-digit regex sees neither a match nor a problem.
2. **Redaction does not eat fees.** A guard that redacts 25000 breaks numeric
   grounding, gets blamed for false positives, and gets switched off. The
   false-positive tests here matter as much as the true-positive ones.
3. **Pattern order is load-bearing**, because the first match consumes the
   digits the later patterns would have matched.
4. **Nothing lexical is in the source** (§19).

The redact-before-embedding test lives in `test_orchestrator.py` rather than
here: proving it needs a pipeline with an embedding call in it, and asserting
it against a stub in this file would prove only that the stub was called.
"""

import ast
from pathlib import Path

import pytest

from moc.agent.guards import PLACEHOLDER_LABELS, redact
from moc.config_store import load

GUARDS_MODULE = Path(__file__).parents[2] / "src" / "moc" / "agent" / "guards.py"

NATIONAL_ID = "29801234567890"
NATIONAL_ID_ARABIC = "٢٩٨٠١٢٣٤٥٦٧٨٩٠"
MOBILE = "01012345678"
MOBILE_ARABIC = "٠١٠١٢٣٤٥٦٧٨"


# ─────────────────────────── national ID ───────────────────────────


def test_redacts_egyptian_national_id():
    result = redact(f"my id is {NATIONAL_ID} thanks")
    assert NATIONAL_ID not in result.text
    assert "national_id" in result.found


def test_redacts_national_id_written_in_arabic_indic_digits():
    """Normalize before matching, or ٢٩٨٠١٢٣٤٥٦٧٨٩٠ walks past a \\d regex.

    This is the test the whole module exists for. A Latin-digit pattern does
    not fail loudly on Arabic-Indic input — it reports a clean message and
    forwards the identifier to two US providers.
    """
    result = redact(f"الرقم القومي {NATIONAL_ID_ARABIC} لو سمحت")
    assert NATIONAL_ID_ARABIC not in result.text
    assert "national_id" in result.found


def test_redacted_output_keeps_the_surrounding_arabic_intact():
    """Only the identifier goes. The message still has to be answerable."""
    result = redact(f"الرقم القومي {NATIONAL_ID_ARABIC} لو سمحت")
    assert "الرقم القومي" in result.text
    assert "لو سمحت" in result.text


# ─────────────────────────── phones ───────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        MOBILE,
        f"+2{MOBILE}",
        f"002{MOBILE}",
        MOBILE_ARABIC,
    ],
)
def test_redacts_phone_numbers_in_both_digit_scripts(raw):
    result = redact(f"كلمني على {raw} بليز")
    assert raw not in result.text
    assert "phone" in result.found


# ─────────────────────────── false positives ───────────────────────────


@pytest.mark.parametrize("figure", ["1400", "25000", "٢٥٠٠٠", "2026", "٤٫٥"])
def test_a_fee_figure_is_never_redacted(figure):
    """The check that keeps this guard switched on.

    Redacting a fee would break numeric grounding on every correctly-answered
    turn, and the team's fix would be to disable the guard rather than narrow
    the pattern. A redactor with false positives protects nobody.
    """
    result = redact(f"المصاريف {figure} جنيه")
    assert figure in result.text
    assert result.found == ()


# ─────────────────────────── ordering ───────────────────────────


def test_national_id_is_matched_before_the_payment_card_pattern():
    """Pattern order is behaviour, not style.

    A 14-digit national ID also satisfies a 13-to-19-digit card pattern. The
    first pattern to match consumes the digits, so whichever runs first decides
    the label — and a national ID logged as a card number is a national ID that
    the PDPL retention rules were never applied to.
    """
    result = redact(f"id {NATIONAL_ID}")
    assert result.found == ("national_id",)


def test_config_order_is_the_order_applied():
    """Reordering the YAML changes behaviour, so the file must be the source."""
    configured = [p["label"] for p in load("agent/redaction")["patterns"]]
    assert configured.index("national_id") < configured.index("payment_card")


# ─────────────────────────── §19: nothing lexical in source ───────────────────────────


def _non_docstring_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, holders) and ast.get_docstring(node, clean=False) is not None:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_redaction_patterns_come_from_config_not_literals():
    """§19: the code holds the algorithm, the config holds the values.

    A regex is a value. Egypt renumbers mobile prefixes and reissues ID
    formats; when that happens this must be a config edit reviewed by whoever
    owns compliance, not a source change that moves config_hash silently.
    """
    offenders = [
        s for s in _non_docstring_strings(GUARDS_MODULE) if "\\d" in s or "[0-9" in s
    ]
    assert offenders == [], (
        f"design §19: regex patterns belong in config/agent/redaction.yaml, "
        f"found in guards.py: {offenders}"
    )


def test_every_configured_label_is_a_known_placeholder():
    """A typo'd label would silently produce an unredacted-looking placeholder."""
    configured = {p["label"] for p in load("agent/redaction")["patterns"]}
    assert configured <= set(PLACEHOLDER_LABELS)


def test_the_guard_would_notice_a_pattern_moved_into_source(tmp_path):
    """Prove the §19 check fires, rather than trusting that it would."""
    planted = tmp_path / "leaky.py"
    planted.write_text('PATTERN = "\\\\d{14}"\n', encoding="utf-8")
    assert [s for s in _non_docstring_strings(planted) if "\\d" in s] != []


# ────────────────── formatting is not arithmetic (2026-08-20) ──────────────────


def test_a_decorated_hallucinated_figure_still_fails_the_gate():
    """**The one that shipped a wrong fee.**

    `hallucinated_figure_rate` is zero-tolerance and read 0.0% throughout,
    because a figure with `**` welded to it was not a figure at all. The same
    sentence without the asterisks was caught correctly, so the gate was doing
    its job on every reply nobody had formatted.
    """
    from moc.agent.guards import check_numeric_grounding

    result = check_numeric_grounding("الرسوم **3000** جنيه", ["رسوم التقديم 2000 جنيه"], [])

    assert not result.passed
    assert 3000 in result.orphan_numbers


@pytest.mark.parametrize(
    "reply",
    [
        "الرسوم **3000** جنيه",
        "الرسوم *3000* جنيه",
        "الرسوم `3000` جنيه",
        "الرسوم _3000_ جنيه",
        "| الرسوم | 3000 |",
        "**Fee: 3000 EGP**",
    ],
)
def test_no_decoration_hides_a_figure(reply):
    """Every character a model reaches for when it formats. The gate must see
    through all of them, because the prompt asking for plain text is an
    instruction and this is the guarantee."""
    from moc.agent.guards import check_numeric_grounding

    assert not check_numeric_grounding(reply, ["رسوم التقديم 2000 جنيه"], []).passed


def test_a_decorated_grounded_figure_still_passes():
    """Seeing through the decoration must not turn correct replies into
    failures — that is the other half of the same bug."""
    from moc.agent.guards import check_numeric_grounding

    assert check_numeric_grounding("الرسوم **2000** جنيه", ["رسوم التقديم 2000 جنيه"], []).passed


def test_a_list_marker_is_structure_not_a_claim():
    """edu-0010: three application methods, numbered.

    `## 1.` / `## 2.` / `## 3.` were read as the figures one, two and three,
    every one of them an orphan, and a correct and fully-grounded answer was
    discarded whole. The customer got "I need to verify this before telling
    you" about a list of three ways to apply.
    """
    from moc.agent.guards import check_numeric_grounding

    numbered = "في ثلاث طرق:\n1. الموقع\n2. الفروع\n3. المكتب"
    assert check_numeric_grounding(numbered, ["ثلاث طرق"], []).passed

    headed = "## 1. عن طريق الموقع\n## 2. في الفروع\n## 3. عن طريق المكتب"
    assert check_numeric_grounding(headed, ["ثلاث طرق"], []).passed

    parenthesised = "1) الموقع\n2) الفروع"
    assert check_numeric_grounding(parenthesised, ["طرق"], []).passed


def test_a_figure_that_merely_starts_a_line_is_still_a_claim():
    """Only a marker — digits then `.` or `)` then the rest of the line — is
    structure. A line that opens with the fee is a line stating the fee."""
    from moc.agent.guards import check_numeric_grounding

    assert not check_numeric_grounding("3000 جنيه رسوم التقديم", ["2000 جنيه"], []).passed


def test_a_marker_digit_elsewhere_in_the_line_is_still_a_claim():
    """`1. الرسوم 3000 جنيه` — the 1 is a marker, the 3000 is a claim."""
    from moc.agent.guards import check_numeric_grounding

    result = check_numeric_grounding("1. الرسوم 3000 جنيه", ["2000 جنيه"], [])
    assert not result.passed
    assert result.orphan_numbers == [3000], "the marker must not be counted"
