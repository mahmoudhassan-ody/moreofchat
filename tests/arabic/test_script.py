"""Script detection for the expected_lang check (harness spec §5.1)."""

import ast
from pathlib import Path

import pytest

from moc.arabic.script import detect_language, script_counts

SCRIPT_MODULE = Path(__file__).parents[2] / "src" / "moc" / "arabic" / "script.py"
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
ALLOWED_NUMERIC_LITERALS = frozenset({0, 1, 2})


def test_no_arabic_literals_in_script_module():
    """Same §19 rule as numerals.py: the ranges live in the lexicon."""
    source = SCRIPT_MODULE.read_text(encoding="utf-8")
    offenders = sorted({c for c in source if any(lo <= ord(c) <= hi for lo, hi in ARABIC_RANGES)})
    assert offenders == [], f"design doc §19: found Arabic literals {offenders}"


def test_no_magic_numbers_in_script_module():
    tree = ast.parse(SCRIPT_MODULE.read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    }
    leaked = sorted(literals - ALLOWED_NUMERIC_LITERALS)
    assert literals <= ALLOWED_NUMERIC_LITERALS, f"§19: codepoint bounds leaked into code: {leaked}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("المصاريف كام لكلية الهندسة؟", "ar"),
        ("Is the property registered?", "en"),
        ("رسوم الساعة 1400 جنيه", "ar"),
    ],
)
def test_detects_the_dominant_script(text, expected):
    assert detect_language(text) == expected


def test_code_switched_reply_reads_as_arabic():
    """Egyptian replies keep English technical terms — that is not an English reply.

    Case edu-0010 says so explicitly: dominant language wins, not the first token.
    """
    reply = "متطلبات الadmission لكلية الbusiness هي شهادة الثانوية العامة"
    assert detect_language(reply) == "ar"


def test_mostly_english_with_a_few_arabic_words_reads_as_english():
    """The boundary, stated deliberately.

    Dominance is by letter count, so a reply that merely sprinkles Arabic over
    English prose is an English reply. That is the F6 catch — a lenient rule
    like "contains any Arabic" would pass exactly the failure being hunted.
    """
    assert detect_language("The admission requirements للbusiness are as follows") == "en"


def test_wholly_english_reply_to_an_arabic_question_is_english():
    """The F6 failure: mirroring broke. The checker must see it."""
    assert detect_language("The admission requirements are as follows") == "en"


def test_digits_and_punctuation_do_not_decide_language():
    assert detect_language("1400 !!! 2026") is None


def test_empty_text_has_no_language():
    assert detect_language("") is None


def test_script_counts_exposes_the_tally():
    counts = script_counts("مرحبا hello")
    assert counts["ar"] == len("مرحبا")
    assert counts["en"] == len("hello")
