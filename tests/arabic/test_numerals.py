"""Numeral extraction — design doc §19 keeps every literal below out of the module.

Arabic strings are fine *here*. The rule is that `src/moc/arabic/numerals.py`
holds the algorithm and `config/arabic/lexicon.yaml` holds the vocabulary;
test_no_arabic_literals_in_numerals_module guards it.
"""

import ast
from pathlib import Path

import pytest

from moc.arabic.numerals import (
    QuantityKind,
    contains_ordinal,
    extract_numbers,
    extract_quantities,
    normalize_digits,
)

NUMERALS_MODULE = Path(__file__).parents[2] / "src" / "moc" / "arabic" / "numerals.py"

# Unicode blocks: Arabic, Arabic Supplement, Arabic Extended-A, Presentation Forms.
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

# Loop counters and slice indices only. A multiplier, a year bound or a digit
# offset appearing here means a lexical value leaked out of the config.
ALLOWED_NUMERIC_LITERALS = frozenset({0, 1, 2})


# ─────────────────────────── §19 guards ───────────────────────────


def test_no_arabic_literals_in_numerals_module():
    source = NUMERALS_MODULE.read_text(encoding="utf-8")
    offenders = sorted(
        {c for c in source if any(lo <= ord(c) <= hi for lo, hi in ARABIC_RANGES)}
    )
    assert offenders == [], (
        f"design doc §19: numerals.py must contain zero Arabic literals, found {offenders}. "
        f"Move them to config/arabic/lexicon.yaml."
    )


def test_no_magic_numbers_in_numerals_module():
    tree = ast.parse(NUMERALS_MODULE.read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    }
    leaked = sorted(literals - ALLOWED_NUMERIC_LITERALS)
    assert literals <= ALLOWED_NUMERIC_LITERALS, (
        f"design doc §19: unexpected numeric literals {leaked}. "
        f"Multipliers, year bounds and digit counts belong in the lexicon."
    )


# ─────────────────────────── DIGITS ───────────────────────────


def test_normalizes_arabic_indic_digits():
    assert normalize_digits("المصاريف ٢٥٠٠٠ جنيه") == "المصاريف 25000 جنيه"


def test_normalizes_extended_arabic_indic():
    assert normalize_digits("۱۲۳") == "123"


def test_normalize_digits_leaves_latin_and_letters_alone():
    assert normalize_digits("fee is 1400 EGP") == "fee is 1400 EGP"


def test_extracts_from_mixed_script():
    assert extract_numbers("الرسوم ١٢٥٠ جنيه للساعة و 8 سنين") == [1250, 8]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,250,000 جنيه", [1_250_000]),
        ("١٬٢٥٠٬٠٠٠ جنيه", [1_250_000]),
        ("2.5 مليون", [2_500_000]),
        ("2,5 مليون", [2_500_000]),
    ],
)
def test_separators_and_decimals(text, expected):
    assert extract_numbers(text) == expected


# ─────────────────────────── UNITS ───────────────────────────


@pytest.mark.parametrize(
    "text",
    ["٥ مليون", "٥ ملیون", "5 melion", "5 million", "5M", "٥ م"],
)
def test_million_unit_words(text):
    """Both yeh spellings: مليون uses Arabic yeh, ملیون uses Farsi yeh."""
    assert extract_numbers(text) == [5_000_000]


@pytest.mark.parametrize("text", ["٥ ألف", "٥ الف", "5k"])
def test_thousand_unit_words(text):
    assert extract_numbers(text) == [5_000]


def test_parses_half_unit_phrase():
    assert extract_numbers("٥ و نص مليون") == [5_500_000]


@pytest.mark.parametrize("text", ["٦ م", "6M"])
def test_six_million_both_scripts(text):
    assert extract_numbers(text) == [6_000_000]


def test_unit_result_is_an_int_when_integral():
    assert extract_numbers("2.5 مليون") == [2_500_000]
    assert isinstance(extract_numbers("2.5 مليون")[0], int)


# ─────────────────────────── REJECT ───────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "الترم الأول لعام 2026",
        "الترم التالتة",
        "السنة ٢٠٢٦",
        "2026",
        "٢٠٢٦",
    ],
)
def test_ignores_ordinals_and_years(text):
    assert extract_numbers(text) == []


@pytest.mark.parametrize(
    "text",
    ["الدور الخامس", "الدور التالت", "الدور 5", "الدور ٥", "الطابق 3"],
)
def test_ignores_floors(text):
    """A floor number is a location, not a quantity — and never a price."""
    assert extract_numbers(text) == []


@pytest.mark.parametrize("text", ["التجمع الخامس", "التجمع 5", "شقة في التجمع الخامس"])
def test_ignores_district_names(text):
    """التجمع الخامس is New Cairo, not the number five."""
    assert extract_numbers(text) == []


def test_currency_beats_the_year_heuristic():
    """A 2000 EGP fee is a fee, even though 2000 looks like a year."""
    assert extract_numbers("الرسوم 2026 جنيه") == [2026]


def test_franco_letter_digits_are_not_numbers():
    """In franco-arab, 3/5/7 are consonants: 5ames is خامس, not the number 5."""
    assert extract_numbers("3ayez sha22a fel tagamo3 el 5ames") == []


def test_franco_digit_still_parses_when_standalone():
    assert extract_numbers("3ayez sha22a b 5 melion") == [5_000_000]


def test_contains_ordinal_reads_from_the_lexicon():
    assert contains_ordinal("الترم الأول") is True
    assert contains_ordinal("الدور التالت") is True
    assert contains_ordinal("المصاريف كام") is False


# ─────────────────────────── EXTRACT + tagging ───────────────────────────


@pytest.mark.parametrize("text", ["مقدم ١٠٪", "مقدم 10%"])
def test_handles_percentages(text):
    assert extract_numbers(text) == [10]
    assert extract_quantities(text)[0].kind is QuantityKind.percent


@pytest.mark.parametrize("text", ["1400 جنيه", "1400 EGP", "1400 ج.م"])
def test_tags_currency_adjacent_figures(text):
    q = extract_quantities(text)
    assert [x.value for x in q] == [1400]
    assert q[0].kind is QuantityKind.currency


def test_tags_bedroom_counts_as_count_not_currency():
    q = extract_quantities("٤ غرف")
    assert [x.value for x in q] == [4]
    assert q[0].kind is QuantityKind.count


def test_bare_figure_has_no_tag():
    assert extract_quantities("و 8 سنين")[0].kind is QuantityKind.bare


def test_marks_approximated_figures():
    q = extract_quantities("حوالي 1500 جنيه")
    assert q[0].approximate is True
    assert extract_quantities("1500 جنيه")[0].approximate is False


def test_multi_word_approximation_marker():
    assert extract_quantities("في حدود ٦ مليون")[0].approximate is True


# ───────── a figure at the end of a sentence (found 2026-08-20) ─────────


@pytest.mark.parametrize(
    "text",
    [
        "رسوم التقديم 2000 جنيه.",
        "الرسوم 2000 جنيه، وهي غير مستردة",
        "the fee is 2000 EGP.",
    ],
    ids=["arabic-period", "arabic-comma", "english-period"],
)
def test_a_currency_marker_still_counts_with_punctuation_welded_to_it(text):
    """`.` and `,` are stripped out of the trim set to protect `3.5` and
    `1,500` — and that also stopped `جنيه.` from folding to `جنيه`.

    So the currency marker was invisible, a four-digit figure in the year
    range fell through `_is_year`, and 2000 at the end of a sentence was not a
    figure at all. `check_numeric_grounding` would have passed an invented
    2000 EGP fee in the commonest sentence shape the product writes.

    3000 was caught, which is why nothing looked wrong: only figures inside
    1900-2100 hit the year heuristic, and the decorated-hallucination
    regression test happens to use 3000.
    """
    assert [q.value for q in extract_quantities(text)] == [2000]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("السعر 3.5 مليون", [3500000.0]),
        ("السعر 1,500 جنيه", [1500]),
        ("الرسوم 2000 ج.م", [2000]),
        ("the fee is 2000 l.e", [2000]),
    ],
    ids=["decimal", "thousands", "arabic-abbrev", "latin-abbrev"],
)
def test_a_separator_inside_a_token_still_survives(text, expected):
    """The reason `.` and `,` were taken out of the trim set in the first
    place. Shaving them off the END of a token cannot reach a separator that
    sits between digits, or the period inside `ج.م` and `l.e`."""
    assert [q.value for q in extract_quantities(text)] == expected


def test_a_year_at_the_end_of_a_sentence_is_still_a_year():
    """The heuristic this restores must not be broken by restoring it. A bare
    four-digit number with no currency or unit around it still qualifies a
    fee rather than asserting one."""
    assert extract_quantities("النسب المطلوبة لعام 2026.") == []
