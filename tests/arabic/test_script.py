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


# ─────────────────────────── franco is not English ───────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "3ayez sha22a fel tagamo3 el 5ames b 6 melion",
        "ana 3ayez sha22a fe el sa7el",
        "3andokom eh fe el sa7el?",
    ],
)
def test_franco_arabic_is_detected_as_arabic_not_english(text):
    """Franco is Arabic typed on a Latin keyboard, and mirroring it as English
    is worse than not mirroring at all.

    re-0016 is `3ayez sha22a fel tagamo3 el 5ames b 6 melion`. The reply came
    back "We have a apartment in HDP North…" because the message is
    Latin-script and the scripted-reply path had just learned to mirror
    language. A customer writing franco reads Arabic.
    """
    from moc.arabic.script import is_franco, reply_language

    assert is_franco(text)
    assert reply_language(text) == "ar"


@pytest.mark.parametrize(
    "text",
    [
        "In any other location",
        "Is the unit registered and does the developer have a valid licence?",
        "Can I see unit A3 please",
    ],
)
def test_english_stays_english(text):
    """Including one with a letter-adjacent digit. A single `A3` is a unit
    reference, not franco — the signal is several such tokens, because franco
    substitutes digits for consonants throughout rather than once."""
    from moc.arabic.script import is_franco, reply_language

    assert not is_franco(text)
    assert reply_language(text) == "en"


def test_arabic_script_is_arabic():
    from moc.arabic.script import reply_language

    assert reply_language("عايز شقة في التجمع") == "ar"


def test_a_message_with_no_letters_has_no_reply_language():
    from moc.arabic.script import reply_language

    assert reply_language("١٢٣ ؟!") is None


# ─────────── franco without substitutions (the edu-0015 gap) ───────────


@pytest.mark.parametrize(
    "text",
    [
        "fe manh fe kantara?",
        "ana ayez sha2a fe masr",
        "3andokom eh",
        "el mawad di kam?",
    ],
)
def test_franco_without_digit_substitutions_is_still_franco(text):
    """`fe manh fe kantara?` has no substituted digit anywhere, and edu-0015
    was answered in English because of it.

    Roughly a third of real franco carries no substitutions: the digits stand
    in for ع ح خ ء, and a sentence without those letters produces none. The
    grammatical skeleton is the other signal, and unlike place names it is a
    closed set — about forty function words covers nearly everything.
    """
    from moc.arabic.script import is_franco

    assert is_franco(text)


@pytest.mark.parametrize(
    "text",
    [
        "Please confirm the fe number on the invoice",
        "Is housing available at the Arish branch?",
        "In any other location",
        "Can I see unit A3 please",
        "El Gouna is lovely",
    ],
)
def test_english_containing_a_franco_word_is_still_english(text):
    """A two-hit minimum is what buys this. One `fe` or one `el` inside an
    English sentence is a coincidence — franco carries its skeleton
    throughout, and a one-hit rule would read half of English as Arabic."""
    from moc.arabic.script import is_franco, reply_language

    assert not is_franco(text)
    assert reply_language(text) == "en"


def test_the_franco_markers_are_config_not_a_literal():
    """§19, and the reason this list is defensible where the location one was
    not: function words are a CLOSED set. Place names grow with the catalogue
    and that list was ten of ninety-four; this one is nearly complete at forty
    and does not grow when a tenant adds inventory."""
    import inspect

    from moc.arabic import script
    from moc.config_store import load

    markers = load("arabic/lexicon")["franco_markers"]
    assert len(markers) >= 20
    source = inspect.getsource(script)
    for marker in markers:
        assert f'"{marker}"' not in source, f"{marker} is a literal in the module"
