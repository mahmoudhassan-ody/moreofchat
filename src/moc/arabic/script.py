"""Script detection for the expected_lang check (harness spec §5.1).

Attribution is by majority of letters, not by first token. Code-switching is
the norm in Egyptian messaging, not an edge case: a reply that keeps English
technical terms inside Arabic prose is a correct Arabic reply, and case
edu-0010 says so in as many words. Counting the first token, or failing on any
Latin character, would flag the natural register as an F6 mirroring failure.

Digits and punctuation are deliberately excluded from the tally. Arabic-Indic
digits sit inside the Arabic block, so counting them would make "1400 2026"
read as Arabic in one script and English in the other, purely from how the
numbers were typed.

Design doc §19 applies here as it does to numerals.py: the codepoint ranges are
lexicon data, not literals. Two tests in tests/arabic/test_script.py enforce it.
"""

from functools import lru_cache

from moc.config_store import load

_LEXICON = "arabic/lexicon"


@lru_cache(maxsize=1)
def _script_ranges() -> dict[str, tuple[tuple[int, int], ...]]:
    scripts = load(_LEXICON)["scripts"]
    return {
        language: tuple((lo, hi) for lo, hi in spec["ranges"])
        for language, spec in scripts.items()
    }


def script_counts(text: str) -> dict[str, int]:
    """Letters per language. Digits, spaces and punctuation are not counted."""
    ranges = _script_ranges()
    counts = dict.fromkeys(ranges, 0)
    for char in text:
        if not char.isalpha():
            continue
        point = ord(char)
        for language, bounds in ranges.items():
            if any(lo <= point <= hi for lo, hi in bounds):
                counts[language] += 1
                break
    return counts


def detect_language(text: str) -> str | None:
    """The dominant language, or None when there are no letters to judge by.

    None rather than a default: a reply of digits and punctuation has no
    detectable language, and guessing one would let a genuinely broken reply
    satisfy the expected_lang check.
    """
    counts = script_counts(text)
    best = max(counts, key=lambda language: counts[language])
    return best if counts[best] else None
