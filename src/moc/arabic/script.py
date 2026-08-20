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

import re
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


#: Digits franco substitutes for Arabic consonants — hamza/qaf, ain, kha, ha,
#: ghain, sad. Named rather than written, because §19 keeps Arabic literals
#: out of this module and the lexicon is where letters live.
#:
#: They are also ordinary digits, which is why the rule below needs one to sit
#: against a letter *inside* a word: `sha22a` is franco, `6 melion` is a price.
_FRANCO_DIGITS = "2345789"

#: How many franco-marked tokens a message needs before it reads as franco.
#: One is a unit reference — `A3`, `B2` — and franco substitutes throughout
#: rather than once, so two is the honest floor.
_FRANCO_TOKENS = 2


def is_franco(text: str) -> bool:
    """Arabic typed on a Latin keyboard.

    Two signals, either of which is enough, both needing two hits.

    **Digit substitution** — digits standing in for consonants inside words:
    `3ayez`, `sha22a`, `tagamo3`, `5ames`. General, and it was the only rule
    at first because a word list is what had just cost us eighty-four
    compounds.

    **Function words** — `fe`, `ana`, `msh`, `ezay`. Added after
    `fe manh fe kantara?` was answered in English: the substituted digits
    stand in for four consonants only, and a sentence containing none of
    those letters produces none of the digits — roughly a third of real
    franco.

    The list is defensible where the location one was not, and the difference
    is worth being explicit about: function words are a CLOSED set. Place
    names grow every time a tenant loads inventory; these do not.

    Two hits either way. One `fe` or one `el` in an English sentence is a
    coincidence, and a one-hit rule reads half of English as Arabic.
    """
    if detect_language(text) != "en":
        return False
    tokens = [t for t in re.split(r"[^0-9A-Za-z]+", text) if t]
    return _substituted(tokens) >= _FRANCO_TOKENS or _markers(tokens) >= _FRANCO_TOKENS


def _substituted(tokens: list[str]) -> int:
    marked = 0
    for token in tokens:
        if len(token) < _FRANCO_TOKENS or not any(c.isalpha() for c in token):
            continue
        if any(
            char in _FRANCO_DIGITS
            and any(neighbour.isalpha() for neighbour in _neighbours(token, index))
            for index, char in enumerate(token)
        ):
            marked += 1
    return marked


def _markers(tokens: list[str]) -> int:
    known = _franco_markers()
    return sum(1 for token in tokens if token.lower() in known)


@lru_cache(maxsize=1)
def _franco_markers() -> frozenset[str]:
    return frozenset(word.lower() for word in load(_LEXICON)["franco_markers"])


def _neighbours(token: str, index: int) -> tuple[str, ...]:
    return tuple(token[i] for i in (index - 1, index + 1) if 0 <= i < len(token))


def reply_language(text: str) -> str | None:
    """Which language a reply to `text` should be written in.

    Not the same question as `detect_language`, and conflating them cost
    re-0016 its language check: franco is Latin script and Arabic language,
    so the script the customer typed in is the wrong thing to mirror.
    """
    return "ar" if is_franco(text) else detect_language(text)
