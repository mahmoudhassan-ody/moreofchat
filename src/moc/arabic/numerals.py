"""Extract quantities from Egyptian-Arabic text.

Feeds the numeric grounding check (eval-harness-spec §5.1), the highest-value
deterministic check in the system: every figure in a reply must trace to a
retrieved chunk or a script constant, and an orphan figure is F1 — the failure
that costs a tenant money.

Design doc §19 binds this module: it holds the algorithm and nothing else. No
Arabic literal, no multiplier, no year bound, no punctuation set. Everything
lexical is read from the platform lexicon through moc.config_store, and two
tests in tests/arabic/test_numerals.py fail if a value leaks back into source.

The hard part is not parsing digits, it is deciding which digits are quantities:

  - ordinals and floors     a fifth-floor flat is a location, never a price
  - place names             a district named "the fifth settlement" is a place
  - years                   an academic year qualifies a fee, it is not one
  - franco-arab consonants  in franco, several digits stand in for consonants,
                            so a digit welded to Latin letters is a letter

Each rejection is a real false positive from Egyptian messaging. A check that
flags them is a check the team turns off within a week, and a disabled check
catches nothing at all.
"""

import enum
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from moc.config_store import load

_LEXICON = "arabic/lexicon"


class QuantityKind(enum.StrEnum):
    """What a figure is about, per the per-dimension checks in spec §5.1."""

    bare = "bare"
    currency = "currency"
    percent = "percent"
    count = "count"


@dataclass(frozen=True)
class Quantity:
    value: int | float
    kind: QuantityKind
    raw: str
    approximate: bool


@dataclass(frozen=True)
class _Lexicon:
    """The lexicon, pre-indexed into the shapes the scanner needs."""

    digit_map: dict[str, str]
    latin_digits: str
    thousands: frozenset[str]
    decimals: frozenset[str]
    separators: str
    group_size: int
    trim: str
    list_markers: str
    list_marker_max_digits: int
    line_prefixes: str
    units: dict[str, int]
    fractions: dict[str, float]
    conjunctions: frozenset[str]
    ordinals: frozenset[str]
    floors: frozenset[str]
    places: frozenset[str]
    approximations: tuple[tuple[str, ...], ...]
    currencies: frozenset[str]
    percents: frozenset[str]
    counts: frozenset[str]
    year_min: int
    year_max: int
    year_digits: int
    year_markers: frozenset[str]


def _fold(token: str) -> str:
    """Case-fold and strip diacritics, so lookups tolerate spelling variation."""
    bare = "".join(c for c in unicodedata.normalize("NFD", token) if not unicodedata.combining(c))
    return bare.casefold()


@lru_cache(maxsize=1)
def _lexicon() -> _Lexicon:
    raw: dict[str, Any] = load(_LEXICON)

    digits = raw["digits"]
    latin = digits["latin"]
    digit_map = {
        glyph: latin[index]
        for name, glyphs in digits.items()
        if name != "latin"
        for index, glyph in enumerate(glyphs)
    }

    units: dict[str, int] = {}
    for entry in raw["unit_multipliers"]:
        for word in entry["words"]:
            units[_fold(word)] = entry["multiplier"]

    separators = raw["separators"]
    thousands = frozenset(separators["thousands"])
    decimals = frozenset(separators["decimal"])
    separator_chars = "".join(sorted(thousands | decimals))
    year = raw["year"]

    return _Lexicon(
        digit_map=digit_map,
        latin_digits=latin,
        thousands=thousands,
        decimals=decimals,
        separators=separator_chars,
        group_size=separators["group_size"],
        trim="".join(c for c in raw["punctuation"]["trim"] if c not in separator_chars),
        list_markers=raw["punctuation"]["list_markers"],
        list_marker_max_digits=raw["punctuation"]["list_marker_max_digits"],
        line_prefixes=raw["punctuation"]["line_prefixes"],
        units=units,
        fractions={_fold(k): v for k, v in raw["fractions"].items()},
        conjunctions=frozenset(_fold(w) for w in raw["conjunctions"]),
        ordinals=frozenset(_fold(w) for w in raw["ordinal_markers"]),
        floors=frozenset(_fold(w) for w in raw["floor_markers"]),
        places=frozenset(_fold(w) for w in raw["place_markers"]),
        approximations=tuple(
            tuple(_fold(part) for part in m.split()) for m in raw["approximation_markers"]
        ),
        currencies=frozenset(_fold(w) for w in raw["currency_markers"]),
        percents=frozenset(_fold(w) for w in raw["percent_markers"]),
        counts=frozenset(_fold(w) for w in raw["count_markers"]),
        year_min=year["min"],
        year_max=year["max"],
        year_digits=year["digit_length"],
        year_markers=frozenset(_fold(w) for w in year["markers"]),
    )


@lru_cache(maxsize=1)
def _number_pattern() -> re.Pattern[str]:
    """A digit run with optional separator groups, plus whatever is welded to it.

    The suffix group is what separates a unit abbreviation from a franco-arab
    consonant: both are digits touching letters, and only the lexicon can tell
    them apart.
    """
    separators = re.escape(_lexicon().separators)
    return re.compile(rf"(\d[\d{separators}]*)([^\d\s]*)")


# ─────────────────────────────── public API ───────────────────────────────


def normalize_digits(text: str) -> str:
    """Map every non-Latin digit script to Latin. Length-preserving, so offsets hold."""
    digit_map = _lexicon().digit_map
    return "".join(digit_map.get(c, c) for c in text)


def contains_ordinal(text: str) -> bool:
    """True if any token is an ordinal. An ordinal never denotes an amount."""
    lex = _lexicon()
    return any(_fold(_shave(t, lex)) in lex.ordinals for t in text.split())


def extract_numbers(text: str) -> list[int | float]:
    """Every figure in `text` that is a quantity, in order of appearance."""
    return [q.value for q in extract_quantities(text)]


def extract_quantities(text: str) -> list[Quantity]:
    """As extract_numbers, but tagged with what each figure is about."""
    lex = _lexicon()
    tokens, markers = _tokenize(normalize_digits(text), lex)
    folded = [_fold(_shave(t, lex)) for t in tokens]

    quantities = []
    for index, token in enumerate(tokens):
        if index in markers:
            # A list marker numbers an item; it does not state an amount.
            continue
        match = _number_pattern().fullmatch(_shave(token, lex))
        if match is None:
            continue
        body, suffix = match.group(1), _fold(match.group(2))

        if _is_franco_consonant(suffix, lex):
            continue
        value = _parse_number(body, lex)
        if value is None:
            continue
        if _is_rejected(index, folded, body, suffix, value, lex):
            continue

        value = _apply_units(value, index, folded, suffix, lex)
        quantities.append(
            Quantity(
                value=_as_int_if_whole(value),
                kind=_classify(index, folded, suffix, lex),
                raw=token,
                approximate=_is_approximated(index, folded, lex),
            )
        )
    return quantities


# ─────────────────────────────── parsing ───────────────────────────────


def _shave(token: str, lex: _Lexicon) -> str:
    """Strip decoration, including a sentence-final separator.

    The decimal point and the thousands comma are held out of the trim set so
    that 3.5 and 1,500 survive — and that also stopped a currency word with a
    full stop welded to it from folding to the bare word. The marker was then
    invisible, a four-digit figure inside the year range fell through
    `_is_year`, and a fee ending a sentence was not a figure at all: an
    invented amount in the commonest sentence shape the product writes would
    have passed a zero-tolerance gate. Only 1900-2100 hits the heuristic, which
    is why the decorated-hallucination regression test — written with 3000 —
    never saw it.

    Right-strip, because a separator only means anything between digits. It
    cannot reach the point in 3.5, the comma in 1,500, or the period inside a
    two-letter currency abbreviation.
    """
    return token.strip(lex.trim).rstrip(lex.separators)


def _tokenize(text: str, lex: _Lexicon) -> tuple[list[str], set[int]]:
    """Whitespace tokens, plus the indices that are list markers.

    Line-aware, which the previous whitespace split was not — and it has to be,
    because "is this digit a claim?" is answered by where on the line it sits.
    `1.` opening a line numbers an item; the same `1.` mid-sentence does not.
    """
    tokens: list[str] = []
    markers: set[int] = set()
    marker = _marker_pattern()
    for line in text.splitlines():
        line_tokens = line.split()
        offset = len(tokens)
        first = 0
        while first < len(line_tokens) and _is_prefix(line_tokens[first], lex):
            first += 1
        if first < len(line_tokens) and marker.fullmatch(line_tokens[first]):
            markers.add(offset + first)
        tokens.extend(line_tokens)
    return tokens, markers


def _is_prefix(token: str, lex: _Lexicon) -> bool:
    """A formatter's leading decoration — `##`, `-`, `>`. Never content."""
    return bool(token) and all(c in lex.line_prefixes for c in token)


@lru_cache(maxsize=1)
def _marker_pattern() -> re.Pattern[str]:
    lex = _lexicon()
    return re.compile(
        rf"\d{{1,{lex.list_marker_max_digits}}}[{re.escape(lex.list_markers)}]"
    )


def _is_franco_consonant(suffix: str, lex: _Lexicon) -> bool:
    """Digits fused to Latin letters are transliterated consonants, not numbers.

    Unless the letters are a unit abbreviation — the same shape means six
    million in one case and a consonant in the other.
    """
    if not suffix or suffix in lex.units or suffix in lex.percents:
        return False
    return any(c.isalpha() and c.isascii() for c in suffix)


def _parse_number(raw: str, lex: _Lexicon) -> float | None:
    """Resolve separators. A comma groups thousands or marks a decimal, never both."""
    body = raw.rstrip(lex.separators)
    if not body:
        return None
    positions = [i for i, c in enumerate(body) if c in lex.separators]
    if not positions:
        return float(body)

    head, *rest = re.split(f"[{re.escape(lex.separators)}]", body)
    grouped = bool(rest) and all(len(group) == lex.group_size for group in rest)
    if grouped and all(body[i] in lex.thousands for i in positions):
        return float(head + "".join(rest))
    if len(positions) == 1 and body[positions[0]] in lex.decimals:
        return float(f"{head}.{rest[0]}")
    return None


def _unit_span(index: int, folded: list[str], lex: _Lexicon) -> tuple[float, int, int]:
    """Scan forward for `[conjunction] [fraction] unit`, e.g. "five and a half million".

    Returns the fraction to add, the multiplier, and how many tokens were used.
    """
    cursor = index + 1
    if cursor < len(folded) and folded[cursor] in lex.conjunctions:
        cursor += 1
    fraction = 0.0
    if cursor < len(folded) and folded[cursor] in lex.fractions:
        fraction = lex.fractions[folded[cursor]]
        cursor += 1
    if cursor < len(folded) and folded[cursor] in lex.units:
        return fraction, lex.units[folded[cursor]], cursor - index
    return 0.0, 0, 0


def _apply_units(value: float, index: int, folded: list[str], suffix: str, lex: _Lexicon) -> float:
    if suffix in lex.units:
        return value * lex.units[suffix]
    fraction, multiplier, _ = _unit_span(index, folded, lex)
    return (value + fraction) * multiplier if multiplier else value


# ─────────────────────────────── rejection ───────────────────────────────


def _is_rejected(
    index: int, folded: list[str], body: str, suffix: str, value: float, lex: _Lexicon
) -> bool:
    previous = folded[index - 1] if index else ""
    if previous in lex.floors or previous in lex.places:
        return True
    return _is_year(index, folded, body, suffix, value, lex)


def _is_year(
    index: int, folded: list[str], body: str, suffix: str, value: float, lex: _Lexicon
) -> bool:
    """A bare four-digit number in the year range qualifies a fee; it is not one.

    A currency marker, a unit word or a percent sign outranks the heuristic, so
    a fee that happens to look like a year still reads as a fee.
    """
    tagged = _classify(index, folded, suffix, lex) is not QuantityKind.bare
    if tagged or _unit_span(index, folded, lex)[1]:
        return False
    if index and folded[index - 1] in lex.year_markers:
        return True
    digits = [c for c in body if c.isdigit()]
    if len(digits) != lex.year_digits or value != int(value):
        return False
    return lex.year_min <= value <= lex.year_max


# ─────────────────────────────── tagging ───────────────────────────────


def _classify(index: int, folded: list[str], suffix: str, lex: _Lexicon) -> QuantityKind:
    """A welded percent sign binds tightest; otherwise the following token decides."""
    if suffix and suffix in lex.percents:
        return QuantityKind.percent

    following = folded[index + 1] if index + 1 < len(folded) else ""
    for vocabulary, kind in (
        (lex.percents, QuantityKind.percent),
        (lex.currencies, QuantityKind.currency),
        (lex.counts, QuantityKind.count),
    ):
        if following in vocabulary:
            return kind
    return QuantityKind.bare


def _is_approximated(index: int, folded: list[str], lex: _Lexicon) -> bool:
    """Look back far enough to cover the widest multi-word marker."""
    for marker in lex.approximations:
        start = index - len(marker)
        if start >= 0 and tuple(folded[start:index]) == marker:
            return True
    return False


def _as_int_if_whole(value: float) -> int | float:
    return int(value) if value == int(value) else value
