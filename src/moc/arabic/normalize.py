"""Search normalization — design §7.1.

Retrieval matches on the normalized form; the reply cites the original. Both
are stored on every chunk, and keeping them apart is the point: a reply
quoting normalized text has rewritten the tenant's own wording, and for a fee
it has rewritten the digits a student will compare against an invoice.

§19 binds this module the way it binds `numerals.py`. Every fold is config —
no Arabic literal appears here, and a test enforces it. The rules exist
because Egyptians type that way: `القنطره` for `القنطرة`, `الشمالى` for
`الشمالي`, `احمد` for `أحمد`. Folding them is not tidying the language, it is
matching the spellings customers actually send.
"""

import unicodedata
from functools import lru_cache

from moc.arabic.numerals import normalize_digits
from moc.config_store import load

_NORMALIZATION = "arabic/normalization"


@lru_cache(maxsize=1)
def _rules() -> tuple[dict[int, str | None], bool, bool, bool]:
    """The folds, pre-compiled into one translation table.

    A single `str.translate` pass rather than a chain of replaces: this runs
    over every chunk at ingest and every query at search time, and the chain
    version walks the string once per rule.
    """
    document = load(_NORMALIZATION)
    table: dict[int, str | None] = {}
    for source, target in document["letter_folding"].items():
        table[ord(source)] = target or None
    for character in document["remove"]:
        table[ord(character)] = None
    return (
        table,
        document["strip_combining_marks"],
        document["casefold_latin"],
        document["collapse_whitespace"],
    )


def normalize(text: str) -> str:
    """Fold `text` into the form retrieval matches against.

    Order matters. Combining marks come off first, because decomposing after
    folding would re-expose marks on characters the folds just replaced.
    Digits normalize last so a figure written in Arabic-Indic reaches the same
    string as the same figure written in Latin — otherwise `٢٥٠٠٠` and `25000`
    are two different fees to the lexical arm.
    """
    table, strip_marks, casefold, collapse = _rules()

    if strip_marks:
        text = "".join(
            character
            for character in unicodedata.normalize("NFD", text)
            if not unicodedata.combining(character)
        )
    text = text.translate(table)
    if casefold:
        text = text.casefold()
    text = normalize_digits(text)
    if collapse:
        text = " ".join(text.split())
    return text


__all__ = ["normalize"]
