"""Reading a broker's own export — design §3.2, §19.2, demo plan Task 41.

`load_units` takes canonical field names. A real export does not have them: it
has whatever the broker's CRM or their sales admin's spreadsheet calls things,
in either language and often both in one file. This is the layer between.

**An unrecognised status is never available.** `available_status` has been
config since P1 because every CRM spells it differently — but the other half
was missing: what happens to a value the map does not know. Defaulted to
available, a sold unit reaches a second buyer, and nothing anywhere raises:
the row *says* available, so the availability filter passes it and
`sold_unit_offered_rate` reads zero because the data is what lied. So an
unmapped value is stored verbatim. It cannot equal `available_status`, so it
cannot be offered, and `preview` names it and counts it.

**Nothing is written before somebody has looked.** `preview` takes no session
and no engine — there is nothing it could write to — and reports what a broker
can act on: which columns did not map, which statuses are unrecognised, which
rows were refused and why, how many units can actually be offered, and a sample
of the parsed rows. A count says the importer ran; only the rows say whether it
ran sensibly, and only the person who owns the stock can tell.

**The output is what `load_units` already takes.** A second ingestion path
would be a second place the availability filter could be bypassed, which is the
one thing §3.2 arranges the whole connector against.
"""

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moc.arabic.normalize import normalize
from moc.arabic.numerals import normalize_digits
from moc.config_store import load

_INVENTORY = "retrieval/inventory"
_SAMPLE = 5

#: Everything that is not part of a number. Applied after digit folding, so
#: `EGP 5,500,000` and `٥٬٥٠٠٬٠٠٠` reduce to the same digits.
_NOT_A_NUMBER = re.compile(r"[^\d.]")

#: Arabic decimal and thousands separators. Folded *before* the strip above,
#: which would otherwise delete them and shift every digit left: `١٢٣٤٫٥`
#: becomes 12345 rather than 1234.5, and a 120.5 m² unit is advertised at
#: 1205 m². `normalize_digits` converts the digits and leaves these two alone.
_ARABIC_SEPARATORS = {ord("\u066b"): ".", ord("\u066c"): None}
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")

_INTEGER_FIELDS = ("price", "bedrooms", "bathrooms")
_NUMERIC_FIELDS = ("unit_area_sqm",)


class UnmappedColumns(ValueError):
    """A required column has no match in the sheet.

    Raised by `to_units` and reported by `preview`. The asymmetry is the point:
    the preview exists to be read by somebody who can fix it, and raising there
    would make a traceback the only way to learn which column is missing.
    """


@dataclass(frozen=True)
class Mapping:
    """Canonical field -> the header it came from, plus the status map."""

    columns: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    #: Headers nothing claimed. Reported rather than dropped silently: a column
    #: the broker considers important and this does not is worth a question.
    unclaimed: tuple[str, ...] = ()


@dataclass(frozen=True)
class Refusal:
    row: int
    reason: str


@dataclass(frozen=True)
class ImportPreview:
    total: int
    refused: tuple[Refusal, ...]
    missing: tuple[str, ...]
    unclaimed: tuple[str, ...]
    #: Sheet value -> how many rows hold it, for values the map does not know.
    unknown_statuses: dict[str, int]
    #: Unit ids appearing more than once, and how often.
    duplicates: dict[str, int]
    #: How many rows would actually be offerable. Not the import count: two
    #: hundred units of which four are available is a bot that says "no stock"
    #: to every question, and the import that produced it reported success.
    offerable: int
    sample: tuple[dict[str, Any], ...]


def _settings() -> dict[str, Any]:
    return load(_INVENTORY)["import"]


def _key(value: str) -> str:
    """A header or a status, reduced to what two spellings of it share.

    Case-folded, punctuation stripped, whitespace collapsed, and run through
    the Arabic normalizer so alef and ta-marbuta variants meet. Without it the
    alias list needs `الحالة` and `الحاله` and `الحالة ` as separate entries,
    and the fourth spelling somebody types is still missed.
    """
    folded = normalize(_PUNCTUATION.sub(" ", value or ""))
    return _SPACES.sub(" ", folded).strip().lower()


def read_sheet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Headers and rows from a CSV export.

    Two encodings of the same file break in ways that look like a mapping bug:

    - **The byte-order mark.** Excel writes one at the start of a UTF-8 CSV.
      Left in, the first header becomes `\\ufeffUnit Code`, the id column fails
      to map, and every row is refused for a missing id — on the one file
      format every broker actually sends.
    - **The delimiter.** Excel on an Arabic or European locale writes `;`. Read
      as commas, the whole file is one column and no header maps at all.
    """
    raw = path.read_bytes().decode("utf-8-sig" if _settings()["strip_byte_order_mark"] else "utf-8")
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        # One column, or a file too small to sniff. Commas is the right guess
        # and a wrong one shows up immediately as unmapped headers.
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(raw), dialect=dialect)
    headers = [name for name in (reader.fieldnames or []) if name is not None]
    return headers, [dict(row) for row in reader]


def propose_mapping(headers: list[str]) -> Mapping:
    """Best effort, and honest about what it could not do.

    Never a partial guess on a required field: an id column matched to the
    wrong header upserts every row onto one unit.
    """
    settings = _settings()
    by_alias: dict[str, str] = {}
    for canonical, aliases in settings["aliases"].items():
        by_alias[_key(canonical)] = canonical
        for alias in aliases:
            by_alias[_key(alias)] = canonical

    columns: dict[str, str] = {}
    unclaimed: list[str] = []
    for header in headers:
        canonical = by_alias.get(_key(header))
        if canonical is None:
            unclaimed.append(header)
        elif canonical not in columns:
            columns[canonical] = header

    statuses: dict[str, str] = {}
    for canonical, aliases in settings["status_aliases"].items():
        statuses[_key(canonical)] = canonical
        for alias in aliases:
            statuses[_key(alias)] = canonical

    missing = tuple(name for name in settings["required"] if name not in columns)
    return Mapping(
        columns=columns, statuses=statuses, missing=missing, unclaimed=tuple(unclaimed)
    )


def _number(value: str | None) -> float | None:
    """A figure from a spreadsheet cell.

    Arabic-Indic digits first: `٥٥٠٠٠٠٠` is what a sheet typed on an Arabic
    keyboard holds, and read as text it is a string that refuses the row.
    Then everything that is not a digit or a decimal point, which removes
    thousands separators, currency words and stray spaces alike.
    """
    if value is None:
        return None
    folded = normalize_digits(str(value)).translate(_ARABIC_SEPARATORS)
    cleaned = _NOT_A_NUMBER.sub("", folded)
    if not cleaned or cleaned == ".":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _status(value: str | None, mapping: Mapping) -> str:
    """The canonical status, or the sheet's own word.

    Returning the raw value is the safe direction and the deliberate one: it
    cannot equal `available_status`, so an unrecognised status can never be
    offered. See the module docstring.
    """
    raw = (value or "").strip()
    return mapping.statuses.get(_key(raw), raw)


def _row_to_unit(row: dict[str, str], mapping: Mapping, as_of: str) -> tuple[dict[str, Any], str]:
    """One sheet row -> one canonical record, or a reason it cannot be one."""
    unit: dict[str, Any] = {"as_of": as_of}
    for canonical, header in mapping.columns.items():
        value = (row.get(header) or "").strip()
        if canonical == "availability":
            unit[canonical] = _status(value, mapping)
        elif canonical in _INTEGER_FIELDS:
            number = _number(value)
            unit[canonical] = int(number) if number is not None else None
        elif canonical in _NUMERIC_FIELDS:
            unit[canonical] = _number(value)
        else:
            unit[canonical] = value or None

    unit.setdefault("currency", None)
    if not unit.get("currency"):
        # Not a guess about the market: every price in this table is a figure
        # the bot will state, and a figure with no currency is one it cannot.
        unit["currency"] = "EGP"

    for required in _settings()["required"]:
        if unit.get(required) in (None, ""):
            return unit, f"{required} is empty"
    return unit, ""


def preview(
    rows: list[dict[str, str]], *, mapping: Mapping, as_of: str
) -> ImportPreview:
    """What this sheet would become, without becoming it.

    Takes no session and no engine. There is nothing here that could write, and
    that is asserted rather than described.
    """
    refused: list[Refusal] = []
    accepted: list[dict[str, Any]] = []
    unknown: dict[str, int] = {}
    seen: dict[str, int] = {}

    available = load(_INVENTORY)["available_status"]
    known = set(_settings()["status_aliases"])

    for index, row in enumerate(rows, start=1):
        unit, reason = _row_to_unit(row, mapping, as_of)
        status = unit.get("availability")
        if status and status not in known:
            unknown[str(status)] = unknown.get(str(status), 0) + 1
        if reason:
            refused.append(Refusal(row=index, reason=reason))
            continue
        accepted.append(unit)
        unit_id = str(unit.get("unit_id"))
        seen[unit_id] = seen.get(unit_id, 0) + 1

    return ImportPreview(
        total=len(rows),
        refused=tuple(refused),
        missing=mapping.missing,
        unclaimed=mapping.unclaimed,
        unknown_statuses=unknown,
        duplicates={unit_id: count for unit_id, count in seen.items() if count > 1},
        offerable=sum(1 for unit in accepted if unit.get("availability") == available),
        sample=tuple(accepted[:_SAMPLE]),
    )


def to_units(
    rows: list[dict[str, str]], *, mapping: Mapping, as_of: str
) -> list[dict[str, Any]]:
    """Canonical records, ready for `load_units`.

    Raises when a required column never mapped: at this point somebody has read
    the preview and pressed go, and importing a sheet whose id column was never
    found would upsert every row onto one unit.
    """
    if mapping.missing:
        raise UnmappedColumns(
            f"no column matched {list(mapping.missing)}. The sheet's own headers "
            f"are {list(mapping.columns.values()) + list(mapping.unclaimed)} — add "
            # The logical name, not the file path. §19's plan is YAML today,
            # a table in P3 and a console control in P4, and a message naming
            # the file becomes a lie at the first of those steps while the
            # logical name stays true. (It is also what the config-store guard
            # checks for, and the guard is right.)
            f"the spelling to the {_INVENTORY!r} config rather than renaming the "
            "broker's file, so the next export from the same CRM also works."
        )
    return [
        unit
        for unit, reason in (_row_to_unit(row, mapping, as_of) for row in rows)
        if not reason
    ]


__all__ = [
    "ImportPreview",
    "Mapping",
    "Refusal",
    "UnmappedColumns",
    "preview",
    "propose_mapping",
    "read_sheet",
    "to_units",
]
