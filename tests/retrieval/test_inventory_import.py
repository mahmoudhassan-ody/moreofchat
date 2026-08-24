"""Reading a broker's own export — demo plan Task 41.

The fixture has canonical field names. A real export does not, and the plan
says so in one line: *their column names will not match the fixtures. This is
where surprises live.*

Two of those surprises are worth stating before the tests, because both fail
without an error.

**An unrecognised status must never mean available.** Every broker CRM spells
it differently, which is why `available_status` has been config since P1 — but
the other half was missing: what happens to a value that is not in the map.
Defaulted to available, a sold unit reaches a second buyer, `sold_unit_offered_rate`
stays at zero because the row *says* available, and the only person who finds
out is the buyer. So an unmapped status is stored verbatim, which cannot equal
`available_status` and therefore cannot be offered, and the preview names it.

**Nothing is written before somebody has looked.** The same rule the knowledge
screen follows: a count says the importer ran, only the rows say whether it ran
sensibly, and only the broker can tell. So `preview` is the whole surface and
`to_units` is what happens after the preview was read.
"""

from pathlib import Path

import pytest

from moc.retrieval.inventory_import import (
    UnmappedColumns,
    preview,
    propose_mapping,
    read_sheet,
    to_units,
)

AS_OF = "2026-08-24"


def sheet(
    tmp_path: Path, body: str, name: str = "units.csv", encoding: str = "utf-8"
) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding=encoding)
    return path


ENGLISH = """Unit Code,Type,Project,Price (EGP),Status,Beds
MD-1,Apartment,Madinaty,"5,500,000",Available,3
MD-2,Apartment,Madinaty,"6,100,000",Sold,3
"""

ARABIC = """كود الوحدة,النوع,المشروع,السعر,الحالة,عدد الغرف
MD-1,شقة,مدينتي,٥٥٠٠٠٠٠,متاح,٣
"""


# ─────────────────────────── reading the file ───────────────────────────


def test_english_headers_map_to_the_canonical_fields(tmp_path):
    headers, rows = read_sheet(sheet(tmp_path, ENGLISH))
    mapping = propose_mapping(headers)
    assert mapping.columns["unit_id"] == "Unit Code"
    assert mapping.columns["price"] == "Price (EGP)"
    assert mapping.columns["availability"] == "Status"


def test_arabic_headers_map_to_the_same_fields(tmp_path):
    headers, _ = read_sheet(sheet(tmp_path, ARABIC))
    mapping = propose_mapping(headers)
    assert mapping.columns["unit_id"] == "كود الوحدة"
    assert mapping.columns["price"] == "السعر"
    assert mapping.columns["availability"] == "الحالة"


def test_an_excel_byte_order_mark_does_not_hide_the_first_column(tmp_path):
    """Excel writes a BOM at the start of a UTF-8 CSV. Left in, the first
    header becomes `﻿Unit Code`, the id column silently fails to map, and
    every row is refused for a missing id — on the one file format every broker
    actually sends."""
    path = sheet(tmp_path, ENGLISH, encoding="utf-8-sig")
    headers, _ = read_sheet(path)
    assert propose_mapping(headers).columns["unit_id"] == "Unit Code"


def test_a_semicolon_separated_export_is_read_as_columns_not_one(tmp_path):
    """Excel on an Arabic or European locale writes `;`. Read as commas, the
    whole file is a single column, no header maps, and the import reports every
    required column missing — which reads as "your file is wrong" rather than
    "we read it wrong"."""
    body = (
        "Unit Code;Type;Project;Price (EGP);Status;Beds\n"
        "MD-1;Apartment;Madinaty;5500000;Available;3\n"
        "MD-2;Apartment;Madinaty;6100000;Sold;3\n"
    )
    headers, rows = read_sheet(sheet(tmp_path, body))
    assert len(headers) == 6, f"the file was read as one column: {headers}"
    mapping = propose_mapping(headers)
    assert mapping.missing == (), f"nothing mapped: {mapping}"
    assert to_units(rows, mapping=mapping, as_of=AS_OF)[0]["price"] == 5_500_000


# ─────────────────────────── the figures ───────────────────────────


def test_a_price_with_thousands_separators_is_a_number(tmp_path):
    headers, rows = read_sheet(sheet(tmp_path, ENGLISH))
    units = to_units(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert units[0]["price"] == 5_500_000


def test_arabic_indic_digits_are_a_number(tmp_path):
    """`٥٥٠٠٠٠٠` is what a sheet typed on an Arabic keyboard holds.

    Python's own `float` reads Arabic-Indic digits, so this passes with or
    without the folding — recorded so nobody removes the folding on the
    strength of this test alone. What the folding is actually for is the test
    below it.
    """
    headers, rows = read_sheet(sheet(tmp_path, ARABIC))
    units = to_units(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert units[0]["price"] == 5_500_000
    assert units[0]["bedrooms"] == 3


def test_the_arabic_decimal_separator_does_not_shift_every_digit_left(tmp_path):
    """`٫` is a decimal point and `٬` is a thousands mark, and neither is an
    ASCII character any digit-stripping regex keeps.

    Deleted rather than translated, `١٢٠٫٥` becomes 1205: a 120.5 m² unit
    advertised at 1205 m², off by a factor of ten with every digit correct.
    """
    body = (
        "Unit Code,Type,Project,Price (EGP),Status,Area sqm\n"
        "MD-9,Apartment,Madinaty,٥٬٥٠٠٬٠٠٠,Available,١٢٠٫٥\n"
    )
    headers, rows = read_sheet(sheet(tmp_path, body))
    units = to_units(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert units[0]["unit_area_sqm"] == 120.5
    assert units[0]["price"] == 5_500_000


def test_a_price_carrying_its_currency_still_parses(tmp_path):
    body = ENGLISH.replace('"5,500,000"', '"EGP 5,500,000"')
    headers, rows = read_sheet(sheet(tmp_path, body))
    units = to_units(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert units[0]["price"] == 5_500_000


def test_a_row_with_no_price_is_refused_rather_than_priced_at_zero(tmp_path):
    body = ENGLISH.replace('"5,500,000"', "")
    headers, rows = read_sheet(sheet(tmp_path, body))
    result = preview(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert [r.row for r in result.refused] == [1]
    assert "price" in result.refused[0].reason


def test_the_snapshot_date_comes_from_the_import_not_the_sheet(tmp_path):
    """§3.2: `as_of` is a column on the row and not a constant, because two
    tenants ingest on different days — and it is the export's date, which a
    sheet almost never carries."""
    headers, rows = read_sheet(sheet(tmp_path, ENGLISH))
    units = to_units(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert {unit["as_of"] for unit in units} == {AS_OF}


# ─────────────────────── the status, which is the dangerous one ───────────────────────


def test_a_recognised_status_maps_to_the_canonical_one(tmp_path):
    headers, rows = read_sheet(sheet(tmp_path, ENGLISH))
    units = to_units(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert [unit["availability"] for unit in units] == ["available", "sold"]


def test_an_unrecognised_status_is_never_available(tmp_path):
    """The one that costs a broker a relationship.

    `Under Offer` is not in the alias list. Defaulted to available, that unit is
    offered to a second buyer and nothing raises — the row *says* available, so
    the availability filter passes it and `sold_unit_offered_rate` reads zero.
    """
    body = ENGLISH.replace(",Available,", ",Under Offer,")
    headers, rows = read_sheet(sheet(tmp_path, body))
    mapping = propose_mapping(headers)

    units = to_units(rows, mapping=mapping, as_of=AS_OF)
    from moc.retrieval.inventory import available_status

    assert units[0]["availability"] != available_status()

    result = preview(rows, mapping=mapping, as_of=AS_OF)
    assert result.unknown_statuses == {"Under Offer": 1}
    assert result.offerable == 0, "an unmapped status was counted as sellable stock"


def test_the_preview_says_how_much_of_the_sheet_can_actually_be_offered(tmp_path):
    """A count of imported rows is not the number a broker cares about. Two
    hundred units of which four are available is a bot that says "no stock" to
    every question, and the import that produced it reported success."""
    headers, rows = read_sheet(sheet(tmp_path, ENGLISH))
    result = preview(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert result.total == 2
    assert result.offerable == 1, "one of the two rows is sold"


# ─────────────────────── refusing to guess ───────────────────────


def test_a_missing_required_column_stops_the_import_rather_than_guessing(tmp_path):
    body = ENGLISH.replace("Unit Code,", "").replace("MD-1,", "").replace("MD-2,", "")
    headers, rows = read_sheet(sheet(tmp_path, body))
    mapping = propose_mapping(headers)

    assert "unit_id" in mapping.missing
    with pytest.raises(UnmappedColumns) as raised:
        to_units(rows, mapping=mapping, as_of=AS_OF)
    assert "unit_id" in str(raised.value)


def test_the_preview_reports_a_missing_column_instead_of_raising(tmp_path):
    """The preview exists to be read by somebody who can fix it. Raising there
    would mean the only way to learn which column is missing is a traceback."""
    body = ENGLISH.replace(",Status", ",Situation")
    headers, rows = read_sheet(sheet(tmp_path, body))
    result = preview(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert "availability" in result.missing
    assert result.offerable == 0


def test_a_repeated_unit_id_is_reported_because_the_upsert_hides_it(tmp_path):
    """`load_units` upserts on `(tenant_id, unit_id)`, so a sheet holding the
    same unit twice imports cleanly and the last row wins. When the duplicate is
    a stale row copied down a spreadsheet, the last row is the wrong one — and
    the import reports the same count either way."""
    body = ENGLISH + 'MD-1,Apartment,Madinaty,"9,900,000",Available,3\n'
    headers, rows = read_sheet(sheet(tmp_path, body))
    result = preview(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert result.duplicates == {"MD-1": 2}


def test_the_preview_shows_rows_and_not_only_counts(tmp_path):
    """The rule the knowledge screen already follows: a count says the importer
    ran, only the rows say whether it ran sensibly."""
    headers, rows = read_sheet(sheet(tmp_path, ENGLISH))
    result = preview(rows, mapping=propose_mapping(headers), as_of=AS_OF)
    assert result.sample, "nothing to look at before confirming"
    assert result.sample[0]["unit_id"] == "MD-1"


def test_nothing_is_written_by_a_preview(tmp_path):
    """Structural. `preview` takes no session and no engine, so there is
    nothing it could write to."""
    import inspect

    assert not {"session", "engine"} & set(inspect.signature(preview).parameters)


# ─────────────────────── and out through the real path ───────────────────────


async def test_an_imported_sheet_loads_through_the_ingestion_path(
    app_engine, engine, tenant_tables, tmp_path
):
    """The output is what `load_units` already takes.

    A second ingestion path would be a second place the availability filter
    could be bypassed, which is the one thing §3.2 arranges the connector
    against — so this drives the sheet all the way to a query and asserts the
    sold unit is not among the answers.
    """
    import json
    import uuid

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.retrieval.inventory import InventoryRepository, UnitQuery, load_units
    from moc.tenancy.context import tenant_session
    from moc.tenancy.models import Tenant

    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="sheet", name="Sheet", vertical="realestate"))
        await s.commit()

    headers, rows = read_sheet(sheet(tmp_path, ENGLISH))
    units = to_units(rows, mapping=propose_mapping(headers), as_of=AS_OF)

    snapshot = tmp_path / "units.jsonl"
    snapshot.write_text(
        "\n".join(json.dumps(unit, ensure_ascii=False) for unit in units),
        encoding="utf-8",
    )
    async with tenant_session(engine, tenant_id) as session:
        assert await load_units(session=session, path=snapshot) == 2
        await session.commit()

    async with tenant_session(app_engine, tenant_id) as session:
        offered = await InventoryRepository(session=session).search(UnitQuery())
    assert [unit.unit_id for unit in offered] == ["MD-1"], "the sold unit was offered"
