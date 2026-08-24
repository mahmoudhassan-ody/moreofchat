"""Load a broker's own export into `inventory_units` — demo plan Task 41.

    uv run python scripts/import_inventory.py --tenant <slug> --file units.csv
    uv run python scripts/import_inventory.py --tenant <slug> --file units.csv --confirm

**Dry run by default.** The first form reads the sheet, maps its columns and
prints what would happen; nothing is written until `--confirm`. That is the
same rule the knowledge screen follows, for the same reason: a count says the
importer ran, only the rows say whether it ran sensibly, and only the person
who owns the stock can tell.

The number to read is **offerable**, not imported. Two hundred units of which
four are available is a bot that answers "no stock" to every question, and the
import that produced it reported success.

The plan says this happens the day before, not during. It is here rather than
in the console because the day before is when column names turn out to be
wrong, and fixing that is a config edit plus a re-run rather than a screen.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from moc.retrieval.inventory import available_status
from moc.retrieval.inventory_import import preview, propose_mapping, read_sheet, to_units

ROOT = Path(__file__).resolve().parents[1]


def report(result, mapping) -> None:
    print(f"\n  rows in the sheet: {result.total}")
    print(f"  would be offerable: {result.offerable}  (status = {available_status()!r})")

    print("\n  columns mapped:")
    for canonical, header in sorted(mapping.columns.items()):
        print(f"    {canonical:<16} <- {header}")
    if mapping.unclaimed:
        print(f"\n  columns nothing claimed: {list(mapping.unclaimed)}")
        print("    Add the spelling to config/retrieval/inventory.yaml if one of")
        print("    these is a field this platform stores.")
    if result.missing:
        print(f"\n  REQUIRED AND MISSING: {list(result.missing)}")
        print("    Nothing can be imported until these map. Add the sheet's own")
        print("    spelling to the aliases rather than renaming the broker's file,")
        print("    so the next export from the same CRM also works.")

    if result.unknown_statuses:
        print("\n  statuses this platform does not recognise:")
        for value, count in sorted(result.unknown_statuses.items()):
            print(f"    {value!r}: {count} row(s) — stored as-is, never offered")
        print("    Unrecognised is never available, on purpose. Map them in")
        print("    config/retrieval/inventory.yaml if any of them means sellable.")

    if result.duplicates:
        print(f"\n  unit ids appearing more than once: {result.duplicates}")
        print("    The upsert keys on (tenant, unit_id), so the last row wins and")
        print("    the import count looks the same either way.")

    if result.refused:
        print(f"\n  rows refused: {len(result.refused)}")
        for refusal in result.refused[:10]:
            print(f"    row {refusal.row}: {refusal.reason}")

    print("\n  a sample of what would be stored:")
    for unit in result.sample:
        print(
            f"    {unit.get('unit_id')}  {unit.get('property_type')}  "
            f"{unit.get('compound')}  {unit.get('price')} {unit.get('currency')}  "
            f"{unit.get('availability')}"
        )


async def write(tenant_slug: str, units: list[dict], as_of: str) -> int:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.config import settings
    from moc.retrieval.inventory import load_units
    from moc.tenancy.context import tenant_session

    engine = create_async_engine(settings.database_url)
    async with engine.connect() as connection:
        tenant_id = (
            await connection.execute(
                text("SELECT id FROM tenants WHERE slug = :slug"), {"slug": tenant_slug}
            )
        ).scalar_one_or_none()
    if tenant_id is None:
        await engine.dispose()
        print(f"\n  no tenant with slug {tenant_slug!r}")
        return 1

    # Through `load_units`, not around it. A second write path is a second
    # place the availability filter could be bypassed.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as handle:
        handle.write("\n".join(json.dumps(unit, ensure_ascii=False) for unit in units))
        snapshot = Path(handle.name)
    try:
        async with tenant_session(engine, tenant_id) as session:
            count = await load_units(session=session, path=snapshot)
            await session.commit()
    finally:
        # Off the loop: a one-line unlink is not worth blocking on, and ruff is
        # right that a coroutine touching the filesystem synchronously is a
        # habit rather than an exception.
        await asyncio.to_thread(snapshot.unlink, True)
        await engine.dispose()

    print(f"\n  {count} rows written for {tenant_slug} at as_of {as_of}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True, help="tenant slug")
    parser.add_argument("--file", required=True, type=Path, help="the broker's export")
    parser.add_argument(
        "--as-of",
        required=True,
        help="the date this export was taken (YYYY-MM-DD). Not a column: it is "
        "the snapshot's date and a sheet almost never carries it, and every "
        "reply states it.",
    )
    parser.add_argument("--confirm", action="store_true", help="write; otherwise dry run")
    args = parser.parse_args(argv)

    headers, rows = read_sheet(args.file)
    mapping = propose_mapping(headers)
    result = preview(rows, mapping=mapping, as_of=args.as_of)
    report(result, mapping)

    if result.missing:
        return 1
    if not args.confirm:
        print("\n  dry run. Re-run with --confirm to write.")
        return 0
    units = to_units(rows, mapping=mapping, as_of=args.as_of)
    return asyncio.run(write(args.tenant, units, args.as_of))


if __name__ == "__main__":
    sys.exit(main())
