"""Build the broker_demo_2026_08_01 fixture from the real listings export.

Run it from anywhere; it reads `source/` next to this file and writes
`units.jsonl` into the current directory. `tests/evals/test_fixture_rebuild.py`
runs it into a temp directory and asserts byte-identical output, which is what
makes the freeze a checked claim rather than a note in a manifest.

Synthetic additions are deterministic (seeded) and documented in MANIFEST.md.
The source export has one availability state, one status, one listing kind and
no payment-plan columns, so the highest-risk real-estate cases could not exist
against it unaltered.

**Stdlib only, deliberately.** This ran on pandas against absolute paths under
/mnt/user-data/uploads, so it could only ever execute on the machine it was
written on. A build script nobody can run is a fixture nobody can regenerate,
and every unit price a case asserts against stops being traceable the day
someone needs to check one.

Two coercions pandas did implicitly and this has to do explicitly, because the
csv module hands back strings and the failure mode of getting either wrong is
a fixture that rebuilds *differently* rather than one that errors:

  - Integers. `int("6500000")` and `int("6500000.0")` do not both work, so
    values go through `as_int`, which accepts either form and refuses anything
    that is not exactly integral.
  - Booleans. `bool("False")` is **True** — the single nastiest line this
    rewrite could have contained. `as_bool` parses the word.
"""

import csv
import json
import pathlib
import random
import statistics
from collections import Counter
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "source"

AS_OF = "2026-08-01"
SEED = 20260801  # fixed: the fixture must be byte-identical on every rebuild


_BOM = b"\xef\xbb\xbf"
_ZWNBSP = "\ufeff"


def _header_diagnosis(path: pathlib.Path, columns: list[str]) -> str:
    """Say whether a byte-order mark is in play.

    Excel writes a UTF-8 BOM, so the first header cell parses as
    '\ufefftitle' under plain utf-8 and every lookup on that column returns
    nothing. pandas stripped it silently; the csv module does not. The symptom
    surfaces as an empty *first field on row 0*, which sends the reader to the
    data when the problem is in the header — so the header check says so out
    loud instead.
    """
    notes = []
    if path.read_bytes()[:3] == _BOM:
        notes.append(
            "the file starts with a UTF-8 BOM (efbbbf), which is normal for an "
            "Excel export and is stripped by encoding='utf-8-sig'"
        )
    if any(_ZWNBSP in column for column in columns):
        notes.append(
            "a parsed column name still contains U+FEFF, so the BOM was NOT "
            "stripped — the file is being opened as 'utf-8' rather than 'utf-8-sig'"
        )
    return (" (" + "; ".join(notes) + ")") if notes else ""


LISTING_COLUMNS = (
    "ref", "title", "listingKind", "propertyType", "compound", "area", "city",
    "price", "currency", "unitAreaSqm", "bedrooms", "bathrooms", "finish",
    "furnished", "deliveryDate", "address",
)
PROJECT_COLUMNS = ("name", "status")


def read_rows(name: str, required: tuple[str, ...]) -> list[dict[str, str]]:
    """Read one source CSV, checking the header before the data.

    `utf-8-sig`, not `utf-8`: the catalogue exports from Excel and every export
    carries a BOM. Stripping it here rather than asking for clean files is the
    right place — the next export will have one too.
    """
    path = SOURCE / name
    if not path.is_file():
        raise SystemExit(
            f"missing source: {path}\n"
            f"The fixture sources live beside this script so a price in a case "
            f"traces to a row in a file that exists. See MANIFEST.md."
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = [column for column in required if column not in columns]
        if missing:
            raise SystemExit(
                f"{name}: missing column(s) {missing}. "
                f"Header parsed as {columns}{_header_diagnosis(path, columns)}."
            )
        return list(reader)


def as_int(text: str, field: str, row_number: int) -> int:
    number = float(text)
    if not number.is_integer():
        raise SystemExit(f"row {row_number}: {field} is not an integer: {text!r}")
    return int(number)


_TRUE = {"true", "1", "yes", "t"}
_FALSE = {"false", "0", "no", "f", ""}


def as_bool(text: str, field: str, row_number: int) -> bool:
    """Parse the word. Never `bool(text)` — `bool("False")` is True."""
    token = text.strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise SystemExit(f"row {row_number}: {field} is not a boolean: {text!r}")


listings = read_rows("listings.csv", LISTING_COLUMNS)
projects = read_rows("projects.csv", PROJECT_COLUMNS)

# developers.csv is deliberately not read. The pandas version loaded it and
# never referenced a column; carrying it forward would mean shipping a source
# file the build does not depend on, which is provenance that misleads.

# noqa S311: a seeded PRNG is the requirement, not an oversight. These are
# synthetic availability states for a test fixture that must rebuild
# byte-identically; a cryptographic source would make it irreproducible,
# and nothing here protects anything.
rng = random.Random(SEED)  # noqa: S311

# ── synthetic addition 1: availability states ────────────────────────────
# 15 sold, 8 reserved, chosen deterministically across a spread of compounds
# and price bands so no case can pass by accident of clustering.
idx = list(range(len(listings)))
rng.shuffle(idx)
SOLD = set(idx[:15])
RESERVED = set(idx[15:23])

def availability_of(i):
    if i in SOLD:
        return "sold"
    if i in RESERVED:
        return "reserved"
    return "available"

# ── synthetic addition 2: payment plans ──────────────────────────────────
# Terms mirror the Egyptian off-plan market: a down payment percentage, an
# instalment horizon in years, quarterly or monthly frequency. Ready units
# are cash-only, which is also how the market behaves.
PLAN_SHAPES = [
    {"down_payment_pct": 10, "years": 8,  "frequency": "quarterly"},
    {"down_payment_pct": 10, "years": 10, "frequency": "monthly"},
    {"down_payment_pct": 15, "years": 7,  "frequency": "quarterly"},
    {"down_payment_pct": 20, "years": 5,  "frequency": "monthly"},
    {"down_payment_pct": 25, "years": 4,  "frequency": "quarterly"},
    {"down_payment_pct": 5,  "years": 12, "frequency": "monthly"},
]

# Last row wins on a duplicate name, which is what dict(zip(...)) did.
proj_status = {row["name"]: row["status"] for row in projects}

def plan_for(compound, i, price):
    status = proj_status.get(compound, "Under Construction")
    if status in ("Ready to Move", "Completed"):
        return None  # cash only
    shape = PLAN_SHAPES[i % len(PLAN_SHAPES)]
    down = round(price * shape["down_payment_pct"] / 100)
    remaining = price - down
    n = shape["years"] * (4 if shape["frequency"] == "quarterly" else 12)
    # Zero-interest instalments, which is the Egyptian off-plan norm. The last
    # payment absorbs the rounding remainder so the schedule sums exactly to
    # the price — a plan that does not sum is a plan a customer can dispute.
    per = remaining // n
    last = remaining - per * (n - 1)
    return {
        "down_payment_pct": shape["down_payment_pct"],
        "down_payment": down,
        "years": shape["years"],
        "frequency": shape["frequency"],
        "installment_count": n,
        "installment_amount": per,
        "final_installment_amount": last,
        "total": down + per * (n - 1) + last,
        "interest_rate": 0,
    }

units = []
for i, row in enumerate(listings):
    price = as_int(row["price"], "price", i)
    units.append({
        "unit_id": row["ref"],
        "fixture": "broker_demo_2026_08_01",
        "as_of": AS_OF,
        "title": row["title"],
        "listing_kind": row["listingKind"],
        "property_type": row["propertyType"],
        "compound": row["compound"],
        "area": row["area"],
        "city": row["city"],
        "price": price,
        "currency": row["currency"],
        "unit_area_sqm": as_int(row["unitAreaSqm"], "unitAreaSqm", i),
        "bedrooms": as_int(row["bedrooms"], "bedrooms", i),
        "bathrooms": as_int(row["bathrooms"], "bathrooms", i),
        "finish": row["finish"],
        "furnished": as_bool(row["furnished"], "furnished", i),
        "availability": availability_of(i),
        "delivery_date": row["deliveryDate"],
        "project_status": proj_status.get(row["compound"]),
        "payment_plan": plan_for(row["compound"], i, price),
        "address": row["address"],
        "source_row": i,
    })

# ── synthetic addition 3: studio and penthouse units ─────────────────────
# The taxonomy decision of 2026-08-17 splits residential into
# studio | apartment | penthouse. The source export types all 140 residential
# units as `apartment`, so two of the three are empty — and `type_substitution_rate`
# is a zero-tolerance gate that cannot fail against an empty type. A gate with
# no data reads green while untested, which is worse than no gate at all.
#
# Compound and area are an explicit table rather than sampled, because these
# are the fixture's identity: a case citing SYN-STUD-003 must find the same
# unit at the same price on every rebuild, and a reseeded sample would move
# both. `source_row` is null, so one field tells a reader a unit is synthetic.
TIERS = {
    "studio": {
        "prefix": "SYN-STUD",
        "label": "Studio",
        "multiplier": 0.95,
        "bedrooms": 0,
        "bathrooms": 1,
    },
    "penthouse": {
        "prefix": "SYN-PENT",
        "label": "Penthouse",
        "multiplier": 1.35,
        "bedrooms": 3,
        "bathrooms": 3,
    },
}

# (compound, unit_area_sqm). Studio areas 45-68, penthouse 210-275 — disjoint
# ranges, asserted below, so the two types stay distinguishable by size rather
# than only by their label.
SYNTHETIC_UNITS = [
    ("studio", "Mivida", 55),
    ("studio", "ZED East", 48),
    ("studio", "Fifth Square", 62),
    ("studio", "Vinci", 58),
    ("studio", "Noor City", 45),
    ("studio", "Joulz", 68),
    ("studio", "Etapa", 52),
    ("studio", "Bloomfields", 60),
    ("penthouse", "Mivida", 240),
    ("penthouse", "Fifth Square", 275),
    ("penthouse", "ZED East", 210),
    ("penthouse", "Vinci", 230),
    ("penthouse", "Joulz", 265),
    ("penthouse", "ZED West", 220),
]

# Fields a synthetic unit inherits from a real neighbour rather than inventing.
# Copying beats generating: a made-up address in a real compound is the same
# class of error as a made-up compound, just harder to spot.
INHERITED = ("area", "city", "finish", "delivery_date", "project_status", "address")


def reference_unit(compound):
    """The first real unit in `compound`.

    First, not random: the choice has to be stable across rebuilds, and an
    arbitrary-but-fixed neighbour is as representative as any other for the
    fields being copied.
    """
    for unit in units:
        if unit["source_row"] is not None and unit["compound"] == compound:
            return unit
    raise SystemExit(
        f"synthetic unit names compound {compound!r}, which is not in the real export"
    )


def median_apartment_rate(city):
    """Median EGP/sqm for real apartments in `city`.

    Prices are derived, not invented. A New Cairo studio is priced off the 71
    real New Cairo apartments in the export, so the synthetic units stay
    internally consistent with the catalogue around them — a case comparing a
    studio to an apartment is comparing two numbers from the same market.
    """
    rates = [
        unit["price"] / unit["unit_area_sqm"]
        for unit in units
        if unit["source_row"] is not None
        and unit["city"] == city
        and unit["property_type"] == "apartment"
    ]
    if not rates:
        raise SystemExit(f"no real apartments in {city!r} to price against")
    return statistics.median(rates)


tier_counts = dict.fromkeys(TIERS, 0)

for i, (tier_name, compound, area_sqm) in enumerate(SYNTHETIC_UNITS):
    tier = TIERS[tier_name]
    tier_counts[tier_name] += 1
    reference = reference_unit(compound)
    price = int(round(median_apartment_rate(reference["city"]) * area_sqm * tier["multiplier"], -4))
    units.append({
        # Numbered within the tier, so ids read SYN-STUD-001..008 and
        # SYN-PENT-001..006 rather than sharing one counter.
        "unit_id": f"{tier['prefix']}-{tier_counts[tier_name]:03d}",
        "fixture": "broker_demo_2026_08_01",
        "as_of": AS_OF,
        "title": f"{tier['label']} for sale in {compound}",
        "listing_kind": "sale",
        "property_type": tier_name,
        "compound": compound,
        "area": reference["area"],
        "city": reference["city"],
        "price": price,
        "currency": "EGP",
        "unit_area_sqm": area_sqm,
        "bedrooms": tier["bedrooms"],
        "bathrooms": tier["bathrooms"],
        "finish": reference["finish"],
        "furnished": False,
        # Synthetic units are never sold or reserved: the availability spread
        # in addition 1 is drawn across the real export, and adding these to it
        # would move the 15/8 counts the cases depend on.
        "availability": "available",
        "delivery_date": reference["delivery_date"],
        "project_status": reference["project_status"],
        "payment_plan": plan_for(compound, i, price),
        "address": reference["address"],
        # Null marks a unit as synthetic in one field, without a parallel list
        # anyone has to keep in step.
        "source_row": None,
    })


with open("units.jsonl", "w", encoding="utf-8") as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")

# ── integrity assertions ─────────────────────────────────────────────────
ids = [u["unit_id"] for u in units]
assert len(ids) == len(set(ids)), "duplicate unit_id"

avail = Counter(u["availability"] for u in units)
assert avail["sold"] == 15 and avail["reserved"] == 8, avail

# every plan must sum exactly to the price — a schedule that does not
# reconcile is one a customer can dispute, and the arithmetic gate would
# then be checking against a wrong total
for u in units:
    p = u["payment_plan"]
    if p:
        assert p["total"] == u["price"], (u["unit_id"], p["total"], u["price"])
        assert p["down_payment"] + p["installment_amount"] * (p["installment_count"] - 1) \
               + p["final_installment_amount"] == u["price"]

# ── synthetic addition 3 assertions ──────────────────────────────────────
synthetic = [u for u in units if u["source_row"] is None]
by_type = Counter(u["property_type"] for u in units)

# A zero-tolerance gate cannot fail against an empty type. If any of the three
# residential types is empty, type_substitution_rate is decorative.
for residential in ("studio", "apartment", "penthouse"):
    assert by_type[residential] > 0, f"no {residential} units — the type gate is untestable"
assert by_type["studio"] == 8, by_type
assert by_type["penthouse"] == 6, by_type

# Disjoint areas. If the ranges overlap the two types stop being
# distinguishable by size, and the distinction becomes assertion rather than
# data — a case could no longer tell a mis-typed unit from a small penthouse.
studio_areas = [u["unit_area_sqm"] for u in units if u["property_type"] == "studio"]
penthouse_areas = [u["unit_area_sqm"] for u in units if u["property_type"] == "penthouse"]
assert max(studio_areas) < min(penthouse_areas), (max(studio_areas), min(penthouse_areas))

# No synthetic unit may name a compound that the real export does not place in
# that city. This caught three errors during authoring — real compound names
# assigned to the wrong city — and each would have produced a unit that looks
# entirely legitimate: right compound, right price band, wrong place. A case
# filtering by city would then return it, and the reply would send a customer
# to a compound that is not there.
real_pairs = {(u["compound"], u["city"]) for u in units if u["source_row"] is not None}
for u in synthetic:
    assert (u["compound"], u["city"]) in real_pairs, (
        f"{u['unit_id']}: {u['compound']!r} is not in {u['city']!r} in the real export"
    )

# ready units must be cash-only, off-plan must have a plan
for u in units:
    if u["project_status"] in ("Ready to Move", "Completed"):
        assert u["payment_plan"] is None, u["unit_id"]

with_plan = sum(1 for u in units if u["payment_plan"])
print(f"{len(units)} units ({len(synthetic)} synthetic)")
print(f"  availability: {dict(avail)}")
print(f"  with payment plan: {with_plan}  cash-only: {len(units) - with_plan}")
print(f"  types: {dict(sorted(by_type.items()))}")
print(f"  as_of: {AS_OF}  seed: {SEED}")
