"""Build the broker_demo_2026_08_01 fixture from the real listings export.

Synthetic additions are deterministic (seeded) and documented in MANIFEST.md.
The source export has one availability state, one status, one listing kind and
no payment-plan columns, so the highest-risk real-estate cases could not exist
against it unaltered.
"""
import json
import random

import pandas as pd

AS_OF = "2026-08-01"
SEED = 20260801  # fixed: the fixture must be byte-identical on every rebuild

listings = pd.read_csv("/mnt/user-data/uploads/listings-egypt-filled.csv")
projects = pd.read_csv("/mnt/user-data/uploads/projects-egypt-filled.csv")
developers = pd.read_csv("/mnt/user-data/uploads/developers-egypt-filled.csv")

rng = random.Random(SEED)

# ── synthetic addition 1: availability states ────────────────────────────
# 15 sold, 8 reserved, chosen deterministically across a spread of compounds
# and price bands so no case can pass by accident of clustering.
idx = list(listings.index)
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

proj_status = dict(zip(projects.name, projects.status))

def plan_for(row, i):
    status = proj_status.get(row.compound, "Under Construction")
    if status in ("Ready to Move", "Completed"):
        return None  # cash only
    shape = PLAN_SHAPES[i % len(PLAN_SHAPES)]
    price = int(row.price)
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
for i, row in listings.iterrows():
    plan = plan_for(row, i)
    units.append({
        "unit_id": row.ref,
        "fixture": "broker_demo_2026_08_01",
        "as_of": AS_OF,
        "title": row.title,
        "listing_kind": row.listingKind,
        "property_type": row.propertyType,
        "compound": row.compound,
        "area": row.area,
        "city": row.city,
        "price": int(row.price),
        "currency": row.currency,
        "unit_area_sqm": int(row.unitAreaSqm),
        "bedrooms": int(row.bedrooms),
        "bathrooms": int(row.bathrooms),
        "finish": row.finish,
        "furnished": bool(row.furnished),
        "availability": availability_of(i),
        "delivery_date": row.deliveryDate,
        "project_status": proj_status.get(row.compound),
        "payment_plan": plan,
        "address": row.address,
        "source_row": int(i),
    })

with open("units.jsonl", "w", encoding="utf-8") as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")

# ── integrity assertions ─────────────────────────────────────────────────
ids = [u["unit_id"] for u in units]
assert len(ids) == len(set(ids)), "duplicate unit_id"

from collections import Counter
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

# ready units must be cash-only, off-plan must have a plan
for u in units:
    if u["project_status"] in ("Ready to Move", "Completed"):
        assert u["payment_plan"] is None, u["unit_id"]

with_plan = sum(1 for u in units if u["payment_plan"])
print(f"{len(units)} units")
print(f"  availability: {dict(avail)}")
print(f"  with payment plan: {with_plan}  cash-only: {len(units) - with_plan}")
print(f"  as_of: {AS_OF}  seed: {SEED}")
