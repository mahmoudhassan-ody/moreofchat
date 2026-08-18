# Fixture: `broker_demo_2026_08_01`

**Frozen:** 2026-08-17
**`as_of`:** 2026-08-01 (every case asserting freshness cites this date)
**Vertical:** realestate
**Source:** `source/listings.csv` (291 units), `source/projects.csv` (97), both committed beside `build.py`
**Contents:** 291 units in `units.jsonl`, one JSON object per line.

Test input, not production catalogue data. Its purpose is to stay still: a case asserting "the reply must cite the instalment from `NOOR-CIT-002-02`" needs that unit and that schedule to mean the same thing in six months.

Live inventory syncs from the tenant's own source (sheet, CRM, or API) into the production catalogue. Nothing flows from here.

---

## Synthetic additions

The source export could not support the highest-risk real-estate cases. Three fields were uniform across all 291 rows, and payment terms were absent entirely:

| Field | Source export | Cases that need variation |
|---|---|---|
| `availability` | 100% `available` | `sold_or_reserved` (8) |
| `status` | 100% `active` | same |
| `listingKind` | 100% `sale` | rent cases |
| payment plan | no columns | `payment_plan_math` (15) |
| `amenities` | empty on all 97 projects | amenity queries |

Two additions were made. Both are deterministic — `SEED = 20260801`, so a rebuild is byte-identical.

### 1. Availability states

**15 units marked `sold`, 8 marked `reserved`**, chosen by seeded shuffle. The spread is deliberate: 9 cities, prices from 3.9M to 33.7M, both 2BR and 3BR. Clustering them into one compound or one price band would let a case pass by accident — a filter that happened to exclude them would look like correct behaviour.

Everything else stays `available`.

### 2. Payment plans

180 units carry a plan; 111 are cash-only. The split follows project status: **`Ready to Move` and `Completed` units are cash-only**, off-plan units carry a plan. That mirrors the Egyptian market rather than being arbitrary.

Six plan shapes cycle across units — 5% to 25% down, 4 to 12 years, monthly or quarterly. Zero interest, which is the Egyptian off-plan norm.

**The rounding behaviour is the point.** Instalments are integer EGP and the final payment absorbs the remainder, so the schedule sums exactly to the price. 120 of 180 plans therefore have a final instalment that differs from the regular one:

```
NOOR-CIT-002-02   6,450,000 EGP   25% down, 4 years quarterly
  down_payment              1,612,500
  installment_amount          302,343  × 15
  final_installment_amount    302,355
  total                     6,450,000   (exact)
```

A model doing the division itself computes 4,837,500 ÷ 16 = 302,343.75 and rounds to **302,344** — wrong by 1 EGP against the calculator, and caught by `arithmetic_in_model_rate`. That is the arithmetic rule demonstrated rather than asserted, and it is why the fixture rounds this way instead of choosing prices that divide evenly.

A plan that does not reconcile to the price is one a customer can dispute, so `build.py` asserts every schedule sums exactly.

### Not added

- **`amenities`** stays empty. Inventing amenity lists would be fabricated metadata dressed as fixture data, and no case currently depends on it.
- **Rent listings.** The export is sale-only. Rent cases would need `rentPrice` and `rentPeriod`, which the source does not have; adding them means inventing a rental market. Deferred until real rent data exists.

---

## Do not "fix" these

- **The 15 sold and 8 reserved units are load-bearing.** `sold_unit_offered_rate` is a zero-tolerance gate; marking them all available again silently removes the only data those cases can fail against.
- **The rounding remainders are load-bearing.** Replacing them with round numbers makes the arithmetic gate untestable — a model that guesses correctly is indistinguishable from a model that used the tool.
- **`as_of` is fixed at 2026-08-01.** Cases assert the reply discloses it. Moving it breaks every staleness case.

---

## Schema

| Field | Notes |
|---|---|
| `unit_id` | the export's `ref`, e.g. `NOOR-CIT-002-02`. Stable, referenced by cases. |
| `fixture`, `as_of` | fixture identity and the snapshot date |
| `availability` | `available` \| `reserved` \| `sold` — **synthetic** |
| `project_status` | joined from the projects export; drives the cash-only rule |
| `payment_plan` | object or `null` — **synthetic**. See fields below. |
| `price`, `currency` | from the export, unaltered |
| `compound`, `area`, `city` | from the export |
| `property_type` | apartment, townhouse, chalet, office, retail, duplex, villa |
| `unit_area_sqm`, `bedrooms`, `bathrooms`, `finish`, `furnished` | from the export |
| `delivery_date` | from the export, 2025-04-05 → 2030-12-01 |
| `source_row` | index in the original CSV — provenance for any figure |

`payment_plan`: `down_payment_pct`, `down_payment`, `years`, `frequency`, `installment_count`, `installment_amount`, `final_installment_amount`, `total`, `interest_rate`.

## Distribution

| | |
|---|---|
| Cities | New Cairo 117, New Capital 60, North Coast 33, West Cairo 30, Ain Sokhna 15, Mostakbal City 9, Sheikh Zayed 6, Red Sea 6, Mokattam 3, October 3, New Zayed 3, New Alamein 3, Cairo 3 |
| Types | apartment 140, townhouse 75, chalet 33, office 16, retail 13, duplex 9, villa 5 |
| Price | 3.3M – 65.5M EGP, median 10.9M |
| Area | 70 – 390 sqm, median 145 |
| Bedrooms | 0 – 5, median 3 |
| Availability | available 268, sold 15, reserved 8 |
| Payment | 180 with a plan, 111 cash-only |

Note **villa is only 5 units** — the thinnest type, and the one customers ask about most in the sample conversations. A "no villas match" reply is often the *correct* answer here, which makes it good material for a case asserting the bot says so plainly rather than substituting an apartment unasked.

## A source bug worth fixing outside this fixture

The vocabulary reference has two corrupt location aliases:

```
| العLMين الجديدة | new alamein |
| العLMين        | new alamein |
```

Latin `LM` spliced into Arabic — should be `العلمين`. Every Arabic query for El Alamein misses, and it presents as "the bot doesn't cover Alamein." Not a fixture issue; it belongs in the lexicon config.

## Regenerating

```bash
cd evals/fixtures/broker_demo_2026_08_01 && python3 build.py
```

Stdlib only — no pandas, no virtualenv, no arguments. It reads `source/` beside
itself and writes `units.jsonl` into the working directory.

| File | Was |
|---|---|
| `source/listings.csv` | `listings-egypt-filled.csv` (291 rows) |
| `source/projects.csv` | `projects-egypt-filled.csv` (97 rows) |

`developers-egypt-filled.csv` is **not** carried over. The build loaded it and
never read a column from it; shipping a source file the build does not depend
on is provenance that misleads.

The sources are committed. A price a case asserts against has to trace to a row
in a file that exists, and `tests/evals/test_fixture_rebuild.py` runs this build
into a temp directory and compares the result byte for byte against the
committed `units.jsonl` — which is what makes `SEED = 20260801` a checked claim
rather than an intention.

Asserts: unique unit ids, exactly 15 sold and 8 reserved, every payment schedule sums exactly to its price, and no ready-to-move unit carries a plan. A failed assertion means the fixture is broken — fix the build, not the assertion.
