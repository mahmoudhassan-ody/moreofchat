# Fixture: `sinai_demo`

**Frozen:** 2026-08-17
**Vertical:** education
**Source:** Sinai University call-centre knowledge base, exported as parallel Arabic/English CSVs (51 Q&A pairs each, row-aligned).
**Contents:** 102 chunks — 51 bilingual pairs, one chunk per language.

This is **test input**, not production KB content. Its purpose is to stay still: when a case asserts "the reply must cite `sinai_fee_application_initial_ar`", that chunk has to mean the same thing in six months, or every eval baseline measured against it becomes meaningless.

Live tenant KB content is separate — uploaded through the console into `kb_documents` / `kb_chunks`, editable per tenant, and never sourced from here.

---

## Do not "fix" these

Four gaps and oddities are load-bearing. Cases fail against them deliberately. Closing one silently turns a passing case into a case that passes for the wrong reason.

### 1. There is no tuition data. At all.

The KB covers the application fee (2000 EGP), the placement-test fee (1000), ministry registration (100), track change (500), and discounts (30% Qantara / 40% Arish, 10% annual increase). It contains **no per-faculty tuition or credit-hour fee**.

So `المصاريف كام لكلية الهندسة؟` — the most-asked question in the vertical — is **unanswerable** from this fixture. That is the point:

- The correct behaviour is **handoff**, not an estimate.
- The specific failure to catch is the model reaching for the 2000 EGP application fee and presenting it as tuition. Adjacent, plausible, wrong, and expensive.
- `build.py` asserts `"credit hour" not in` the corpus, so tuition data cannot drift in unnoticed.

Real tuition figures belong in the live tenant KB via the admin UI (design §19), not here.

### 2. Fee chunks carry no academic year

`effective_from` and `effective_to` are `null` on every chunk except the admission thresholds, because the source states no year. Cases assert the reply **discloses the absence** rather than inventing a year — a stronger test than asserting a year the data never had.

### 3. The admission thresholds are 2025 figures relabelled 2026

Row 38. The source says 2025; the owner's decision was to treat them as the 2026 cycle. `build.py` replaces every `2025` with `2026` in both languages and asserts the relabel is complete, since a chunk saying "2026" in the heading and "2025" in the body would be self-contradictory and would break any case asserting the year.

**These are not the real 2026 Tansik thresholds.** Nobody knows what those are yet. Correct for testing grounding; **wrong as production content** — a student choosing a faculty on a wrong threshold is a real harm, not a test artifact. Real figures go in through the UI.

### 4. Arabic titles were disambiguated

Rows 22–25 and 28–31 had **identical Arabic titles** with different content — domestic applicants versus international. The English titles distinguished them; the Arabic ones did not, so an Arabic query hit two chunks with conflicting answers and fusion picked one at random. Broken on the primary language.

Added: `(للطلاب المصريين)` to the domestic set, `(للطلاب الوافدين)` to the international set. `build.py` asserts no duplicate titles remain in either language.

---

## Edits applied to the source

Every change, with its reason. `build.py` is the only thing that writes `chunks.jsonl` — regenerate rather than hand-edit.

| Row | Lang | Edit | Why |
|---|---|---|---|
| 36 | ar | `كل ماهو يخص السكن` → `كل ماهو يخص المواصلات` | Source copy-paste: the Arabic transport entry described housing. The English was correct. Left as-is, an Arabic transport query retrieves a chunk whose text is about housing — the lexical arm misses on the language that matters most. |
| 38 | ar, en | `2025` → `2026` throughout | Owner decision (see gap 3) |
| 22–25 | ar | title + `(للطلاب المصريين)` | Disambiguation (see gap 4) |
| 28–31 | ar | title + `(للطلاب الوافدين)` | Disambiguation (see gap 4) |

Not edited, but worth knowing:

- **`categoryId` is unusable** — `NaN` in the Arabic export, the literal string `"English "` in the English one. Dropped; `topic` is derived from content instead.
- **Rows 39–47 are department contact lists**, not Q&A. Different shape; `topic: contacts`. They will want different chunking when real ingestion lands in P1.
- **Institutional contact details are retained** — role addresses (`oss.a@su.edu.eg`) and departmental phone numbers (`01013331500`). Work contact info, no personal data. Note the numeric extractor treats phone numbers as non-figures, so they do not enter grounding checks.
- **`entity_ref` is sparse on purpose.** Only branch and transfer-direction tags, where the source states them. Inventing entity tags would be fabricated metadata dressed as fixture data.

---

## Schema

One JSON object per line in `chunks.jsonl`:

| Field | Notes |
|---|---|
| `chunk_id` | `sinai_<slug>_<lang>` — stable, referenced by cases. Never reuse or rename. |
| `doc_id` | `sinai_<slug>` — shared by the ar/en pair |
| `tenant_fixture` | `sinai_demo` |
| `vertical` | `education` |
| `lang` | `ar` or `en` |
| `topic` | fees, discounts, documents, transfer, programs, admission, campus_services, contacts, locations, institution |
| `entity_ref` | `branch:arish`, `branch:qantara`, `branch:sama_tower`, `transfer:domestic`, `transfer:international`, or `null` |
| `title` | the question, as the KB states it |
| `content` | the answer |
| `effective_from` / `effective_to` | `null` except the admission thresholds |
| `source_row` | index in the original CSV — provenance, so any figure traces back |

## Figure-bearing chunks

Grounding targets. Any figure in a reply must trace to one of these:

| Chunk | Figure |
|---|---|
| `sinai_fee_application_initial_*` | 2000 EGP |
| `sinai_fee_application_service_*` | 1000 EGP (application office) |
| `sinai_fee_english_placement_test_*` | 1000 EGP |
| `sinai_fee_ministry_registration_*` | 100 EGP |
| `sinai_fee_track_change_*` | 500 EGP |
| `sinai_discount_tuition_*` | 30% Qantara, 40% Arish |
| `sinai_tuition_annual_increase_*` | 10% per year |
| `sinai_differentiators_*` | admission 5% below other universities |
| `sinai_admission_thresholds_2026_*` | per-faculty per-branch percentages |
| `sinai_founding_*` | 2006 (a year, not a quantity — the extractor must reject it) |

## Regenerating

```bash
python3 build.py    # reads the two source CSVs, writes chunks.jsonl
```

The build asserts: unique `chunk_id`s, no duplicate titles within a language, no tuition data present, and a complete year relabel. A failed assertion means the fixture is broken — fix the build, not the assertion.
