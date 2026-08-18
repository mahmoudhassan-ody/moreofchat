import json

import pandas as pd

ar = pd.read_csv("/mnt/user-data/uploads/kb_translated_full_Arabic.csv")
en = pd.read_csv("/mnt/user-data/uploads/kb_translated_full_English1.csv")

SLUGS = [
    "founding", "president", "faculties", "accreditation", "differentiators",
    "application_methods", "application_eligibility",
    "fee_application_initial", "fee_application_service", "fee_application_refund",
    "fee_english_placement_test", "english_test_absence",
    "fee_ministry_registration", "fee_ministry_refund", "fee_track_change",
    "discount_tuition", "tuition_annual_increase", "discount_combination",
    "discount_gpa_requirement", "discount_summer_internship",
    "discount_faculty_transfer", "discount_withdraw_reapply",
    "docs_general_azhari", "docs_arab_equivalent", "docs_american_diploma", "docs_igcse",
    "transfer_inside_egypt", "transfer_outside_egypt",
    "docs_intl_general_azhari", "docs_intl_arab_equivalent",
    "docs_intl_american_diploma", "docs_intl_igcse",
    "programs_arish", "programs_qantara",
    "housing_availability", "housing_booking",
    "transport_availability", "transport_booking",
    "admission_thresholds_2026",
    "contact_transportation", "contact_dorms", "contact_oss",
    "contact_financial_affairs", "contact_internship", "contact_hiring",
    "contact_marketing", "contact_alumni", "accreditations_international",
    "location_arish", "location_qantara", "location_sama_tower",
]
assert len(SLUGS) == 51

# Rows 28-31: identical Arabic titles to 22-25. Disambiguate the international set.
INTL_QUALIFIER = " (للطلاب الوافدين)"
INTL_ROWS = {28, 29, 30, 31}
# and mark the domestic set explicitly, so neither is the ambiguous default
DOM_QUALIFIER = " (للطلاب المصريين)"
DOM_ROWS = {22, 23, 24, 25}

TOPIC = {
    "fees": {7, 8, 9, 10, 12, 13, 14},
    "discounts": {15, 16, 17, 18, 19, 20, 21},
    "documents": {22, 23, 24, 25, 28, 29, 30, 31},
    "transfer": {26, 27},
    "programs": {2, 32, 33},
    "admission": {5, 6, 11, 38},
    "campus_services": {34, 35, 36, 37},
    "contacts": {39, 40, 41, 42, 43, 44, 45, 46, 47},
    "locations": {48, 49, 50},
    "institution": {0, 1, 3, 4},
}
def topic_of(i):
    for t, rows in TOPIC.items():
        if i in rows:
            return t
    raise AssertionError(i)

# Only row 38 carries a year in the source. Everything else has none.
EFFECTIVE = {38: {"effective_from": "2026-01-01", "effective_to": "2026-12-31"}}
DEFAULT_EFF = {"effective_from": None, "effective_to": None}

# Entities a case can filter on. Deliberately sparse — invented entity tags
# would be fabricated metadata.
ENTITY = {
    32: "branch:arish", 33: "branch:qantara",
    48: "branch:arish", 49: "branch:qantara", 50: "branch:sama_tower",
    26: "transfer:domestic", 27: "transfer:international",
}

# ---- documented edits to the source, each with a reason ----
# Every change here is recorded in MANIFEST.md. The fixture is test input,
# not a copy of production KB content.
EDITS = []

def apply_edits(i, lang, content):
    # 1. Relabel the 2025 admission cycle as 2026, per the owner's decision.
    #    These are 2025 figures wearing a 2026 label: correct for testing
    #    grounding (the case asserts the reply cites the chunk), WRONG as
    #    production KB content. See MANIFEST.md.
    if i == 38:
        before = content
        content = content.replace("2025", "2026")
        if content != before:
            EDITS.append((i, lang, "relabelled 2025 -> 2026"))

    # 2. Source bug: the Arabic transport entry says "السكن" (housing) where
    #    the English correctly says transportation — copy-pasted from the
    #    housing entry. Left as-is, an Arabic transport query retrieves a
    #    chunk whose text is about housing, so the lexical arm misses on the
    #    one language that matters most.
    if i == 36 and lang == "ar":
        before = content
        content = content.replace("كل ماهو يخص السكن", "كل ماهو يخص المواصلات")
        if content != before:
            EDITS.append((i, lang, "source copy-paste: housing -> transport"))

    return content


chunks = []
for i in range(51):
    slug = SLUGS[i]
    topic = topic_of(i)
    eff = EFFECTIVE.get(i, DEFAULT_EFF)

    ar_title = str(ar.title[i]).strip()
    en_title = str(en.title[i]).strip()
    if i in INTL_ROWS:
        ar_title += INTL_QUALIFIER
    if i in DOM_ROWS:
        ar_title += DOM_QUALIFIER

    for lang, title, content in (
        ("ar", ar_title, apply_edits(i, "ar", str(ar.content[i]).strip())),
        ("en", en_title, apply_edits(i, "en", str(en.content[i]).strip())),
    ):
        chunks.append({
            "chunk_id": f"sinai_{slug}_{lang}",
            "doc_id": f"sinai_{slug}",
            "tenant_fixture": "sinai_demo",
            "vertical": "education",
            "lang": lang,
            "topic": topic,
            "entity_ref": ENTITY.get(i),
            "title": title,
            "content": content,
            "effective_from": eff["effective_from"],
            "effective_to": eff["effective_to"],
            "source_row": i,
        })

with open("chunks.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# ---- integrity assertions: the fixture must not ship broken ----
ids = [c["chunk_id"] for c in chunks]
assert len(ids) == len(set(ids)), "duplicate chunk_id"

for lang in ("ar", "en"):
    titles = [c["title"] for c in chunks if c["lang"] == lang]
    dupes = {t for t in titles if titles.count(t) > 1}
    assert not dupes, f"{lang} duplicate titles remain: {dupes}"

# the deliberate gap
joined = " ".join(c["content"] for c in chunks)
assert "credit hour" not in joined.lower(), "tuition data leaked into the fixture"

# the relabel must be complete, not partial
thr = [c for c in chunks if c["doc_id"] == "sinai_admission_thresholds_2026"]
for c in thr:
    assert "2025" not in c["content"], "partial relabel leaves the chunk self-contradictory"
    assert "2026" in c["content"], "relabel did not apply"

print(f"{len(chunks)} chunks written ({len(chunks)//2} bilingual pairs)")
print("edits applied:")
for e in EDITS:
    print("  row", e[0], e[1], "-", e[2])
print("no duplicate ids, no duplicate titles in either language")
