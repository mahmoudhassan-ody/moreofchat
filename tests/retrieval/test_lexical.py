"""The Meilisearch arm against a real Meilisearch — design §7.4.

Real, not mocked. Everything worth testing here is behaviour of the engine:
whether the synonym map actually bridges منحة to خصم, whether typo tolerance
survives the normalizer, whether a tenant-scoped search genuinely cannot see
another tenant's documents. A mock confirms the settings this code sends and
nothing about what they do.

**The acceptance test is `test_synonym_bridges_a_zero_overlap_query`** —
edu-0015, "fe manh fe kantara" reaching `sinai_discount_tuition_ar`. The
question and the answer share no letters: the student says منحة, the knowledge
base says خصم and never says منحة. Without the synonym map every student who
asks about a scholarship gets nothing, and the failure reads as "the bot
doesn't cover scholarships" rather than as a missing config entry.
"""

import json
import time
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from moc.config_store import load
from moc.retrieval.fusion import Candidate, fuse
from moc.retrieval.lexical import (
    LexicalDocument,
    MeilisearchAdmin,
    MeilisearchRepository,
    index_for,
    strip_stop_words,
)

CONFIG = load("retrieval/lexical")
FUSION = CONFIG["fusion"]

SINAI = Path(__file__).parents[2] / "evals" / "fixtures" / "sinai_demo" / "chunks.jsonl"

TEST_INDEXES = {"education": "test_lex_education", "realestate": "test_lex_realestate"}
TEST_CONFIG = {
    **CONFIG,
    "meilisearch": {**CONFIG["meilisearch"], "indexes": TEST_INDEXES},
}

DISCOUNT = "sinai_discount_tuition_ar"
HOUSING = "sinai_housing_availability_ar"


def fixture_chunks() -> list[dict]:
    return [json.loads(line) for line in SINAI.read_text(encoding="utf-8").splitlines()]


@pytest_asyncio.fixture(loop_scope="session")
async def client():
    """Real Meilisearch, on indexes named apart from the dev ones.

    Fails in CI, skips locally — the same rule as Valkey and Qdrant. A skipped
    retrieval test is green and proves nothing, and this file holds the P1
    milestone.
    """
    import os

    from meilisearch_python_sdk import AsyncClient

    from moc.config import settings

    connection = AsyncClient(
        f"http://{settings.meili_host}:{settings.meili_port}", settings.meili_key or None
    )
    try:
        await connection.health()
    except Exception as exc:
        message = f"meilisearch unreachable: {exc}"
        if os.environ.get("CI"):
            pytest.fail(
                f"{message}. CI brings the stack up with compose, so this is a broken "
                f"run rather than a missing dependency."
            )
        pytest.skip(f"{message}. Start it with: docker compose up -d meilisearch")

    for name in TEST_INDEXES.values():
        await connection.delete_index_if_exists(name)
    await MeilisearchAdmin(client=connection, config=TEST_CONFIG).ensure_indexes()
    yield connection
    for name in TEST_INDEXES.values():
        await connection.delete_index_if_exists(name)
    await connection.aclose()


@pytest_asyncio.fixture(loop_scope="session")
async def corpus(client):
    """The frozen sinai fixture, loaded once, under one tenant."""
    tenant = uuid.uuid4()
    repository = MeilisearchRepository(client=client, config=TEST_CONFIG)
    await repository.add(
        tenant_id=tenant,
        vertical="education",
        documents=[
            LexicalDocument(
                point_id=f"{tenant}-{record['chunk_id']}",
                chunk_id=record["chunk_id"],
                content=record["content"],
                title=record["title"],
                payload={"lang": record["lang"], "topic": record["topic"]},
            )
            for record in fixture_chunks()
        ],
    )
    return repository, tenant


async def chunk_ids(repository, tenant, query: str, limit: int = 20) -> list[str]:
    hits = await repository.search(
        tenant_id=tenant, vertical="education", query=query, limit=limit
    )
    return [hit.chunk_id for hit in hits]


# ─────────────────────────── the acceptance test ───────────────────────────


async def test_synonym_bridges_a_zero_overlap_query(corpus, capsys):
    """edu-0015 — the P1 retrieval milestone.

    "fe manh fe kantara" against a corpus that says خصم and never says منحة.
    Zero lexical overlap: no shared word, no shared root, no shared letter
    sequence. It reaches the chunk only because the synonym map says a منحة is
    a خصم in this tenant's vocabulary, and because `kantara` maps to القنطرة.

    If this fails, every student who says منحة gets nothing.
    """
    repository, tenant = corpus
    found = await chunk_ids(repository, tenant, "fe manh fe kantara?")

    with capsys.disabled():
        print(f"\n  edu-0015 'fe manh fe kantara?' -> {found[:3]}")

    assert DISCOUNT in found, (
        f"the synonym map did not bridge منحة to خصم; got {found[:5]}. "
        f"Check config/retrieval/lexical.yaml synonyms."
    )


async def test_the_query_and_the_chunk_really_share_nothing(corpus):
    """Guards the acceptance test against passing for the wrong reason.

    If the query and the chunk ever come to share a word, the test above stops
    measuring the synonym map and starts measuring ordinary matching — still
    green, no longer the milestone.
    """
    chunk = next(c for c in fixture_chunks() if c["chunk_id"] == DISCOUNT)
    body = set(chunk["content"].split()) | set(chunk["title"].split())
    assert not body & set("fe manh fe kantara?".split()), (
        "the query now shares a token with the chunk — the synonym map is no longer "
        "what makes edu-0015 pass"
    )


# ─────────────────────────── the other retrieval cases ───────────────────────────


async def test_franco_arab_query_retrieves_the_arabic_chunk(corpus):
    """edu-0005: '3ayez a3raf el khasm f far3 el arish'."""
    repository, tenant = corpus
    assert DISCOUNT in await chunk_ids(repository, tenant, "3ayez a3raf el khasm f far3 el arish")


async def test_ta_marbuta_and_hamza_folding_still_match(corpus):
    """edu-0006: القنطره vs القنطرة.

    The corpus writes القنطرة; the customer types القنطره. Only the normalized
    field carries both to the same form.
    """
    repository, tenant = corpus
    assert HOUSING in await chunk_ids(repository, tenant, "السكن الجامعي موجود في القنطره؟")


async def test_the_arabic_spelling_in_the_corpus_also_matches(corpus):
    """Folding must not break the spelling the tenant actually wrote."""
    repository, tenant = corpus
    assert HOUSING in await chunk_ids(repository, tenant, "السكن الجامعي في القنطرة")


# ─────────────────────────── config ───────────────────────────


async def test_arabic_stopwords_and_synonyms_come_from_config(client):
    """§19. A vocabulary entry is a tenant's own knowledge, and it must not
    take a deploy to add."""
    settings = await MeilisearchAdmin(client=client, config=TEST_CONFIG).settings_for(
        vertical="education"
    )
    # Compared by count and by round-trip, not by exact string equality:
    # Meilisearch stores its own normalized form of an Arabic stop word
    # (`في` comes back as `فی`, with the Persian yeh), so an equality
    # assertion would fail on a setting that was applied correctly.
    assert len(settings.stop_words) == len(CONFIG["stop_words"])
    assert {"el", "fe", "the"} <= set(settings.stop_words), (
        "the Latin stop words round-trip unchanged, so a mismatch here is a real one"
    )
    assert "منحة" in settings.synonyms
    assert "خصم" in settings.synonyms["منحة"]


async def test_both_text_forms_are_searchable(client):
    settings = await MeilisearchAdmin(client=client, config=TEST_CONFIG).settings_for(
        vertical="education"
    )
    assert "content" in settings.searchable_attributes
    assert "content_normalized" in settings.searchable_attributes


def test_indexes_are_per_vertical_not_per_tenant():
    """§7.4: index-per-tenant multiplies Meilisearch's per-index overhead by
    the tenant count, which on 3.3 GB binds early."""
    assert index_for("education", config=TEST_CONFIG) != index_for(
        "realestate", config=TEST_CONFIG
    )
    with pytest.raises(KeyError):
        index_for("healthcare", config=TEST_CONFIG)


# ─────────────────────────── tenancy ───────────────────────────


async def test_tenant_scoped_search_isolates_indexes(client):
    """One shared index, two tenants, the same words in both documents.

    Ranking cannot separate them — only the scope can.
    """
    repository = MeilisearchRepository(client=client, config=TEST_CONFIG)
    a, b = uuid.uuid4(), uuid.uuid4()
    for tenant, chunk_id in ((a, "a-doc"), (b, "b-doc")):
        await repository.add(
            tenant_id=tenant,
            vertical="education",
            documents=[
                LexicalDocument(
                    point_id=f"{tenant}-{chunk_id}",
                    chunk_id=chunk_id,
                    content="خصم على المصروفات الدراسية",
                )
            ],
        )

    assert await chunk_ids(repository, a, "خصم") == ["a-doc"]
    assert await chunk_ids(repository, b, "خصم") == ["b-doc"]


async def test_a_tenant_with_no_documents_gets_nothing(client):
    repository = MeilisearchRepository(client=client, config=TEST_CONFIG)
    assert await chunk_ids(repository, uuid.uuid4(), "خصم") == []


# ─────────────────────────── fusion, end to end ───────────────────────────


async def test_the_lexical_arm_feeds_fusion_and_the_gate_opens(corpus, capsys):
    """The whole retrieval path for edu-0015, with the dense arm absent.

    §7.3's degraded shape: embeddings down, Meilisearch only, and the turn
    still answers. Latency is recorded here rather than discovered later —
    this is the measurement that includes a real network round trip.
    """
    repository, tenant = corpus
    started = time.perf_counter()
    hits = await repository.search(
        tenant_id=tenant,
        vertical="education",
        query="fe manh fe kantara?",
        limit=FUSION["candidates_per_arm"],
    )
    search_ms = (time.perf_counter() - started) * 1000

    result = await fuse(
        query="fe manh fe kantara?",
        dense=None,
        sparse=[
            Candidate(
                chunk_id=hit.chunk_id,
                rank=hit.rank,
                # Rank-derived relevance: Meilisearch ranks but does not return
                # a calibrated 0-1 score by default. Honest stand-in until the
                # dense arm supplies a real one — and it is the dense arm's
                # cosine that the §7.5 threshold is ultimately tuned against.
                relevance=1.0 / hit.rank,
                content=hit.content,
            )
            for hit in hits
        ],
        config=CONFIG,
    )

    with capsys.disabled():
        print(
            f"\n  lexical search: {search_ms:.1f} ms | fusion: {result.elapsed_ms:.2f} ms "
            f"| total {search_ms + result.elapsed_ms:.1f} ms "
            f"| target {FUSION['latency']['target_ms']} ms"
        )

    assert result.degraded_arms == ("dense",)
    assert result.gate_closed is False
    assert result.passages, "the gate closed on the milestone query"
    assert search_ms + result.elapsed_ms < FUSION["latency"]["ceiling_ms"]


# ─────────────────────── query-side stop words (§7.4) ───────────────────────
#
# Meilisearch normalizes the `stopWords` setting on write — ي and ى become ی
# (U+06CC), ك becomes ک — but matches tokens against the stored list without
# applying that same normalization. A stop word containing any of those
# letters is therefore stored in a form nothing will ever equal, and is
# silently inert. Measured on 1.53.1 with a three-document index: `عاوز` as a
# stop word makes `عاوز` unsearchable, `عايز` does not.
#
# That is not cosmetic. The default matching strategy drops query words from
# the *end*, and Masri puts the filler at the *start*, so one inert filler
# ANDs the whole query to nothing. Six of the sixteen Arabic stop words in
# config are affected, including `في` — which is exactly the word the franco
# `fe`/`fi`/`f` entries were added to mirror.


async def test_meilisearch_ignores_a_stop_word_containing_yeh(client):
    """The engine defect the query-side strip exists to work around.

    Asserted rather than described, so that if a future Meilisearch fixes it
    this test fails and tells us the workaround can go — rather than the
    workaround quietly outliving its reason.
    """
    index_name = "test_lex_yeh"
    await client.delete_index_if_exists(index_name)
    await client.create_index(index_name, primary_key="point_id")
    index = client.index(index_name)
    await client.wait_for_task(
        (await index.update_searchable_attributes(["content"])).task_uid
    )
    # `عايز` carries a yeh; `عاوز` is the same word one letter apart and does not.
    await client.wait_for_task((await index.update_stop_words(["عايز", "عاوز"])).task_uid)
    await client.wait_for_task(
        (
            await index.add_documents(
                [
                    {"point_id": "1", "content": "عايز اسكن هنا"},
                    {"point_id": "2", "content": "عاوز اسكن هنا"},
                ]
            )
        ).task_uid
    )

    without_yeh = await index.search("عاوز", limit=10)
    with_yeh = await index.search("عايز", limit=10)
    await client.delete_index_if_exists(index_name)

    assert not without_yeh.hits, "a stop word with no yeh is honoured"
    assert with_yeh.hits, (
        "meilisearch still matches a stop word containing yeh — if this now "
        "returns nothing, the engine was fixed and `strip_stop_words` can go"
    )


def test_stop_words_are_stripped_by_normalized_form():
    """`أعرف` and `اعرف` are one word. Config carries one spelling."""
    assert strip_stop_words("عايز أعرف الحد الأدنى للقبول", TEST_CONFIG) == (
        "الحد الأدنى للقبول"
    )
    assert strip_stop_words("عايز اعرف الحد الأدنى للقبول", TEST_CONFIG) == (
        "الحد الأدنى للقبول"
    )


def test_stripping_preserves_the_original_spelling_of_what_survives():
    """The surviving words keep their own form — normalization is used to
    *decide*, never to rewrite. A rewritten query no longer matches the
    synonym keys, which are stored as the customer types them."""
    assert strip_stop_words("عايز منحة في القنطرة", TEST_CONFIG) == "منحة القنطرة"


def test_a_query_of_only_stop_words_is_searched_unchanged():
    """Stripping to nothing would turn a weak query into a match-everything
    one. Better to search the words the customer sent and rank badly."""
    assert strip_stop_words("عايز أعرف", TEST_CONFIG) == "عايز أعرف"


async def test_a_leading_masri_filler_does_not_empty_the_result(corpus):
    """The four-of-twelve failure, as a test.

    `عايز أعرف` in front of a real question returned zero lexical hits — not
    a bad ranking, an empty list — because the filler survived every step of
    the end-first word drop.
    """
    repository, tenant = corpus
    bare = await chunk_ids(repository, tenant, "الحد الأدنى للقبول")
    with_filler = await chunk_ids(repository, tenant, "عايز أعرف الحد الأدنى للقبول")
    assert bare, "the question alone must retrieve something for this test to mean anything"
    assert with_filler == bare
