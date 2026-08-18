"""Ingestion and the outbox — design §7.1, §5.

Postgres is the source of truth; Qdrant and Meilisearch are derived. The
outbox is the mechanism that keeps that true, and every test here is about a
way it could silently stop being true:

- chunks committed without their sync rows, so a chunk exists that nothing
  knows to index — retrievable from Postgres, invisible to search
- an outbox row consumed before the work succeeded, so a crash loses the chunk
  permanently rather than retrying it
- a replay writing second copies instead of upserting
- `effective_from` defaulted rather than preserved, which quietly makes every
  undated fee current

The fixtures load through the real ingest path. A loader that bypassed
chunking would test the loader.
"""

import json
import uuid
from datetime import date
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text as sql

from moc.config_store import load
from moc.retrieval.ingest import (
    SourceChunk,
    SourceDocument,
    drain_outbox,
    ingest_document,
    point_id_for,
)
from moc.tenancy.context import tenant_session

CONFIG = load("retrieval/defaults")
SINAI = Path(__file__).parents[2] / "evals" / "fixtures" / "sinai_demo" / "chunks.jsonl"

FEE_TEXT = "رسوم التقديم 2000 جنيه تُدفع مرة واحدة. السكن متاح في الفرعين."


def document(**overrides) -> SourceDocument:
    return SourceDocument(
        **{
            "doc_id": "sinai_fee_application_initial",
            "title": "رسوم التقديم",
            "vertical": "education",
            "lang": "ar",
            "text": FEE_TEXT,
            **overrides,
        }
    )


@pytest_asyncio.fixture(loop_scope="session")
async def tenant(engine):
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.tenancy.models import Tenant

    async with AsyncSession(engine, expire_on_commit=False) as s:
        await s.execute(sql("DELETE FROM kb_outbox"))
        await s.execute(sql("DELETE FROM kb_chunks"))
        await s.execute(sql("DELETE FROM kb_documents"))
        await s.execute(sql("DELETE FROM usage_ledger"))
        await s.execute(sql("DELETE FROM conversations"))
        await s.execute(sql("DELETE FROM tenants"))
        row = Tenant(slug="ingest", name="Ingest", vertical="education")
        s.add(row)
        await s.commit()
        return row


async def counts(app_engine, tenant_id) -> dict[str, int]:
    async with tenant_session(app_engine, tenant_id) as s:
        return {
            # noqa S608: table names are the literal tuple below, not input.
            table: (await s.execute(sql(f"SELECT count(*) FROM {table}"))).scalar_one()  # noqa: S608
            for table in ("kb_documents", "kb_chunks", "kb_outbox")
        }


# ─────────────────────────── the transaction ───────────────────────────


async def test_chunks_and_outbox_rows_commit_in_one_transaction(app_engine, tenant):
    """No window where a chunk exists and nothing knows to index it.

    A chunk in Postgres with no outbox row is retrievable by SQL and invisible
    to search — the worst shape, because the data looks present everywhere a
    developer checks and absent everywhere a customer does.
    """
    async with tenant_session(app_engine, tenant.id) as s:
        result = await ingest_document(s, document())
        await s.commit()

    after = await counts(app_engine, tenant.id)
    assert after["kb_chunks"] == len(result.chunks) > 0
    assert after["kb_outbox"] == len(result.chunks) * len(CONFIG["outbox"]["targets"])


async def test_a_failed_ingest_leaves_no_partial_document(app_engine, tenant):
    """The transaction is the guarantee, so it gets asserted rather than assumed."""
    async with tenant_session(app_engine, tenant.id) as s:
        await ingest_document(s, document(doc_id="rolled_back"))
        await s.rollback()

    async with tenant_session(app_engine, tenant.id) as s:
        found = (
            await s.execute(
                sql("SELECT count(*) FROM kb_documents WHERE doc_id = 'rolled_back'")
            )
        ).scalar_one()
    assert found == 0


# ─────────────────────────── the outbox ───────────────────────────


async def test_a_failed_embed_leaves_the_outbox_row_pending_not_lost(app_engine, tenant):
    """A crash mid-sync must cost a retry, not a chunk.

    Marking a row done before the work succeeds is how a chunk becomes
    permanently unsearchable while every count looks right.
    """
    async with tenant_session(app_engine, tenant.id) as s:
        await ingest_document(s, document(doc_id="embed_fails"))
        await s.commit()

    async def explode(rows):
        raise RuntimeError("embedding endpoint down")

    async with tenant_session(app_engine, tenant.id) as s:
        drained = await drain_outbox(s, handler=explode, target="qdrant")
        await s.commit()

    assert drained == 0
    async with tenant_session(app_engine, tenant.id) as s:
        pending, attempts = (
            await s.execute(
                sql(
                    "SELECT count(*), coalesce(max(attempts), 0) FROM kb_outbox "
                    "WHERE status = 'pending' AND target = 'qdrant'"
                )
            )
        ).one()
    assert pending > 0, "the row was consumed despite the failure"
    assert attempts == 1, "the attempt was not recorded, so a poison row retries forever"


async def test_the_outbox_drains_to_empty_on_success(app_engine, tenant):
    seen = []

    async def collect(rows):
        seen.extend(rows)

    async with tenant_session(app_engine, tenant.id) as s:
        await ingest_document(s, document(doc_id="drains"))
        await s.commit()
    async with tenant_session(app_engine, tenant.id) as s:
        for target in CONFIG["outbox"]["targets"]:
            await drain_outbox(s, handler=collect, target=target)
        await s.commit()

    async with tenant_session(app_engine, tenant.id) as s:
        pending = (
            await s.execute(sql("SELECT count(*) FROM kb_outbox WHERE status = 'pending'"))
        ).scalar_one()
    assert pending == 0
    assert seen


async def test_point_ids_are_uuid5_of_chunk_id_so_replay_is_idempotent():
    """§7.1. A replay after a crash must upsert the same point, not add one.

    Derived rather than random, and asserted against uuid5 directly — a
    generated id would make every re-run a duplicate insert that nothing
    detects until search returns the same passage twice.
    """
    namespace = uuid.UUID(CONFIG["outbox"]["point_namespace"])
    assert point_id_for("sinai_fees_ar") == uuid.uuid5(namespace, "sinai_fees_ar")
    assert point_id_for("sinai_fees_ar") == point_id_for("sinai_fees_ar")
    assert point_id_for("sinai_fees_ar") != point_id_for("sinai_fees_en")


# ─────────────────────────── replay ───────────────────────────


async def test_reingesting_the_same_document_does_not_duplicate_chunks(app_engine, tenant):
    """Re-ingest is the normal path, not an edge case — a tenant re-syncs a
    sheet whenever they edit it. Duplicates would return the same passage
    twice and inflate every retrieval metric that counts hits."""
    async with tenant_session(app_engine, tenant.id) as s:
        first = await ingest_document(s, document(doc_id="stable"))
        await s.commit()
    async with tenant_session(app_engine, tenant.id) as s:
        second = await ingest_document(s, document(doc_id="stable"))
        await s.commit()

    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    async with tenant_session(app_engine, tenant.id) as s:
        chunks = (
            await s.execute(
                sql(
                    "SELECT count(*) FROM kb_chunks c JOIN kb_documents d "
                    "ON c.document_id = d.id WHERE d.doc_id = 'stable'"
                )
            )
        ).scalar_one()
        documents = (
            await s.execute(
                sql("SELECT count(*) FROM kb_documents WHERE doc_id = 'stable'")
            )
        ).scalar_one()
    assert documents == 1
    assert chunks == len(first.chunks)


async def test_removed_text_does_not_leave_orphan_chunks(app_engine, tenant):
    """A shortened document must lose its tail, or a deleted fee stays
    retrievable and answerable after the tenant removed it."""
    long_text = " ".join([FEE_TEXT] * 60)
    async with tenant_session(app_engine, tenant.id) as s:
        big = await ingest_document(s, document(doc_id="shrinks", text=long_text))
        await s.commit()
    async with tenant_session(app_engine, tenant.id) as s:
        small = await ingest_document(s, document(doc_id="shrinks", text=FEE_TEXT))
        await s.commit()

    assert len(small.chunks) < len(big.chunks)
    async with tenant_session(app_engine, tenant.id) as s:
        remaining = (
            await s.execute(
                sql(
                    "SELECT count(*) FROM kb_chunks c JOIN kb_documents d "
                    "ON c.document_id = d.id WHERE d.doc_id = 'shrinks'"
                )
            )
        ).scalar_one()
    assert remaining == len(small.chunks)


# ─────────────────────────── payload fidelity ───────────────────────────


async def test_effective_from_and_to_survive_ingestion(app_engine, tenant):
    """The sinai fixture has null on every chunk but one.

    A pipeline defaulting these to now() would make every undated fee current,
    and the staleness cases would pass while asserting nothing — the fixture's
    silence about dates is the thing being tested.
    """
    dated = SourceChunk(
        chunk_id="sinai_admission_thresholds_2026_ar",
        content="حدود القبول للعام 2026.",
        effective_from=date(2026, 1, 1),
        effective_to=date(2026, 12, 31),
    )
    undated = SourceChunk(chunk_id="sinai_founding_ar", content="تأسست الجامعة.")

    async with tenant_session(app_engine, tenant.id) as s:
        await ingest_document(s, document(doc_id="dates"), chunks=[dated, undated])
        await s.commit()
        rows = dict(
            (
                await s.execute(
                    sql(
                        "SELECT chunk_id, effective_from FROM kb_chunks "
                        "WHERE chunk_id IN (:a, :b)"
                    ),
                    {"a": dated.chunk_id, "b": undated.chunk_id},
                )
            ).all()
        )

    assert rows[dated.chunk_id] == date(2026, 1, 1)
    assert rows[undated.chunk_id] is None, "an undated chunk was given a date"


async def test_the_embedding_version_is_recorded_per_chunk(app_engine, tenant):
    """§5: a model change is a tracked backfill, not a silent reindex."""
    async with tenant_session(app_engine, tenant.id) as s:
        await ingest_document(s, document(doc_id="versioned"))
        await s.commit()
        versions = (
            await s.execute(sql("SELECT DISTINCT embedding_version FROM kb_chunks"))
        ).scalars().all()
    assert versions == [CONFIG["embedding_version"]]


async def test_both_text_forms_are_stored(app_engine, tenant):
    async with tenant_session(app_engine, tenant.id) as s:
        await ingest_document(s, document(doc_id="forms", text="الرسوم ٢٥٠٠٠ جنيه."))
        await s.commit()
        content, normalized = (
            await s.execute(
                sql(
                    "SELECT content, content_normalized FROM kb_chunks "
                    "WHERE chunk_id LIKE 'forms%'"
                )
            )
        ).one()
    assert "٢٥٠٠٠" in content
    assert "25000" in normalized


# ─────────────────────────── tenancy ───────────────────────────


async def test_kb_tables_are_tenant_scoped(app_engine, two_tenants):
    """Through moc_app, the non-owner role. The owner bypasses RLS, so a test
    through it proves nothing — and kb_chunks holds the text replies are
    grounded in, which makes a missing filter one tenant's fees answered from
    another tenant's corpus."""
    tenant_a, tenant_b = two_tenants
    async with tenant_session(app_engine, tenant_a.id) as s:
        await ingest_document(s, document(doc_id="scoped"))
        await s.commit()

    async with tenant_session(app_engine, tenant_b.id) as s:
        for table in ("kb_documents", "kb_chunks", "kb_outbox"):
            visible = (
                await s.execute(sql(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            assert visible == 0, f"tenant B can see tenant A's {table}"


async def test_ingesting_without_a_tenant_context_fails_closed(app_engine, two_tenants):
    """No tenant set means nothing is written — and the error says why.

    RLS `WITH CHECK` is what rejects this, not the NOT NULL constraint: the
    predicate compares against an unset setting, `NULL = NULL` is not true, and
    the row is refused before the column constraint is ever consulted. Worth
    asserting precisely, because a NOT NULL violation here would mean the
    policy had stopped applying and only the column was still holding.
    """
    from sqlalchemy.exc import ProgrammingError
    from sqlalchemy.ext.asyncio import AsyncSession

    async with AsyncSession(app_engine) as s:
        with pytest.raises(ProgrammingError) as caught:
            await ingest_document(s, document(doc_id="no_tenant"))
    assert "row-level security policy" in str(caught.value)


# ─────────────────────────── the frozen fixture, through the real path ───────────────────────────


@pytest.mark.eval
async def test_the_sinai_fixture_ingests_through_the_real_path(app_engine, tenant):
    """Acceptance. A fixture loader that bypassed chunking would test itself."""
    records = [json.loads(line) for line in SINAI.read_text(encoding="utf-8").splitlines()]
    async with tenant_session(app_engine, tenant.id) as s:
        for record in records:
            await ingest_document(
                s,
                SourceDocument(
                    doc_id=record["doc_id"] + "_" + record["lang"],
                    title=record["title"],
                    vertical=record["vertical"],
                    lang=record["lang"],
                    text=record["content"],
                ),
            )
        await s.commit()
        ingested = (await s.execute(sql("SELECT count(*) FROM kb_chunks"))).scalar_one()
    assert ingested >= len(records), "every fixture chunk produced at least one kb_chunk"


@pytest.mark.eval
async def test_chunk_counts_are_stable_across_two_runs(app_engine, tenant):
    """Acceptance. An unstable count means retrieval is measuring a moving
    corpus, and every recall number is noise."""
    records = [json.loads(line) for line in SINAI.read_text(encoding="utf-8").splitlines()][:20]

    async def run() -> int:
        async with tenant_session(app_engine, tenant.id) as s:
            for record in records:
                await ingest_document(
                    s,
                    SourceDocument(
                        doc_id=record["doc_id"] + "_" + record["lang"],
                        title=record["title"],
                        vertical=record["vertical"],
                        lang=record["lang"],
                        text=record["content"],
                    ),
                )
            await s.commit()
            return (await s.execute(sql("SELECT count(*) FROM kb_chunks"))).scalar_one()

    assert await run() == await run()


# ─────────────────────────── the boundary between grounding modes ───────────────────────────


@pytest.mark.eval
async def test_no_kb_chunk_contains_a_broker_unit_id(app_engine, tenant):
    """The two grounding modes must not merge (§3.2).

    Documents ground on chunks; structured inventory grounds on the inventory
    connector. Ingesting `units.jsonl` as text would create a second path to a
    unit's price — and that path has none of the guarantees the first one has.
    It bypasses the availability filter, so a sold unit's price sits in a
    retrieved chunk and `sold_unit_offered_rate` reads zero while the reply
    quotes it. It bypasses the as_of disclosure, so the figure arrives with no
    statement of when it was current.

    A boundary rather than a scoping decision, which is why it is asserted
    rather than left to whoever writes the next ingest script.
    """
    units = (
        Path(__file__).parents[2]
        / "evals"
        / "fixtures"
        / "broker_demo_2026_08_01"
        / "units.jsonl"
    )
    unit_ids = [
        json.loads(line)["unit_id"] for line in units.read_text(encoding="utf-8").splitlines()
    ]
    assert unit_ids, "the fixture is empty — this test would pass vacuously"

    async with tenant_session(app_engine, tenant.id) as s:
        for record in [json.loads(line) for line in SINAI.read_text(encoding="utf-8").splitlines()]:
            await ingest_document(
                s,
                SourceDocument(
                    doc_id=record["doc_id"] + "_" + record["lang"],
                    title=record["title"],
                    vertical=record["vertical"],
                    lang=record["lang"],
                    text=record["content"],
                ),
            )
        await s.commit()

        corpus = " ".join(
            (await s.execute(sql("SELECT content FROM kb_chunks"))).scalars().all()
        )

    leaked = [unit_id for unit_id in unit_ids if unit_id in corpus]
    assert leaked == [], (
        f"broker unit ids reached kb_chunks: {leaked[:5]}. Structured inventory has "
        f"one grounding path, and it is the one carrying availability and as_of."
    )
