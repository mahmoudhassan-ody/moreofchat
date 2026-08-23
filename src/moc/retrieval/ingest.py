"""Ingestion and the outbox — design §7.1 and §5.

Postgres is the source of truth; Qdrant and Meilisearch are derived. This
module is what makes that statement true rather than aspirational: chunks and
the rows describing their sync commit in **one transaction**, so there is no
instant at which a chunk exists and nothing knows to index it.

Writing to the search stores directly from here would be the dual-write
problem D2 chose the outbox to avoid. It fails in the shape nobody notices: a
chunk present in Postgres and absent from Qdrant is retrievable by every query
a developer runs and invisible to every query a customer runs.

Three properties carry the design:

**One transaction.** `ingest_document` writes the document, its chunks and its
outbox rows through the caller's session and never commits. The caller owns
the boundary, so ingesting several documents atomically is the default rather
than a special path.

**Replay is idempotent.** Chunk ids derive from the document id and the
ordinal, point ids are UUIDv5 of the chunk id (§7.1), and a re-ingest deletes
the tail a shortened document no longer covers. A tenant re-syncing an edited
sheet is the normal path, not an edge case, and duplicates would return the
same passage twice and inflate every retrieval metric that counts hits.

**A failed sync costs a retry, not a chunk.** `drain_outbox` marks rows done
only after the handler returns. A row consumed before the work succeeded is a
chunk that becomes permanently unsearchable while every count looks right.
"""

import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.config_store import load
from moc.retrieval.chunker import chunk_text
from moc.retrieval.records import point_id_for

_DEFAULTS = "retrieval/defaults"

_PENDING = "pending"
_DONE = "done"
_UPSERT = "upsert"


@dataclass(frozen=True)
class SourceDocument:
    """What a tenant supplies. `text` is chunked unless chunks are passed in."""

    doc_id: str
    vertical: str
    text: str = ""
    title: str | None = None
    lang: str | None = None
    source_uri: str | None = None


@dataclass(frozen=True)
class SourceChunk:
    """A pre-chunked passage — the frozen fixtures arrive this way.

    `effective_from` and `effective_to` default to None and must stay that
    way. Defaulting them to today would make every undated fee current, and
    the staleness cases would pass while asserting nothing (§7.1).
    """

    chunk_id: str
    content: str
    lang: str | None = None
    topic: str | None = None
    entity_ref: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    ordinal: int
    point_id: uuid.UUID


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    chunks: list[StoredChunk] = field(default_factory=list)


#: The current tenant, from the session rather than from a parameter — the
#: same rule every other query in this codebase follows. The point id needs it
#: (see `vectors.point_id_for`), and a `tenant_id` argument here would be one
#: more place for a caller to pass the wrong one.
_CURRENT_TENANT = text("SELECT nullif(current_setting('moc.tenant_id', true), '')::uuid")


_UPSERT_DOCUMENT = text(
    """
INSERT INTO kb_documents (id, tenant_id, doc_id, title, vertical, lang, source_uri)
VALUES (
  :id, nullif(current_setting('moc.tenant_id', true), '')::uuid,
  :doc_id, :title, :vertical, :lang, :source_uri
)
ON CONFLICT (tenant_id, doc_id) DO UPDATE SET
  title = EXCLUDED.title,
  vertical = EXCLUDED.vertical,
  lang = EXCLUDED.lang,
  source_uri = EXCLUDED.source_uri
RETURNING id
"""
)

_UPSERT_CHUNK = text(
    """
INSERT INTO kb_chunks (
  id, tenant_id, document_id, chunk_id, ordinal, content, content_normalized,
  lang, topic, entity_ref, effective_from, effective_to, embedding_version
) VALUES (
  :id, nullif(current_setting('moc.tenant_id', true), '')::uuid,
  :document_id, :chunk_id, :ordinal, :content, :content_normalized,
  :lang, :topic, :entity_ref, :effective_from, :effective_to, :embedding_version
)
ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
  document_id = EXCLUDED.document_id,
  ordinal = EXCLUDED.ordinal,
  content = EXCLUDED.content,
  content_normalized = EXCLUDED.content_normalized,
  lang = EXCLUDED.lang,
  topic = EXCLUDED.topic,
  entity_ref = EXCLUDED.entity_ref,
  effective_from = EXCLUDED.effective_from,
  effective_to = EXCLUDED.effective_to,
  embedding_version = EXCLUDED.embedding_version
"""
)

# A shortened document must lose its tail, or a fee the tenant deleted stays
# retrievable and answerable.
_DELETE_ORPHANS = text(
    "DELETE FROM kb_chunks WHERE document_id = :document_id AND ordinal >= :kept"
)

_INSERT_OUTBOX = text(
    """
INSERT INTO kb_outbox (id, tenant_id, chunk_id, target, op, point_id)
VALUES (
  :id, nullif(current_setting('moc.tenant_id', true), '')::uuid,
  :chunk_id, :target, :op, :point_id
)
"""
)


async def ingest_document(
    session: AsyncSession,
    document: SourceDocument,
    *,
    chunks: Sequence[SourceChunk] | None = None,
    config: dict[str, Any] | None = None,
) -> IngestResult:
    """Write a document, its chunks and its outbox rows. Does not commit.

    The caller owns the transaction boundary. That is what lets a tenant sync
    of forty documents be one atomic unit, and it is why a partial failure
    leaves no half-ingested document behind.
    """
    settings = config or load(_DEFAULTS)
    tenant_id = (await session.execute(_CURRENT_TENANT)).scalar_one()
    document_id = (
        await session.execute(
            _UPSERT_DOCUMENT,
            {
                "id": uuid.uuid4(),
                "doc_id": document.doc_id,
                "title": document.title,
                "vertical": document.vertical,
                "lang": document.lang,
                "source_uri": document.source_uri,
            },
        )
    ).scalar_one()

    prepared = list(chunks) if chunks is not None else _chunks_from_text(document, settings)
    stored: list[StoredChunk] = []

    for ordinal, source in enumerate(prepared):
        point_id = point_id_for(tenant_id, source.chunk_id, config=settings)
        await session.execute(
            _UPSERT_CHUNK,
            {
                "id": uuid.uuid4(),
                "document_id": document_id,
                "chunk_id": source.chunk_id,
                "ordinal": ordinal,
                "content": source.content,
                "content_normalized": _normalized(source.content),
                "lang": source.lang or document.lang,
                "topic": source.topic,
                "entity_ref": source.entity_ref,
                "effective_from": source.effective_from,
                "effective_to": source.effective_to,
                "embedding_version": settings["embedding_version"],
            },
        )
        stored.append(StoredChunk(chunk_id=source.chunk_id, ordinal=ordinal, point_id=point_id))

    await session.execute(
        _DELETE_ORPHANS, {"document_id": document_id, "kept": len(prepared)}
    )

    # Same transaction as the chunks above. This is the whole outbox pattern:
    # the sync instruction cannot be lost separately from the data it describes.
    for entry in stored:
        for target in settings["outbox"]["targets"]:
            await session.execute(
                _INSERT_OUTBOX,
                {
                    "id": uuid.uuid4(),
                    "chunk_id": entry.chunk_id,
                    "target": target,
                    "op": _UPSERT,
                    "point_id": entry.point_id,
                },
            )

    return IngestResult(document_id=document_id, chunks=stored)


def _chunks_from_text(
    document: SourceDocument, settings: dict[str, Any]
) -> list[SourceChunk]:
    """Chunk ids are the document id plus the ordinal.

    Deterministic, so re-ingesting unchanged text upserts the same rows rather
    than writing a second copy under fresh ids.
    """
    return [
        SourceChunk(
            chunk_id=f"{document.doc_id}#{piece.ordinal:04d}",
            content=piece.content,
            lang=document.lang,
        )
        for piece in chunk_text(document.text, config=settings)
    ]


def _normalized(content: str) -> str:
    from moc.arabic.normalize import normalize

    return normalize(content)


# ─────────────────────────── draining ───────────────────────────

OutboxHandler = Callable[[Sequence[dict[str, Any]]], Awaitable[None]]

_CLAIM = text(
    """
SELECT id, chunk_id, point_id, op FROM kb_outbox
WHERE status = :pending AND target = :target
ORDER BY created_at
LIMIT :limit
FOR UPDATE SKIP LOCKED
"""
)

_MARK_DONE = text(
    "UPDATE kb_outbox SET status = :done, processed_at = :now WHERE id = ANY(:ids)"
)

_RECORD_FAILURE = text(
    """
UPDATE kb_outbox SET attempts = attempts + 1, last_error = :error
WHERE id = ANY(:ids)
"""
)


async def drain_outbox(
    session: AsyncSession,
    *,
    handler: OutboxHandler,
    target: str,
    limit: int = 100,
) -> int:
    """Hand pending rows for `target` to `handler`, marking them done only on success.

    Returns how many rows were completed.

    `FOR UPDATE SKIP LOCKED` so two sync workers can drain concurrently
    without either waiting on the other or both claiming the same row.

    On failure the attempt is recorded and the rows stay pending. Recording it
    matters as much as not consuming the row: without a count a poisoned entry
    retries forever, and the queue stops draining behind it.
    """
    claimed = (
        await session.execute(
            _CLAIM, {"pending": _PENDING, "target": target, "limit": limit}
        )
    ).mappings().all()
    if not claimed:
        return 0

    ids = [row["id"] for row in claimed]
    try:
        await handler([dict(row) for row in claimed])
    except Exception as exc:  # noqa: BLE001 - any failure means "not done", uniformly
        await session.execute(_RECORD_FAILURE, {"ids": ids, "error": repr(exc)[:500]})
        return 0

    await session.execute(_MARK_DONE, {"done": _DONE, "now": datetime.now(UTC), "ids": ids})
    return len(ids)


__all__ = [
    "IngestResult",
    "SourceChunk",
    "SourceDocument",
    "StoredChunk",
    "drain_outbox",
    "ingest_document",
    "point_id_for",
]
