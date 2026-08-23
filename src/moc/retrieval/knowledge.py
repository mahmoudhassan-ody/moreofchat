"""The knowledge screen's engine — demo plan Task 31.

Where "they feed their own data" becomes true. Four operations, and each one
exists because of a failure that does not raise.

**Preview before confirm.** `preview` chunks and describes; it writes nothing
and embeds nothing. The order is the whole point: a screen that ingests first
and reports afterwards has already spent the money and already put a broken
corpus behind the bot. `moc.retrieval.preview` holds the analysis and has no
session to write through.

**Unchanged is answered before a provider call.** A tenant fixes one row in a
spreadsheet and exports the whole sheet — that is the normal path, not an edge
case. The document's content hash sits on `kb_documents` (migration 0014), so a
re-upload of unedited text costs nothing at all: no embedding, no ledger row,
no reindex. The embedding cache makes the vectors free one level down; this
makes the *call* unnecessary.

**A failed sync is reported per document, with the reason.** Chunks live in
Postgres and the search stores are derived from them, so a sync that fails
leaves a document that is retrievable by every query a developer runs and
invisible to every query a customer runs. The outbox keeps the rows pending for
a retry — and the screen says which document and why, because a silent
"uploaded" over a half-indexed corpus is the worst of the available outcomes.

**Ingest is metered.** What onboarding a tenant costs is a question somebody
asks, and it has to be answerable by a query rather than an estimate.

The drain runs inline, inside the request. That is right for one document from
a console and wrong for a forty-document sync, which belongs to a worker — the
outbox row is the same either way, which is what makes moving it later a change
of caller rather than a change of design.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import text

from moc.config_store import load
from moc.retrieval.chunker import embedding_text
from moc.retrieval.ingest import SourceChunk, SourceDocument, ingest_document
from moc.retrieval.lexical import LexicalDocument
from moc.retrieval.preview import DocumentPreview, preview_document
from moc.retrieval.records import VectorPoint, point_id_for
from moc.tenancy.context import tenant_session
from moc.tenancy.metering import UsageKind, record_usage
from moc.tenancy.pricing import price

_DEFAULTS = "retrieval/defaults"


class Embedder(Protocol):
    async def embed(self, *, texts: Any) -> Any: ...


class Dense(Protocol):
    async def upsert(self, *, tenant_id: uuid.UUID, vertical: str, points: Any) -> int: ...
    async def delete(
        self, *, tenant_id: uuid.UUID, vertical: str, chunk_ids: Any
    ) -> int: ...


class Lexical(Protocol):
    async def add(self, *, tenant_id: uuid.UUID, vertical: str, documents: Any) -> int: ...
    async def remove(
        self, *, tenant_id: uuid.UUID, vertical: str, point_ids: Any
    ) -> int: ...


@dataclass(frozen=True)
class Failure:
    """Which document, and why. Never a count.

    A screen that says "1 document failed" sends the tenant back into their own
    corpus to work out which one.
    """

    doc_id: str
    reason: str


@dataclass(frozen=True)
class IngestOutcome:
    doc_id: str
    chunk_count: int
    #: True when the content hash matched what is already stored. No embedding
    #: was bought, no row was rewritten, and the console says so rather than
    #: showing a progress bar over work that achieves nothing.
    unchanged: bool = False
    failures: list[Failure] = field(default_factory=list)


@dataclass(frozen=True)
class StoredDocument:
    doc_id: str
    title: str | None
    vertical: str
    chunk_count: int
    created_at: Any


class KnowledgeService:
    def __init__(
        self, *, engine: Any, embedder: Embedder, dense: Dense, lexical: Lexical
    ) -> None:
        self._engine = engine
        self._embedder = embedder
        self._dense = dense
        self._lexical = lexical

    # ─────────────────────────── before confirming ───────────────────────────

    async def preview(
        self, *, tenant_id: uuid.UUID, title: str | None, text_body: str, doc_id: str
    ) -> tuple[DocumentPreview, bool]:
        """What the chunker produced, and whether this is already stored.

        Returns the preview and `unchanged`, so the screen can say "nothing to
        do" before the tenant clicks confirm rather than after.
        """
        result = preview_document(text=text_body, title=title)
        return result, await self._is_unchanged(
            tenant_id=tenant_id, doc_id=doc_id, content_hash=result.content_hash
        )

    # ─────────────────────────── confirming ───────────────────────────

    async def ingest(
        self,
        *,
        tenant_id: uuid.UUID,
        doc_id: str,
        title: str | None,
        text_body: str,
        vertical: str,
        lang: str | None = None,
    ) -> IngestOutcome:
        analysis = preview_document(text=text_body, title=title)
        if await self._is_unchanged(
            tenant_id=tenant_id, doc_id=doc_id, content_hash=analysis.content_hash
        ):
            return IngestOutcome(
                doc_id=doc_id, chunk_count=analysis.chunk_count, unchanged=True
            )

        settings = load(_DEFAULTS)
        async with tenant_session(self._engine, tenant_id) as session:
            result = await ingest_document(
                session,
                SourceDocument(
                    doc_id=doc_id,
                    vertical=vertical,
                    title=title,
                    lang=lang,
                    text=text_body,
                ),
                chunks=[
                    SourceChunk(chunk_id=f"{doc_id}#{chunk.ordinal}", content=chunk.content)
                    for chunk in analysis.chunks
                ],
                config=settings,
            )
            await session.execute(
                text("UPDATE kb_documents SET content_hash = :hash WHERE id = :id"),
                {"hash": analysis.content_hash, "id": result.document_id},
            )
            # Committed before the sync is attempted, deliberately. Postgres is
            # the source of truth and the outbox rows are already written; if
            # the sync then fails the chunks are safe and pending a retry,
            # which is the whole reason the outbox exists.
            await session.commit()

            failures = await self._sync(
                session=session,
                tenant_id=tenant_id,
                vertical=vertical,
                doc_id=doc_id,
                title=title,
                chunks=analysis.chunks,
                stored=result.chunks,
            )
            await session.commit()

        return IngestOutcome(
            doc_id=doc_id, chunk_count=analysis.chunk_count, failures=failures
        )

    async def _sync(
        self,
        *,
        session: Any,
        tenant_id: uuid.UUID,
        vertical: str,
        doc_id: str,
        title: str | None,
        chunks: Any,
        stored: Any,
    ) -> list[Failure]:
        texts = [embedding_text(title=title, content=chunk.content) for chunk in chunks]
        if not texts:
            return []
        try:
            embedded = await self._embedder.embed(texts=texts)
        except Exception as failed:  # noqa: BLE001 - any failure is "not indexed"
            return [Failure(doc_id=doc_id, reason=repr(failed)[:300])]

        await self._meter(session, embedded, count=len(texts))

        try:
            await self._dense.upsert(
                tenant_id=tenant_id,
                vertical=vertical,
                points=[
                    VectorPoint(chunk_id=entry.chunk_id, vector=vector)
                    for entry, vector in zip(stored, embedded.vectors, strict=True)
                ],
            )
            await self._lexical.add(
                tenant_id=tenant_id,
                vertical=vertical,
                documents=[
                    LexicalDocument(
                        point_id=str(entry.point_id),
                        chunk_id=entry.chunk_id,
                        content=chunk.content,
                        title=title or "",
                    )
                    for entry, chunk in zip(stored, chunks, strict=True)
                ],
            )
        except Exception as failed:  # noqa: BLE001 - see the module docstring
            return [Failure(doc_id=doc_id, reason=repr(failed)[:300])]
        return []

    async def _meter(self, session: Any, embedded: Any, *, count: int) -> None:
        await record_usage(
            session,
            kind=UsageKind.embedding_call,
            quantity=count,
            model=embedded.model,
            provider=embedded.provider,
            input_tokens=embedded.input_tokens,
            provider_cost_usd=price(
                model=embedded.model, input_tokens=embedded.input_tokens
            ).usd,
        )

    # ─────────────────────────── after ───────────────────────────

    async def documents(self, *, tenant_id: uuid.UUID) -> list[StoredDocument]:
        async with tenant_session(self._engine, tenant_id) as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT d.doc_id, d.title, d.vertical, d.created_at, "
                        "count(c.id) AS chunk_count "
                        "FROM kb_documents d LEFT JOIN kb_chunks c ON c.document_id = d.id "
                        "GROUP BY d.id ORDER BY d.created_at DESC"
                    )
                )
            ).all()
        return [
            StoredDocument(
                doc_id=row.doc_id,
                title=row.title,
                vertical=row.vertical,
                chunk_count=row.chunk_count,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def remove(self, *, tenant_id: uuid.UUID, doc_id: str) -> bool:
        """Delete a document and everything derived from it.

        All three stores, not just Postgres. A chunk left in Meilisearch after
        its row is gone is one arm of fusion still answering with a figure the
        tenant withdrew, which reads to a customer as the bot inventing it.
        """
        async with tenant_session(self._engine, tenant_id) as session:
            found = (
                await session.execute(
                    text(
                        "SELECT c.chunk_id, d.vertical FROM kb_documents d "
                        "JOIN kb_chunks c ON c.document_id = d.id WHERE d.doc_id = :doc_id"
                    ),
                    {"doc_id": doc_id},
                )
            ).all()
            deleted = (
                await session.execute(
                    text("DELETE FROM kb_documents WHERE doc_id = :doc_id RETURNING id"),
                    {"doc_id": doc_id},
                )
            ).all()
            if not deleted:
                return False
            await session.commit()

        if found:
            vertical = found[0].vertical
            chunk_ids = [row.chunk_id for row in found]
            await self._dense.delete(
                tenant_id=tenant_id, vertical=vertical, chunk_ids=chunk_ids
            )
            await self._lexical.remove(
                tenant_id=tenant_id,
                vertical=vertical,
                point_ids=[str(point_id_for(tenant_id, chunk)) for chunk in chunk_ids],
            )
        return True

    # ─────────────────────────── internals ───────────────────────────

    async def _is_unchanged(
        self, *, tenant_id: uuid.UUID, doc_id: str, content_hash: str
    ) -> bool:
        async with tenant_session(self._engine, tenant_id) as session:
            stored = (
                await session.execute(
                    text("SELECT content_hash FROM kb_documents WHERE doc_id = :doc_id"),
                    {"doc_id": doc_id},
                )
            ).scalar_one_or_none()
        # NULL means the row predates migration 0014 and nothing verified it,
        # so the first re-upload is a full ingest rather than a false claim.
        return stored is not None and stored == content_hash


__all__ = ["Failure", "IngestOutcome", "KnowledgeService", "StoredDocument"]
