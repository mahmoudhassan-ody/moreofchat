"""The knowledge screen over HTTP — demo plan Task 31.

Thin, like every router here: the tenant comes from the session and the work
happens in `moc.retrieval.knowledge`.

**Two routes where one would do, deliberately.** `POST /knowledge/preview`
describes what the chunker did and writes nothing; `POST /knowledge/documents`
confirms it. Collapsing them into one call with a `dryRun` flag would put the
difference between "look at this" and "spend money on this" inside a boolean
that a client sets — and the flag would be wrong exactly once.

`docId` is in the delete path and that is fine: it is the tenant's own name for
their file, scoped by RLS under the session's tenant, and a document that is
not theirs does not exist to be deleted. A `tenantId` would be a different
thing entirely, and there is none here for the same reason there is none in
`moc.api.tenant`.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from moc.api.inbox import AgentPrincipal
from moc.retrieval.knowledge import KnowledgeService

_NOT_FOUND = 404


class Upload(BaseModel):
    """What the screen sends. Text, not a file.

    Extraction — PDF, DOCX, a spreadsheet — happens before this and is its own
    problem with its own failure modes; the console does it and sends the
    result, so the one thing this module can be sure of is that what it chunks
    is what the tenant saw in the preview.
    """

    docId: str  # noqa: N815 - the wire is camelCase, the console reads it
    text: str
    title: str | None = None
    vertical: str = "education"
    lang: str | None = None


def build_knowledge_router(*, service: KnowledgeService, authenticate: Any) -> APIRouter:
    router = APIRouter(prefix="/knowledge")

    @router.post("/preview")
    async def preview(upload: Upload, request: Request) -> dict[str, Any]:
        """Chunk count, a sample, and what is wrong with it. Nothing is stored.

        The sample is the part that matters. A count says the chunker ran; only
        the text says whether it ran sensibly, and only somebody looking at
        their own corpus can tell.
        """
        principal: AgentPrincipal = await authenticate(request)
        result, unchanged = await service.preview(
            tenant_id=principal.tenant_id,
            title=upload.title,
            text_body=upload.text,
            doc_id=upload.docId,
        )
        return {
            "chunkCount": result.chunk_count,
            "sample": [
                {"ordinal": chunk.ordinal, "content": chunk.content}
                for chunk in result.sample
            ],
            "warnings": [
                {"name": str(finding.name), "ordinal": finding.ordinal,
                 "reason": finding.reason}
                for finding in result.warnings
            ],
            "contentHash": result.content_hash,
            # Answered here so the screen can say "nothing to do" before the
            # tenant confirms rather than after.
            "unchanged": unchanged,
        }

    @router.post("/documents")
    async def confirm(upload: Upload, request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        outcome = await service.ingest(
            tenant_id=principal.tenant_id,
            doc_id=upload.docId,
            title=upload.title,
            text_body=upload.text,
            vertical=upload.vertical,
            lang=upload.lang,
        )
        # 200 with failures listed rather than 500. The chunks committed and
        # the outbox rows are pending a retry; the request did not fail, the
        # sync did, and those are different things to a tenant looking at a
        # screen.
        return {
            "docId": outcome.doc_id,
            "chunkCount": outcome.chunk_count,
            "unchanged": outcome.unchanged,
            "failures": [
                {"docId": failure.doc_id, "reason": failure.reason}
                for failure in outcome.failures
            ],
        }

    @router.get("/documents")
    async def listing(request: Request) -> list[dict[str, Any]]:
        principal: AgentPrincipal = await authenticate(request)
        return [
            {
                "docId": document.doc_id,
                "title": document.title,
                "vertical": document.vertical,
                "chunkCount": document.chunk_count,
                # §7.1's staleness question, in the console: a broker looking
                # at inventory should see the date without asking for it.
                "createdAt": document.created_at.isoformat(),
            }
            for document in await service.documents(tenant_id=principal.tenant_id)
        ]

    @router.delete("/documents/{doc_id}")
    async def remove(doc_id: str, request: Request) -> dict[str, bool]:
        principal: AgentPrincipal = await authenticate(request)
        if not await service.remove(tenant_id=principal.tenant_id, doc_id=doc_id):
            # 404, never 403. Another tenant's document does not exist under
            # this session, and 403 would confirm that it does somewhere.
            raise HTTPException(status_code=_NOT_FOUND, detail="no such document")
        return {"ok": True}

    return router


__all__ = ["Upload", "build_knowledge_router"]
