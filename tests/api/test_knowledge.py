"""The knowledge screen — demo plan Task 31.

Where "they feed their own data" becomes true, and where a corpus quietly
breaks. Every test here is about a failure that does not raise:

- a PDF that extracted to whitespace ingests perfectly and teaches the bot
  nothing;
- a fee table with no full stops becomes one chunk that every query returns;
- a re-uploaded corpus buys the same vectors a second time;
- a failed sync leaves chunks in Postgres that no search will ever see.

`test_the_screen_shows_chunk_count_and_a_sample_before_confirming` is the one
that matters, and its point is the *order*: preview, then confirm. A screen
that ingests first and reports afterwards is a screen that has already spent
the money and already broken the corpus.
"""

import uuid
from contextlib import asynccontextmanager

import httpx
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.api.auth import build_auth_router, cookie_authenticator
from moc.api.knowledge import build_knowledge_router
from moc.config_store import load
from moc.llm.base import Embedding
from moc.retrieval.knowledge import KnowledgeService

COOKIE = load("security/agents")["session"]["cookie"]["name"]
PASSWORD = "correct horse battery staple"  # noqa: S105 - a test fixture, not a secret

CLEAN = (
    "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026. "
    "رسوم التقديم 2000 جنيه وهي غير مستردة."
)


class RecordingEmbedder:
    """Counts calls and texts, because "costs no embedding" is a claim about
    calls made rather than about how the code reads."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, *, texts):
        self.batches.append(list(texts))
        return Embedding(
            vectors=[[0.1] * 8 for _ in texts],
            provider="openai",
            model="text-embedding-3-large",
            input_tokens=7 * len(texts),
        )

    @property
    def calls(self) -> int:
        return len(self.batches)


class RecordingDense:
    """`QdrantRepository`'s tenant-facing shape, keyword for keyword.

    Matched to the real signature on purpose. A double with a convenient
    signature lets the service call something that does not exist, and every
    test passes until the day it runs against the real repository — the
    mismatch `FakeExtractor` and `RecordingJudge` both hid.

    `fail_with` makes a sync fail the way a real one does: after the chunks are
    already committed to Postgres.
    """

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.upserted: list = []
        self.deleted: list[str] = []
        self.fail_with = fail_with

    async def upsert(self, *, tenant_id, vertical: str, points) -> int:
        if self.fail_with is not None:
            raise self.fail_with
        self.upserted.extend(points)
        return len(list(points))

    async def delete(self, *, tenant_id, vertical: str, chunk_ids) -> int:
        self.deleted.extend(chunk_ids)
        return len(list(chunk_ids))


class RecordingLexical:
    """`MeilisearchRepository`'s tenant-facing shape, for the same reason."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.upserted: list = []
        self.deleted: list[str] = []
        self.fail_with = fail_with

    async def add(self, *, tenant_id, vertical: str, documents) -> int:
        if self.fail_with is not None:
            raise self.fail_with
        self.upserted.extend(documents)
        return len(list(documents))

    async def remove(self, *, tenant_id, vertical: str, point_ids) -> int:
        self.deleted.extend(point_ids)
        return len(list(point_ids))


def test_the_doubles_match_the_real_repositories():
    """The doubles above claim to stand in for two real classes. This is what
    makes the claim checkable rather than a comment."""
    import inspect

    from moc.retrieval.lexical import MeilisearchRepository
    from moc.retrieval.vectors import QdrantRepository

    for double, real, methods in (
        (RecordingDense, QdrantRepository, ("upsert", "delete")),
        (RecordingLexical, MeilisearchRepository, ("add", "remove")),
    ):
        for name in methods:
            assert set(inspect.signature(getattr(double, name)).parameters) == set(
                inspect.signature(getattr(real, name)).parameters
            ), f"{double.__name__}.{name} does not match {real.__name__}.{name}"


@pytest_asyncio.fixture(loop_scope="session")
async def console(engine, app_engine, lookup_engine, tenant_tables):
    from moc.tenancy.agent_auth import AgentDirectory
    from moc.tenancy.models import Tenant

    ids = {"a": uuid.uuid4(), "b": uuid.uuid4()}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add_all(
            [
                Tenant(id=ids["a"], slug="kb-a", name="A", vertical="education"),
                Tenant(id=ids["b"], slug="kb-b", name="B", vertical="education"),
            ]
        )
        await s.commit()

    directory = AgentDirectory(engine=app_engine, lookup=lookup_engine)
    await directory.create_agent(tenant_id=ids["a"], email="ali@kb-a.example",
                                 password=PASSWORD, display_name="Ali")
    await directory.create_agent(tenant_id=ids["b"], email="basma@kb-b.example",
                                 password=PASSWORD, display_name="Basma")

    embedder, dense, lexical = RecordingEmbedder(), RecordingDense(), RecordingLexical()
    service = KnowledgeService(
        engine=app_engine, embedder=embedder, dense=dense, lexical=lexical
    )
    app = FastAPI()
    app.include_router(build_auth_router(directory=directory))
    app.include_router(
        build_knowledge_router(
            service=service, authenticate=cookie_authenticator(directory=directory)
        )
    )
    yield app, ids, service, embedder, dense, lexical

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


@asynccontextmanager
async def signed_in(app, email: str):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        await http.post("/auth/login", json={"email": email, "password": PASSWORD})
        yield http


def document(**overrides) -> dict:
    return {
        "docId": "fees-2026",
        "title": "الرسوم",
        "text": CLEAN,
        "vertical": "education",
        **overrides,
    }


# ─────────────────────── preview, then confirm ───────────────────────


async def test_the_screen_shows_chunk_count_and_a_sample_before_confirming(console):
    """Chunking is where a corpus quietly breaks. A sentence severed from its
    number surfaces months later as an orphan figure in a correct reply."""
    app, _, _, embedder, dense, _ = console

    async with signed_in(app, "ali@kb-a.example") as http:
        preview = (await http.post("/knowledge/preview", json=document())).json()

    assert preview["chunkCount"] >= 1
    assert preview["sample"], "a count with no text is a number nobody can act on"
    assert preview["sample"][0]["content"].startswith("رسوم الساعة")
    assert preview["warnings"] == []
    # And nothing happened. The whole point of a preview is the option to say
    # no, which a preview that ingested would have already spent.
    assert embedder.calls == 0
    assert dense.upserted == []


async def test_a_preview_writes_nothing_to_the_corpus(console, app_engine):
    app, ids, _, _, _, _ = console

    async with signed_in(app, "ali@kb-a.example") as http:
        await http.post("/knowledge/preview", json=document(docId="never-ingested"))

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, ids["a"]) as session:
        count = (
            await session.execute(
                text("SELECT count(*) FROM kb_documents WHERE doc_id = 'never-ingested'")
            )
        ).scalar_one()
    assert count == 0


async def test_the_preview_names_what_is_wrong_with_a_real_export(console):
    """A fee table exported to plain text has no full stop in it, and the
    tenant is the only person who can look at the result and know."""
    app, _, _, _, _, _ = console
    table = " ".join(f"الهندسة {1000 + i} 2026" for i in range(200))

    async with signed_in(app, "ali@kb-a.example") as http:
        preview = (
            await http.post("/knowledge/preview", json=document(text=table))
        ).json()

    assert "no_sentence_boundaries" in [w["name"] for w in preview["warnings"]]
    assert preview["warnings"][0]["reason"]


# ─────────────────────────── ingestion ───────────────────────────


async def test_an_uploaded_document_becomes_searchable_chunks(console):
    app, _, _, embedder, dense, lexical = console

    async with signed_in(app, "ali@kb-a.example") as http:
        result = (await http.post("/knowledge/documents", json=document())).json()

    assert result["chunkCount"] >= 1
    assert result["unchanged"] is False
    assert embedder.calls == 1
    assert len(dense.upserted) == result["chunkCount"]
    assert len(lexical.upserted) == result["chunkCount"]


async def test_ingest_writes_an_embedding_call_row_to_the_ledger(console, app_engine):
    """What onboarding costs is the number somebody asks for, and it has to
    be a query rather than an estimate."""
    app, ids, _, _, _, _ = console

    async with signed_in(app, "ali@kb-a.example") as http:
        await http.post("/knowledge/documents", json=document(docId="metered"))

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, ids["a"]) as session:
        rows = (
            await session.execute(
                text(
                    "SELECT model, input_tokens, provider_cost_usd FROM usage_ledger "
                    "WHERE kind = 'embedding_call'"
                )
            )
        ).all()
    assert rows, "the ingest bought embeddings and the ledger shows none"
    assert rows[0].model == "text-embedding-3-large"
    assert rows[0].input_tokens > 0
    assert rows[0].provider_cost_usd is not None


async def test_re_uploading_an_unchanged_document_costs_no_embedding(console):
    """The content-addressed cache, visible in the UI as 'unchanged'.

    A tenant fixes one row in a spreadsheet and exports the whole sheet. That
    is the normal path, not an edge case.
    """
    app, _, _, embedder, _, _ = console

    async with signed_in(app, "ali@kb-a.example") as http:
        first = (await http.post("/knowledge/documents", json=document())).json()
        again = (await http.post("/knowledge/documents", json=document())).json()

    assert first["unchanged"] is False
    assert again["unchanged"] is True
    assert embedder.calls == 1, "the second upload bought embeddings again"


async def test_an_edited_document_is_not_unchanged(console):
    app, _, _, embedder, _, _ = console

    async with signed_in(app, "ali@kb-a.example") as http:
        await http.post("/knowledge/documents", json=document())
        edited = (
            await http.post("/knowledge/documents", json=document(text=CLEAN + " تم."))
        ).json()

    assert edited["unchanged"] is False
    assert embedder.calls == 2


async def test_a_failed_ingest_names_the_row_and_the_reason(console, app_engine):
    """A sync that fails leaves chunks in Postgres that no search will see —
    retrievable by every query a developer runs and invisible to every query a
    customer runs. The screen has to say which document and why."""
    app, ids, service, _, _, _ = console
    service._dense.fail_with = RuntimeError("qdrant refused the batch")

    try:
        async with signed_in(app, "ali@kb-a.example") as http:
            response = await http.post("/knowledge/documents", json=document(docId="broken"))
    finally:
        service._dense.fail_with = None

    body = response.json()
    assert response.status_code == 200
    assert body["failures"], "a failed sync reported as a success"
    assert body["failures"][0]["docId"] == "broken"
    assert "qdrant refused" in body["failures"][0]["reason"]


async def test_a_document_can_be_removed_and_its_chunks_go_with_it(console, app_engine):
    app, ids, _, _, _, lexical = console

    async with signed_in(app, "ali@kb-a.example") as http:
        await http.post("/knowledge/documents", json=document(docId="temporary"))
        removed = await http.delete("/knowledge/documents/temporary")

    assert removed.status_code == 200

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, ids["a"]) as session:
        chunks = (
            await session.execute(
                text(
                    "SELECT count(*) FROM kb_chunks c JOIN kb_documents d "
                    "ON d.id = c.document_id WHERE d.doc_id = 'temporary'"
                )
            )
        ).scalar_one()
        documents = (
            await session.execute(
                text("SELECT count(*) FROM kb_documents WHERE doc_id = 'temporary'")
            )
        ).scalar_one()
    assert (chunks, documents) == (0, 0)
    assert lexical.deleted, "the search index still holds the removed chunks"


async def test_the_document_list_shows_what_the_tenant_uploaded(console):
    app, _, _, _, _, _ = console

    async with signed_in(app, "ali@kb-a.example") as http:
        await http.post("/knowledge/documents", json=document(docId="listed", title="قائمة"))
        listed = (await http.get("/knowledge/documents")).json()

    row = next(d for d in listed if d["docId"] == "listed")
    assert row["title"] == "قائمة"
    assert row["chunkCount"] >= 1


# ─────────────────────────── isolation ───────────────────────────


async def test_upload_is_tenant_scoped(console):
    app, _, _, _, _, _ = console

    async with signed_in(app, "ali@kb-a.example") as http:
        await http.post("/knowledge/documents", json=document(docId="a-only"))
    async with signed_in(app, "basma@kb-b.example") as http:
        theirs = (await http.get("/knowledge/documents")).json()
        # The header a frontend "already sends".
        spoofed = await http.get(
            "/knowledge/documents", headers={"X-Tenant-Id": "whatever"}
        )
        gone = await http.delete("/knowledge/documents/a-only")

    assert theirs == []
    assert spoofed.json() == []
    assert gone.status_code == 404, "tenant B deleted a document that is not theirs"


async def test_every_knowledge_route_requires_a_session(console):
    app, _, _, _, _, _ = console

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        assert (await http.get("/knowledge/documents")).status_code == 401
        assert (await http.post("/knowledge/preview", json=document())).status_code == 401
        assert (await http.post("/knowledge/documents", json=document())).status_code == 401
        assert (await http.delete("/knowledge/documents/x")).status_code == 401
