"""The Qdrant repository against a real Qdrant — design §7.2.

Real, not mocked. A mock would confirm the arguments this repository passes;
what needs confirming is what Qdrant does with them — that the payload index
exists, that the filter partitions, that a search from tenant B genuinely
cannot see tenant A's points. A filter that is syntactically present and
semantically wrong looks identical in a mock.

Collections are created and dropped per module, named apart from the dev ones
so a test run cannot touch working data.
"""

import uuid

import pytest
import pytest_asyncio

from moc.config_store import load
from moc.retrieval.vectors import (
    QdrantAdmin,
    QdrantRepository,
    VectorPoint,
    collection_for,
)

CONFIG = load("retrieval/defaults")
QDRANT = CONFIG["qdrant"]
SIZE = QDRANT["vector_size"]

#: Test collections, distinct from the configured ones. A test that dropped
#: `kb_education` would delete the corpus a developer had just ingested.
TEST_COLLECTIONS = {
    "education": "test_kb_education",
    "realestate": "test_kb_realestate",
}


def vector(seed: float) -> list[float]:
    """Distinct but deterministic. Identical vectors would let a repository
    that ignores the query still rank plausibly."""
    return [seed] + [0.0] * (SIZE - 1)


TEST_CONFIG = {**CONFIG, "qdrant": {**QDRANT, "collections": TEST_COLLECTIONS}}


@pytest_asyncio.fixture(loop_scope="session")
async def client():
    """Real Qdrant, on collections named apart from the dev ones.

    Unreachable Qdrant fails in CI and skips locally — same rule as the Valkey
    fixture. A skipped cross-tenant test is the one skip this project can
    least afford: it is green and it proves nothing.
    """
    import os

    from qdrant_client import AsyncQdrantClient

    from moc.config import settings

    connection = AsyncQdrantClient(
        url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
        api_key=settings.qdrant_key or None,
    )
    try:
        await connection.get_collections()
    except Exception as exc:
        message = f"qdrant unreachable: {exc}"
        if os.environ.get("CI"):
            pytest.fail(
                f"{message}. CI brings the stack up with compose, so this is a broken "
                f"run rather than a missing dependency — and skipping would hide every "
                f"cross-tenant test behind a green tick."
            )
        pytest.skip(f"{message}. Start it with: docker compose up -d qdrant")

    for collection in TEST_COLLECTIONS.values():
        await connection.delete_collection(collection)
    await QdrantAdmin(client=connection, config=TEST_CONFIG).ensure_collections()
    yield connection
    for collection in TEST_COLLECTIONS.values():
        await connection.delete_collection(collection)
    await connection.close()


@pytest_asyncio.fixture(loop_scope="session")
async def repository(client) -> QdrantRepository:
    return QdrantRepository(client=client, config=TEST_CONFIG)


@pytest_asyncio.fixture(loop_scope="session")
async def admin(client) -> QdrantAdmin:
    return QdrantAdmin(client=client, config=TEST_CONFIG)


def point(chunk_id: str, seed: float = 1.0) -> VectorPoint:
    return VectorPoint(chunk_id=chunk_id, vector=vector(seed), payload={"topic": "fees"})


# ─────────────────────────── isolation ───────────────────────────


async def test_search_never_returns_another_tenants_points(repository):
    """The failure this whole file exists for.

    Both tenants store a point at the same vector, so ranking cannot separate
    them — only the filter can. If it is missing, each tenant sees two hits.
    """
    a, b = uuid.uuid4(), uuid.uuid4()
    await repository.upsert(tenant_id=a, vertical="education", points=[point("a-1")])
    await repository.upsert(tenant_id=b, vertical="education", points=[point("b-1")])

    for tenant, expected in ((a, "a-1"), (b, "b-1")):
        hits = await repository.search(
            tenant_id=tenant, vertical="education", vector=vector(1.0), limit=10
        )
        assert [hit.chunk_id for hit in hits] == [expected]


async def test_a_tenant_with_no_points_gets_nothing_not_everything(repository):
    """The shape a broken filter takes when the store is asked for "no tenant":
    an empty predicate matches every point rather than none."""
    stranger = uuid.uuid4()
    await repository.upsert(
        tenant_id=uuid.uuid4(), vertical="education", points=[point("owned")]
    )
    hits = await repository.search(
        tenant_id=stranger, vertical="education", vector=vector(1.0), limit=10
    )
    assert hits == []


async def test_delete_cannot_reach_another_tenants_points(repository):
    """Isolation is not only about reads. A delete that ignores the filter is
    a tenant silently destroying another tenant's corpus."""
    a, b = uuid.uuid4(), uuid.uuid4()
    await repository.upsert(tenant_id=a, vertical="education", points=[point("keep")])
    await repository.upsert(tenant_id=b, vertical="education", points=[point("keep")])

    await repository.delete(tenant_id=b, vertical="education", chunk_ids=["keep"])

    assert await repository.count(tenant_id=a, vertical="education") == 1
    assert await repository.count(tenant_id=b, vertical="education") == 0


async def test_counts_are_per_tenant(repository):
    a, b = uuid.uuid4(), uuid.uuid4()
    await repository.upsert(
        tenant_id=a, vertical="education", points=[point("x"), point("y")]
    )
    await repository.upsert(tenant_id=b, vertical="education", points=[point("z")])
    assert await repository.count(tenant_id=a, vertical="education") == 2
    assert await repository.count(tenant_id=b, vertical="education") == 1


# ─────────────────────────── layout ───────────────────────────


async def test_collections_are_per_vertical_not_per_tenant(repository):
    """§7.2: collection-per-tenant eats RAM at ~40 tenants, sooner on 3.3 GB.

    Two tenants in one vertical share a collection; two verticals do not.
    """
    a, b = uuid.uuid4(), uuid.uuid4()
    await repository.upsert(tenant_id=a, vertical="education", points=[point("edu")])
    await repository.upsert(tenant_id=b, vertical="realestate", points=[point("re")])

    education = collection_for("education", config=TEST_CONFIG)
    realestate = collection_for("realestate", config=TEST_CONFIG)
    assert education != realestate
    assert education == TEST_COLLECTIONS["education"]
    # The same vertical is one collection regardless of how many tenants use it.
    await repository.upsert(tenant_id=b, vertical="education", points=[point("edu-2")])
    assert collection_for("education", config=TEST_CONFIG) == TEST_COLLECTIONS["education"]


async def test_an_unknown_vertical_is_a_config_error_not_a_new_collection(repository):
    """Creating a collection on demand is how a typo becomes a silent corpus
    nobody searches."""
    with pytest.raises(KeyError):
        collection_for("healthcare")


async def test_the_tenant_field_is_indexed(admin):
    """Without the payload index the filter still works and scans everyone's
    points to do it — correct, and quadratic in tenants."""
    info = await admin.collection_info(vertical="education")
    assert QDRANT["tenant_field"] in info.payload_schema


# ─────────────────────────── replay ───────────────────────────


async def test_upsert_is_idempotent_on_chunk_id(repository):
    """Point ids are UUIDv5 of chunk_id (§7.1), so a re-sync overwrites."""
    tenant = uuid.uuid4()
    await repository.upsert(tenant_id=tenant, vertical="education", points=[point("same")])
    await repository.upsert(
        tenant_id=tenant, vertical="education", points=[point("same", seed=2.0)]
    )
    assert await repository.count(tenant_id=tenant, vertical="education") == 1


async def test_two_tenants_may_hold_the_same_chunk_id(repository):
    """Chunk ids are only unique per tenant. Point ids derived from chunk_id
    alone would make one tenant's re-sync overwrite another's point."""
    a, b = uuid.uuid4(), uuid.uuid4()
    await repository.upsert(tenant_id=a, vertical="education", points=[point("shared")])
    await repository.upsert(tenant_id=b, vertical="education", points=[point("shared")])
    assert await repository.count(tenant_id=a, vertical="education") == 1
    assert await repository.count(tenant_id=b, vertical="education") == 1


# ─────────────────────────── reconciliation ───────────────────────────


async def test_reconciliation_detects_a_missing_point(repository):
    """Postgres has a chunk Qdrant does not. The chunk is unsearchable and
    every count elsewhere looks right — the outbox's failure mode, seen from
    the other end."""
    tenant = uuid.uuid4()
    await repository.upsert(tenant_id=tenant, vertical="education", points=[point("present")])

    report = await repository.reconcile(
        tenant_id=tenant, vertical="education", expected_chunk_ids=["present", "absent"]
    )
    assert report.missing == ("absent",)
    assert report.orphaned == ()
    assert report.in_sync is False


async def test_reconciliation_detects_an_orphan_point(repository):
    """Qdrant has a point Postgres does not — a deleted chunk still answerable."""
    tenant = uuid.uuid4()
    await repository.upsert(
        tenant_id=tenant, vertical="education", points=[point("live"), point("deleted")]
    )
    report = await repository.reconcile(
        tenant_id=tenant, vertical="education", expected_chunk_ids=["live"]
    )
    assert report.orphaned == ("deleted",)
    assert report.missing == ()


async def test_reconciliation_reports_in_sync_when_they_agree(repository):
    tenant = uuid.uuid4()
    await repository.upsert(
        tenant_id=tenant, vertical="education", points=[point("one"), point("two")]
    )
    report = await repository.reconcile(
        tenant_id=tenant, vertical="education", expected_chunk_ids=["one", "two"]
    )
    assert report.in_sync is True


async def test_reconciliation_is_scoped_to_its_tenant(repository):
    """Otherwise every tenant's points read as orphans of every other tenant,
    and the nightly job proposes deleting the entire corpus."""
    a, b = uuid.uuid4(), uuid.uuid4()
    await repository.upsert(tenant_id=a, vertical="education", points=[point("a-only")])
    await repository.upsert(tenant_id=b, vertical="education", points=[point("b-only")])

    report = await repository.reconcile(
        tenant_id=a, vertical="education", expected_chunk_ids=["a-only"]
    )
    assert report.in_sync is True, f"saw another tenant's points: {report.orphaned}"
