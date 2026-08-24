"""The Qdrant repository — design §7.2.

**Flagged for line-by-line human review. The single most dangerous file in P1.**

One collection holds every tenant's points (§7.2 — collection-per-tenant eats
RAM at a few dozen tenants, and sooner on 3.3 GB). So the tenant filter is the
only thing separating a university's students from a property developer's
catalogue, and its absence has no behavioural signature: an unfiltered search
returns results, ranked plausibly, and they look correct right up until
somebody notices they belong to another company.

The design goal is therefore not "always pass the filter" but **make the
unfiltered query unrepresentable**, the way `verify_signature` takes only
bytes so a re-serialized body cannot be handed to it. Three facts together:

1. **No public method accepts a filter.** There is no parameter through which
   a caller could supply one, so there is none through which they could supply
   an empty one.
2. **The only object that can talk to Qdrant is `_TenantScope`, and it cannot
   be constructed without a tenant id.** `QdrantRepository` holds the client
   solely to hand it to a scope and never calls it. There is no expression in
   this module that reaches Qdrant without a tenant id having been supplied.
3. **Every `models.Filter` is built by `_TenantScope._filter`**, which has one
   return and always emits the tenant clause. A conditionally-applied filter is
   the version of this bug that survives review — right on every path anyone
   tested, absent on the one they did not.

`tests/retrieval/test_vector_tenancy.py` asserts all three against the source,
because none of them can be observed by calling the code.

Point ids are UUIDv5 over **tenant id and chunk id together**. Chunk ids are
unique per tenant, not globally, so deriving the point id from the chunk id
alone would make one tenant's re-sync overwrite another tenant's point — a
cross-tenant write through the idempotency mechanism.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import models

from moc.config_store import load

# Shapes and ids with no vendor SDK behind them, so a caller needing a
# point id does not pull qdrant_client into its import graph. Re-exported
# here because this module is still the place people look for them.
from moc.retrieval.records import VectorPoint, point_id_for

_DEFAULTS = "retrieval/defaults"


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconciliationReport:
    """What Postgres and Qdrant disagree about, for one tenant and vertical.

    Both directions matter and they fail differently. `missing` is a chunk
    that exists and cannot be found — the outbox's failure seen from the far
    end. `orphaned` is a point whose chunk was deleted, still retrievable and
    still answerable, which is how a withdrawn fee keeps being quoted.
    """

    tenant_id: uuid.UUID
    vertical: str
    missing: tuple[str, ...] = ()
    orphaned: tuple[str, ...] = ()

    @property
    def in_sync(self) -> bool:
        return not self.missing and not self.orphaned


@dataclass(frozen=True)
class _TenantScope:
    """A Qdrant client bound to exactly one tenant and one collection.

    The only object in this module with query methods. `tenant_id` has no
    default, so an unscoped scope is a `TypeError` rather than a quiet
    everything-query.
    """

    client: Any
    tenant_id: uuid.UUID
    collection: str
    tenant_field: str = "tenant_id"
    chunk_field: str = "chunk_id"

    def _filter(self, chunk_ids: Sequence[str] | None = None) -> models.Filter:
        """The one place a filter is built. Always carries the tenant clause.

        Extra conditions append to the tenant condition and can never replace
        it — there is no argument here that could remove or override it, which
        is what keeps the single-return shape meaningful.
        """
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key=self.tenant_field,
                match=models.MatchValue(value=str(self.tenant_id)),
            )
        ]
        if chunk_ids is not None:
            conditions.append(
                models.FieldCondition(
                    key=self.chunk_field, match=models.MatchAny(any=list(chunk_ids))
                )
            )
        return models.Filter(must=conditions)

    async def search(self, *, vector: list[float], limit: int) -> list[SearchHit]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=self._filter(),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                chunk_id=hit.payload[self.chunk_field],
                score=hit.score,
                payload=dict(hit.payload),
            )
            for hit in response.points
        ]

    async def upsert(self, *, points: Sequence[VectorPoint]) -> int:
        await self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=str(point_id_for(self.tenant_id, point.chunk_id)),
                    vector=point.vector,
                    # The tenant lands in the payload here and nowhere else, so
                    # a point cannot be written under a tenant the caller did
                    # not name.
                    payload={
                        **dict(point.payload),
                        self.tenant_field: str(self.tenant_id),
                        self.chunk_field: point.chunk_id,
                    },
                )
                for point in points
            ],
            wait=True,
        )
        return len(points)

    async def delete(self, *, chunk_ids: Sequence[str]) -> int:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._filter(chunk_ids)),
            wait=True,
        )
        return len(chunk_ids)

    async def count(self) -> int:
        response = await self.client.count(
            collection_name=self.collection, count_filter=self._filter(), exact=True
        )
        return response.count

    async def chunk_ids(self) -> set[str]:
        """Every chunk id this tenant holds, for reconciliation."""
        found: set[str] = set()
        offset = None
        while True:
            records, offset = await self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._filter(),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            found.update(record.payload[self.chunk_field] for record in records)
            if offset is None:
                return found


def collection_for(vertical: str, *, config: dict[str, Any] | None = None) -> str:
    """One collection per vertical (§7.2). A `KeyError` on an unknown one.

    A pure lookup, deliberately not a method on the repository: it needs no
    client and no tenant, and keeping it off that class is what lets "every
    public method takes a tenant id" stay an absolute statement rather than a
    rule with two exceptions nobody re-examines.

    Creating a collection on demand would be how a typo becomes a corpus
    nobody searches — ingestion succeeds, the points land somewhere real, and
    every query against the intended collection returns nothing.
    """
    return (config or load(_DEFAULTS))["qdrant"]["collections"][vertical]


def qdrant_client(**overrides: Any) -> Any:
    """The one place an `AsyncQdrantClient` is constructed.

    Here rather than in the composition root because the contract is "the
    Qdrant client is imported by exactly one module" — and that contract is
    what makes the tenant filter checkable. A composition root that imported
    the SDK to build a client would be a second module able to query, which is
    a second place a filter can be forgotten.
    """
    from qdrant_client import AsyncQdrantClient

    from moc.config import settings

    return AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_key or None, **overrides
    )


class QdrantAdmin:
    """Collection topology. Reads and writes **no points**.

    Split from `QdrantRepository` so the object that touches tenant data has
    no untenanted method at all. Creating a collection and setting up an index
    are genuinely tenant-free operations, and the honest way to say so is to
    put them somewhere that cannot reach a point — rather than adding
    exceptions to the rule that guards the dangerous surface.

    A test asserts this class never calls a point-level client method.
    """

    def __init__(self, *, client: Any, config: dict[str, Any] | None = None) -> None:
        self._client = client
        self._config = config or load(_DEFAULTS)
        self._settings = self._config["qdrant"]

    async def ensure_collections(self) -> None:
        """Create what is missing, with the memory shape §7.2 requires.

        Vectors on disk and scalar quantization in RAM from day one. The
        alternative on 3.3 GB is a box that swaps under the first real corpus,
        and retrofitting the layout means re-uploading every point.
        """
        existing = {
            collection.name for collection in (await self._client.get_collections()).collections
        }
        for name in self._settings["collections"].values():
            if name in existing:
                continue
            await self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self._settings["vector_size"],
                    distance=models.Distance[self._settings["distance"].upper()],
                    on_disk=self._settings["on_disk_vectors"],
                ),
                quantization_config=(
                    models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=self._settings["quantile"],
                            always_ram=self._settings["always_ram_quantized"],
                        )
                    )
                    if self._settings["scalar_quantization"]
                    else None
                ),
            )
            # Tenant-aware partitioning (§7.2). Without the index the filter is
            # still correct and scans every tenant's points to apply it.
            await self._client.create_payload_index(
                collection_name=name,
                field_name=self._settings["tenant_field"],
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=True
                ),
            )

    async def collection_info(self, *, vertical: str) -> Any:
        return await self._client.get_collection(collection_for(vertical, config=self._config))


class QdrantRepository:
    """The sole entry point to the vector store (§7.2).

    Holds the client only to hand it to a `_TenantScope`. It never queries
    directly, and a test asserts that against the source — otherwise the
    scope's guarantee would be a convention this class could step around.
    """

    def __init__(self, *, client: Any, config: dict[str, Any] | None = None) -> None:
        settings = (config or load(_DEFAULTS))["qdrant"]
        self._client = client
        self._settings = settings
        self._config = config or load(_DEFAULTS)

    # ─────────────────────────── scoped operations ───────────────────────────

    def _scope(self, tenant_id: uuid.UUID, vertical: str) -> _TenantScope:
        return _TenantScope(
            client=self._client,
            tenant_id=tenant_id,
            collection=collection_for(vertical, config=self._config),
            tenant_field=self._settings["tenant_field"],
            chunk_field=self._settings["chunk_field"],
        )

    async def upsert(
        self, *, tenant_id: uuid.UUID, vertical: str, points: Sequence[VectorPoint]
    ) -> int:
        return await self._scope(tenant_id, vertical).upsert(points=points)

    async def search(
        self, *, tenant_id: uuid.UUID, vertical: str, vector: list[float], limit: int = 20
    ) -> list[SearchHit]:
        return await self._scope(tenant_id, vertical).search(vector=vector, limit=limit)

    async def delete(
        self, *, tenant_id: uuid.UUID, vertical: str, chunk_ids: Sequence[str]
    ) -> int:
        return await self._scope(tenant_id, vertical).delete(chunk_ids=chunk_ids)

    async def count(self, *, tenant_id: uuid.UUID, vertical: str) -> int:
        return await self._scope(tenant_id, vertical).count()

    async def reconcile(
        self, *, tenant_id: uuid.UUID, vertical: str, expected_chunk_ids: Sequence[str]
    ) -> ReconciliationReport:
        """Compare this tenant's points against what Postgres says it has (§7.2).

        Scoped like everything else. An unscoped reconciliation would read
        every other tenant's points as orphans and propose deleting the whole
        corpus — the nightly job turning a missing filter into data loss.
        """
        actual = await self._scope(tenant_id, vertical).chunk_ids()
        expected = set(expected_chunk_ids)
        return ReconciliationReport(
            tenant_id=tenant_id,
            vertical=vertical,
            missing=tuple(sorted(expected - actual)),
            orphaned=tuple(sorted(actual - expected)),
        )


__all__ = [
    "QdrantAdmin",
    "QdrantRepository",
    "collection_for",
    "ReconciliationReport",
    "SearchHit",
    "VectorPoint",
    "point_id_for",
]
