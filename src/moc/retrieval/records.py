"""The shapes and ids the stores agree on, with no vendor SDK behind them.

`VectorPoint` is three fields and `point_id_for` is a uuid5; neither needs a
Qdrant client to exist. They lived in `vectors.py` until Task 31, which was
fine while the only caller was the repository — and stopped being fine the
moment the knowledge service needed a point id, because that pulled
`qdrant_client` into the import graph of `moc.api` and broke the contract
saying the vendor SDK lives in its adapter and nowhere else.

Import-linter caught it. That is the contract working rather than a nuisance:
an API package that transitively requires a vector-database driver is one
refactor away from importing it directly.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_DEFAULTS = "retrieval/defaults"


@dataclass(frozen=True)
class VectorPoint:
    """One embedded chunk, before it acquires a tenant."""

    chunk_id: str
    vector: list[float]
    payload: Mapping[str, Any] = field(default_factory=dict)


def point_id_for(
    tenant_id: uuid.UUID, chunk_id: str, *, config: dict[str, Any] | None = None
) -> uuid.UUID:
    """UUIDv5 over tenant and chunk together. **The only derivation there is.**

    Both halves are required. Chunk ids are unique per tenant only — two
    tenants ingesting the same source file produce the same chunk id — so a
    point id derived from the chunk alone would let one tenant's upsert
    silently overwrite another's point. That is a cross-tenant *write*, and it
    would arrive through the mechanism that exists to make replay safe.

    **There used to be two of these and they disagreed.** `ingest.point_id_for`
    hashed the chunk id alone under the configured namespace and wrote the
    result into `kb_outbox.point_id`; the repository hashed tenant and chunk
    under `NAMESPACE_URL`. Nothing had drained the outbox yet, so the column
    was unread and the disagreement invisible — and the first person to write a
    sync worker would have taken the point id off the row it is stored on,
    upserted with it, and reintroduced exactly the collision the paragraph
    above exists to prevent. Task 31 is the first code to drain the outbox,
    which is what surfaced it.

    The namespace is config (§19). It was already config on one side, and a
    UUID namespace is precisely the kind of value that must not be a literal in
    two places.
    """
    settings = config or _load()
    namespace = uuid.UUID(settings["outbox"]["point_namespace"])
    return uuid.uuid5(namespace, f"{tenant_id}/{chunk_id}")


def _load() -> dict[str, Any]:
    from moc.config_store import load

    return load(_DEFAULTS)


__all__ = ["VectorPoint", "point_id_for"]
