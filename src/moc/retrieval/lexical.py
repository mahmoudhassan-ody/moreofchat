"""The Meilisearch arm — design §7.4.

The lexical half of retrieval, and the half that decides Arabic recall. Dense
similarity is good at paraphrase and bad at the thing Egyptian customers do
most: writing a word the corpus never uses. edu-0015 asks about a منحة against
a knowledge base that says خصم and never says منحة — no shared letters, no
shared root. Nothing in an embedding fixes that reliably; a synonym map does,
and it is auditable by the tenant who knows their own vocabulary.

**Tenant isolation is a scoped search token, not a filter the caller passes**
(§7.4). One shared index per vertical — index-per-tenant multiplies
Meilisearch's per-index overhead by the tenant count, which on 3.3 GB binds
early — and the token embeds the tenant predicate server-side, so a search
issued with it cannot widen its own scope. Same discipline as `vectors.py`:
`_TenantScope` is the only object that can search, and it cannot be built
without a tenant id.

Both text forms are indexed (§7.1). The original carries the tenant's own
spelling; the normalized form carries the spelling customers type. A query for
القنطره has to reach a chunk written القنطرة, and only one of those two fields
makes that happen.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from uuid import UUID

from moc.arabic.normalize import normalize
from moc.config_store import load

_LEXICAL = "retrieval/lexical"

#: Stripped before a token is compared against the stop list or against a
#: tenant's synonym key, never from the query itself. Both scripts' marks,
#: because a Masri message mixes them.
_PUNCTUATION = "؟،؛«»…\"'()[]{}!?.,:;-—–"


@dataclass(frozen=True)
class LexicalDocument:
    """One chunk as Meilisearch stores it."""

    point_id: str
    chunk_id: str
    content: str
    title: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    rank: int
    content: str
    payload: Mapping[str, Any] = field(default_factory=dict)


def _comparable(word: str) -> str:
    """The form a stop word is recognised by: normalized, punctuation off.

    Customers type `كام؟` and `عامة،`, not `كام ؟`. Splitting on whitespace
    alone leaves the mark glued to the word, the token never equals the config
    entry, and the filler survives — the same empty result this module exists
    to prevent, reintroduced by a space that is not there.
    """
    return normalize(word.strip(_PUNCTUATION))


@lru_cache(maxsize=8)
def _stop_word_forms(stop_words: tuple[str, ...]) -> frozenset[str]:
    return frozenset(_comparable(word) for word in stop_words)


def expand_query(query: str, synonyms: Mapping[str, Sequence[str]]) -> str:
    """Append a tenant's own words for what the corpus calls something else.

    **Query-side because the index is shared.** One Meilisearch index per
    vertical serves every tenant in it, so a synonym written into the index
    settings is one broker's word for an area changing another broker's
    ranking. Applied here, the expansion reaches only the query it was scoped
    to — which is the only scoping a shared index allows.

    It is deliberately *not* identical to Meilisearch's own synonym handling.
    The expansion is appended rather than substituted, so the customer's own
    words survive and `matching_strategy: frequency` drops whichever side
    matches nothing. The platform synonym map still lives in the index, where
    it is curated and shared on purpose.

    Matched on whole words after normalization and after the same punctuation
    strip the stop-word check uses, so a tenant's `التجمع` reaches a query
    written `التجمع؟` — and so a two-letter entry cannot fire inside every
    word that happens to contain it.
    """
    if not synonyms:
        return query
    folded = {normalize(key): values for key, values in synonyms.items()}
    additions: list[str] = []
    for word in query.split():
        for value in folded.get(normalize(word.strip(_PUNCTUATION)), ()):
            if value not in query and value not in additions:
                additions.append(value)
    return " ".join([query, *additions]) if additions else query


def strip_stop_words(query: str, config: dict[str, Any] | None = None) -> str:
    """Remove stop words from the query before it reaches Meilisearch.

    A workaround for an engine defect, and it has to live here rather than in
    config. Meilisearch normalizes the `stopWords` setting on write — ي and ى
    become ی (U+06CC), ك becomes ک — but matches tokens against the stored
    list without applying that normalization, so any stop word containing one
    of those letters is stored in a form nothing equals and never fires.
    Measured on 1.53.1: `عاوز` as a stop word makes `عاوز` unsearchable,
    `عايز` does not. Six of the sixteen Arabic entries in config are affected,
    `في` among them. Writing the entry in the U+06CC form does not help — all
    three spellings collapse to the same stored value, and none of them work.

    Why an inert filler is not a small loss: the default matching strategy
    drops query words from the *end*, and Masri puts the scaffolding at the
    *start*. So `عايز أعرف الحد الأدنى للقبول` keeps `عايز` through every drop
    step, and since it matches no document the whole query ANDs to nothing.
    Not a bad ranking — an empty result.

    Matching is on the normalized form so config carries one spelling of a
    word rather than every hamza variant. The *surviving* words keep their
    original form: normalization decides what to drop, it never rewrites what
    stays, because the synonym keys are stored as customers type them and a
    rewritten query stops matching them.
    """
    settings = config or load(_LEXICAL)
    stop = _stop_word_forms(tuple(settings["stop_words"]))
    kept = [word for word in query.split() if _comparable(word) not in stop]
    # Stripping to nothing turns a weak query into a match-everything one.
    # Searching the words the customer actually sent and ranking badly is the
    # better failure.
    return " ".join(kept) if kept else query


def index_for(vertical: str, *, config: dict[str, Any] | None = None) -> str:
    """One index per vertical. A `KeyError` on an unknown one.

    A pure lookup, off the searching class for the same reason as
    `vectors.collection_for`: it keeps "every method that searches takes a
    tenant id" an absolute statement rather than a rule with an exception.
    """
    return (config or load(_LEXICAL))["meilisearch"]["indexes"][vertical]


@dataclass(frozen=True)
class _TenantScope:
    """A Meilisearch index bound to one tenant.

    `tenant_id` has no default. There is no way to construct a scope that
    searches everyone, which is the same guarantee `vectors._TenantScope`
    makes and for the same reason: the filter's absence has no behavioural
    signature, it just returns somebody else's documents.
    """

    client: Any
    tenant_id: UUID
    index: str
    tenant_field: str = "tenant_id"
    #: Carried so the scope can strip stop words query-side. Defaults to the
    #: loaded config rather than being required, so a scope is still cheap to
    #: construct in a test.
    config: dict[str, Any] | None = None
    #: §7.4. `last` drops query words from the end, which is the wrong end for
    #: Masri. See the config comment for the measured comparison.
    matching_strategy: str = "frequency"

    def _filter(self) -> str:
        """The one place a filter is built, and it always names the tenant."""
        return f"{self.tenant_field} = '{self.tenant_id}'"

    async def search(self, *, query: str, limit: int) -> list[LexicalHit]:
        index = self.client.index(self.index)
        response = await index.search(
            strip_stop_words(query, self.config),
            limit=limit,
            filter=self._filter(),
            matching_strategy=self.matching_strategy,
        )
        return [
            LexicalHit(
                chunk_id=hit["chunk_id"],
                rank=position,
                content=hit.get("content", ""),
                payload=hit,
            )
            for position, hit in enumerate(response.hits, start=1)
        ]

    async def add(self, *, documents: Sequence[LexicalDocument]) -> int:
        index = self.client.index(self.index)
        task = await index.add_documents(
            [
                {
                    "point_id": document.point_id,
                    "chunk_id": document.chunk_id,
                    self.tenant_field: str(self.tenant_id),
                    "content": document.content,
                    # Indexed alongside the original so a query in the
                    # spelling customers type reaches a chunk in the spelling
                    # the tenant wrote (§7.1).
                    "content_normalized": normalize(document.content),
                    "title": document.title,
                    "title_normalized": normalize(document.title),
                    **dict(document.payload),
                }
                for document in documents
            ]
        )
        await self.client.wait_for_task(task.task_uid)
        return len(documents)

    async def delete(self, *, point_ids: Sequence[str]) -> int:
        """Remove named documents. By primary key, which is `point_id`.

        Added for Task 31: removing a document from the console has to remove
        it from the lexical arm too, and the only delete this repository had
        was `delete_all`. Without it a deleted fee stayed searchable in
        Meilisearch while being gone from Postgres and Qdrant — one arm of
        fusion still answering with a figure the tenant withdrew, which reads
        as the bot inventing it.

        The point id already carries the tenant (see `vectors.point_id_for`),
        so naming another tenant's document means first deriving its id from
        that tenant's uuid — which is the thing a caller does not have.
        """
        if not point_ids:
            return 0
        index = self.client.index(self.index)
        task = await index.delete_documents(ids=list(point_ids))
        await self.client.wait_for_task(task.task_uid)
        return len(point_ids)

    async def delete_all(self) -> None:
        """Only this tenant's documents. Scoped like everything else — an
        unscoped clear is one tenant wiping the shared index."""
        index = self.client.index(self.index)
        task = await index.delete_documents_by_filter(self._filter())
        await self.client.wait_for_task(task.task_uid)


class MeilisearchAdmin:
    """Index topology and settings. Searches nothing.

    Split from the searching class for the same reason as `QdrantAdmin`:
    creating an index and applying a synonym map are genuinely tenant-free,
    and the honest way to say so is to put them where they cannot reach a
    document.
    """

    def __init__(self, *, client: Any, config: dict[str, Any] | None = None) -> None:
        self._client = client
        self._config = config or load(_LEXICAL)
        self._settings = self._config["meilisearch"]

    async def ensure_indexes(self) -> None:
        """Create each index and push the settings that decide Arabic recall.

        Synonyms and stop words come from config on every run, not once at
        creation: a tenant adding a vocabulary entry must not require dropping
        and rebuilding an index to take effect.
        """
        for name in self._settings["indexes"].values():
            await self._client.create_index(name, primary_key=self._settings["primary_key"])
            index = self._client.index(name)
            tenant_field = self._settings["tenant_field"]
            await self._client.wait_for_task(
                (await index.update_filterable_attributes([tenant_field])).task_uid
            )
            await self._client.wait_for_task(
                (
                    await index.update_searchable_attributes(
                        self._settings["searchable_attributes"]
                    )
                ).task_uid
            )
            await self._client.wait_for_task(
                (await index.update_stop_words(self._config["stop_words"])).task_uid
            )
            await self._client.wait_for_task(
                (await index.update_synonyms(self._config["synonyms"])).task_uid
            )

    async def settings_for(self, *, vertical: str) -> Any:
        index = self._client.index(index_for(vertical, config=self._config))
        return await index.get_settings()


class MeilisearchRepository:
    """The sole search entry point. Holds the client only to scope it."""

    def __init__(self, *, client: Any, config: dict[str, Any] | None = None) -> None:
        self._client = client
        self._config = config or load(_LEXICAL)
        self._settings = self._config["meilisearch"]

    def _scope(self, tenant_id: UUID, vertical: str) -> _TenantScope:
        return _TenantScope(
            client=self._client,
            tenant_id=tenant_id,
            index=index_for(vertical, config=self._config),
            tenant_field=self._settings["tenant_field"],
            config=self._config,
            matching_strategy=self._settings["matching_strategy"],
        )

    async def add(
        self, *, tenant_id: UUID, vertical: str, documents: Sequence[LexicalDocument]
    ) -> int:
        return await self._scope(tenant_id, vertical).add(documents=documents)

    async def search(
        self, *, tenant_id: UUID, vertical: str, query: str, limit: int = 20
    ) -> list[LexicalHit]:
        """Search this tenant's slice of the index.

        The query is searched as written. Normalization happens on the *index*
        side, and Meilisearch's own typo tolerance plus the synonym map cover
        the query side — normalizing the query here as well would fold it into
        a form the synonym keys no longer match.
        """
        return await self._scope(tenant_id, vertical).search(query=query, limit=limit)

    async def remove(
        self, *, tenant_id: UUID, vertical: str, point_ids: Sequence[str]
    ) -> int:
        return await self._scope(tenant_id, vertical).delete(point_ids=point_ids)

    async def clear(self, *, tenant_id: UUID, vertical: str) -> None:
        await self._scope(tenant_id, vertical).delete_all()


__all__ = [
    "LexicalDocument",
    "LexicalHit",
    "MeilisearchAdmin",
    "MeilisearchRepository",
    "index_for",
    "expand_query",
    "strip_stop_words",
]
