"""Hybrid fusion — design §7.5.

Two arms, reciprocal rank fusion, then a confidence gate. Each part exists for
a failure the other cannot cover:

**Two arms.** Dense similarity handles paraphrase and fails at vocabulary the
corpus never uses; lexical search handles exact words and fails at rephrasing.
Egyptian customers do both in one message. edu-0015 — a منحة asked against a
corpus that says خصم — is the case that only the lexical arm's synonym map can
answer, and edu-0005 is one where typo tolerance and the dense arm carry it.

**RRF rather than score blending.** The two arms produce scores on
incomparable scales; normalizing them means choosing a normalization that
nobody can defend when the ranking changes. Rank is the one thing both arms
agree on the meaning of.

**The gate returns nothing, not something.** §7.5 is a hard rule: a turn that
retrieval cannot support does not reach answer composition, and the gate needs
an empty result to route on rather than a weak passage the caller might use
anyway.

What the gate decides is *presence*, not similarity. It was specified as a
similarity threshold, and that was measured against the 17 education cases
and does not work: cases that must be answered scored 0.218-0.610 and cases
that must not scored 0.280-0.779. The populations overlap almost entirely and
overlap the wrong way round — the single most similar result in the suite
(0.779) is a case requiring clarification, and the least similar (0.218) is
the acceptance case retrieving exactly the right chunk. No cutoff exists.

The reason is structural rather than a matter of tuning. Cosine answers "is
this chunk similar to the query", and edu-0015 is the case built to show that
is the wrong question: منحة and خصم share no letters, the synonym map does the
work, and the dense arm cannot see it. Meanwhile answer-versus-clarify is
decided by whether the slots are known — which is the script engine's job, not
retrieval's.

So the gate now closes on what it can actually determine: nothing was
retrieved, or nothing retrieved has any text to ground on. `min_score` stays
in config, at the floor, because a tenant whose corpus does separate cleanly
should be able to raise it without a deploy.

**One arm is a degraded result, not an error** (§7.3). If embeddings are down
the correct behaviour is Meilisearch-only, and the caller is told which arm was
missing so a quality dip is attributable rather than mysterious.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from moc.config_store import load

_LEXICAL = "retrieval/lexical"


@dataclass(frozen=True)
class Candidate:
    """One arm's opinion about one chunk: a rank for ordering, a score for the gate.

    Both, because they answer different questions. Rank is what RRF combines —
    the arms score on incomparable scales and only their orderings mean the
    same thing. But rank cannot drive the confidence gate: RRF is deliberately
    flat, so with k=60 a bottom-of-list hit still reads three quarters of a
    top-of-list one, and a threshold set on that closes on nothing.

    `relevance` is the arm's own calibrated 0-1 judgement — cosine similarity
    from Qdrant. `None` means this arm has no calibrated score to give, which
    is Meilisearch's actual position: it ranks, it does not score.

    None rather than a rank-derived stand-in, because the stand-in was worse
    than nothing. `1.0 / rank` made every chunk the lexical arm ranked first
    score exactly 1.0 — perfect confidence derived from position in a list.
    Across the 17 education cases 7 read 1.000, three of them cases the suite
    requires *not* be answered. An arm with nothing calibrated to say must
    contribute nothing, not a number that outranks every real measurement.
    """

    chunk_id: str
    rank: int
    relevance: float | None = None
    content: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedHit:
    #: RRF score — for ordering only. Not comparable between queries.
    chunk_id: str
    score: float
    #: Best relevance any *calibrated* arm gave this chunk, or None when no
    #: arm had a score. This is what the gate reads.
    relevance: float | None = None
    content: str = ""
    arms: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionResult:
    """What retrieval hands the orchestrator.

    `passages` is empty whenever the gate closed — the caller must not have to
    re-apply the threshold, because a caller that forgets is a caller that
    grounds an answer on a weak passage.

    `degraded_arms` names what was missing. A quality dip during an embedding
    outage should be attributable to the outage rather than investigated as a
    ranking regression.
    """

    hits: tuple[FusedHit, ...] = ()
    #: None when no arm supplied a calibrated score — an embedding outage, or
    #: a lexical-only deployment. Distinct from 0.0, which means the dense arm
    #: answered and found nothing similar.
    confidence: float | None = None
    degraded_arms: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    gate_closed: bool = False
    #: Figures the script itself may state (§3.1). Always empty here — the
    #: retrieval layer has no script — but present so this object satisfies
    #: the orchestrator's `Retriever` port directly.
    #:
    #: The port is structural, and it has to be: `Retrieval` is defined in
    #: `moc.agent`, and importing it here would make the retrieval layer
    #: depend on the agent layer. Satisfying the shape rather than the name is
    #: what keeps the dependency pointing agent -> retrieval.
    script_constants: tuple[str, ...] = ()
    #: What the query embedding cost, for the ledger. Zero when the dense arm
    #: did not run — a lexical-only deployment, or an embedding outage (§7.3).
    #:
    #: Carried rather than metered here for the same reason `Completion` is:
    #: the retrieval layer has no session and no tenant, and a ledger write
    #: outside the caller's transaction would survive a turn that rolled back.
    embedding_model: str = ""
    embedding_tokens: int = 0

    @property
    def passages(self) -> tuple[str, ...]:
        return tuple(hit.content for hit in self.hits)

    @property
    def titles(self) -> tuple[str, ...]:
        """What each retrieved passage is *about*, in the corpus's own words.

        `passages` is bodies, and a body does not say which question it
        answers — so a clarification could only ask "what exactly do you
        need?" of a customer who had already asked clearly (edu-0009:
        "المواعيد إيه؟", which could be branch hours, bus times or a
        deadline). A Q&A corpus keeps its subject in the title, which is the
        one field that can be offered back as an option.

        Untitled hits contribute nothing rather than an empty string: a blank
        option reads as a rendering fault, and a corpus without titles has to
        degrade to the generic clarification rather than to a list of
        nothing. Duplicates collapse — the fixture carries each fact in two
        languages and they retrieve together, so offering both asks the
        customer to choose between a thing and itself.
        """
        seen: dict[str, None] = {}
        for hit in self.hits:
            title = str(hit.payload.get("title") or "").strip()
            if title:
                seen.setdefault(title)
        return tuple(seen)


class Reranker(Protocol):
    """Cross-encoder rerank, via API (§7.5 notes).

    A seam, not an omission. Self-hosting a cross-encoder does not fit in
    3.3 GB, and no rerank vendor is configured yet — so RRF order stands until
    one is. Wiring a placeholder that returns the input unchanged would look
    like a reranker in every trace and improve nothing.
    """

    async def rerank(
        self, *, query: str, hits: Sequence[FusedHit], top_k: int
    ) -> list[FusedHit]: ...


def reciprocal_rank_fusion(
    arms: dict[str, Sequence[Candidate]], *, config: dict[str, Any] | None = None
) -> list[FusedHit]:
    """Combine ranked lists by 1/(k + rank), summed across arms.

    A chunk both arms rank must outrank one that either arm ranks alone, even
    when the solo chunk is first in its own list — agreement between two
    independent retrievers is stronger evidence than confidence within one.
    """
    settings = (config or load(_LEXICAL))["fusion"]
    k = settings["rrf_k"]

    scores: dict[str, float] = {}
    relevance: dict[str, float] = {}
    contents: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    contributing: dict[str, list[str]] = {}

    for arm, candidates in arms.items():
        for candidate in candidates:
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (
                k + candidate.rank
            )
            contributing.setdefault(candidate.chunk_id, []).append(arm)
            # Best across arms, not summed: two arms each 60% sure is not
            # 120%. An arm with no calibrated score is skipped rather than
            # counted as zero — it has no opinion, which is not the same as a
            # low one, and averaging silence in would drag every score down.
            if candidate.relevance is not None:
                relevance[candidate.chunk_id] = max(
                    relevance.get(candidate.chunk_id, candidate.relevance),
                    candidate.relevance,
                )
            if candidate.content and candidate.chunk_id not in contents:
                contents[candidate.chunk_id] = candidate.content
            if candidate.payload:
                # Merged across arms, first writer wins per key. The arms
                # describe the same chunk and index different fields of it —
                # the lexical document carries `title`, the vector point
                # carries `content` — so taking whichever arm answered first
                # and discarding the other's keys loses whichever field that
                # arm does not index. `titles` read empty in the live suite
                # for exactly that reason, and the clarification it feeds was
                # silently dead while every unit test passed.
                merged = payloads.setdefault(candidate.chunk_id, {})
                for key, value in candidate.payload.items():
                    merged.setdefault(key, value)

    return [
        FusedHit(
            chunk_id=chunk_id,
            score=score,
            relevance=relevance.get(chunk_id),
            content=contents.get(chunk_id, ""),
            arms=tuple(contributing[chunk_id]),
            payload=payloads.get(chunk_id, {}),
        )
        # Ties broken by chunk id so the order is stable across runs. An
        # unstable order makes every recall measurement slightly noisy and the
        # noise is indistinguishable from a real regression.
        for chunk_id, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def confidence_of(hits: Sequence[FusedHit]) -> float | None:
    """The §7.5 gate input: how relevant the best hit actually is.

    The top hit's own relevance, not its RRF score. RRF scores depend on how
    many arms answered and how deep their lists were, so the same passage
    scores differently during an embedding outage than outside one — a
    threshold set on that would silently tighten whenever an arm went down,
    which is the opposite of degrading gracefully.

    `None` when nothing was found, and `None` when no arm supplied a
    calibrated score. That is not the same as zero: zero is the dense arm
    reporting it found nothing similar, `None` is nobody having looked with an
    instrument that reads. Conflating them is what let a rank-derived 1.0 pass
    for certainty.
    """
    return hits[0].relevance if hits else None


async def fuse(
    *,
    query: str,
    dense: Sequence[Candidate] | None,
    sparse: Sequence[Candidate] | None,
    reranker: Reranker | None = None,
    config: dict[str, Any] | None = None,
    clock: Any = time.perf_counter,
) -> FusionResult:
    """Fuse both arms, gate on confidence, and record how long it took.

    `None` for an arm means it was unavailable — distinct from an empty list,
    which means it answered and found nothing. Conflating them would make an
    embedding outage look like a corpus with no match, and the two want
    opposite responses.
    """
    settings = (config or load(_LEXICAL))["fusion"]
    started = clock()

    available: dict[str, Sequence[Candidate]] = {}
    degraded: list[str] = []
    for name, candidates in (("dense", dense), ("sparse", sparse)):
        if candidates is None:
            degraded.append(name)
        else:
            available[name] = candidates[: settings["candidates_per_arm"]]

    fused = reciprocal_rank_fusion(available, config=config)
    confidence = confidence_of(fused)

    top = fused[: settings["final_k"]]

    # Presence, not similarity. See the module docstring for the measurement
    # that moved this: nothing retrieved, or nothing with text on it, means
    # composition would have to invent every figure in the reply.
    nothing_to_ground_on = not any(hit.content.strip() for hit in top)

    # An uncalibrated result cannot be thresholded. Skipping the comparison
    # rather than defaulting it means an embedding outage degrades to
    # lexical-only (§7.3) instead of closing on every turn. At the configured
    # floor this never fires; it stays wired for a tenant who raises it.
    below = confidence is not None and confidence < settings["min_score"]

    if nothing_to_ground_on or below:
        # §7.5. Empty, not low-ranked: the gate needs a result it can route on,
        # and a weak passage handed onward is a fee grounded in something
        # merely adjacent.
        return FusionResult(
            confidence=confidence,
            degraded_arms=tuple(degraded),
            elapsed_ms=(clock() - started) * 1000,
            gate_closed=True,
        )

    if reranker is not None:
        top = await reranker.rerank(query=query, hits=top, top_k=settings["final_k"])

    return FusionResult(
        hits=tuple(top),
        confidence=confidence,
        degraded_arms=tuple(degraded),
        elapsed_ms=(clock() - started) * 1000,
    )


class FusionRetriever:
    """Fusion behind the orchestrator's `Retriever` port and the runner's
    `ChunkSource` port.

    Two methods returning two shapes of the same search, deliberately. The
    orchestrator needs passage *text* to ground an answer; the recall metric
    needs chunk *ids*. Handing the orchestrator ids it does not use would put
    gold-shaped data one refactor away from the prompt, which is the failure
    the runner's whole design is arranged against.

    The dense arm is optional. Absent, this is §7.3's degraded shape —
    Meilisearch only — which is a correct turn rather than an outage.
    """

    def __init__(
        self,
        *,
        lexical: Any,
        tenant_id: Any,
        vertical: str,
        dense: Any = None,
        embedder: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._lexical = lexical
        self._dense = dense
        self._embedder = embedder
        self._tenant_id = tenant_id
        self._vertical = vertical
        self._config = config or load(_LEXICAL)
        self._last: FusionResult | None = None

    async def _fuse(self, query: str) -> FusionResult:
        settings = self._config["fusion"]
        hits = await self._lexical.search(
            tenant_id=self._tenant_id,
            vertical=self._vertical,
            query=query,
            limit=settings["candidates_per_arm"],
        )
        sparse = [
            # No relevance: Meilisearch ranks, it does not score. The
            # rank-derived `1.0 / rank` that used to sit here made every
            # top-ranked lexical hit read as total certainty, which is how
            # the gate came to be decorative.
            # The payload travels. It is where `title` lives, and a candidate
            # built without one makes `FusionResult.titles` structurally empty
            # — which it was, in the only path that reads it, while `fuse` and
            # `titles` both passed their own tests.
            Candidate(
                chunk_id=hit.chunk_id,
                rank=hit.rank,
                content=hit.content,
                payload=dict(getattr(hit, "payload", None) or {}),
            )
            for hit in hits
        ]

        dense: list[Candidate] | None = None
        embedding_model, embedding_tokens = "", 0
        if self._dense is not None and self._embedder is not None:
            embedding = await self._embedder.embed(texts=[query])
            vector = embedding.vectors[0]
            embedding_model, embedding_tokens = embedding.model, embedding.input_tokens
            dense = [
                Candidate(
                    chunk_id=point.chunk_id,
                    rank=position,
                    relevance=point.score,
                    content=point.payload.get("content", ""),
                    payload=dict(point.payload),
                )
                for position, point in enumerate(
                    await self._dense.search(
                        tenant_id=self._tenant_id,
                        vertical=self._vertical,
                        vector=vector,
                        limit=settings["candidates_per_arm"],
                    ),
                    start=1,
                )
            ]

        fused = await fuse(query=query, dense=dense, sparse=sparse, config=self._config)
        self._last = replace(
            fused, embedding_model=embedding_model, embedding_tokens=embedding_tokens
        )
        return self._last

    async def search(self, *, query: str) -> FusionResult:
        """The orchestrator's port, satisfied structurally.

        Returns the `FusionResult` itself: it already carries `passages`,
        `confidence` and `script_constants`, which is the whole shape the
        orchestrator reads. Constructing the agent's own `Retrieval` here
        would point the retrieval layer at the agent layer for the sake of a
        name.
        """
        return await self._fuse(query)

    async def chunk_ids_for(self, *, query: str) -> tuple[str, ...]:
        """The runner's port, for measuring recall against gold chunks.

        Reuses the last fusion when the query matches, so a run does not
        search twice per turn — and so the ids measured are the ids the
        orchestrator actually saw.
        """
        result = self._last if self._last is not None else await self._fuse(query)
        return tuple(hit.chunk_id for hit in result.hits)


__all__ = [
    "Candidate",
    "FusionRetriever",
    "FusedHit",
    "FusionResult",
    "Reranker",
    "confidence_of",
    "fuse",
    "reciprocal_rank_fusion",
]
