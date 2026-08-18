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

**The confidence gate returns nothing, not something.** §7.5 is a hard rule:
below the threshold the turn does not reach answer composition. Handing back a
weak passage would let composition ground a fee on something merely adjacent,
which is precisely the shape of an invented figure that traces to a real
chunk. The gate needs an empty result to route on.

**One arm is a degraded result, not an error** (§7.3). If embeddings are down
the correct behaviour is Meilisearch-only, and the caller is told which arm was
missing so a quality dip is attributable rather than mysterious.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
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

    `relevance` is the arm's own 0-1 judgement — cosine similarity from
    Qdrant, ranking score from Meilisearch. It defaults to zero, so an arm
    that supplies no relevance signal contributes no confidence and the gate
    fails closed rather than open.
    """

    chunk_id: str
    rank: int
    relevance: float = 0.0
    content: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedHit:
    #: RRF score — for ordering only. Not comparable between queries.
    chunk_id: str
    score: float
    #: Best relevance any arm gave this chunk. This is what the gate reads.
    relevance: float = 0.0
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
    confidence: float = 0.0
    degraded_arms: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    gate_closed: bool = False

    @property
    def passages(self) -> tuple[str, ...]:
        return tuple(hit.content for hit in self.hits)


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
            # Best across arms, not summed: two arms each 60% sure is not 120%.
            relevance[candidate.chunk_id] = max(
                relevance.get(candidate.chunk_id, 0.0), candidate.relevance
            )
            if candidate.content and candidate.chunk_id not in contents:
                contents[candidate.chunk_id] = candidate.content
            if candidate.payload and candidate.chunk_id not in payloads:
                payloads[candidate.chunk_id] = dict(candidate.payload)

    return [
        FusedHit(
            chunk_id=chunk_id,
            score=score,
            relevance=relevance.get(chunk_id, 0.0),
            content=contents.get(chunk_id, ""),
            arms=tuple(contributing[chunk_id]),
            payload=payloads.get(chunk_id, {}),
        )
        # Ties broken by chunk id so the order is stable across runs. An
        # unstable order makes every recall measurement slightly noisy and the
        # noise is indistinguishable from a real regression.
        for chunk_id, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def confidence_of(hits: Sequence[FusedHit]) -> float:
    """The §7.5 gate input: how relevant the best hit actually is.

    The top hit's own relevance, not its RRF score. RRF scores depend on how
    many arms answered and how deep their lists were, so the same passage
    scores differently during an embedding outage than outside one — a
    threshold set on that would silently tighten whenever an arm went down,
    which is the opposite of degrading gracefully.

    Zero when nothing was found, and zero when the arms supplied no relevance
    signal at all. Both are "we have no reason to believe this answers the
    question", and both must close the gate.
    """
    return hits[0].relevance if hits else 0.0


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

    if confidence < settings["min_score"]:
        # §7.5. Empty, not low-ranked: the gate needs a result it can route on,
        # and a weak passage handed onward is a fee grounded in something
        # merely adjacent.
        return FusionResult(
            confidence=confidence,
            degraded_arms=tuple(degraded),
            elapsed_ms=(clock() - started) * 1000,
            gate_closed=True,
        )

    top = fused[: settings["final_k"]]
    if reranker is not None:
        top = await reranker.rerank(query=query, hits=top, top_k=settings["final_k"])

    return FusionResult(
        hits=tuple(top),
        confidence=confidence,
        degraded_arms=tuple(degraded),
        elapsed_ms=(clock() - started) * 1000,
    )


__all__ = [
    "Candidate",
    "FusedHit",
    "FusionResult",
    "Reranker",
    "confidence_of",
    "fuse",
    "reciprocal_rank_fusion",
]
