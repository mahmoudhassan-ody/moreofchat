"""Hybrid fusion — design §7.5.

Unit-level: the arms are lists, because what needs testing is the combination
rule and the gate, not whether two search engines are reachable. The arms
themselves are exercised in `test_lexical.py` against a real Meilisearch.

The gate is the part with teeth. §7.5 says a below-threshold turn does not
reach answer composition, and the mechanism is that fusion hands back *no*
passages rather than weak ones — a caller that has to re-apply the threshold
is a caller that will one day forget.
"""

import time

import pytest

from moc.config_store import load
from moc.retrieval.fusion import (
    Candidate,
    FusedHit,
    confidence_of,
    fuse,
    reciprocal_rank_fusion,
)

CONFIG = load("retrieval/lexical")
FUSION = CONFIG["fusion"]


def ranked(*chunk_ids: str, relevance: float = 0.9) -> list[Candidate]:
    """Ranked candidates that also carry an arm relevance.

    Relevance is supplied because the gate reads it. A helper that omitted it
    would build candidates the gate always rejects, and every test here would
    be asserting against a closed gate without saying so.
    """
    return [
        Candidate(
            chunk_id=chunk_id,
            rank=position,
            relevance=relevance,
            content=f"body of {chunk_id}",
        )
        for position, chunk_id in enumerate(chunk_ids, start=1)
    ]


# ─────────────────────────── the combination rule ───────────────────────────


def test_rrf_ranks_a_chunk_both_arms_agree_on_above_either_alone():
    """Agreement between two independent retrievers beats confidence within one.

    `agreed` is second on both lists and still wins, because `dense_only` and
    `sparse_only` each have one arm's vote at rank 1 while `agreed` has two.
    """
    fused = reciprocal_rank_fusion(
        {
            "dense": ranked("dense_only", "agreed"),
            "sparse": ranked("sparse_only", "agreed"),
        },
        config=CONFIG,
    )
    assert fused[0].chunk_id == "agreed"
    assert set(fused[0].arms) == {"dense", "sparse"}


def test_rrf_uses_rank_not_score():
    """The arms score on incomparable scales. Blending them means choosing a
    normalization nobody can defend when the ranking changes."""
    fused = reciprocal_rank_fusion({"dense": ranked("first", "second")}, config=CONFIG)
    expected = 1.0 / (FUSION["rrf_k"] + 1)
    assert fused[0].score == pytest.approx(expected)


def test_ties_break_deterministically():
    """An unstable order makes every recall measurement noisy, and the noise
    is indistinguishable from a regression."""
    arms = {"dense": ranked("b", "a"), "sparse": ranked("a", "b")}
    first = [hit.chunk_id for hit in reciprocal_rank_fusion(arms, config=CONFIG)]
    second = [hit.chunk_id for hit in reciprocal_rank_fusion(arms, config=CONFIG)]
    assert first == second == sorted(first)


def test_only_the_configured_depth_per_arm_is_considered():
    deep = ranked(*[f"c{n}" for n in range(FUSION["candidates_per_arm"] + 10)])
    result = reciprocal_rank_fusion({"dense": deep[: FUSION["candidates_per_arm"]]}, config=CONFIG)
    assert len(result) == FUSION["candidates_per_arm"]


# ─────────────────────────── the gate ───────────────────────────


async def test_below_threshold_confidence_returns_no_passages():
    """§7.5, and the reason it returns nothing rather than something.

    A weak passage handed to composition is a fee grounded in a chunk that
    merely mentions the topic — an invented figure that traces to a real
    source, which is the hardest kind to catch afterwards.
    """
    weak = [Candidate(chunk_id="weak", rank=19, relevance=0.2, content="barely related")]
    result = await fuse(query="q", dense=weak, sparse=None, config=CONFIG)
    assert result.passages == ()
    assert result.hits == ()
    assert result.gate_closed is True
    assert result.confidence < FUSION["min_score"]


async def test_a_strong_agreed_hit_passes_the_gate():
    result = await fuse(
        query="q", dense=ranked("top"), sparse=ranked("top"), config=CONFIG
    )
    assert result.gate_closed is False
    assert result.passages == ("body of top",)


async def test_the_threshold_comes_from_config():
    strict = {**CONFIG, "fusion": {**FUSION, "min_score": 1.01}}
    result = await fuse(query="q", dense=ranked("top"), sparse=ranked("top"), config=strict)
    assert result.gate_closed is True


async def test_no_results_at_all_is_a_closed_gate_not_a_crash():
    result = await fuse(query="q", dense=[], sparse=[], config=CONFIG)
    assert result.gate_closed is True
    assert result.confidence == 0.0


def test_confidence_survives_a_missing_arm():
    """A one-armed result must not be penalised by the arithmetic.

    Raw RRF scores depend on how many arms answered, so thresholding them
    would silently tighten the gate during an embedding outage — closing on
    every turn instead of degrading to lexical-only, which is the opposite of
    what §7.3 asks for.
    """
    one_arm = reciprocal_rank_fusion({"dense": ranked("x")}, config=CONFIG)
    two_arms = reciprocal_rank_fusion(
        {"dense": ranked("x"), "sparse": ranked("x")}, config=CONFIG
    )
    assert confidence_of(one_arm) == pytest.approx(confidence_of(two_arms))


def test_an_arm_with_no_relevance_signal_closes_the_gate():
    """Fails closed. No relevance is "no reason to believe this answers the
    question", which must not read the same as a confident hit."""
    silent = [Candidate(chunk_id="x", rank=1, content="body")]
    assert confidence_of(reciprocal_rank_fusion({"dense": silent}, config=CONFIG)) == 0.0


def test_relevance_is_the_best_across_arms_not_the_sum():
    """Two arms each 60% sure is not 120% sure."""
    fused = reciprocal_rank_fusion(
        {
            "dense": ranked("x", relevance=0.6),
            "sparse": ranked("x", relevance=0.7),
        },
        config=CONFIG,
    )
    assert fused[0].relevance == pytest.approx(0.7)


# ─────────────────────────── degradation ───────────────────────────


async def test_a_missing_dense_arm_degrades_rather_than_errors():
    """§7.3: embeddings down means Meilisearch only, which is correct
    behaviour rather than an outage the customer should feel."""
    result = await fuse(query="q", dense=None, sparse=ranked("lexical"), config=CONFIG)
    assert result.degraded_arms == ("dense",)
    assert result.passages == ("body of lexical",)


async def test_an_empty_arm_is_not_the_same_as_a_missing_one():
    """Conflating them makes an embedding outage look like a corpus with no
    match, and the two want opposite responses — retry versus hand off."""
    empty = await fuse(query="q", dense=[], sparse=ranked("x"), config=CONFIG)
    missing = await fuse(query="q", dense=None, sparse=ranked("x"), config=CONFIG)
    assert empty.degraded_arms == ()
    assert missing.degraded_arms == ("dense",)


async def test_both_arms_missing_is_reported_not_silently_empty():
    result = await fuse(query="q", dense=None, sparse=None, config=CONFIG)
    assert set(result.degraded_arms) == {"dense", "sparse"}
    assert result.gate_closed is True


# ─────────────────────────── rerank seam ───────────────────────────


async def test_the_reranker_is_consulted_when_supplied():
    class Reverse:
        async def rerank(self, *, query, hits, top_k):
            return list(reversed(list(hits)))[:top_k]

    result = await fuse(
        query="q",
        dense=ranked("a", "b"),
        sparse=ranked("a", "b"),
        reranker=Reverse(),
        config=CONFIG,
    )
    assert [hit.chunk_id for hit in result.hits] == ["b", "a"]


async def test_no_reranker_leaves_rrf_order_intact():
    """RRF order stands until a rerank vendor is configured. A placeholder
    that returned its input unchanged would appear in every trace as a
    reranker and improve nothing."""
    result = await fuse(
        query="q", dense=ranked("a", "b"), sparse=ranked("a", "b"), config=CONFIG
    )
    assert [hit.chunk_id for hit in result.hits] == ["a", "b"]


async def test_only_final_k_hits_survive():
    many = ranked(*[f"c{n}" for n in range(10)])
    result = await fuse(query="q", dense=many, sparse=many, config=CONFIG)
    assert len(result.hits) == FUSION["final_k"]


# ─────────────────────────── latency ───────────────────────────


async def test_fusion_latency_is_recorded_on_every_result(capsys):
    """Recorded from the first test rather than discovered at the end.

    Fusion sits inside §2.5's turn budget next to a model call that already
    measures 2.7-5.0 s. A retrieval layer that quietly costs a second is a
    turn budget blown before composition starts.
    """
    many = ranked(*[f"c{n}" for n in range(FUSION["candidates_per_arm"])])
    started = time.perf_counter()
    result = await fuse(query="q", dense=many, sparse=many, config=CONFIG)
    wall_ms = (time.perf_counter() - started) * 1000

    with capsys.disabled():
        print(
            f"\n  fusion (in-process, {FUSION['candidates_per_arm']} per arm): "
            f"{result.elapsed_ms:.2f} ms | target {FUSION['latency']['target_ms']} ms"
        )

    assert result.elapsed_ms > 0
    assert result.elapsed_ms <= wall_ms + 1
    assert result.elapsed_ms < FUSION["latency"]["ceiling_ms"]


async def test_latency_is_recorded_even_when_the_gate_closes():
    """A closed gate is still work done, and it is the path that runs on every
    unanswerable turn — the one most likely to be slow without anyone noticing."""
    result = await fuse(query="q", dense=[], sparse=[], config=CONFIG)
    assert result.elapsed_ms > 0


def test_a_fused_hit_carries_which_arms_found_it():
    """Attribution. A recall regression traced to one arm is a fix; a recall
    regression traced to "retrieval" is a week."""
    fused = reciprocal_rank_fusion({"dense": ranked("x")}, config=CONFIG)
    assert isinstance(fused[0], FusedHit)
    assert fused[0].arms == ("dense",)
