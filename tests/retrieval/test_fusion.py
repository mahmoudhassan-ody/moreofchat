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
import uuid
from types import SimpleNamespace

import pytest

from moc.config_store import load
from moc.retrieval.fusion import (
    Candidate,
    FusedHit,
    FusionRetriever,
    confidence_of,
    fuse,
    reciprocal_rank_fusion,
)

CONFIG = load("retrieval/lexical")
FUSION = CONFIG["fusion"]


class StubLexical:
    """A Meilisearch repository that returns fixed ranked hits."""

    def __init__(self, hits):
        self._hits = hits

    async def search(self, *, tenant_id, vertical, query, limit):
        return self._hits[:limit]


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


async def test_the_gate_closes_when_there_is_nothing_to_ground_on():
    """§7.5, narrowed to what the gate can actually decide.

    Composition must not be handed an empty passage set: with nothing to
    ground on, every figure in the reply would be the model's own. That is a
    question the gate can answer from what it holds.
    """
    empty = [Candidate(chunk_id="a", rank=1, relevance=0.9, content="   ")]
    result = await fuse(query="q", dense=empty, sparse=None, config=CONFIG)
    assert result.passages == ()
    assert result.gate_closed is True


async def test_a_low_similarity_score_does_not_close_the_gate():
    """The measured reason `min_score` sits at the floor.

    0.218 is edu-0015's real cosine — "fe manh fe kantara?" retrieving
    `sinai_discount_tuition_ar`, correctly, as the top hit. The question and
    the answer share no letters; the synonym map does the work and cosine
    cannot see it. Any threshold that would close on 0.218 closes on the
    project's own acceptance case.
    """
    weak = [Candidate(chunk_id="right", rank=1, relevance=0.218, content="خصم 25%")]
    result = await fuse(query="fe manh fe kantara?", dense=weak, sparse=None, config=CONFIG)
    assert result.gate_closed is False
    assert result.passages == ("خصم 25%",)


def test_min_score_is_at_the_floor_and_says_why():
    """§19. Kept as a value so it stays visible and tunable, set to the floor
    because measurement showed no cutoff separates the two populations."""
    assert FUSION["min_score"] == 0.0


async def test_a_strong_agreed_hit_passes_the_gate():
    result = await fuse(
        query="q", dense=ranked("top"), sparse=ranked("top"), config=CONFIG
    )
    assert result.gate_closed is False
    assert result.passages == ("body of top",)


async def test_the_threshold_still_comes_from_config_when_a_tenant_raises_it():
    """At the floor it never fires, but the mechanism stays wired: a tenant
    whose corpus does separate cleanly may raise it, and that must not need a
    deploy."""
    strict = {**CONFIG, "fusion": {**FUSION, "min_score": 1.01}}
    result = await fuse(query="q", dense=ranked("top"), sparse=ranked("top"), config=strict)
    assert result.gate_closed is True


async def test_no_results_at_all_is_a_closed_gate_not_a_crash():
    result = await fuse(query="q", dense=[], sparse=[], config=CONFIG)
    assert result.gate_closed is True
    assert result.confidence is None


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


def test_an_arm_with_no_relevance_signal_reports_no_confidence():
    """Deliberate contract change, recorded here because it reverses one.

    This previously read "no relevance closes the gate", on the reasoning that
    silence should fail closed. Measurement showed the opposite risk was the
    live one: the lexical arm was *not* silent, it was supplying `1.0 / rank`,
    so the gate read certainty for 7 of 17 cases and closed on none of them.

    Silence must therefore be representable. It now reports `None` — no
    instrument read this — which the caller may not threshold. Failing closed
    on it would turn an embedding outage into a total outage, which §7.3
    explicitly rejects.
    """
    silent = [Candidate(chunk_id="x", rank=1, content="body")]
    assert confidence_of(reciprocal_rank_fusion({"dense": silent}, config=CONFIG)) is None


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


# ───────────── confidence comes from a calibrated arm, or not at all ─────────────


def test_a_top_ranked_lexical_hit_is_not_read_as_certainty():
    """The defect this section exists for.

    The lexical arm supplied `1.0 / rank` as its relevance, so any chunk
    Meilisearch ranked first scored exactly 1.0 — perfect confidence, derived
    from nothing but position in a list. Measured across the 17 education
    cases, 7 read 1.000, including three the suite requires *not* be answered.
    A gate whose input is 1.0 for the cases it must close is decorative.
    """
    sparse_first = reciprocal_rank_fusion(
        {"sparse": [Candidate(chunk_id="x", rank=1)]}, config=CONFIG
    )
    assert confidence_of(sparse_first) is None


def test_confidence_is_the_dense_arms_cosine():
    fused = reciprocal_rank_fusion(
        {"dense": [Candidate(chunk_id="x", rank=1, relevance=0.42)]}, config=CONFIG
    )
    assert confidence_of(fused) == pytest.approx(0.42)


def test_confidence_is_absent_rather_than_zero_when_no_arm_is_calibrated():
    """Absent and zero are different claims.

    Zero says "we looked and found nothing similar". Absent says "nothing
    here can answer that question" — which is the honest report during an
    embedding outage, and the two must not be conflated by a gate that
    treats both as failure.
    """
    fused = reciprocal_rank_fusion(
        {"sparse": [Candidate(chunk_id="x", rank=1)]}, config=CONFIG
    )
    assert fused[0].relevance is None
    assert confidence_of(fused) is None
    assert confidence_of(()) is None


def test_an_uncalibrated_arm_does_not_dilute_a_calibrated_one():
    """Best across arms, ignoring arms that have no opinion to give."""
    fused = reciprocal_rank_fusion(
        {
            "dense": [Candidate(chunk_id="x", rank=2, relevance=0.31)],
            "sparse": [Candidate(chunk_id="x", rank=1)],
        },
        config=CONFIG,
    )
    assert fused[0].relevance == pytest.approx(0.31)


async def test_an_embedding_outage_reports_no_confidence_and_still_answers():
    """§7.3: one arm down is a degraded turn, not a refusal.

    With no dense arm there is no calibrated score, so the gate has nothing
    to read. It must hand the passages on rather than close — the alternative
    is that an embedding outage silently becomes a total outage.
    """
    result = await fuse(
        query="رسوم التقديم",
        dense=None,
        sparse=[Candidate(chunk_id="x", rank=1, content="2000 جنيه")],
        config=CONFIG,
    )
    assert result.confidence is None
    assert result.degraded_arms == ("dense",)
    assert result.passages, "a degraded arm must not empty the result"


async def test_the_retriever_does_not_invent_relevance_for_the_lexical_arm():
    """The stand-in removed at its source, not just where it was read.

    `FusionRetriever` built each sparse candidate with `relevance = 1.0/rank`.
    Leaving that in place and only hardening `confidence_of` would fix
    nothing: the fake number is manufactured here, and every consumer
    downstream would keep receiving it.
    """
    retriever = FusionRetriever(
        lexical=StubLexical(
            [
                SimpleNamespace(chunk_id="a", rank=1, content="first"),
                SimpleNamespace(chunk_id="b", rank=2, content="second"),
            ]
        ),
        tenant_id=uuid.uuid4(),
        vertical="education",
        config=CONFIG,
    )
    result = await retriever.search(query="رسوم التقديم")

    assert result.confidence is None, "a rank is not a score"
    assert all(hit.relevance is None for hit in result.hits)
    assert result.passages, "and the turn still gets its passages"
