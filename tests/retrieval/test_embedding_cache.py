"""Content-addressed embedding cache — the corpus that was paid for every time.

The live education fixture re-embedded all 102 chunks on every pytest
invocation. Cents at this size, and the wrong shape: the chunks do not change
between runs, and at three tenants with real corpora it is the ingest bill
rather than a rounding error.

Keyed per text rather than per corpus, which is the property that matters. A
corpus with one edited chunk should cost one embedding, not a hundred and two
— and a per-corpus key turns any edit into a full re-ingest.
"""

import json

import pytest

from moc.retrieval.embedding_cache import EmbeddingCache


class CountingEmbedder:
    """Records what it was actually asked to embed.

    The count is the whole assertion: a cache that returns correct vectors
    while still calling the provider is a cache that costs money and looks
    like it works.
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, *, texts):
        self.batches.append(list(texts))
        from moc.llm.base import Embedding

        return Embedding(
            vectors=[[float(len(t))] * 4 for t in texts],
            provider="openai",
            model="text-embedding-3-large",
            input_tokens=sum(len(t) for t in texts),
        )


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(root=tmp_path, model="text-embedding-3-large", dimensions=4)


async def test_a_second_run_over_the_same_corpus_embeds_nothing(cache):
    embedder = CountingEmbedder()
    texts = ["الرسوم 2000", "السكن متاح", "الخصم 30%"]

    first = await cache.embed(embedder, texts)
    second = await cache.embed(embedder, texts)

    assert first.vectors == second.vectors
    assert len(embedder.batches) == 1, "the corpus was paid for twice"


async def test_one_changed_chunk_costs_one_embedding(cache):
    """The reason the key is per text. A per-corpus key makes any edit a full
    re-ingest, which is the behaviour this replaces wearing a hash."""
    embedder = CountingEmbedder()
    await cache.embed(embedder, ["a", "b", "c"])
    await cache.embed(embedder, ["a", "b", "CHANGED"])

    assert embedder.batches[1] == ["CHANGED"], "the unchanged chunks were re-embedded"


async def test_vectors_come_back_in_the_order_they_were_asked_for(cache):
    """A cache that returns the right vectors attached to the wrong chunks is
    worse than no cache: retrieval degrades and nothing raises."""
    embedder = CountingEmbedder()
    await cache.embed(embedder, ["bb"])
    mixed = await cache.embed(embedder, ["aaa", "bb", "c"])

    assert [v[0] for v in mixed.vectors] == [3.0, 2.0, 1.0]


async def test_a_different_model_is_a_different_cache(cache, tmp_path):
    """Vectors from two models are not comparable — §7.3's whole reason for
    pinning embeddings to one provider. Serving one model's vector for
    another's request would collapse retrieval quality silently."""
    embedder = CountingEmbedder()
    await cache.embed(embedder, ["x"])

    other = EmbeddingCache(root=tmp_path, model="some-other-model", dimensions=4)
    await other.embed(embedder, ["x"])

    assert len(embedder.batches) == 2


async def test_a_different_dimension_is_a_different_cache(cache, tmp_path):
    """Matryoshka truncation means the same model at 1024 and at 256 returns
    differently-normalized vectors."""
    embedder = CountingEmbedder()
    await cache.embed(embedder, ["x"])

    narrower = EmbeddingCache(root=tmp_path, model="text-embedding-3-large", dimensions=2)
    await narrower.embed(embedder, ["x"])

    assert len(embedder.batches) == 2


async def test_an_empty_request_calls_nothing(cache):
    embedder = CountingEmbedder()
    assert (await cache.embed(embedder, [])).vectors == []
    assert embedder.batches == []


async def test_a_corrupt_entry_is_reembedded_rather_than_raising(cache, tmp_path):
    """A half-written file from an interrupted run must not make the next one
    fail. The cache is an optimisation and has to degrade to the thing it
    optimises."""
    embedder = CountingEmbedder()
    await cache.embed(embedder, ["x"])
    for path in tmp_path.rglob("*.json"):
        path.write_text("{not json", encoding="utf-8")

    again = await cache.embed(embedder, ["x"])
    assert again.vectors == [[1.0] * 4]
    assert len(embedder.batches) == 2


async def test_the_stored_entry_records_what_produced_it(cache, tmp_path):
    """So a cache directory is readable by a person debugging why a vector
    looks wrong, rather than a pile of hashes."""
    embedder = CountingEmbedder()
    await cache.embed(embedder, ["الرسوم"])
    entry = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert entry["model"] == "text-embedding-3-large"
    assert entry["dimensions"] == 4
    assert len(entry["vector"]) == 4


async def test_the_batch_preserves_duplicates_without_paying_twice(cache):
    """A corpus with two identical chunks is one embedding and two rows."""
    embedder = CountingEmbedder()
    result = await cache.embed(embedder, ["same", "same"])
    assert result.vectors == [[4.0] * 4, [4.0] * 4]
    assert embedder.batches == [["same"]]


async def test_the_result_reports_only_what_was_actually_billed(cache):
    """A cached corpus costs nothing, and the ledger has to say so.

    The point of metering ingest is to answer "what does onboarding a tenant
    cost". If a cache hit reported the tokens it would have spent, the answer
    would be the same on the hundredth run as on the first — a number that
    describes the corpus rather than the bill.
    """
    embedder = CountingEmbedder()
    first = await cache.embed(embedder, ["الرسوم 2000", "السكن متاح"])
    assert first.input_tokens > 0
    assert first.model == "text-embedding-3-large"

    again = await cache.embed(embedder, ["الرسوم 2000", "السكن متاح"])
    assert again.input_tokens == 0, "a cache hit was reported as spend"
    assert again.vectors == first.vectors


async def test_a_partial_hit_reports_only_the_miss(cache):
    """One edited chunk in a hundred is one chunk's worth of spend."""
    embedder = CountingEmbedder()
    await cache.embed(embedder, ["a", "b"])
    partial = await cache.embed(embedder, ["a", "b", "cccc"])
    assert partial.input_tokens == len("cccc")
