"""Content-addressed embedding cache — design §7.3, §19.

The education fixture re-embedded all 102 chunks on every pytest invocation.
Cents at this size, and the wrong shape: the chunks do not change between runs,
and at three tenants with real corpora the ingest is the bill rather than a
rounding error.

**Keyed per text, not per corpus.** That is the property that matters. A corpus
with one edited chunk should cost one embedding, not a hundred and two — and a
per-corpus key turns any edit into a full re-ingest, which is the behaviour
this replaces wearing a hash.

**The model and the dimension count are part of the key.** Vectors from two
models are not comparable, which is §7.3's whole reason for pinning embeddings
to one provider and one width; Matryoshka truncation means the same model at
1024 and at 256 returns differently-normalized vectors. Serving one for the
other would collapse retrieval quality without erroring, the worst failure mode
available.

**It degrades to the thing it optimises.** A half-written file from an
interrupted run re-embeds rather than raising. A cache that can fail a run it
was added to speed up is a cache nobody keeps.

No tenant in the key, deliberately. The vector for a piece of text is the same
for everyone who holds that text, and tenant isolation is enforced where it
belongs — the Qdrant payload filter and the RLS policy. Keying by tenant here
would multiply the cost by the number of tenants sharing a corpus, which is
exactly the case this exists for.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from moc.llm.base import Embedding, Vector

#: Two levels of fan-out on the digest. A single directory of a hundred
#: thousand files is slow to list and unpleasant to inspect by hand.
_FANOUT = 2
#: §7.3 pins embeddings to one provider by design — there is no failover, so a
#: cached result can name it without asking who answered.
_PROVIDER = "openai"


class Embeds(Protocol):
    """The one-method shape fusion and the fixtures already pass around."""

    async def embed(self, *, texts: Sequence[str]) -> Any: ...


class EmbeddingCache:
    def __init__(self, *, root: Path, model: str, dimensions: int) -> None:
        self._root = Path(root)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, embedder: Embeds, texts: Sequence[str]) -> Embedding:
        """Vectors for `texts`, in order, embedding only what is not cached.

        Duplicates within one batch are embedded once and returned twice: a
        corpus holding the same sentence in two chunks is one call and two
        rows.

        **`input_tokens` is what was actually billed, not what the corpus would
        cost.** The point of metering ingest is to answer "what does onboarding
        a tenant cost"; a cache hit that reported the tokens it avoided would
        give the same answer on the hundredth run as on the first, which
        describes the corpus rather than the bill. A full hit reports zero and
        writes no ledger row.
        """
        cached = {text: self._read(text) for text in dict.fromkeys(texts)}
        missing = [text for text, vector in cached.items() if vector is None]
        billed, model = 0, self._model
        if missing:
            fresh = await embedder.embed(texts=missing)
            billed, model = fresh.input_tokens, fresh.model
            for text, vector in zip(missing, fresh.vectors, strict=True):
                cached[text] = vector
                self._write(text, vector)
        return Embedding(
            # Rebuilt from `texts` rather than from the batch. A cache that
            # returns correct vectors attached to the wrong chunks degrades
            # retrieval and raises nothing, which is worse than no cache.
            vectors=[cached[text] for text in texts],
            provider=_PROVIDER,
            model=model,
            input_tokens=billed,
        )

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(
            "\0".join((self._model, str(self._dimensions), text)).encode("utf-8")
        ).hexdigest()
        return self._root / digest[:_FANOUT] / digest[_FANOUT:_FANOUT * 2] / f"{digest}.json"

    def _read(self, text: str) -> Vector | None:
        path = self._path(text)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            vector = entry["vector"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            # Anything unreadable is a miss. See the module docstring: an
            # interrupted write must cost one embedding, never a failed run.
            return None
        return vector if len(vector) == self._dimensions else None

    def _write(self, text: str, vector: Vector) -> None:
        path = self._path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole to a sibling and moved, so a reader never sees half a
        # file. Same directory, because rename is only atomic within one.
        temporary = path.with_suffix(".partial")
        temporary.write_text(
            json.dumps(
                {
                    "model": self._model,
                    "dimensions": self._dimensions,
                    # The text itself, so a cache directory is readable by a
                    # person debugging a vector rather than a pile of hashes.
                    "text": text,
                    "vector": vector,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


__all__ = ["EmbeddingCache"]
