"""Chunking — design §7.1.

One rule governs everything here: **a sentence is never split.** Naive
character splitting leaves `2000` at the end of one chunk and
`جنيه رسوم التقديم` at the start of the next, and two things break at once.
Retrieval returns a chunk holding a bare number with nothing to identify it,
and the grounding gate reports an orphan figure in a reply that was correct.
The second is the expensive one: it makes the gate look wrong, and a gate that
looks wrong gets relaxed.

Chunks carry both forms of their text (§7.1). Retrieval matches on the
normalized one; a reply quotes the original. Only the original may reach a
customer.

Every value is config (§19). A chunk size compiled into source is a
retrieval-quality knob that needs a deploy to turn, and it is the first knob
anyone reaches for when answers come back subtly wrong.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from moc.arabic.normalize import normalize
from moc.config_store import load

_DEFAULTS = "retrieval/defaults"


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage.

    `ordinal` is position within the document, not a global id — it is what
    lets a reader reassemble the source and what makes a chunk id stable
    across a re-ingest of unchanged text.
    """

    ordinal: int
    content: str
    content_normalized: str

    @property
    def char_count(self) -> int:
        return len(self.content)


def embedding_text(*, title: str | None, content: str) -> str:
    """The text a chunk is embedded as: its title, then its body.

    One function because there must be exactly one answer. The dense arm can
    only retrieve what its vector describes, and a chunk embedded on the wrong
    text fails silently — it simply never comes back, at any k, for any query,
    and looks like a corpus that does not cover the topic.

    edu-0002 is that failure in full. `sinai_fee_application_initial_ar` is
    the title "ما قيمة رسوم التقديم المبدئي؟" over the content "2000 جنيه
    مصري": fourteen characters naming neither the fee nor the word رسوم.
    Embedded on content alone its vector says "a sum of money in EGP", so
    "رسوم التقديم كام؟" could not reach it — while the *refund* chunk, whose
    body repeats رسوم التقديم, came back first in both arms. A Q&A corpus
    keeps its subject in the title, and a passage that drops the question
    keeps only the answer to a question nobody can now find.

    The original text, never the normalized form: normalization exists to
    match spelling variants in the lexical arm, and folding ة to ه before
    embedding just hands the model a misspelling.
    """
    heading = (title or "").strip()
    if not heading or content.lstrip().startswith(heading):
        # A heading duplicated into its own body is one passage, not two.
        # Repeating it raises its term frequency in the embedded text and
        # tilts the vector toward the heading over the substance.
        return content
    return f"{heading}\n{content}"


@lru_cache(maxsize=8)
def _splitter(terminators: str) -> re.Pattern[str]:
    """Split *after* a terminator, but only when whitespace or the end follows.

    That condition is the whole difference between a sentence splitter and a
    corpus-destroying regex: `6.5` keeps its decimal, `su.edu.eg` keeps its
    dots, and `للوحدة.` still ends a sentence.
    """
    return re.compile(f"(?<=[{re.escape(terminators)}])(?=\\s|$)")


def split_sentences(text: str, terminators: str) -> list[str]:
    return [part.strip() for part in _splitter(terminators).split(text) if part.strip()]


def chunk_text(text: str, *, config: dict[str, Any] | None = None) -> list[Chunk]:
    """Split `text` into overlapping chunks that respect sentence boundaries.

    Greedy packing: sentences accumulate until the next one would exceed the
    target, then a new chunk starts carrying the tail of the previous one as
    overlap. A sentence longer than the target becomes its own chunk rather
    than being cut — an oversized chunk is a retrieval-quality problem the
    reranker can absorb, a severed one is a correctness problem that surfaces
    at answer time.
    """
    settings = (config or load(_DEFAULTS))["chunking"]
    target = int(settings["target_tokens"] * settings["chars_per_token"])
    overlap_budget = int(target * settings["overlap_ratio"])

    sentences = split_sentences(text, settings["sentence_terminators"])
    if not sentences:
        # Whitespace only. Returning one empty chunk here would embed a point
        # that matches everything weakly — unrelated results at low scores,
        # which reads as a ranking problem rather than an ingest bug.
        return []

    chunks: list[list[str]] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        if current and length + len(sentence) + 1 > target:
            chunks.append(current)
            current = _overlap_tail(current, overlap_budget)
            length = sum(len(s) + 1 for s in current)
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(current)

    return [
        Chunk(
            ordinal=ordinal,
            content=(joined := " ".join(sentences_in_chunk)),
            content_normalized=normalize(joined),
        )
        for ordinal, sentences_in_chunk in enumerate(chunks)
    ]


def _overlap_tail(sentences: list[str], budget: int) -> list[str]:
    """The trailing sentences a new chunk repeats from the previous one.

    §7.1's 15%. Without it a question landing on a boundary sees half its
    context, and the chunk that should have answered it ranks below one that
    merely mentions the topic.

    At least one sentence whenever the budget is non-zero, even if that
    sentence exceeds the budget: an overlap of nothing is not an overlap, and
    silently producing one when the config asks for 15% is the kind of gap
    that is only found by wondering why boundary questions answer badly.
    """
    if budget <= 0:
        return []
    tail: list[str] = []
    used = 0
    for sentence in reversed(sentences):
        if tail and used + len(sentence) > budget:
            break
        tail.insert(0, sentence)
        used += len(sentence) + 1
    return tail


__all__ = ["Chunk", "chunk_text", "embedding_text", "split_sentences"]
