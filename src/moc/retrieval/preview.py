"""What the chunker did, before anything is written — demo plan Task 31.

**Chunking is where a corpus quietly breaks.** Nothing raises: the document
ingests, the chunks embed, retrieval returns something, and months later a
figure surfaces with no sentence around it and §19.3's gate discards a reply
that was otherwise correct. By then the corpus is large and nobody remembers
which upload did it.

So the knowledge screen shows the chunk count *and* the text, before the tenant
confirms and before a single embedding is bought. A count on its own says the
chunker ran; only the text says whether it ran sensibly, and only somebody
looking at their own corpus can tell.

**This module writes nothing and embeds nothing, structurally.** It takes text
and returns a description of it — no session parameter, no embedder, nothing to
commit. A test asserts the source does not so much as mention them, because a
preview that ingested would make "cancel" a lie and would bill a tenant for a
corpus they rejected.

**The warnings reuse `extract_quantities`, the detector §19.3's figure gate
already uses.** That is the design rather than an economy: what this flags is
exactly what the runtime gate would later refuse to label. A second heuristic
would warn about figures the gate tolerates and stay silent on ones it does
not, which is worse than no warning.

The failure this is really aimed at is the one waiting in the real corpus:
**a fee table exported to plain text has no full stop in it.** The sentence
splitter finds nothing to split on, the whole document becomes one enormous
chunk, and every query retrieves all of it — retrieval that looks like it is
working and discriminates nothing.
"""

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any

from moc.arabic.numerals import QuantityKind, extract_quantities
from moc.retrieval.chunker import Chunk, chunk_text

_DEFAULTS = "retrieval/defaults"

#: How many chunks travel to the browser. The count is the whole truth; this is
#: a window on it, and the response says which is which — a thousand-chunk
#: corpus must not go through a JSON body to be looked at.
SAMPLE_SIZE = 5

#: A chunk beyond this multiple of the configured target is oversized. Two,
#: not 1.1: the chunker packs greedily and a chunk that overshoots slightly is
#: ordinary, while one at twice the target contains a "sentence" that is not a
#: sentence.
_OVERSIZE_FACTOR = 2

#: Below this many quantities a chunk is prose that happens to hold a number,
#: not a column that lost its header.
_BARE_FIGURE_FLOOR = 3

#: **Measured, not chosen.** The first version of this check flagged any chunk
#: holding three unlabelled figures, and it fired on 16 of the 102 chunks in
#: the corpus this product already ships — document-requirement lists like
#: "عدد 6 صور شخصية" and "نموذج 2 جند", where the number is labelled by the
#: words around it rather than by a currency marker. A warning with a 16%
#: false-positive rate on our own corpus is a warning the first tenant learns
#: to ignore.
#:
#: What separates a severed column from prose is not how many figures there
#: are but how much text is around them. Over those 102 chunks the highest
#: unlabelled-figure-per-word density is 0.333 (the fee chunks, which are
#: terse by nature); a column of values with its header row lost measures
#: 0.875. 0.6 sits between, at roughly twice the highest reading real Arabic
#: produced here and well under a true column.
#:
#: Re-measure this against a real tenant corpus when one lands. It is a
#: property of the text, not a constant.
_BARE_DENSITY = 0.6


class Warning(enum.StrEnum):
    """Named rather than free text so the console can translate them.

    A reason string is written once, in English, by whoever added the check.
    The tenant reading it is an admissions officer in Egypt.
    """

    empty = "empty"
    no_sentence_boundaries = "no_sentence_boundaries"
    oversized = "oversized"
    bare_figures = "bare_figures"


@dataclass(frozen=True)
class Finding:
    """One problem, and which chunk has it.

    `ordinal` rather than a count: a warning that says "3 oversized chunks"
    sends the reader back into their corpus to find them.
    """

    name: Warning
    ordinal: int
    reason: str


@dataclass(frozen=True)
class DocumentPreview:
    chunk_count: int
    chunks: list[Chunk] = field(default_factory=list)
    #: The first `SAMPLE_SIZE` chunks, verbatim. Not truncated and not
    #: summarised — the question the reader is answering is whether the text
    #: survived extraction intact.
    sample: list[Chunk] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    #: Content-addressed, over the title and the body. Lets "unchanged" be
    #: answered before an embedding is bought, and the title is in it because
    #: `embedding_text` prepends the title to every chunk — a re-titled
    #: document embeds differently and is not unchanged.
    content_hash: str = ""


def preview_document(
    *, text: str, title: str | None = None, config: dict[str, Any] | None = None
) -> DocumentPreview:
    """Chunk `text` and describe the result. Nothing is stored."""
    settings = _settings(config)
    chunks = chunk_text(text, config=settings)
    return DocumentPreview(
        chunk_count=len(chunks),
        chunks=chunks,
        sample=chunks[:SAMPLE_SIZE],
        warnings=_findings(chunks, settings),
        content_hash=_hash(title, text),
    )


def _settings(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return config
    from moc.config_store import load

    return load(_DEFAULTS)


def _hash(title: str | None, text: str) -> str:
    digest = hashlib.sha256()
    digest.update((title or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _findings(chunks: list[Chunk], settings: dict[str, Any]) -> list[Finding]:
    if not chunks:
        return [
            Finding(
                name=Warning.empty,
                ordinal=0,
                reason=(
                    "No text was found. A PDF that extracted to whitespace "
                    "ingests perfectly and leaves a knowledge base that knows "
                    "nothing about the document."
                ),
            )
        ]

    chunking = settings["chunking"]
    target = int(chunking["target_tokens"] * chunking["chars_per_token"])
    findings: list[Finding] = []

    # The loud one, and it is a property of the document rather than of any
    # chunk: one chunk that is far over target means the splitter found no
    # terminator to work with, not that one sentence ran long.
    if len(chunks) == 1 and len(chunks[0].content) > target * _OVERSIZE_FACTOR:
        findings.append(
            Finding(
                name=Warning.no_sentence_boundaries,
                ordinal=0,
                reason=(
                    "The whole document became one chunk: no sentence terminator "
                    f"was found in {len(chunks[0].content)} characters. A table "
                    "exported to plain text does this, and every search then "
                    "returns the entire document."
                ),
            )
        )

    for chunk in chunks:
        if len(chunk.content) > target * _OVERSIZE_FACTOR:
            findings.append(
                Finding(
                    name=Warning.oversized,
                    ordinal=chunk.ordinal,
                    reason=(
                        f"{len(chunk.content)} characters against a {target}-character "
                        "target. One 'sentence' here is not a sentence, and a chunk "
                        "this size ranks against everything."
                    ),
                )
            )
        bare = [
            quantity
            for quantity in extract_quantities(chunk.content)
            if quantity.kind is QuantityKind.bare
        ]
        words = len(chunk.content.split())
        density = len(bare) / words if words else 0.0
        if len(bare) >= _BARE_FIGURE_FLOOR and density >= _BARE_DENSITY:
            findings.append(
                Finding(
                    name=Warning.bare_figures,
                    ordinal=chunk.ordinal,
                    reason=(
                        f"{len(bare)} of {words} tokens here are figures with no "
                        "currency, percent or unit near them — a column whose header "
                        "row did not come with it. A reply quoting one states a "
                        "number the figure gate cannot label."
                    ),
                )
            )
    return findings


__all__ = ["SAMPLE_SIZE", "DocumentPreview", "Finding", "Warning", "preview_document"]
