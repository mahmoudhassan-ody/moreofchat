"""Chunking — design §7.1.

The rule that matters is that a sentence is never split. Naive character
splitting puts `2000` at the end of one chunk and `جنيه رسوم التقديم` at the
start of the next, and then two things go wrong at once: retrieval returns a
chunk containing a bare number with nothing to identify it, and the grounding
check reports an orphan figure in a reply that was actually correct. The
second is worse, because it makes the gate look wrong and gets it relaxed.

Everything numeric here is config (§19). A chunk size baked into source is a
retrieval-quality knob that needs a deploy to turn.
"""

import pytest

from moc.config_store import load
from moc.retrieval.chunker import Chunk, chunk_text

CONFIG = load("retrieval/defaults")
CHUNKING = CONFIG["chunking"]

FEE = "رسوم التقديم 2000 جنيه تُدفع مرة واحدة."
HOUSING = "السكن الجامعي متاح في فرعي العريش والقنطرة."


def target_chars() -> int:
    return int(CHUNKING["target_tokens"] * CHUNKING["chars_per_token"])


# ─────────────────────────── boundaries ───────────────────────────


def test_respects_arabic_sentence_boundaries():
    """A figure and the words naming it end up in the same chunk.

    Built to force a boundary exactly where the number sits: the filler ahead
    of it fills the chunk, so a size-driven splitter would cut mid-sentence.
    """
    filler = "الجامعة تأسست في سيناء وتضم عدة كليات. " * 40
    chunks = chunk_text(filler + FEE)

    holding = [c for c in chunks if "2000" in c.content]
    assert holding, "the figure vanished"
    for held in holding:
        assert "جنيه" in held.content, "the number was severed from its unit"
        assert "رسوم التقديم" in held.content, "the number was severed from its label"


def test_no_chunk_ends_mid_sentence():
    text = " ".join([FEE, HOUSING] * 60)
    for produced in chunk_text(text):
        assert produced.content.strip()[-1] in CHUNKING["sentence_terminators"]


def test_a_decimal_is_not_treated_as_a_sentence_end():
    """`6.5` and `su.edu.eg` must survive. A terminator ends a sentence only
    when whitespace or the end of the text follows it."""
    chunks = chunk_text("السعر 6.5 مليون جنيه للوحدة. راسلنا على oss.a@su.edu.eg اليوم.")
    assert len(chunks) == 1
    assert "6.5" in chunks[0].content
    assert "oss.a@su.edu.eg" in chunks[0].content


def test_a_sentence_longer_than_the_target_becomes_its_own_chunk():
    """Oversized beats severed.

    An oversized chunk is a retrieval-quality problem the reranker can absorb.
    A severed one is a correctness problem, and only one of those is
    recoverable at answer time.
    """
    monster = "كلمة " * (target_chars() // 2)
    chunks = chunk_text(monster.strip() + ".")
    assert len(chunks) == 1
    assert len(chunks[0].content) > target_chars()


# ─────────────────────────── size and overlap ───────────────────────────


def test_chunk_size_and_overlap_come_from_config():
    """Change the config, change the output — asserted, not assumed."""
    text = " ".join([FEE, HOUSING] * 80)
    wide = chunk_text(text, config={**CONFIG, "chunking": {**CHUNKING, "target_tokens": 2000}})
    narrow = chunk_text(text, config={**CONFIG, "chunking": {**CHUNKING, "target_tokens": 100}})
    assert len(narrow) > len(wide)


def test_overlap_repeats_trailing_sentences_from_the_previous_chunk():
    """§7.1's 15%. A question landing on a boundary otherwise sees half its
    context, and the arm that should have retrieved it ranks it below a chunk
    that merely mentions the topic."""
    text = " ".join(f"جملة رقم {n} عن الرسوم والمصاريف الجامعية." for n in range(120))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        tail = set(earlier.content.split(".")) - {""}
        head = set(later.content.split(".")) - {""}
        assert tail & head, "consecutive chunks share no sentence — overlap is not applied"


def test_zero_overlap_is_honoured_rather_than_defaulted():
    text = " ".join(f"جملة رقم {n} عن الرسوم." for n in range(120))
    chunks = chunk_text(
        text, config={**CONFIG, "chunking": {**CHUNKING, "overlap_ratio": 0.0}}
    )
    joined = "".join(c.content for c in chunks)
    assert len(joined) <= len(text) + len(chunks), "overlap appeared with the ratio at zero"


# ─────────────────────────── degenerate input ───────────────────────────


def test_a_short_document_yields_one_chunk_not_an_empty_tail():
    """The off-by-one that ships an empty vector.

    An empty chunk embeds to a point that matches everything weakly, which
    surfaces as unrelated results at low scores rather than as an error.
    """
    chunks = chunk_text(FEE)
    assert len(chunks) == 1
    assert chunks[0].content == FEE


@pytest.mark.parametrize("empty", ["", "   ", "\n\n"])
def test_whitespace_only_input_yields_no_chunks(empty):
    assert chunk_text(empty) == []


def test_ordinals_are_contiguous_from_zero():
    chunks = chunk_text(" ".join([FEE, HOUSING] * 60))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


# ─────────────────────────── the two forms ───────────────────────────


def test_preserves_the_original_text_alongside_the_normalized_form():
    """§7.1: retrieval matches on normalized, the reply cites the original.

    Only one of the two may ever reach a customer. A reply quoting the
    normalized form has silently rewritten the tenant's own wording — and for
    a fee, rewritten the digits a student will compare against an invoice.
    """
    original = "الرسوم ٢٥٠٠٠ جنيه"
    produced = chunk_text(original)[0]
    assert produced.content == original
    assert produced.content_normalized != original
    assert "25000" in produced.content_normalized
    assert "٢٥٠٠٠" in produced.content


def test_normalization_folds_spelling_variation():
    """`القنطره` and `القنطرة` must reach the same normalized form, or the
    lexical arm misses on the spelling customers actually type."""
    with_ta = chunk_text("السكن في القنطرة متاح.")[0]
    without = chunk_text("السكن في القنطره متاح.")[0]
    assert with_ta.content_normalized == without.content_normalized
    assert with_ta.content != without.content


def test_the_chunk_carries_what_the_payload_needs():
    produced = chunk_text(FEE)[0]
    assert isinstance(produced, Chunk)
    assert produced.char_count == len(FEE)
