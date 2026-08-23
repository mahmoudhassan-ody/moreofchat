"""Chunk preview — demo plan Task 31.

**Chunking is where a corpus quietly breaks.** Nothing errors: the document
ingests, the chunks embed, retrieval returns something, and months later a
figure surfaces with no sentence around it and the grounding gate discards a
reply that was otherwise correct. By then the corpus is large and nobody
remembers which upload did it.

So the screen shows what the chunker produced *before* anything is written or
embedded, and it says what is wrong with it in the tenant's terms.

**The warnings reuse `extract_quantities`, the same detector §19.3's figure
gate uses.** That is the point rather than an economy: what the preview flags
is exactly what the runtime gate would later refuse. A separate heuristic here
would warn about things the gate tolerates and stay silent on things it does
not, which is worse than no warning at all.

The failure this is really aimed at: **Sinai's real KB will not look like the
fixture.** A fee table exported to text has no full stops in it, so the
sentence splitter finds nothing to split on, the whole document becomes one
oversized chunk, and every query retrieves the entire document.
"""

from moc.retrieval.preview import Warning as PreviewWarning
from moc.retrieval.preview import preview_document

# Two ordinary Arabic sentences: labelled figures, real terminators.
CLEAN = (
    "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026. "
    "رسوم التقديم 2000 جنيه وهي غير مستردة. "
    "الحد الأدنى للقبول في كلية الهندسة 64 بالمئة للثانوية العامة."
)

# What a fee table looks like after it has been exported to plain text: no
# terminator anywhere, so the splitter has nothing to work with.
TABLE = " ".join(f"الهندسة {1000 + index} 2026" for index in range(200))

# A column of values whose header row did not come with it. This is the shape
# `bare_figures` is about, and it is NOT the same shape as the table above —
# that one's problem is the missing sentence boundaries, and it is reported as
# that. Measured at 0.875 unlabelled figures per token against 0.333 for the
# most figure-dense real Arabic in the shipped corpus.
COLUMN = " ".join(str(1000 + index * 50) for index in range(40))


def names(preview) -> set[str]:
    return {warning.name for warning in preview.warnings}


def test_a_clean_document_previews_without_warnings():
    preview = preview_document(text=CLEAN, title="الرسوم")

    assert preview.chunk_count >= 1
    assert preview.warnings == []


def test_the_screen_shows_chunk_count_and_a_sample_before_confirming():
    """The whole reason this module exists.

    Count *and* sample, not one or the other. A count says the chunker ran; the
    text says whether it ran sensibly, and only a person looking at their own
    corpus can tell — "رسوم الساعة المعتمدة" ending a chunk and "1400 جنيه"
    opening the next looks fine in a count and is a broken corpus.
    """
    preview = preview_document(text=CLEAN, title="الرسوم")

    assert preview.chunk_count == len(preview.chunks)
    assert preview.sample, "a count with no text is a number nobody can act on"
    assert all(chunk.content.strip() for chunk in preview.sample)
    # The first chunk, verbatim. Not a truncation and not a summary: the
    # question the reader is answering is whether the text is intact.
    assert preview.sample[0].content.startswith("رسوم الساعة المعتمدة")


def test_nothing_is_written_or_embedded_by_a_preview():
    """Structural, because the cost of getting this wrong is silent.

    A preview that ingested would make "cancel" a lie and would bill a tenant
    for a corpus they rejected. The module takes no session and no embedder,
    so there is nothing for either to happen through.
    """
    import ast
    import inspect

    from moc.retrieval import preview as module

    parameters = set(inspect.signature(preview_document).parameters)
    assert parameters == {"text", "title", "config"}

    # Over the AST, not the text. The module's docstring explains at length
    # why it does not touch a session, and a substring scan reads its own
    # rationale as a violation — the same trap the console's CSS scanner fell
    # into. Names in code are the claim; prose about them is not.
    tree = ast.parse(inspect.getsource(module))
    used = {
        *(alias.name.split(".")[0] for node in ast.walk(tree)
          if isinstance(node, ast.Import) for alias in node.names),
        *(node.module.split(".")[0] for node in ast.walk(tree)
          if isinstance(node, ast.ImportFrom) and node.module),
        *(node.id for node in ast.walk(tree) if isinstance(node, ast.Name)),
        *(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)),
    }
    forbidden = {"sqlalchemy", "AsyncSession", "session", "execute", "commit",
                 "embed", "ingest_document"}
    assert not (used & forbidden), sorted(used & forbidden)


def test_a_document_with_no_sentence_boundaries_is_the_loud_warning():
    """The one that catches a real KB.

    A fee table exported to text has no full stop in it. The splitter finds
    nothing to split on, the document becomes one enormous chunk, and every
    query then retrieves the whole document — retrieval that looks like it is
    working and discriminates nothing.
    """
    preview = preview_document(text=TABLE, title="جدول الرسوم")

    assert PreviewWarning.no_sentence_boundaries in names(preview)
    detail = next(w for w in preview.warnings if w.name == PreviewWarning.no_sentence_boundaries)
    assert detail.ordinal == 0
    assert "sentence" in detail.reason.lower()


def test_a_clean_document_does_not_trip_the_boundary_warning():
    """The complement, and it is the half that decides whether the warning is
    usable: one that fires on ordinary Arabic gets turned off in a week."""
    assert PreviewWarning.no_sentence_boundaries not in names(
        preview_document(text=CLEAN, title="الرسوم")
    )


def test_figures_with_no_unit_around_them_are_flagged():
    """`QuantityKind.bare` is the detector §19.3 already uses.

    A column that lost its header row is numbers with no currency and no
    percent anywhere near them. A reply built from one states a figure the gate
    cannot label, which is the failure the whole script-first design exists to
    prevent — and it starts here, at ingest, where it is still cheap.
    """
    preview = preview_document(text=COLUMN, title="عمود")

    assert PreviewWarning.bare_figures in names(preview)


def test_a_labelled_figure_is_not_flagged_as_bare():
    """`1400 جنيه` carries its unit. Flagging it would make the warning noise,
    and noise is how a check gets disabled."""
    preview = preview_document(text=CLEAN, title="الرسوم")

    assert PreviewWarning.bare_figures not in names(preview)


def test_an_empty_document_is_refused_rather_than_ingested_as_nothing():
    """A PDF that extracted to whitespace is the common shape here, and it
    ingests perfectly: zero chunks, no error, and a knowledge base that
    silently knows nothing about the thing the tenant just uploaded."""
    preview = preview_document(text="   \n\t  ", title="empty.pdf")

    assert preview.chunk_count == 0
    assert PreviewWarning.empty in names(preview)


def test_an_oversized_chunk_names_its_ordinal():
    """Which chunk, not just that there is one. A count sends the reader back
    to a corpus to find it themselves."""
    long_sentence = "كلمة " * 2000 + "."
    preview = preview_document(text=long_sentence + " تم.", title="طويل")

    oversized = [w for w in preview.warnings if w.name == PreviewWarning.oversized]
    assert oversized
    assert oversized[0].ordinal == 0


def test_the_sample_is_bounded_but_the_count_is_not():
    """A thousand-chunk corpus must not put a thousand chunks through a JSON
    response and into a browser. The count is the whole truth; the sample is a
    window on it, and the response says which it is."""
    from moc.retrieval.preview import SAMPLE_SIZE

    many = " ".join(f"الجملة رقم {index} هنا." for index in range(400))
    preview = preview_document(text=many, title="كبير")

    assert preview.chunk_count > SAMPLE_SIZE
    assert len(preview.sample) == SAMPLE_SIZE
    assert len(preview.chunks) == preview.chunk_count


def test_the_content_hash_is_stable_and_content_addressed():
    """The same bytes give the same hash, so "unchanged" is answerable before
    a single embedding is bought."""
    first = preview_document(text=CLEAN, title="الرسوم")
    again = preview_document(text=CLEAN, title="الرسوم")
    edited = preview_document(text=CLEAN + " تم.", title="الرسوم")

    assert first.content_hash == again.content_hash
    assert first.content_hash != edited.content_hash


def test_the_title_is_part_of_the_hash():
    """`embedding_text` prepends the title to every chunk, so two documents
    with the same body and different titles embed differently. A hash that
    ignored the title would report "unchanged" for a re-titled document and
    leave the old vectors in place."""
    assert (
        preview_document(text=CLEAN, title="الرسوم").content_hash
        != preview_document(text=CLEAN, title="المصاريف").content_hash
    )


def test_the_shipped_corpus_previews_without_noise():
    """The false-positive rate, measured against real Arabic rather than
    argued about.

    A warning that fires on the corpus we already ship is a warning the first
    tenant learns to ignore, and an ignored check catches nothing. 102 real
    chunks, every one of them content somebody wrote for this product.
    """
    import json
    from pathlib import Path

    fixture = Path(__file__).parents[2] / "evals" / "fixtures" / "sinai_demo" / "chunks.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]

    noisy = [
        (row["chunk_id"], warning.name)
        for row in rows
        for warning in preview_document(text=row["content"], title=row["title"]).warnings
        if warning.name != PreviewWarning.empty
    ]
    assert noisy == [], f"{len(noisy)} of {len(rows)} real chunks tripped a warning"
