"""Which chunk grounded which figure — demo plan Task 32.

**The differentiator, and the data already exists.** Every composed turn runs
`check_numeric_grounding` over the reply and the retrieved passages; it decides
whether each figure has a source and then throws the mapping away, keeping only
a pass/fail and two lists of numbers. A dean asking "where did 1400 come from?"
is asking a question the system answered a millisecond before it discarded the
answer.

So this module keeps the mapping: figure -> the chunk that contained it, its
title, and its as-of date. That is what the inbox's source pane renders, and it
is the thing that makes a dean believe the product rather than the demo.

**It reuses the grounding check's own comparison.** A second implementation of
"is this figure in that passage" would show a source for a figure the gate
called an orphan, or show nothing for one it passed — a provenance pane that
disagrees with the gate is worse than none, because it is the pane people
believe.
"""

from datetime import date

from moc.agent.provenance import trace_figures

FEE = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه."
APPLICATION = "رسوم التقديم 2000 جنيه وهي غير مستردة."


def passages(*rows):
    from moc.agent.provenance import Passage

    return [Passage(**row) for row in rows]


SOURCES = passages(
    {"chunk_id": "sinai_fee_hour_ar", "content": FEE, "title": "رسوم الساعة",
     "as_of": date(2026, 1, 1)},
    {"chunk_id": "sinai_fee_application_ar", "content": APPLICATION,
     "title": "رسوم التقديم", "as_of": None},
)


def test_a_grounded_figure_links_to_the_chunk_that_grounded_it():
    """The whole point. 1400 came from somewhere, and the somewhere has a name.

    A dean who can click a figure and see the sentence it came from, in their
    own document, with the date it was current, is a dean who believes the
    rest of the screen. One who is told "the system is grounded" is not.
    """
    traced = trace_figures(reply="الرسوم 1400 جنيه.", passages=SOURCES)

    assert len(traced) == 1
    assert traced[0].value == 1400
    assert traced[0].chunk_id == "sinai_fee_hour_ar"
    assert traced[0].title == "رسوم الساعة"
    assert traced[0].as_of == date(2026, 1, 1)
    # The sentence, not the whole chunk. A pane showing a paragraph makes the
    # reader find the figure again themselves.
    assert "1400" in traced[0].excerpt


def test_the_right_chunk_when_two_could_have_supplied_it():
    """2000 is in the application-fee chunk and nowhere else. Naming the first
    passage regardless would be a link that is usually right, which is the
    worst kind."""
    traced = trace_figures(reply="رسوم التقديم 2000 جنيه.", passages=SOURCES)

    assert traced[0].chunk_id == "sinai_fee_application_ar"


def test_an_orphan_figure_is_traced_to_nothing_rather_than_omitted():
    """Present with no source, never absent.

    Omitting it would make the pane show a reply whose figures all have sources
    — which is exactly the claim §19.3 exists to check, arriving as a UI
    default rather than as a measurement.
    """
    traced = trace_figures(reply="الرسوم 9999 جنيه.", passages=SOURCES)

    assert len(traced) == 1
    assert traced[0].value == 9999
    assert traced[0].chunk_id is None
    assert traced[0].grounded is False


def test_a_script_constant_is_a_source_with_no_chunk():
    """§3.1: a figure held in a script node is as legitimate a source as a
    retrieved chunk. It has no chunk id because it is not a chunk, and the
    pane says so rather than showing it as ungrounded."""
    traced = trace_figures(
        reply="رسوم التقديم 500 جنيه.", passages=SOURCES, script_constants=(500,)
    )

    assert traced[0].grounded is True
    assert traced[0].chunk_id is None
    assert traced[0].source == "script"


def test_arabic_indic_digits_match_a_latin_source():
    """The comparison happens after digit normalization, like the gate's. A
    pane that could not match ١٤٠٠ to 1400 would show every correctly grounded
    Arabic reply as ungrounded."""
    traced = trace_figures(reply="الرسوم ١٤٠٠ جنيه.", passages=SOURCES)

    assert traced[0].chunk_id == "sinai_fee_hour_ar"


def test_a_reply_with_no_figures_traces_to_nothing():
    """Not an empty pane with a heading — nothing to show is a different state
    from nothing found, and the caller decides how to render it."""
    assert trace_figures(reply="أهلاً بحضرتك.", passages=SOURCES) == []


def test_the_trace_agrees_with_the_grounding_gate():
    """The property that matters more than any single mapping.

    A pane that shows a source for a figure the gate called an orphan — or
    nothing for one it passed — is a pane that contradicts the system it
    describes, and people believe the pane.
    """
    from moc.agent.guards import check_numeric_grounding

    for reply in (
        "الرسوم 1400 جنيه.",
        "الرسوم 9999 جنيه.",
        "رسوم التقديم 2000 والساعة 1400.",
        "أهلاً بحضرتك.",
    ):
        gate = check_numeric_grounding(reply, [p.content for p in SOURCES], ())
        traced = trace_figures(reply=reply, passages=SOURCES)

        assert [f.value for f in traced if not f.grounded] == gate.orphan_numbers, reply
        assert sorted(f.value for f in traced) == sorted(gate.reply_numbers), reply
