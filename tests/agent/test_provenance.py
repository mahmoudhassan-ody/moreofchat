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


# ─────────────────────── the other grounding mode ───────────────────────
#
# Demo plan Task 41b. A document answer traces to a chunk; an inventory answer
# traces to a row and a calculator output. Same promise, same `figures` list,
# same renderer — a second provenance shape would be a second thing the pane
# can fail to render, and the promise made to all three tenants is one promise.


def test_a_price_traces_to_the_row_it_was_read_from():
    from datetime import date

    from moc.agent.provenance import INVENTORY, Row, trace_figures

    traced = trace_figures(
        reply="عندنا شقة في مدينتي بسعر 5,500,000 جنيه.",
        rows=(
            Row(
                unit_id="MD-1",
                values={"price": 5_500_000, "bedrooms": 3},
                title="Madinaty",
                as_of=date(2026, 8, 1),
            ),
        ),
    )
    assert [figure.value for figure in traced] == [5_500_000]
    assert traced[0].source == INVENTORY
    assert traced[0].chunk_id == "MD-1"
    assert traced[0].title == "Madinaty"
    assert "price" in traced[0].excerpt


def test_an_instalment_traces_to_the_calculator_inputs_that_produced_it():
    """Not to the row. §19.3: the arithmetic is the tool's, and the evidence
    for a figure the model never composed is the computation — the tool that
    ran and what it ran with."""
    from moc.agent.provenance import CALCULATOR, Computation, trace_figures

    traced = trace_figures(
        reply="المقدم 1,100,000 والقسط 137,500 على 8 سنين.",
        computations=(
            Computation(
                tool="payment_plan_calculator",
                unit_id="MD-1",
                values={
                    "down_payment": 1_100_000,
                    "installment_amount": 137_500,
                    "years": 8,
                },
                inputs={"price": 5_500_000, "down_payment_pct": 20, "years": 8},
            ),
        ),
    )
    # The term is absent since 2026-08-26, and deliberately: a span of time
    # stopped being a figure when eight study durations in a faculty list cost
    # a customer their answer. The pane shows the money and not the term.
    #
    # It is not left unchecked. `check_computation` in the inventory runner
    # reads the reply with a raw digit regex rather than through
    # `extract_numbers`, so every digit run in a payment-plan reply is still
    # string-matched against the calculator's own output — the term included.
    assert {figure.value for figure in traced} == {1_100_000, 137_500}
    assert {figure.source for figure in traced} == {CALCULATOR}
    down = next(f for f in traced if f.value == 1_100_000)
    assert "payment_plan_calculator" in down.excerpt
    assert "price=5,500,000" in down.excerpt


def test_the_row_wins_when_the_calculator_only_echoed_it():
    """A schedule carries the total it was built from, so a price appears in
    both. The row is where the figure originated and the calculator is where it
    passed through, and a pane that named the calculator would send a broker
    looking for a computation that did not compute anything."""
    from moc.agent.provenance import INVENTORY, Computation, Row, trace_figures

    traced = trace_figures(
        reply="السعر 5,500,000.",
        rows=(Row(unit_id="MD-1", values={"price": 5_500_000}),),
        computations=(
            Computation(
                tool="payment_plan_calculator",
                unit_id="MD-1",
                values={"total": 5_500_000},
                inputs={},
            ),
        ),
    )
    assert traced[0].source == INVENTORY


def test_the_as_of_travels_with_the_figure_rather_than_beside_it():
    """A price separated from its date is a price the tenant cannot stand
    behind. The pane shows it on the figure, not once at the top, because a
    reply can quote two units snapshotted on different days."""
    from datetime import date

    from moc.agent.provenance import Row, trace_figures

    traced = trace_figures(
        reply="مدينتي 5,500,000 وزايد 7,200,000.",
        rows=(
            Row(unit_id="MD-1", values={"price": 5_500_000}, as_of=date(2026, 8, 1)),
            Row(unit_id="SZ-9", values={"price": 7_200_000}, as_of=date(2026, 7, 2)),
        ),
    )
    assert {f.value: f.as_of for f in traced} == {
        5_500_000: date(2026, 8, 1),
        7_200_000: date(2026, 7, 2),
    }


def test_a_figure_from_neither_a_row_nor_a_computation_is_an_orphan():
    """The same honesty the chunk path already has. A pane where every figure
    has a source would be making §19.3's claim as a rendering default."""
    from moc.agent.provenance import Row, trace_figures

    traced = trace_figures(
        reply="السعر 5,500,000 والمقدم 999,999.",
        rows=(Row(unit_id="MD-1", values={"price": 5_500_000}),),
    )
    orphan = next(f for f in traced if f.value == 999_999)
    assert orphan.grounded is False
    assert orphan.source is None


def test_an_inventory_figure_uses_the_wire_shape_the_pane_already_renders():
    """One renderer. The keys are the ones `SourcePane` reads today; only the
    value of `source` is new."""
    from datetime import date

    from moc.agent.provenance import Row, trace_figures

    figure = trace_figures(
        reply="5,500,000",
        rows=(Row(unit_id="MD-1", values={"price": 5_500_000}, title="Madinaty",
                  as_of=date(2026, 8, 1)),),
    )[0].to_dict()
    assert set(figure) == {
        "value", "raw", "grounded", "source", "chunkId", "title", "asOf", "excerpt"
    }
    assert figure["asOf"] == "2026-08-01"
