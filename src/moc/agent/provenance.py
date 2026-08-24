"""Which chunk grounded which figure — demo plan Task 32.

**The data already exists and is thrown away.** Every composed turn runs
`check_numeric_grounding` over the reply and the retrieved passages. It decides,
figure by figure, whether each one has a source — and then keeps a pass/fail
and two lists of numbers, discarding the mapping. A dean asking "where did 1400
come from?" is asking a question the system answered a millisecond earlier.

This module keeps it: figure -> the chunk that held it, that chunk's title, its
as-of date, and the sentence it appeared in. The inbox's source pane renders
that, and it is the thing that makes a dean believe the product rather than the
demo — a claim of groundedness is a brochure, a click-through to their own
sentence is evidence.

**The comparison is the gate's own.** `is_grounded` and `extract_quantities`
are imported from `guards` and `numerals` rather than reimplemented: a pane
that showed a source for a figure the gate called an orphan, or nothing for one
it passed, would contradict the system it describes — and people believe the
pane. `test_the_trace_agrees_with_the_grounding_gate` holds the two together
over the same replies.

An ungrounded figure is reported with no source rather than left out. Omitting
it would make every pane show a reply whose figures all trace, which is exactly
the claim §19.3 exists to check, arriving as a UI default instead of as a
measurement.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from moc.agent.guards import is_grounded
from moc.arabic.numerals import extract_numbers, extract_quantities, normalize_digits

#: What a figure traced to. `chunk` is a retrieved passage, `script` is §3.1's
#: figure held in a script node — as legitimate a source, and not a chunk, so
#: the pane says which rather than showing the second as ungrounded.
#:
#: `inventory` and `calculator` are §3.2's grounding mode (Task 41b). A price
#: traces to a **row** — a unit id, what it is called, and the `as_of` it was
#: snapshotted at — and an instalment traces to a **calculator output** with
#: the inputs it ran with. Same promise as a chunk and a different shape, in
#: one `figures` list: a second provenance shape would be a second thing the
#: pane can fail to render, and the promise made to a university and to a
#: broker is one promise.
CHUNK = "chunk"
SCRIPT = "script"
INVENTORY = "inventory"
CALCULATOR = "calculator"

_SENTENCE_ENDINGS = ".!?؟؛۔\n"


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, with what the pane needs to name it.

    `as_of` is here because §7.1 makes staleness a first-class property: a
    broker looking at a figure should see the date it was current without
    asking, and the pane is where that becomes true.
    """

    chunk_id: str
    content: str
    title: str | None = None
    as_of: date | None = None


@dataclass(frozen=True)
class Row:
    """One inventory row, as evidence (§3.2).

    `values` is field name -> figure, because a reply can state a price, a
    bedroom count and an area from the same row and the pane has to say which
    of them each number is. Naming the row alone would answer "where did
    5,500,000 come from?" with "that unit", which is the question restated.
    """

    unit_id: str
    values: Mapping[str, float]
    #: What a person calls it — the compound, usually. The pane shows this
    #: where a document answer shows the document's title.
    title: str | None = None
    as_of: date | None = None


@dataclass(frozen=True)
class Computation:
    """One tool output, as evidence (§19.3).

    The arithmetic is the calculator's and never the model's, so the evidence
    for an instalment is not a sentence anywhere — it is the tool that ran and
    what it ran with. `inputs` is therefore part of the trace rather than
    context beside it: "137,500" is not checkable, and "137,500 from
    payment_plan_calculator(price=5,500,000, down_payment_pct=20, years=8)" is.
    """

    tool: str
    values: Mapping[str, float]
    inputs: Mapping[str, Any] = field(default_factory=dict)
    unit_id: str | None = None
    as_of: date | None = None


@dataclass(frozen=True)
class FigureSource:
    """One figure in a reply, and where it came from."""

    value: int | float
    raw: str
    grounded: bool
    #: `chunk`, `script`, or None when nothing supplied it.
    source: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    as_of: date | None = None
    #: The sentence the figure appeared in, not the whole chunk. A pane showing
    #: a paragraph makes the reader find the figure again themselves.
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The wire shape, for `messages.provenance` and the inbox response."""
        return {
            "value": self.value,
            "raw": self.raw,
            "grounded": self.grounded,
            "source": self.source,
            "chunkId": self.chunk_id,
            "title": self.title,
            "asOf": self.as_of.isoformat() if self.as_of else None,
            "excerpt": self.excerpt,
        }


def trace_figures(
    *,
    reply: str,
    passages: Sequence[Passage] = (),
    script_constants: Iterable[float | str] = (),
    rows: Sequence[Row] = (),
    computations: Sequence[Computation] = (),
) -> list[FigureSource]:
    """Every figure in `reply`, with what produced it.

    Sources are tried in one order: inventory row, tool output, retrieved
    passage, script constant. The first two and the third never arrive
    together — a turn is either a document answer or an inventory answer, and
    §3.2 is explicit that the broker fixture is deliberately absent from
    `kb_chunks` so a price cannot come from a passage — so the order between
    those groups decides nothing. The order *within* the pair does.

    **Rows before computations, deliberately.** A payment schedule carries the
    total it was built from, so a price appears in both — and the row is where
    the figure originated while the calculator is where it passed through.
    Naming the calculator would send a broker looking for a computation that
    computed nothing.
    """
    constants = _constant_values(script_constants)
    traced: list[FigureSource] = []

    for quantity in extract_quantities(reply):
        from_row = _in_row(quantity.value, rows)
        if from_row is not None:
            traced.append(_from_row(quantity, *from_row))
            continue
        from_tool = _in_computation(quantity.value, computations)
        if from_tool is not None:
            traced.append(_from_computation(quantity, *from_tool))
            continue
        found = _find(quantity.value, passages)
        if found is not None:
            passage, excerpt = found
            traced.append(
                FigureSource(
                    value=quantity.value,
                    raw=quantity.raw,
                    grounded=True,
                    source=CHUNK,
                    chunk_id=passage.chunk_id,
                    title=passage.title,
                    as_of=passage.as_of,
                    excerpt=excerpt,
                )
            )
            continue
        if is_grounded(quantity.value, constants):
            traced.append(
                FigureSource(
                    value=quantity.value,
                    raw=quantity.raw,
                    grounded=True,
                    source=SCRIPT,
                )
            )
            continue
        traced.append(
            FigureSource(value=quantity.value, raw=quantity.raw, grounded=False)
        )
    return traced


def _in_row(value: float, rows: Sequence[Row]) -> tuple[Row, str] | None:
    """The first row stating `value`, and which of its fields did.

    First rather than best, for the same reason the passage search takes the
    first: the rows arrive in the order the reply presented them.
    """
    for row in rows:
        for field_name, held in row.values.items():
            if held is not None and is_grounded(value, {float(held)}):
                return row, field_name
    return None


def _from_row(quantity: Any, row: Row, field_name: str) -> FigureSource:
    return FigureSource(
        value=quantity.value,
        raw=quantity.raw,
        grounded=True,
        source=INVENTORY,
        # The unit id, in the field the pane already links from. A row is this
        # vertical's chunk.
        chunk_id=row.unit_id,
        title=row.title,
        as_of=row.as_of,
        excerpt=f"{field_name} = {_readable(quantity.value)}",
    )


def _in_computation(
    value: float, computations: Sequence[Computation]
) -> tuple[Computation, str] | None:
    for computation in computations:
        for field_name, held in computation.values.items():
            if held is not None and is_grounded(value, {float(held)}):
                return computation, field_name
    return None


def _from_computation(
    quantity: Any, computation: Computation, field_name: str
) -> FigureSource:
    arguments = ", ".join(
        f"{name}={_readable(held)}" for name, held in sorted(computation.inputs.items())
    )
    return FigureSource(
        value=quantity.value,
        raw=quantity.raw,
        grounded=True,
        source=CALCULATOR,
        chunk_id=computation.unit_id,
        # Deliberately no title. The pane falls back to a translated label for
        # the source kind, and a broker reading `payment_plan_calculator` where
        # a document answer shows their own document's name is developer-speak
        # in front of a customer's figures. The tool name is in the excerpt,
        # where it belongs: as part of what makes the number checkable.
        title=None,
        as_of=computation.as_of,
        # The inputs are the evidence. A number alone is not checkable.
        excerpt=f"{field_name} = {_readable(quantity.value)} — {computation.tool}({arguments})",
    )


def _readable(value: Any) -> str:
    """Thousands separators on figures, and everything else verbatim.

    `5500000` and `5,500,000` are the same number and only one of them can be
    checked against a reply at a glance.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = int(value) if float(value).is_integer() else value
        return f"{number:,}"
    return str(value)


def _find(value: float, passages: Sequence[Passage]) -> tuple[Passage, str] | None:
    """The first passage stating `value`, and the sentence it stated it in.

    First rather than best: the passages arrive in fusion's rank order, so the
    first match is the highest-ranked chunk that could have supplied the
    figure. Naming a fixed one regardless would be a link that is usually
    right, which is the worst kind of link to put in front of a customer.
    """
    for passage in passages:
        numbers = {float(number) for number in extract_numbers(passage.content)}
        if is_grounded(value, numbers):
            return passage, _excerpt(value, passage.content)
    return None


def _excerpt(value: float, content: str) -> str:
    """The sentence holding `value`, or the whole passage if none does.

    Split on the same terminators the chunker uses; matched on normalized
    digits so an Arabic-Indic source is found by a Latin figure and the other
    way round.
    """
    sentences = _split(content)
    for sentence in sentences:
        if is_grounded(
            value, {float(number) for number in extract_numbers(sentence)}
        ):
            return sentence.strip()
    return content.strip()


def _split(content: str) -> list[str]:
    parts, current = [], []
    for character in normalize_digits(content):
        current.append(character)
        if character in _SENTENCE_ENDINGS:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return parts


def _constant_values(script_constants: Iterable[float | str]) -> set[float]:
    """§3.1's other grounding surface, parsed the way `guards` parses it."""
    numbers: set[float] = set()
    for constant in script_constants:
        if isinstance(constant, str):
            numbers.update(float(n) for n in extract_numbers(constant))
        else:
            numbers.add(float(constant))
    return numbers


__all__ = [
    "CALCULATOR",
    "CHUNK",
    "INVENTORY",
    "SCRIPT",
    "Computation",
    "FigureSource",
    "Passage",
    "Row",
    "trace_figures",
]
