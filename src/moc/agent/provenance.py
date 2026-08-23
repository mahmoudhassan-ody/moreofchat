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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from moc.agent.guards import is_grounded
from moc.arabic.numerals import extract_numbers, extract_quantities, normalize_digits

#: What a figure traced to. `chunk` is a retrieved passage, `script` is §3.1's
#: figure held in a script node — as legitimate a source, and not a chunk, so
#: the pane says which rather than showing the second as ungrounded.
CHUNK = "chunk"
SCRIPT = "script"

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
) -> list[FigureSource]:
    """Every figure in `reply`, with the chunk that grounded it."""
    constants = _constant_values(script_constants)
    traced: list[FigureSource] = []

    for quantity in extract_quantities(reply):
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


__all__ = ["CHUNK", "SCRIPT", "FigureSource", "Passage", "trace_figures"]
