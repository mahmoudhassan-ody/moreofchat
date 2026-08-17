"""Stage 1 grading — deterministic checks, no LLM (eval-harness-spec §5.1).

Runs before the judge because it is free, fast, non-flaky, and it covers the
hard gate that matters most: `hallucinated_figure_rate` must be zero.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from moc.arabic.numerals import QuantityKind, extract_numbers, extract_quantities


@dataclass(frozen=True)
class GroundingResult:
    passed: bool
    orphan_numbers: list[int | float] = field(default_factory=list)
    approximations: list[str] = field(default_factory=list)
    reply_numbers: list[int | float] = field(default_factory=list)
    source_numbers: list[int | float] = field(default_factory=list)


def check_numeric_grounding(
    reply: str,
    retrieved_passages: Sequence[str],
    script_constants: Iterable[float | str],
) -> GroundingResult:
    """Every figure in `reply` must appear in a passage or a script constant.

    Two ways to fail:

    - **Orphan figure.** A number with no source. This is F1 — a fee the
      knowledge base never stated, reaching a student on WhatsApp.
    - **Approximation.** A hedged figure fails even when the number is correct.
      Design doc §19.3 makes this non-negotiable: the markers are configurable,
      the fact that an approximation trips the check is not. "Roughly 1400"
      invites the customer to treat a fixed fee as an opening position.

    Comparison happens after digit normalization, so a reply in Arabic-Indic
    digits matches a source in Latin ones — otherwise the check would fire on
    every correctly-grounded Arabic reply and be switched off within a week.
    """
    quantities = extract_quantities(reply)

    # Years qualify a figure rather than asserting one, and the extractor already
    # drops them. Percentages and counts still need a source: a down-payment
    # share and a bedroom count are both claims a tenant can be held to.
    source_numbers = _collect_sources(retrieved_passages, script_constants)
    reply_numbers = [q.value for q in quantities]

    orphans = [q.value for q in quantities if not _is_grounded(q.value, source_numbers)]
    approximations = [q.raw for q in quantities if q.approximate]

    return GroundingResult(
        passed=not orphans and not approximations,
        orphan_numbers=orphans,
        approximations=approximations,
        reply_numbers=reply_numbers,
        source_numbers=sorted(source_numbers),
    )


def _collect_sources(
    retrieved_passages: Sequence[str], script_constants: Iterable[float | str]
) -> set[float]:
    """Numbers the reply is allowed to state, from both grounding surfaces."""
    numbers: set[float] = set()
    for passage in retrieved_passages:
        numbers.update(float(n) for n in extract_numbers(passage))
    for constant in script_constants:
        # Constants arrive as numbers from a calculator tool or as strings from
        # a script node; parse strings through the same extractor so "١٤٠٠"
        # and 1400 are the same permission.
        if isinstance(constant, str):
            numbers.update(float(n) for n in extract_numbers(constant))
        else:
            numbers.add(float(constant))
    return numbers


def _is_grounded(value: float, sources: set[float]) -> bool:
    """Exact match only.

    No tolerance window on purpose. A figure that is merely close is the
    failure this check exists to catch — spec §3.2 says the same about
    payment-plan arithmetic, and a tolerance here would let a rounded fee pass.
    """
    return float(value) in sources


__all__ = ["GroundingResult", "QuantityKind", "check_numeric_grounding"]
