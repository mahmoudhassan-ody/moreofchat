"""Stage 1 grading — deterministic checks, no LLM (eval-harness-spec §5.1).

Runs before the judge because it is free, fast, non-flaky, and it covers the
hard gate that matters most: `hallucinated_figure_rate` must be zero.

Every check returns a `CheckResult` naming the §2.1 metric it feeds, so the
report aggregates by metric without a second mapping table that can drift from
the checks themselves.

A check that has nothing to assert returns `skipped=True` rather than a pass.
The distinction matters at the gate: a metric no case exercised must read as
unmeasured, not as a perfect score.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from moc.arabic.numerals import QuantityKind, extract_numbers, extract_quantities, normalize_digits
from moc.arabic.script import detect_language
from moc.config_store import load
from moc.evals.schema import Action, ExpectedToolCall

_LEXICON = "arabic/lexicon"


@dataclass(frozen=True)
class CheckResult:
    name: str
    metric: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class ToolCall:
    """A call the orchestrator actually made, as recorded on the turn."""

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InventorySnapshot:
    """A frozen real-estate fixture (§8.1).

    **P1 integration seam.** Nothing in src/ builds one of these yet — the
    loader that reads `evals/fixtures/broker_demo_2026_08_01/` lands with the
    fixture work in P1. Until then the caller supplies it, which means a
    missing fixture is a TypeError at the call site rather than an inventory
    check that quietly passes on empty data. Tests construct it directly.
    """

    fixture: str
    as_of: str
    unit_status: Mapping[str, str]


@dataclass(frozen=True)
class GroundingResult:
    """Two independent zero-tolerance gates, never conflated.

    `orphan_numbers` feeds `hallucinated_figure_rate` (§2.1) — the figure has
    no source. `hedged_numbers` feeds `hedged_figure_rate` — the figure has a
    source but the reply hedged it.

    They are separate because the fixes are different. An orphan means
    retrieval or the script failed to supply the figure; a hedge means the
    generation step editorialized over a figure it had. A single number
    combining both tells you a regression happened and nothing about where.

    The lists are disjoint. A figure that is both orphan and hedged counts
    only as an orphan, so one incident cannot move both rates.
    """

    passed: bool
    orphan_numbers: list[int | float] = field(default_factory=list)
    hedged_numbers: list[int | float] = field(default_factory=list)
    reply_numbers: list[int | float] = field(default_factory=list)
    source_numbers: list[int | float] = field(default_factory=list)


def check_numeric_grounding(
    reply: str,
    retrieved_passages: Sequence[str],
    script_constants: Iterable[float | str],
) -> GroundingResult:
    """Every figure in `reply` must appear in a passage or a script constant.

    Two ways to fail, reported separately and gated separately (§2.1):

    - **Orphan figure** -> `hallucinated_figure_rate`. A number with no source.
      This is F1 — a fee the knowledge base never stated, reaching a student on
      WhatsApp.
    - **Hedged figure** -> `hedged_figure_rate`. A *grounded* number the reply
      hedged anyway. Design doc §19.3 makes this non-negotiable: the markers
      are configurable, the fact that an approximation trips the check is not.
      "Roughly 1400" invites the customer to treat a fixed fee as an opening
      position, which is how a tenant ends up honouring a figure they never set.

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

    grounded = [q for q in quantities if _is_grounded(q.value, source_numbers)]
    orphans = [q.value for q in quantities if not _is_grounded(q.value, source_numbers)]
    # Only grounded figures can be hedged. An orphan that is also hedged is one
    # incident, and counting it twice would move both rates for a single fault.
    hedged = [q.value for q in grounded if q.approximate]

    return GroundingResult(
        passed=not orphans and not hedged,
        orphan_numbers=orphans,
        hedged_numbers=hedged,
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


# ─────────────────────────── action, language, slots ───────────────────────────


def check_action(expected: Action, actual: Action) -> CheckResult:
    """Answer vs clarify vs handoff vs refuse (§5.1).

    The F2 failure lives here: a confident answer where handoff was correct
    destroys tenant trust faster than an obvious error does.
    """
    passed = expected == actual
    return CheckResult(
        name="action",
        metric="expected_action_accuracy",
        passed=passed,
        detail="" if passed else f"expected {expected}, got {actual}",
    )


def check_language(reply: str, expected_lang: str | None) -> CheckResult:
    """Reply script matches expected_lang (§5.1), by majority of letters."""
    if expected_lang is None:
        return CheckResult(
            name="language",
            metric="language_mirror_accuracy",
            passed=True,
            skipped=True,
            detail="case pins no expected_lang",
        )
    detected = detect_language(reply)
    passed = detected == expected_lang
    return CheckResult(
        name="language",
        metric="language_mirror_accuracy",
        passed=passed,
        detail="" if passed else f"expected {expected_lang}, detected {detected}",
    )


def _slot_value(value: Any) -> frozenset:
    """Normalize a slot value so order never decides the comparison.

    A slot can hold several values at once — re-0009 keeps two areas after the
    customer names both — and the assertion is about which values are held, not
    the order they arrived in. Scalars normalize to a one-item set so a scalar
    and a single-item list are the same state.
    """
    if isinstance(value, list | tuple | set | frozenset):
        return frozenset(value)
    return frozenset({value})


def check_slots(
    expected: Mapping[str, Any] | None, actual: Mapping[str, Any] | None
) -> CheckResult:
    """Compare slot state against expected_slots (§5.1).

    Exact on keys, not a subset: `expected_slots: {}` is how edu-0004 turn 1
    asserts that nothing has been captured yet, and under subset semantics that
    assertion would be unwriteable.
    """
    if expected is None:
        return CheckResult(
            name="slots",
            metric="slot_retention_accuracy",
            passed=True,
            skipped=True,
            detail="turn pins no expected_slots",
        )
    actual = actual or {}
    differences = [
        key
        for key in set(expected) | set(actual)
        if key not in expected
        or key not in actual
        or _slot_value(expected[key]) != _slot_value(actual[key])
    ]
    return CheckResult(
        name="slots",
        metric="slot_retention_accuracy",
        passed=not differences,
        detail="" if not differences else f"slot mismatch: {', '.join(sorted(differences))}",
    )


# ─────────────────────────── inventory grounding (§3.2) ───────────────────────────


def check_tool_calls(
    expected: Sequence[ExpectedToolCall], actual: Sequence[ToolCall]
) -> CheckResult:
    """Every expected call was made with at least the arguments the case pins.

    `args_contain` is a subset check by design (§3.2): the orchestrator may pass
    more arguments than a case cares about, and pinning the full signature would
    break every inventory case the first time an unrelated parameter is added.
    """
    if not expected:
        return CheckResult(
            name="tool_calls",
            metric="arithmetic_in_model_rate",
            passed=True,
            skipped=True,
            detail="case expects no tool calls",
        )
    missing = [
        f"{want.name}({want.args_contain})"
        for want in expected
        if not any(_matches(want, got) for got in actual)
    ]
    return CheckResult(
        name="tool_calls",
        metric="arithmetic_in_model_rate",
        passed=not missing,
        detail="" if not missing else f"no matching call for {'; '.join(missing)}",
    )


def _matches(expected: ExpectedToolCall, actual: ToolCall) -> bool:
    if expected.name != actual.name:
        return False
    return all(actual.args.get(key) == value for key, value in expected.args_contain.items())


def check_availability(
    presented_unit_ids: Sequence[str], snapshot: InventorySnapshot
) -> CheckResult:
    """No sold or reserved unit may be presented as available (§5.1).

    Takes the units the orchestrator actually offered — read off the turn's
    tool calls — rather than scraping ids out of the reply text. What was
    offered is recorded; what a regex thinks was mentioned is a guess, and this
    is a hard gate.

    A unit absent from the frozen snapshot fails. It cannot be asserted
    available, and inventing one is the same commercial incident as offering a
    sold one.
    """
    if not presented_unit_ids:
        return CheckResult(
            name="availability",
            metric="sold_unit_offered_rate",
            passed=True,
            skipped=True,
            detail="no units presented",
        )
    offenders = [
        f"{unit_id}={snapshot.unit_status.get(unit_id, 'not in snapshot')}"
        for unit_id in presented_unit_ids
        if snapshot.unit_status.get(unit_id) != _available_status()
    ]
    return CheckResult(
        name="availability",
        metric="sold_unit_offered_rate",
        passed=not offenders,
        detail="" if not offenders else f"presented unavailable: {', '.join(offenders)}",
    )


def check_asof_disclosure(
    reply: str, snapshot: InventorySnapshot, required: bool
) -> CheckResult:
    """Does the reply say when the inventory data was current (§3.2)?

    Satisfied by the snapshot date in any digit script, or by a temporal
    qualifier from the lexicon. §5.1 accepts "an equivalent temporal
    qualifier", because a reply reading "prices as of last update" discloses
    staleness as honestly as one quoting the date.
    """
    if not required:
        return CheckResult(
            name="asof_disclosure",
            metric="asof_disclosure_rate",
            passed=True,
            skipped=True,
            detail="case does not require disclosure",
        )
    normalized = normalize_digits(reply).casefold()
    disclosed = snapshot.as_of.casefold() in normalized or any(
        qualifier in normalized for qualifier in _temporal_qualifiers()
    )
    return CheckResult(
        name="asof_disclosure",
        metric="asof_disclosure_rate",
        passed=disclosed,
        detail="" if disclosed else f"reply states neither {snapshot.as_of} nor any qualifier",
    )


@lru_cache(maxsize=1)
def _temporal_qualifiers() -> tuple[str, ...]:
    return tuple(q.casefold() for q in load(_LEXICON)["temporal_qualifiers"])


@lru_cache(maxsize=1)
def _available_status() -> str:
    return load("evals/inventory")["available_status"]


__all__ = [
    "CheckResult",
    "GroundingResult",
    "InventorySnapshot",
    "QuantityKind",
    "ToolCall",
    "check_action",
    "check_asof_disclosure",
    "check_availability",
    "check_language",
    "check_numeric_grounding",
    "check_slots",
    "check_tool_calls",
]
