"""Stage 1 grading — deterministic checks, no LLM (eval-harness-spec §5.1).

Runs before the judge because it is free, fast, non-flaky, and it covers the
hard gate that matters most: `hallucinated_figure_rate` must be zero.

Every check returns a `CheckResult` naming the §2.1 metric it feeds, so the
report aggregates by metric without a second mapping table that can drift from
the checks themselves.

A check that has nothing to assert returns `skipped=True` rather than a pass.
The distinction matters at the gate: a metric no case exercised must read as
unmeasured, not as a perfect score.

`check_numeric_grounding` is re-exported from `moc.agent.guards` rather than
implemented here. It is a runtime gate first and a measurement second: the
orchestrator refuses to send a reply that fails it, and this package reports
how often that happens. Two implementations would drift, and the drift would
present as a green eval suite while production emitted an orphan figure.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from moc.agent.guards import (
    GroundingResult,
    check_named_entities,
    check_numeric_grounding,
    check_type_substitution,
)
from moc.arabic.numerals import QuantityKind, normalize_digits
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
    #: unit id -> `property_type`, for the substitution gate.
    unit_type: Mapping[str, str] = field(default_factory=dict)
    #: unit id -> `compound`. Doubles as the catalogue of real compound names,
    #: which is what an invented one is checked against.
    unit_compound: Mapping[str, str] = field(default_factory=dict)

    def compounds(self) -> frozenset[str]:
        return frozenset(self.unit_compound.values())


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


def check_property_type(
    requested_type: str | None,
    presented_unit_ids: Sequence[str],
    snapshot: InventorySnapshot,
) -> CheckResult:
    """The type asked for is the type offered — never a substitute (§19.3).

    A chalet request answered with a townhouse, an office request with retail:
    the closest-priced unit of another type is the wrong answer wearing the
    right price, and the customer finds out at the viewing. A different
    *compound* is a legitimate alternative and is not flagged here; a different
    *type* never is.

    Skips only when nothing was presented. A turn that offered units without a
    resolved type is *not* clean — there is nothing to compare against, so it
    fails rather than passing quietly.
    """
    if not presented_unit_ids:
        return CheckResult(
            name="property_type",
            metric="type_substitution_rate",
            passed=True,
            skipped=True,
            detail="no units presented",
        )
    presented = {
        unit_id: snapshot.unit_type.get(unit_id, "not in snapshot")
        for unit_id in presented_unit_ids
    }
    result = check_type_substitution(requested_type, presented)
    if result.requested_type is None:
        return CheckResult(
            name="property_type",
            metric="type_substitution_rate",
            passed=False,
            detail="units were presented without a resolved property_type to check against",
        )
    return CheckResult(
        name="property_type",
        metric="type_substitution_rate",
        passed=result.passed,
        detail=""
        if result.passed
        else "asked for {}, offered {}".format(
            result.requested_type,
            "; ".join(f"{unit}={kind}" for unit, kind in result.substituted),
        ),
    )


def check_compound_grounding(
    named_compounds: Sequence[str], snapshot: InventorySnapshot
) -> CheckResult:
    """A compound named in a reply must exist in the catalogue (§19.3).

    The same class of failure as an invented price, and more convincing: a
    plausible Egyptian compound name reads as local knowledge, so nobody
    questions it until a customer drives to somewhere that does not exist.
    """
    if not named_compounds:
        return CheckResult(
            name="compound_grounding",
            metric="invented_compound_rate",
            passed=True,
            skipped=True,
            detail="reply named no compound",
        )
    result = check_named_entities(named_compounds, snapshot.compounds())
    return CheckResult(
        name="compound_grounding",
        metric="invented_compound_rate",
        passed=result.passed,
        detail="" if result.passed else f"not in the catalogue: {', '.join(result.invented)}",
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
    "check_compound_grounding",
    "check_property_type",
    "check_language",
    "check_numeric_grounding",
    "check_slots",
    "check_tool_calls",
]
