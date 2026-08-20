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
    """One check's verdict, and which §2.1 metric it feeds.

    `observational` marks a check that grades nothing the case pinned. It is
    still reported — the whole point is that the population stays visible —
    but it must not decide whether the case passed, or a case that correctly
    exercises the observed condition carries a permanent red mark and someone
    eventually deletes the observation to make it green. `check_type_resolved`
    is the one: re-0023 names no property type on purpose.
    """

    name: str
    metric: str
    passed: bool
    detail: str = ""
    skipped: bool = False
    observational: bool = False


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


def check_figures(
    reply: str,
    retrieved_passages: Sequence[str],
    script_constants: Sequence[Any] = (),
) -> tuple[CheckResult, CheckResult]:
    """§2.1's two figure gates, measured on the reply that was **sent**.

    The delivered reply, not the composed one, and the distinction decides
    what the metric means. §19.3 has the runtime gate discard a composition
    containing an orphan figure whole and send a scripted reply instead, so
    the customer never saw the number. Scoring the discarded text would fail
    the gate on exactly the turns where the protection worked — the harder the
    gate bit, the worse the metric would read. `hallucinated_figure_rate` is
    "a fee the knowledge base never stated, reaching a student on WhatsApp",
    and reaching them is the part being counted.

    Measuring the sent text also audits the scripted replies themselves, which
    the runtime gate never inspects because nothing composed them. A figure
    typed into a script node is as unsourced as one a model invented.

    Delegates to `check_numeric_grounding` rather than reimplementing: two
    implementations would drift, and the drift presents as a green eval suite
    while production emits an orphan.

    Two results from one extraction, because the gates are separate and their
    fixes are different. An orphan means retrieval or the script failed to
    supply the figure; a hedge means generation editorialized over a figure it
    had. One number covering both says a regression happened and nothing about
    where.

    Both skip when the reply states no figure. Most education replies contain
    no number at all, and counting those as passes would report a flawless
    zero on a suite that never put a figure in front of the gate — a metric
    that looks safest exactly when it is measuring nothing.
    """
    grounding = check_numeric_grounding(reply, retrieved_passages, script_constants)
    if not grounding.reply_numbers:
        detail = "reply states no figure"
        return (
            CheckResult("hallucinated_figure", "hallucinated_figure_rate", True, detail, True),
            CheckResult("hedged_figure", "hedged_figure_rate", True, detail, True),
        )
    orphans, hedged = grounding.orphan_numbers, grounding.hedged_numbers
    return (
        CheckResult(
            name="hallucinated_figure",
            metric="hallucinated_figure_rate",
            passed=not orphans,
            detail="" if not orphans else f"no source for {', '.join(map(str, orphans))}",
        ),
        CheckResult(
            name="hedged_figure",
            metric="hedged_figure_rate",
            passed=not hedged,
            detail="" if not hedged else f"hedged a grounded {', '.join(map(str, hedged))}",
        ),
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

    **Feeds `tool_call_accuracy`, not `arithmetic_in_model_rate`.** It used to
    feed the latter, and P1b's first run showed why that was wrong: the gate
    read 60% over 15 observations, 14 of which were this check. A tool called
    with the wrong arguments is an extraction failure — it says nothing about
    whether a model did arithmetic, and the calculator may not have run at
    all. Zero tolerance belongs on the commercial incident; misreading which
    city was asked about is a different problem with a different fix, and
    gating a keyword stub produces pressure to weaken the gate rather than to
    build the extraction prompt.
    """
    if not expected:
        return CheckResult(
            name="tool_calls",
            metric="tool_call_accuracy",
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
        metric="tool_call_accuracy",
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

    Skips when nothing was presented, and when no type was resolved.

    That second skip is a correction. A turn presenting units without a
    resolved `property_type` used to fail this check, on the reasoning that
    an unverifiable turn is not a clean one. But it put an extraction miss
    inside a zero-tolerance commercial gate: re-0005 is "the unit in Noor City
    at six and a half million", which identifies a row without naming a type,
    and no substitution can have occurred where the customer named nothing to
    substitute for. The miss is still reported, by `check_type_resolved`,
    under its own tracked metric.
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
            passed=True,
            skipped=True,
            detail="no property_type resolved — see unresolved_type_rate",
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


def check_type_resolved(
    requested_type: str | None, presented_unit_ids: Sequence[str]
) -> CheckResult:
    """Did the turn know what kind of unit was being asked for?

    Split out of `check_property_type` so an extraction miss is visible
    without being counted as a substitution. Tracked rather than gated: it
    measures the extractor, and the extraction prompt does not exist yet.
    """
    if not presented_unit_ids:
        return CheckResult(
            name="type_resolved",
            metric="unresolved_type_rate",
            passed=True,
            skipped=True,
            detail="no units presented, so nothing needed resolving",
            observational=True,
        )
    resolved = requested_type is not None
    return CheckResult(
        name="type_resolved",
        metric="unresolved_type_rate",
        passed=resolved,
        detail="" if resolved else "units presented without a resolved property_type",
        observational=True,
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


def check_wrong_compound(
    expected: str | None, named_compounds: Sequence[str], snapshot: InventorySnapshot
) -> CheckResult:
    """A reply that answers about a real compound other than the one asked about.

    **The failure `invented_compound_rate` structurally cannot see.** re-0010
    asked about `كريك تاون`; Creek Town sat outside the extractor's closed
    vocabulary, so the model emitted the nearest value it was permitted to and
    the reply described a Jefaira townhouse — right price, right availability,
    wrong development. Jefaira exists, so the invented-compound gate read 0.0%
    for as long as the substitution ran.

    Graded against the case's expectation rather than the resolved slot, for
    the same reason `check_availability` reads the fixture: the slot is what
    the thing under test decided, and asking it to supply its own answer key
    is how a substitution grades itself as correct.

    Naming a *second* compound is not a substitution. §19.3's reply shape for
    no-match is "no villa in <asked> — we have one in <other>", and both
    compounds must appear. Naming only the other one is the failure.
    """
    if not expected or not named_compounds:
        return CheckResult(
            name="wrong_compound",
            metric="wrong_compound_rate",
            passed=True,
            skipped=True,
            detail="the turn pins no compound" if not expected else "reply named none",
        )
    if expected in named_compounds:
        return CheckResult(
            name="wrong_compound", metric="wrong_compound_rate", passed=True
        )
    substituted = [c for c in named_compounds if c in snapshot.compounds()]
    return CheckResult(
        name="wrong_compound",
        metric="wrong_compound_rate",
        passed=not substituted,
        detail=(
            ""
            if not substituted
            else f"asked about {expected}, reply named {', '.join(sorted(substituted))}"
        ),
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
    "check_figures",
    "check_availability",
    "check_compound_grounding",
    "check_property_type",
    "check_language",
    "check_numeric_grounding",
    "check_slots",
    "check_tool_calls",
    "check_type_resolved",
]
