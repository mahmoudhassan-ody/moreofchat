"""Grading inventory turns, and reporting the five gates — harness §3.2, §5.1.

**Unmeasured is not zero, and this module exists because that was not true.**

`arithmetic_in_model_rate`, `type_substitution_rate`, `invented_compound_rate`
and `sold_unit_offered_rate` are zero-tolerance gates in
`config/evals/gates.yaml`; `asof_disclosure_rate` sits at 0.98. Every one of
them had zero observations for as long as it existed — not passing, never run.
A gate nothing feeds cannot fail, and a report that renders it as 0.0% is
worse than one that says nothing, because 0.0% is what success looks like.

So `GateResult` carries its observation count, and a gate with none renders
"not measured" rather than a rate. The distinction is the whole deliverable.

**The snapshot is eval ground truth, read from the fixture.** Not from the
connector: the connector cannot return a sold unit, so asking it for the
status map would be asking the thing under test to grade itself. The fixture
file is to inventory what `gold_chunks` are to retrieval — the answer key,
which never reaches the agent.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moc.agent.state import ConversationState
from moc.evals.deterministic import (
    CheckResult,
    InventorySnapshot,
    ToolCall,
    check_action,
    check_asof_disclosure,
    check_availability,
    check_compound_grounding,
    check_language,
    check_property_type,
    check_slots,
    check_tool_calls,
    check_type_resolved,
)
from moc.evals.schema import EvalCase, Turn

#: Every gate this module is responsible for feeding. Listed so one dropped
#: from the report fails a test rather than vanishing from a summary.
GATES = (
    "arithmetic_in_model_rate",
    "type_substitution_rate",
    "invented_compound_rate",
    "sold_unit_offered_rate",
    "asof_disclosure_rate",
)

#: Reported alongside the gates, never gated. Both measure slot extraction,
#: which is a documented keyword stub until the extraction prompt exists —
#: and both were split out of zero-tolerance gates they were distorting
#: (2026-08-19). Gating a stub produces pressure to weaken the gate rather
#: than to build the prompt.
TRACKED = ("tool_call_accuracy", "unresolved_type_rate")

#: Gates where the metric counts failures (`direction: max` in gates.yaml).
#: `asof_disclosure_rate` is the one that counts successes, and rendering it
#: the same way would invert it.
_FAILURE_RATE = {
    "arithmetic_in_model_rate",
    "type_substitution_rate",
    "invented_compound_rate",
    "sold_unit_offered_rate",
    "unresolved_type_rate",
}


@dataclass(frozen=True)
class GateResult:
    gate: str
    observations: int
    failures: int

    @property
    def rate(self) -> float | None:
        """None when nothing fed it. Not 0.0 — that is the number a passing
        gate shows, and the two must never render alike."""
        if not self.observations:
            return None
        if self.gate in _FAILURE_RATE:
            return self.failures / self.observations
        return (self.observations - self.failures) / self.observations

    def render(self) -> str:
        if self.rate is None:
            return f"{self.gate:26} not measured  (0 observations)"
        return f"{self.gate:26} {self.rate:>6.1%}  ({self.observations} observations)"


@dataclass(frozen=True)
class InventoryTurnOutcome:
    turn_index: int
    reply: str
    action: str
    state: ConversationState
    tool_calls: tuple[Any, ...] = ()
    presented_unit_ids: tuple[str, ...] = ()
    named_compounds: tuple[str, ...] = ()
    passages: tuple[str, ...] = ()
    computation: Any = None
    checks: tuple[CheckResult, ...] = ()

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


@dataclass(frozen=True)
class InventoryCaseOutcome:
    case_id: str
    vertical: str
    category: str
    turns: tuple[InventoryTurnOutcome, ...] = ()
    errored: bool = False
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.errored and bool(self.turns) and all(t.passed for t in self.turns)


def snapshot_from_fixture(path: Path) -> InventorySnapshot:
    """The answer key: every unit's status, type and compound.

    Includes sold and reserved rows, which is the point — a snapshot holding
    only what the connector will return could never detect an unavailable unit
    being offered.
    """
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return InventorySnapshot(
        fixture=rows[0]["fixture"] if rows else "",
        as_of=rows[0]["as_of"] if rows else "",
        unit_status={row["unit_id"]: row["availability"] for row in rows},
        unit_type={row["unit_id"]: row["property_type"] for row in rows},
        unit_compound={
            row["unit_id"]: row["compound"] for row in rows if row.get("compound")
        },
    )


class InventoryCaseRunner:
    """Drives `grounding_mode: inventory` cases and grades them.

    Separate from `CaseRunner` rather than a branch inside it. The two paths
    share no grounding mode, no tool surface and no notion of a passage, and
    folding them together would mean a document case could accidentally be
    graded against an inventory snapshot — or, worse, an inventory price
    grounded on a retrieved passage, which is the failure the fixture is kept
    out of `kb_chunks` to prevent.
    """

    def __init__(self, *, agent: Any, snapshot: InventorySnapshot, script: str) -> None:
        self._agent = agent
        self.snapshot = snapshot
        self._script = script

    async def run(self, cases: Sequence[EvalCase]) -> list[InventoryCaseOutcome]:
        return [await self.run_case(case) for case in cases]

    async def run_case(self, case: EvalCase) -> InventoryCaseOutcome:
        from moc.agent.script_engine import ScriptEngine

        state = ScriptEngine.from_config(self._script).start()
        turns: list[InventoryTurnOutcome] = []

        for index, turn in enumerate(case.turns):
            try:
                result = await self._agent.handle(state=state, text=turn.user)
            except Exception as exc:  # noqa: BLE001 - an error is not a quality signal
                return InventoryCaseOutcome(
                    case_id=case.id,
                    vertical=str(case.vertical),
                    category=str(case.category),
                    turns=tuple(turns),
                    errored=True,
                    error=repr(exc)[:500],
                )
            state = result.state
            turns.append(
                InventoryTurnOutcome(
                    turn_index=index,
                    reply=result.reply,
                    action=str(result.action),
                    state=state,
                    tool_calls=result.tool_calls,
                    presented_unit_ids=result.presented_unit_ids,
                    named_compounds=result.named_compounds,
                    computation=result.computation,
                    checks=tuple(self._checks(turn, result)),
                )
            )

        return InventoryCaseOutcome(
            case_id=case.id,
            vertical=str(case.vertical),
            category=str(case.category),
            turns=tuple(turns),
        )

    def _checks(self, turn: Turn, result: Any) -> list[CheckResult]:
        """Every check an inventory turn feeds, including all five gates."""
        checks = [
            check_action(turn.expected_action, result.action),
            check_language(result.reply, turn.expected_lang),
            check_slots(turn.expected_slots, result.state.slots),
            check_tool_calls(
                turn.expected_tool_calls,
                [ToolCall(name=c.name, args=dict(c.args)) for c in result.tool_calls],
            ),
            check_availability(result.presented_unit_ids, self.snapshot),
            check_property_type(
                result.state.slots.get("property_type"),
                result.presented_unit_ids,
                self.snapshot,
            ),
            check_type_resolved(
                result.state.slots.get("property_type"), result.presented_unit_ids
            ),
            check_compound_grounding(result.named_compounds, self.snapshot),
            check_asof_disclosure(
                result.reply, self.snapshot, required=turn.expected_asof_disclosure
            ),
        ]
        if turn.expected_computation is not None:
            checks.append(self.check_computation(result.reply, result.computation))
        return checks

    def check_computation(self, reply: str, computation: Any) -> CheckResult:
        """Every figure in the reply must string-match the calculator's output.

        Not "close". 302,344 against 302,343 is a failure, and that one-EGP gap
        is the whole demonstration that a tool produced the number rather than
        a model dividing.
        """
        if computation is None:
            return CheckResult(
                name="computation",
                metric="arithmetic_in_model_rate",
                passed=False,
                detail="the case pins a computation and the turn made none",
            )
        allowed = {
            str(value)
            for value in computation.to_json().values()
            if isinstance(value, int)
        }
        allowed |= {f"{int(value):,}" for value in allowed if value.isdigit()}
        import re

        stated = set(re.findall(r"[\d,]*\d", reply))
        orphans = sorted(stated - allowed - _dates(reply))
        return CheckResult(
            name="computation",
            metric="arithmetic_in_model_rate",
            passed=not orphans,
            detail="" if not orphans else f"not in calculator output: {', '.join(orphans)}",
        )


def _dates(reply: str) -> set[str]:
    """A snapshot date is a disclosure, not a computed figure. Excluded so
    `as_of` does not read as arithmetic the model invented."""
    import re

    parts: set[str] = set()
    for match in re.findall(r"\d{4}-\d{2}-\d{2}", reply):
        parts |= set(match.split("-")) | {match}
    return parts


def gate_report(outcomes: Sequence[InventoryCaseOutcome]) -> dict[str, GateResult]:
    """One entry per gate, always, with its observation count.

    Always all five gates plus the tracked metrics, even from an empty run: a
    gate absent from a report reads as a gate that passed, and this whole
    module exists because five of them read that way for a month.
    """
    counts: dict[str, list[int]] = {name: [0, 0] for name in (*GATES, *TRACKED)}
    for outcome in outcomes:
        for turn in outcome.turns:
            for check in turn.checks:
                if check.metric not in counts or check.skipped:
                    continue
                counts[check.metric][0] += 1
                counts[check.metric][1] += 0 if check.passed else 1
    return {
        name: GateResult(gate=name, observations=seen, failures=failed)
        for name, (seen, failed) in counts.items()
    }


__all__ = [
    "GATES",
    "TRACKED",
    "GateResult",
    "InventoryCaseOutcome",
    "InventoryCaseRunner",
    "InventoryTurnOutcome",
    "gate_report",
    "snapshot_from_fixture",
]
