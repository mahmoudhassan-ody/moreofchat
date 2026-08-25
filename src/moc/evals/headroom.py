"""Whether a graded run can honestly finish before it starts.

Written after two failures an hour apart on 2026-08-25, the second caused by
the fix for the first.

**A graded run exhausted the account.** Nineteen cases over three runs, and the
budget went partway through run 1. Runs 2 and 3 then put thirty-eight more
cases to a provider that had already refused, and reported `0.0% (0 scored, 19
errored)` twice. Nothing checked what the run would cost, nothing compared it
to what was left, and nothing stopped after the first refusal.

**Then quota exhaustion became `ProviderUnavailable`,** because a spend cap is
exactly what failover exists for. Right for a customer's turn — the reply still
goes out — and wrong for a measurement. A graded run whose primary is exhausted
no longer errors; it completes on the failover and reports a number for a model
that never ran. §2.3 pins `prompt_version` and `config_hash` so a run stays
comparable, and a silently substituted model defeats both without touching
either.

So `check_primaries` asks a different question from "is the provider up". It
asks *which* provider answered, and refuses when that is not the one the run
will actually use. A working failover is not a reason to measure.

"The one the run will actually use" is not always the primary. §5.2 forbids a
provider from grading its own output, so `Judge.grade` excludes whichever
provider composed — and with composition on Anthropic the judge is on
`gpt-5.6-sol` for every turn of every graded run, while `eval_grading`'s
primary is Opus. The first version of this check probed the primary, reported
it healthy, and described a model the run would not call once.

What this module deliberately does **not** do is ask the vendor how much budget
is left. Neither provider exposes it, and a check built on a number nobody
publishes would be a check that quietly stops working. The honest substitute is
a projection from what turns have actually cost, printed next to the run about
to be started, so the person starting it is the one who decides.
"""

from dataclasses import dataclass
from typing import Any

from moc.llm.base import Message, ProviderUnavailable, Role, Task

#: Enough to get a completion and no more. This runs before a suite that costs
#: dollars, and a check that is itself expensive is one people switch off.
#:
#: It was declared and never passed, so every probe inherited its task's
#: ceiling: `eval_grading` carries 2048 with `effort: high`, and the word "ok"
#: cost $0.0035 of a $0.0042 check — 84% of it, on the one task whose answer is
#: thrown away. 16 rather than 4 because both vendors floor `max_tokens` on a
#: reasoning model, and a probe that 400s reads as an outage.
_PROBE = "ok"
_MAX_TOKENS = 16

#: The task whose answering provider §5.2 bars from grading, and the task it
#: bars it from. Named rather than spelled inline: the coupling between them is
#: the thing this module kept getting wrong.
_ANSWERING = "answer_composition"
_JUDGE = "eval_grading"


class NoHeadroom(RuntimeError):
    """The run must not start: it would not measure what it claims to."""


@dataclass(frozen=True)
class Estimate:
    """What a run is about to cost, from what turns have actually cost.

    `low`/`high` rather than a mean alone: the observed spread between a turn
    that writes a prompt cache and one that reuses it is more than 1.5x, and a
    single figure hides which end a run is likely to land on.
    """

    turns: int
    samples: int
    low: float | None
    high: float | None

    def render(self) -> str:
        if self.samples == 0:
            return (
                f"{self.turns} turns, cost not measured — no priced turn in the "
                f"ledger to project from"
            )
        return (
            f"{self.turns} turns, ${self.low:,.2f}-${self.high:,.2f} "
            f"(n={self.samples} measured turns)"
        )


def projected_cost(*, turn_costs: list[float], turns: int) -> Estimate:
    """What `turns` turns will cost, given what turns have cost.

    Returns an `Estimate` with `None` bounds rather than zero when nothing has
    been measured. §2.4's distinction: a run nobody has priced is not a free
    run, and reporting `$0.00` would read as one.
    """
    priced = [c for c in turn_costs if c]
    if not priced:
        return Estimate(turns=turns, samples=0, low=None, high=None)
    return Estimate(
        turns=turns,
        samples=len(priced),
        low=min(priced) * turns,
        high=max(priced) * turns,
    )


#: The logical config key. Named once rather than spelled at each use, so the
#: error message and the loader cannot disagree about which file is meant.
_BUDGET = "evals/budget"


def ceiling_usd() -> float:
    """What one invocation may spend. §19: it changes with whose budget pays."""
    from moc.config_store import load

    return float(load(_BUDGET)["ceiling_usd"])


def within_budget(estimate: Estimate, *, ceiling_usd: float) -> None:
    """Raise unless the run's **worst case** fits under the ceiling.

    The high end rather than the mean, because a ceiling that admits a run on
    its average and overshoots on its spread fires after the money is gone.

    An uncosted run is refused rather than waved through. `unmeasured is not
    zero` — a run nobody can price is not a free one, and treating it as one is
    how the ledger came to hold no record of the spending that mattered.
    """
    if estimate.high is None:
        raise NoHeadroom(
            f"this run's cost is not measured — {estimate.render()}; "
            f"nothing to compare against the ${ceiling_usd:,.2f} ceiling"
        )
    if estimate.high > ceiling_usd:
        raise NoHeadroom(
            f"this run could cost ${estimate.high:,.2f}, over the "
            # The logical key, not the path. `test_config_store_is_the_only_
            # reader_of_the_config_directory` scans for the directory by name
            # and cannot tell a message from a read — and the key is what
            # survives §19's move of this tier into the database anyway.
            f"${ceiling_usd:,.2f} ceiling in the {_BUDGET!r} config "
            f"({estimate.render()})"
        )


def _expected(spec: dict[str, Any], *, excluded: str | None) -> dict[str, Any] | None:
    """Which candidate should answer, given who is not allowed to.

    For every task but the judge this is simply the primary. For the judge it
    is the first candidate §5.2 leaves standing — which is not the primary
    whenever composition runs on Anthropic, i.e. always, in the configuration
    every baseline so far was measured under.
    """
    candidates = [spec["primary"]]
    if spec.get("failover"):
        candidates.append(spec["failover"])
    if excluded is not None:
        candidates = [c for c in candidates if c["provider"] != excluded]
    return candidates[0] if candidates else None


async def check_primaries(*, router: Any, routing: dict[str, Any]) -> None:
    """Raise unless every task is answered by the model the run will use.

    One cheap completion per task with a primary, capped at `_MAX_TOKENS` — the
    cost of asking is a few tokens; the cost of not asking is a baseline
    attributed to the wrong model, which is worse than no baseline because it
    is quotable.

    **The judge is not checked against its primary.** `Judge.grade` passes
    `exclude_provider=<the provider that composed>`, so with composition on
    Anthropic the judge lands on `gpt-5.6-sol` every single time and Opus never
    grades a turn. Probing `eval_grading` without that exclusion reported
    `anthropic/claude-opus-5` serving: true, green, and about a model the run
    it was gating would not call once.

    The excluded provider is read from `answer_composition`'s configured
    primary rather than from what just answered. Those differ only when
    composition is itself substituted — and that is already a refusal on the
    line above, so the judge's verdict changes nothing.

    A `ProviderUnavailable` from the router means no candidate answered at all,
    which is also a refusal — there is nothing to measure either way.
    """
    served: list[str] = []
    composer = routing["tasks"][_ANSWERING]["primary"]["provider"]
    for name, spec in routing["tasks"].items():
        if not spec.get("primary"):
            continue
        if not spec.get("max_tokens"):
            # Not a completion task. `embedding` is routed and has a primary
            # like everything else, and the router reads `max_tokens` off the
            # spec — so probing it here is a KeyError rather than a check.
            continue
        try:
            task = Task(name)
        except ValueError:
            # A routed task with no `Task` member is not one a suite exercises.
            continue
        excluded = composer if name == _JUDGE else None
        expected = _expected(spec, excluded=excluded)
        if expected is None:
            served.append(
                f"{name}: §5.2 bars {composer!r} and nothing else is routed — "
                f"a graded run structurally needs two healthy providers"
            )
            continue
        try:
            completion = await router.complete(
                task=task,
                messages=[Message(role=Role.user, content=_PROBE)],
                system=None,
                max_tokens=_MAX_TOKENS,
                exclude_provider=excluded,
            )
        except ProviderUnavailable as exc:
            served.append(f"{name}: nothing answered ({str(exc)[:80]})")
            continue
        if completion.provider != expected["provider"]:
            because = (
                f"§5.2 puts the judge on {expected['provider']}/{expected['model']}"
                if excluded is not None
                else f"primary is {expected['provider']}/{expected['model']}"
            )
            served.append(
                f"{name}: answered by {completion.provider}/{completion.model}, "
                f"{because}"
            )
    if served:
        raise NoHeadroom(
            "a graded run would not measure the configured models — "
            + "; ".join(served)
        )


__all__ = [
    "Estimate",
    "NoHeadroom",
    "ceiling_usd",
    "check_primaries",
    "projected_cost",
    "within_budget",
]
