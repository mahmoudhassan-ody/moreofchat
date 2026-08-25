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
asks *which* provider answered, and refuses when that is not the one the
routing table names as primary. A working failover is not a reason to measure.

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
_PROBE = "ok"
_MAX_TOKENS = 4


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


async def check_primaries(*, router: Any, routing: dict[str, Any]) -> None:
    """Raise unless every task's *primary* is the thing answering.

    One cheap completion per task with a primary. The cost of asking is a few
    tokens; the cost of not asking is a baseline attributed to the wrong model,
    which is worse than no baseline because it is quotable.

    A `ProviderUnavailable` from the router means no candidate answered at all,
    which is also a refusal — there is nothing to measure either way.
    """
    served: list[str] = []
    for name, spec in routing["tasks"].items():
        primary = spec.get("primary")
        if not primary:
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
        try:
            completion = await router.complete(
                task=task,
                messages=[Message(role=Role.user, content=_PROBE)],
                system=None,
            )
        except ProviderUnavailable as exc:
            served.append(f"{name}: nothing answered ({str(exc)[:80]})")
            continue
        if completion.provider != primary["provider"]:
            served.append(
                f"{name}: answered by {completion.provider}/{completion.model}, "
                f"primary is {primary['provider']}/{primary['model']}"
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
