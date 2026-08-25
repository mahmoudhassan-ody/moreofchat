"""What a run cost, aggregated once and stored where it outlives the run.

**`usage_ledger` has never seen an eval run.** They execute against `moc_test`,
which `tests/conftest.py` drops and recreates at the start of every session,
and the fixtures truncate the ledger between tests on top of that. Both are
right — a suite must not read another suite's rows, and a test database that
persists is a test database that lies — and together they meant the one
instrument built to answer "what does this cost" was blind to the activity
doing most of the spending.

The consequence arrived on 2026-08-25 as an exhausted account and a spend
report that could account for $0.017 of live traffic and nothing else.

So a run summarises its own ledger before that database goes away, and writes
one durable row to the *real* one. The cross-database write is deliberate and
is the whole point: the run's own database is disposable by design, and a
number that dies with the thing it measures is not a measurement.

**Tokens are stored beside the money.** Prices change — composition is on
introductory rates until 2026-08-31 and goes to list after — and a run recorded
only in dollars cannot be repriced. A run recorded in tokens can.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

#: `(model, provider, input, output, cached, cache_write, usd)` — the shape a
#: ledger row reduces to. A tuple rather than the ORM object so this module
#: does not need a session to be tested, and so a caller can summarise rows it
#: read any way it likes.
LedgerRow = tuple[str | None, str | None, int, int, int, int, float]


@dataclass(frozen=True)
class Spend:
    total_usd: float
    calls: int
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    def render(self) -> str:
        if not self.by_model:
            return "no priced calls — the run reached no provider"
        lines = [f"${self.total_usd:,.4f} over {self.calls} calls"]
        for model, row in sorted(
            self.by_model.items(), key=lambda kv: -kv[1]["usd"]
        ):
            lines.append(
                f"  {model:30} {row['calls']:4} calls  "
                f"{row['input']:8,} in  {row['output']:6,} out  "
                f"${row['usd']:,.4f}"
            )
        return "\n".join(lines)


def summarise(rows: list[LedgerRow]) -> Spend:
    """Fold a run's priced ledger rows into one durable summary.

    An empty run is zero and zero calls, which is a different statement from
    "not measured": it is what a run that errored every case before reaching a
    provider actually cost, and that shape is exactly the 2026-08-25 failure.
    """
    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0, "input": 0, "output": 0,
            "cached": 0, "cache_write": 0, "usd": 0.0, "provider": None,
        }
    )
    total = 0.0
    calls = 0
    for model, provider, tokens_in, tokens_out, cached, cache_write, usd in rows:
        if model is None:
            # `message_in`/`message_out` are unpriced by design and carry no
            # model. Counting them as calls would inflate the denominator of
            # every per-call figure downstream.
            continue
        entry = by_model[model]
        entry["calls"] += 1
        entry["provider"] = provider
        entry["input"] += tokens_in or 0
        entry["output"] += tokens_out or 0
        entry["cached"] += cached or 0
        entry["cache_write"] += cache_write or 0
        entry["usd"] += float(usd or 0)
        total += float(usd or 0)
        calls += 1
    return Spend(total_usd=total, calls=calls, by_model=dict(by_model))


def not_primary(spend: Spend, *, routing: dict[str, Any]) -> list[str]:
    """Models this run used that no task names as its primary.

    Derived from what actually ran rather than from whether a check fired.
    The first version took this from the headroom check, which only runs when
    grading — so a stage-1 run on failover recorded nothing while `by_model`
    plainly showed OpenAI, and a durable row that needs a second column read to
    be understood is one that gets quoted without it.
    """
    primaries = {
        spec["primary"]["model"]
        for spec in routing["tasks"].values()
        if spec.get("primary")
    }
    return sorted(model for model in spend.by_model if model not in primaries)


async def collect(session: Any) -> Spend:
    """Summarise every priced row this run's database holds.

    Called before the run's database is torn down. No tenant filter: the run
    owns the database, and a suite that seeded two tenants spent on both.
    """
    from sqlalchemy import text

    rows = (
        await session.execute(
            text(
                "SELECT model, provider, coalesce(input_tokens, 0), "
                "       coalesce(output_tokens, 0), coalesce(cached_tokens, 0), "
                "       coalesce(cache_write_tokens, 0), "
                "       coalesce(provider_cost_usd, 0) "
                "FROM usage_ledger"
            )
        )
    ).all()
    return summarise([tuple(row) for row in rows])


async def record(
    *,
    spend: Spend,
    suite: str,
    graded: bool,
    runs: int,
    cases: int,
    turns: int,
    started_at: datetime,
    substituted: list[str] | None = None,
) -> uuid.UUID:
    """Write the summary to the durable database, not the run's own.

    `substituted` names any task a different provider answered — a run served
    by failover is not a baseline, and a row that does not say so would be
    quoted as one later.
    """
    import json

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings
    from moc.config_store import load

    # Derived here rather than taken on trust: the caller may not have run a
    # headroom check at all, and the row still has to say what answered.
    used = not_primary(spend, routing=load("llm/routing"))
    if used and not substituted:
        substituted = [f"ran on {model}, which no task names as primary" for model in used]

    run_id = uuid.uuid4()
    engine = create_async_engine(settings.database_url)
    try:
        async with AsyncSession(engine) as session:
            await session.execute(
                text(
                    "INSERT INTO eval_runs (id, suite, graded, runs, cases, turns, "
                    "  cost_usd, by_model, substituted, started_at) "
                    "VALUES (:id, :suite, :graded, :runs, :cases, :turns, :cost, "
                    "  cast(:by_model as jsonb), cast(:substituted as jsonb), :started)"
                ),
                {
                    "id": run_id,
                    "suite": suite,
                    "graded": graded,
                    "runs": runs,
                    "cases": cases,
                    "turns": turns,
                    "cost": spend.total_usd,
                    "by_model": json.dumps(spend.by_model),
                    "substituted": json.dumps(substituted) if substituted else None,
                    "started": started_at,
                },
            )
            await session.commit()
    finally:
        await engine.dispose()
    return run_id


__all__ = ["LedgerRow", "Spend", "collect", "not_primary", "record", "summarise"]
