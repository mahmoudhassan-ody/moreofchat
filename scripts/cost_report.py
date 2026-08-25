"""What has actually been spent, and what a tenant-month costs — from the
ledger, not from arithmetic in a chat window.

**Actual spend and derived projection are printed apart on purpose.** The first
half is rows; the second is rows times an assumption, and the assumption —
turns per conversation — is the one number here nobody has measured. It is an
input, printed as an input.

Two things this report is structurally unable to see, both stated in its own
output rather than left for the reader to notice:

- **Eval runs.** They execute against `moc_test`, whose fixtures truncate
  `usage_ledger` between tests. Every graded run's spend is destroyed at the
  end of the run that produced it, which is how an account was exhausted on
  2026-08-25 by the one activity the ledger keeps no record of.
- **Anything deleted.** Probe cleanups delete their own rows, so this is what
  survives rather than what happened.

Run with `.env` sourced:  uv run python scripts/cost_report.py
"""

import asyncio
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta

#: What a conversation is, in turns. **An assumption, not a measurement.** The
#: rehearsal script runs 2-5 turns per tenant and no real conversation has been
#: observed end to end. Printed with every projection so the number it produces
#: is never read as an observation.
TURNS_PER_CONVERSATION = 5
CONVERSATIONS = 1000


def _money(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


async def main() -> int:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings
    from moc.config_store import load

    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        rows = (
            await session.execute(
                text(
                    "SELECT kind, provider, model, input_tokens, output_tokens, "
                    "       cached_tokens, cache_write_tokens, provider_cost_usd, "
                    "       created_at, channel, degraded "
                    "FROM usage_ledger ORDER BY created_at"
                )
            )
        ).all()
    await engine.dispose()

    print("\ncost report\n")
    if not rows:
        print("  the ledger is empty — nothing to report, and that is a finding")
        return 1

    total = sum(float(r.provider_cost_usd or 0) for r in rows)
    week = datetime.now(UTC) - timedelta(days=7)
    recent = sum(float(r.provider_cost_usd or 0) for r in rows if r.created_at >= week)
    print(f"  rows            {len(rows)}")
    print(f"  window          {rows[0].created_at:%Y-%m-%d %H:%M} .. "
          f"{rows[-1].created_at:%Y-%m-%d %H:%M}")
    print(f"  all time        {_money(total)}")
    print(f"  last 7 days     {_money(recent)}")

    print("\n  by provider and model")
    by_model: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_model[(r.provider or "-", r.model or "-")].append(r)
    for (provider, model), group in sorted(
        by_model.items(), key=lambda kv: -sum(float(r.provider_cost_usd or 0) for r in kv[1])
    ):
        cost = sum(float(r.provider_cost_usd or 0) for r in group)
        tok_in = sum(r.input_tokens or 0 for r in group)
        tok_out = sum(r.output_tokens or 0 for r in group)
        print(f"    {provider:10} {model:28} {len(group):4} calls  "
              f"{tok_in:7} in  {tok_out:6} out  {_money(cost)}")

    print("\n  by kind")
    by_kind: dict[str, list] = defaultdict(list)
    for r in rows:
        by_kind[r.kind].append(r)
    for kind, group in sorted(by_kind.items(), key=lambda kv: -sum(
        float(r.provider_cost_usd or 0) for r in kv[1]
    )):
        cost = sum(float(r.provider_cost_usd or 0) for r in group)
        print(f"    {kind:16} {len(group):4} rows  {_money(cost)}")

    largest = max(rows, key=lambda r: float(r.provider_cost_usd or 0))
    print("\n  largest single line item")
    print(f"    {largest.created_at:%Y-%m-%d %H:%M:%S}  {largest.model}  "
          f"{_money(float(largest.provider_cost_usd or 0))}")
    print(f"    {largest.input_tokens} in, {largest.output_tokens} out, "
          f"{largest.cached_tokens} cached, {largest.cache_write_tokens} cache-write")
    if largest.cache_write_tokens:
        print("    the cache write is most of it, and it is an investment: a "
              "cached prefix\n    is billed at a tenth of the input rate on "
              "every turn that reuses it")

    # ── turns, and what one costs ────────────────────────────────────────────
    turns: dict[datetime, list] = defaultdict(list)
    for r in rows:
        turns[r.created_at].append(r)
    priced = {
        when: sum(float(r.provider_cost_usd or 0) for r in group)
        for when, group in turns.items()
    }
    composed = {
        when: any("sonnet" in (r.model or "") or "gpt-5.6-sol" in (r.model or "")
                  for r in group)
        for when, group in turns.items()
    }
    full = [c for when, c in priced.items() if composed[when]]
    scripted = [c for when, c in priced.items() if not composed[when]]

    print(f"\n  turns observed   {len(priced)}")
    if full:
        print(f"    composed       {len(full)}  "
              f"{_money(min(full))} .. {_money(max(full))}  "
              f"mean {_money(sum(full) / len(full))}")
    if scripted:
        print(f"    scripted       {len(scripted)}  "
              f"mean {_money(sum(scripted) / len(scripted))}   "
              f"(no composition call — a greeting, a refusal, a handoff)")

    if not full:
        print("\n  no composed turn in the ledger; a projection would be guesswork")
        return 0

    # ── the projection, and everything it assumes ────────────────────────────
    low, high = min(full), max(full)
    mean = sum(full) / len(full)
    turns_total = CONVERSATIONS * TURNS_PER_CONVERSATION
    print(f"\n  {CONVERSATIONS} conversations, one tenant, one month")
    print(f"    ASSUMED: {TURNS_PER_CONVERSATION} turns per conversation "
          f"— not measured, no real conversation has been observed end to end")
    print(f"    = {turns_total:,} turns")
    print(f"    at the cheapest observed turn  {_money(low)}   {_money(low * turns_total)}")
    print(f"    at the mean observed turn      {_money(mean)}   {_money(mean * turns_total)}")
    print(f"    at the dearest observed turn   {_money(high)}   {_money(high * turns_total)}")
    print(f"    n = {len(full)} composed turns. §2.4's rule applies to this as to "
          f"anything\n        else: three samples is a range, not a distribution.")

    # ── the price change that is already scheduled ───────────────────────────
    pricing = load("billing/pricing")["per_million_tokens"]["claude-sonnet-5"]
    if pricing["input"] < 3.00:
        factor = 3.00 / pricing["input"]
        print("\n  ⚠ composition is on introductory pricing until 2026-08-31")
        print(f"    {pricing['input']:.2f}/{pricing['output']:.2f} per Mtok now, "
              f"list is 3.00/15.00 — {factor:.1f}x")
        print("    every figure above understates from 2026-09-01 until "
              "config/billing/pricing.yaml is edited")

    await _eval_runs()

    print("\n  what this report cannot see")
    print("    deleted rows     probe cleanups remove their own; this is what survived")
    print("    provider totals  the vendor's own console is the only actual total\n")
    return 0


async def _eval_runs() -> None:
    """Graded runs, from `eval_runs` rather than from the ledger.

    They cannot be in the ledger: a run's rows live in `moc_test`, which the
    test session drops and recreates. Each run therefore summarises itself
    before that database goes away and writes one durable row here.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings

    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        runs = (
            await session.execute(
                text(
                    "SELECT suite, graded, runs, cases, turns, cost_usd, "
                    "       substituted, started_at, by_model "
                    "FROM eval_runs ORDER BY started_at"
                )
            )
        ).all()
    await engine.dispose()

    print("\n  eval runs")
    if not runs:
        print("    none recorded — every graded run before 2026-08-25 spent")
        print("    without leaving a trace, which is what this table is for")
        return
    total = sum(float(r.cost_usd) for r in runs)
    for r in runs:
        mark = "graded" if r.graded else "stage-1"
        flag = "  ⚠ SUBSTITUTED MODELS" if r.substituted else ""
        print(f"    {r.started_at:%Y-%m-%d %H:%M}  {r.suite:12} {mark:8} "
              f"{r.cases:3} cases x{r.runs}  {_money(float(r.cost_usd))}{flag}")
        if r.substituted:
            for part in r.substituted:
                print(f"        {part}")
    print(f"    {len(runs)} run(s), {_money(total)} — and this is the activity")
    print("    that exhausted an account on 2026-08-25")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
