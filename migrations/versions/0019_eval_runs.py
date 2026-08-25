"""eval_runs — what a graded run actually cost, where it survives the run

Revision ID: 0019
Revises: 0018
Create Date: 2026-08-25

`usage_ledger` answers "what does a conversation cost" and has never once seen
an eval run. Graded runs execute against `moc_test`, which `tests/conftest.py`
**drops and recreates** at the start of every session — so the ledger rows a
run writes are destroyed before anyone could read them, and the fixtures
truncate the table between tests on top of that.

The consequence was not theoretical. An account was exhausted on 2026-08-25 by
graded runs, and the spend report could account for $0.017 of live traffic and
nothing else. Roughly seven dollars over three weeks was measurement nobody
recorded, by the one instrument built to record spend.

**Not tenant-scoped, and therefore no RLS.** CLAUDE.md's rule is about tables
holding a `tenant_id`; this holds none. A run is platform telemetry — it
belongs to the harness, not to a customer — and giving it a tenant column to
satisfy a rule it is outside of would invent an owner for a row that has none.
It is also why it is absent from `TENANT_SCOPED_TABLES`: the truncation list is
checked against the catalogue of tenant-scoped tables, and a row here surviving
between tests is the entire point.

One row per run rather than per call. The per-call detail is `usage_ledger`'s
job and it does it well for the turns it sees; what was missing is a durable
total that outlives a disposable database. `by_model` is jsonb for the same
reason `provenance` is: the set of models in a run is not fixed, and a column
per model would need a migration every time routing changes.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0019"
down_revision: Union[str, Sequence[str], None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE eval_runs (
            id uuid PRIMARY KEY,
            suite text NOT NULL,
            graded boolean NOT NULL,
            runs integer NOT NULL,
            cases integer NOT NULL,
            turns integer NOT NULL,
            cost_usd numeric(12, 6) NOT NULL,
            by_model jsonb NOT NULL,
            substituted jsonb,
            started_at timestamptz NOT NULL,
            finished_at timestamptz NOT NULL DEFAULT clock_timestamp()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE eval_runs")
