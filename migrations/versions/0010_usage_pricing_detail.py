"""usage_ledger: cache writes and the pricing tier

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-21

Two columns the billing table needed and did not have.

**`cache_write_tokens`.** `cached_tokens` is cache *reads* on both adapters,
and a read is a tenth of base input where a write is a quarter more than it.
One column for both was wrong in whichever direction the turn happened to go,
and this workload caches the tenant script and policy preamble on every
composed turn — so it was wrong on almost every row.

**`pricing_tier`.** OpenAI bills 2x input and 1.5x output for a request whose
prompt exceeds 272K tokens, for the whole request rather than the excess. Which
side of that line a row fell on cannot be recovered later from its token
counts, because the threshold is a vendor policy that moves — so it is
recorded at write time or not at all.

Nullable with no default, deliberately. Every row written before this migration
was priced under a table that had neither, and a backfilled zero or a
'short' would state something about those rows that nobody measured. NULL is
the honest value for "this predates the column", and it is the same rule
`provider_cost_usd` already follows for a model with no confirmed rate.

No RLS change: the policy is on the table, not per column, and adding a column
does not widen it. One statement per op.execute() — asyncpg rejects
multi-command strings.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE usage_ledger "
        "ADD COLUMN cache_write_tokens integer NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE usage_ledger ADD COLUMN pricing_tier text")


def downgrade() -> None:
    op.execute("ALTER TABLE usage_ledger DROP COLUMN pricing_tier")
    op.execute("ALTER TABLE usage_ledger DROP COLUMN cache_write_tokens")
