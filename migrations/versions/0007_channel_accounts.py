"""channel_accounts and the moc_lookup bootstrap

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-19

**The one table whose RLS cannot carry the whole boundary.**

CLAUDE.md's rule is that every tenant-scoped table gets RLS with FORCE and the
nullif-guarded predicate. `channel_accounts` keeps all of that — but it cannot
be the only protection, because the resolving read runs *before* a tenant
context exists. Establishing which tenant an inbound message belongs to is the
query's whole purpose, and under the standard predicate with nothing set,
`nullif` yields NULL, the comparison yields NULL, and the query returns no
rows. Setting the tenant first requires the answer.

So resolution goes through a second door: `moc_lookup`, a login role whose
only privilege in the database is SELECT on `channel_account_lookup`, a view
exposing the five columns resolution needs. Not the signing secret, not the
base table, not any other table.

Two things make that safe rather than a hole in the wall:

- The view is `security_invoker = false` (the default, stated here because it
  is load-bearing). It executes as its owner, so the owner's RLS treatment
  applies rather than `moc_lookup`'s — which is what lets a role with no rows
  visible to it read the rows it needs. Should the view ever be re-owned to a
  non-superuser, FORCE RLS applies to that owner and the view returns nothing:
  resolution fails and inbound traffic is refused. That is the safe direction
  for this to break in.
- The column list is asserted in `tests/tenancy/test_channel_accounts.py`. An
  ALTER adding a column here is a privilege escalation with no other signal —
  the feature still works and every other test still passes.

`signing_secret` lives on the base table and never in the view. `secret_ref`
names a secret; it is not one. The role that runs before authentication must
not be able to reach the value an attacker would need in order to forge that
tenant's inbound traffic.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid"

#: The view's whole surface. Kept as a list so the migration and the test read
#: the same shape, and so adding one is visibly a change to a security object.
_RESOLVING_COLUMNS = ("id", "tenant_id", "channel", "address", "secret_ref")


def upgrade() -> None:
    op.execute("""CREATE TABLE channel_accounts (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  channel text NOT NULL,
  address text NOT NULL,
  secret_ref text NOT NULL,
  signing_secret text,
  display_name text,
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    op.execute("CREATE INDEX ix_channel_accounts_tenant_id ON channel_accounts (tenant_id)")

    # Global, not per tenant. An inbound message carries an address and nothing
    # else; if two tenants could claim the same one, resolution would have to
    # guess which tenant a customer is talking to.
    op.execute(
        "CREATE UNIQUE INDEX uq_channel_accounts_address "
        "ON channel_accounts (channel, address)"
    )

    op.execute("ALTER TABLE channel_accounts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE channel_accounts FORCE ROW LEVEL SECURITY")
    op.execute(f"""CREATE POLICY tenant_isolation ON channel_accounts
  USING ({_PREDICATE})
  WITH CHECK ({_PREDICATE})""")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON channel_accounts TO moc_app")

    # ─────────────────────── the bootstrap door ───────────────────────

    op.execute(
        "CREATE VIEW channel_account_lookup AS SELECT {} FROM channel_accounts".format(
            ", ".join(_RESOLVING_COLUMNS)
        )
    )
    # Stated rather than left to the default: flipping this to true would make
    # the view run as the caller, and `moc_lookup` has no privilege on the base
    # table — resolution would stop working. Fail-closed, but worth naming so
    # the choice is not mistaken for an oversight.
    op.execute("ALTER VIEW channel_account_lookup SET (security_invoker = false)")
    op.execute(
        "COMMENT ON VIEW channel_account_lookup IS "
        "'Pre-tenant bootstrap read for moc_lookup. Adding a column here widens "
        "what an unauthenticated resolution path can read - see migration 0007.'"
    )

    op.execute("""DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'moc_lookup') THEN
    CREATE ROLE moc_lookup LOGIN PASSWORD 'CHANGE_ME_VIA_ENV';
  END IF;
END $$""")
    op.execute("GRANT USAGE ON SCHEMA public TO moc_lookup")
    op.execute("GRANT SELECT ON channel_account_lookup TO moc_lookup")


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS channel_account_lookup")
    op.execute("DROP TABLE IF EXISTS channel_accounts")
    # The role is not dropped: it may own privileges in another database on the
    # same cluster, and DROP ROLE fails on any that remain. Revoking is the
    # reversible half.
    op.execute("REVOKE USAGE ON SCHEMA public FROM moc_lookup")
