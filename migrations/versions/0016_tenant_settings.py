"""tenant_settings, tenant_scripts and settings_audit — the console's editable tier

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-22

Demo plan Task 33. §19's config layering has always named a tenant tier; this
is the table it lives in.

**Two settings to start, and the pair is the point.** `min_score` may be
raised and never lowered — a tenant whose corpus separates cleanly can make
retrieval stricter, and a change in the other direction produces no error, no
log and no failing test, just more answers with some of them wrong. `synonyms`
is unbounded because a tenant's own vocabulary cannot be wrong.

**Synonyms are here rather than in the search index because the index is
shared.** One Meilisearch index per vertical serves every tenant in it, so a
synonym written into the index settings is one broker's word for an area
changing another broker's ranking. They are applied to the query instead,
which is the only place a shared index can be scoped per tenant.

`settings_audit` answers "the bot got worse yesterday" — the same question
§19.4's run metadata answers for eval runs, arriving through the console
instead of through a deploy. Old and new value both, because "someone raised
it" and "someone raised it from the floor" are different facts and only the
second says whether the change mattered.

Values are text. They are read as JSON by one module and never compared,
summed or indexed inside the database; typed columns would be four columns
where a tenant setting can be a number, a string, a flag or a map.

Both tables tenant-scoped, with the predicate from CLAUDE.md. One statement
per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0016"
down_revision: Union[str, Sequence[str], None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid"
_TABLES = ("tenant_settings", "settings_audit", "tenant_scripts")


def upgrade() -> None:
    op.execute("""CREATE TABLE tenant_settings (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  setting text NOT NULL,
  -- JSON text. See the module docstring: a tenant setting can be a number, a
  -- string, a flag or a map, and nothing in the database reads inside it.
  value text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
)""")
    op.execute(
        "CREATE UNIQUE INDEX uq_tenant_settings ON tenant_settings (tenant_id, setting)"
    )

    op.execute("""CREATE TABLE settings_audit (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  setting text NOT NULL,
  old_value text,
  new_value text NOT NULL,
  -- Who. Not nullable: a settings change with no author is an audit row that
  -- answers half the question it exists for.
  agent_id text NOT NULL,
  changed_at timestamptz NOT NULL DEFAULT clock_timestamp()
)""")
    op.execute(
        "CREATE INDEX ix_settings_audit_tenant ON settings_audit (tenant_id, changed_at DESC)"
    )

    op.execute("""CREATE TABLE tenant_scripts (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  script_id text NOT NULL,
  version integer NOT NULL,
  body jsonb NOT NULL,
  -- NULL means draft. A draft is a version that exists and has never been
  -- reachable by a customer, which is what makes "preview before publishing"
  -- a state rather than a promise.
  published_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    op.execute(
        "CREATE UNIQUE INDEX uq_tenant_scripts ON tenant_scripts "
        "(tenant_id, script_id, version)"
    )

    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""CREATE POLICY tenant_isolation ON {table}
  USING ({_PREDICATE})
  WITH CHECK ({_PREDICATE})""")

    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON tenant_settings TO moc_app")
    # No DELETE on scripts: a published version a conversation is pinned to
    # must still be loadable, and deleting one strands every conversation
    # holding it at a node that no longer exists.
    op.execute("GRANT SELECT, INSERT, UPDATE ON tenant_scripts TO moc_app")
    # No UPDATE and no DELETE on the audit. An audit an application can rewrite
    # is a log, not an audit, and the one time it matters is the one time
    # somebody wants it changed.
    op.execute("GRANT SELECT, INSERT ON settings_audit TO moc_app")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tenant_scripts")
    op.execute("DROP TABLE IF EXISTS settings_audit")
    op.execute("DROP TABLE IF EXISTS tenant_settings")
