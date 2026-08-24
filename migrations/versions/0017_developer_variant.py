"""tenants.project, sales_teams, and the lead columns on handoffs

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-24

Demo plan Task 38, design §11.2. A developer is a tenant flag, not a second
product: brokers search across projects, developers answer deeply about one and
route the lead to the right sales team.

**The flag and its value are one column.** `tenants.project` is NULL for a
broker and holds the project's name for a developer. A boolean beside a
separate name column would admit "developer, project not set", and the only
sensible readings of that state are "scope to nothing" — a bot with no stock —
or "scope to everything", which is broker behaviour with the guarantee
silently off. One nullable column makes the incoherent state unrepresentable.

**Why the scope is not a no-op even though RLS already exists.** RLS says which
tenant. Nothing said which project, and inventory does not arrive one project
at a time — it arrives as a sheet, and sheets hold a second phase, a sister
development, and the row somebody pasted in to compare against.

**A routing rule is a column on the team it routes to.** `sales_teams` carries
`property_type`, so a rule cannot name a team that does not exist: they are the
same row. Two partial unique indexes then make the two ways this goes wrong
unrepresentable rather than merely unlikely — two teams claiming the same type,
and two fallback teams, are refused by the database instead of by whichever
router happened to read them first. Neither has a behavioural signature: the
router returns *a* team either way, and the lead lands with the wrong closer.

`handoffs` gains `team`, `lead_qualified` and `lead_score`. NULL on all three
means the handoff was not a lead — a bot that ran out of clarifications is not
somebody wanting to buy a villa. `qualified leads per 100 conversations` is a
§11.2 KPI and the analytics report reads handoffs, so an unpersisted score is a
KPI nobody can compute.

Tenant-scoped, with the predicate from CLAUDE.md. One statement per
op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0017"
down_revision: Union[str, Sequence[str], None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid"


def upgrade() -> None:
    # NULL is a broker. See the module docstring for why this is one column.
    op.execute("ALTER TABLE tenants ADD COLUMN project text")

    op.execute("""CREATE TABLE sales_teams (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  team_key text NOT NULL,
  name text NOT NULL,
  -- Where a routed lead is sent. An address, not a person: people leave and
  -- the leads keep arriving.
  contact text NOT NULL,
  -- The routing rule, on the team it routes to. NULL is the fallback.
  property_type text,
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    op.execute(
        "CREATE UNIQUE INDEX uq_sales_teams_key ON sales_teams (tenant_id, team_key)"
    )
    # One team per property type. Two would make routing depend on read order.
    op.execute(
        "CREATE UNIQUE INDEX uq_sales_teams_property_type "
        "ON sales_teams (tenant_id, property_type) WHERE property_type IS NOT NULL"
    )
    # Exactly one fallback, at most. Zero is allowed and handled in code: a
    # lead that routes nowhere still reaches the inbox, because an unroutable
    # lead is not a dropped lead.
    op.execute(
        "CREATE UNIQUE INDEX uq_sales_teams_fallback "
        "ON sales_teams (tenant_id) WHERE property_type IS NULL"
    )
    op.execute("ALTER TABLE sales_teams ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE sales_teams FORCE ROW LEVEL SECURITY")
    op.execute(f"""CREATE POLICY tenant_isolation ON sales_teams
  USING ({_PREDICATE})
  WITH CHECK ({_PREDICATE})""")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON sales_teams TO moc_app")

    op.execute("ALTER TABLE handoffs ADD COLUMN team text")
    op.execute("ALTER TABLE handoffs ADD COLUMN lead_qualified boolean")
    op.execute("ALTER TABLE handoffs ADD COLUMN lead_score integer")
    # The KPI's own index: qualified leads per 100 conversations, per team.
    op.execute(
        "CREATE INDEX ix_handoffs_team ON handoffs (tenant_id, team) "
        "WHERE team IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_handoffs_team")
    op.execute("ALTER TABLE handoffs DROP COLUMN IF EXISTS lead_score")
    op.execute("ALTER TABLE handoffs DROP COLUMN IF EXISTS lead_qualified")
    op.execute("ALTER TABLE handoffs DROP COLUMN IF EXISTS team")
    op.execute("DROP TABLE IF EXISTS sales_teams")
    op.execute("ALTER TABLE tenants DROP COLUMN IF EXISTS project")
