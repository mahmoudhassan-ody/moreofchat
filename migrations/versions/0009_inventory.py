"""inventory_units — structured inventory (§3.2)

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-19

**Why this is a table and not chunks.** P1 deliberately kept the broker
fixture out of `kb_chunks`, and this is the other half of that decision. §3.2
answers a unit's price from a *row*: filtered on availability, carrying an
`as_of`, with the payment terms attached. A passage carrying the same price
bypasses both filters and nothing raises — `sold_unit_offered_rate` reads zero
while a sold unit's price sits in a retrieved chunk.

`availability` is a plain column and the filter lives in
`retrieval/inventory.py`, which is the only module that reads this table and
has no expression that reads it unfiltered. A CHECK constraint cannot express
"never SELECT these rows", so the guarantee is structural in the code and
tested there.

`as_of` is a column, not a constant. Two tenants ingest on different days, and
a snapshot date baked into config would attach one tenant's freshness to
another's inventory.

Tenant-scoped, with the nullif-guarded predicate from CLAUDE.md. One statement
per op.execute() — asyncpg rejects multi-command strings.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PREDICATE = "tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.execute("""CREATE TABLE inventory_units (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  unit_id text NOT NULL,
  fixture text,
  as_of date NOT NULL,
  title text,
  listing_kind text,
  property_type text NOT NULL,
  compound text,
  area text,
  city text,
  price bigint NOT NULL,
  currency text NOT NULL,
  unit_area_sqm numeric,
  bedrooms integer,
  bathrooms integer,
  finish text,
  furnished boolean,
  availability text NOT NULL,
  delivery_date date,
  project_status text,
  payment_plan jsonb,
  address text,
  source_row integer,
  ingested_at timestamptz NOT NULL DEFAULT now()
)""")
    # Stable per tenant, so re-ingesting a snapshot updates rather than
    # accumulating a second copy. Two rows for one unit is one unit offered
    # twice, which is the failure this whole table is filtered to prevent.
    op.execute(
        "CREATE UNIQUE INDEX uq_inventory_units_tenant_unit "
        "ON inventory_units (tenant_id, unit_id)"
    )
    # The shape every search has: this tenant's available rows, narrowed by
    # city and type. Availability leads the trailing columns because it is on
    # every query by construction.
    op.execute(
        "CREATE INDEX ix_inventory_units_search "
        "ON inventory_units (tenant_id, availability, city, property_type)"
    )
    op.execute(
        "CREATE INDEX ix_inventory_units_compound ON inventory_units (tenant_id, compound)"
    )

    op.execute("ALTER TABLE inventory_units ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE inventory_units FORCE ROW LEVEL SECURITY")
    op.execute(f"""CREATE POLICY tenant_isolation ON inventory_units
  USING ({_PREDICATE})
  WITH CHECK ({_PREDICATE})""")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON inventory_units TO moc_app")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS inventory_units")
