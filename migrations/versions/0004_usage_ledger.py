"""usage ledger

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-16 23:31:44.201883

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0004'
down_revision: Union[str, Sequence[str], None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""CREATE TYPE usage_kind AS ENUM (
  'message_in', 'message_out', 'llm_call', 'embedding_call', 'tool_call'
)""")
    op.execute("""CREATE TABLE usage_ledger (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  kind usage_kind NOT NULL,
  channel text,
  quantity integer NOT NULL DEFAULT 1,
  model text,
  provider text,
  input_tokens integer NOT NULL DEFAULT 0,
  output_tokens integer NOT NULL DEFAULT 0,
  cached_tokens integer NOT NULL DEFAULT 0,
  provider_cost_usd numeric(14, 6),
  degraded boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    # Billing rolls up per tenant over a period; that is the only read pattern.
    op.execute("""CREATE INDEX ix_usage_ledger_tenant_created
  ON usage_ledger (tenant_id, created_at)""")
    op.execute("""ALTER TABLE usage_ledger ENABLE ROW LEVEL SECURITY""")
    op.execute("""ALTER TABLE usage_ledger FORCE ROW LEVEL SECURITY""")
    op.execute("""CREATE POLICY tenant_isolation ON usage_ledger
  USING (tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid)""")
    # The ledger is append-only: no UPDATE or DELETE grant, so no code path can
    # rewrite history that a tenant will be invoiced from.
    op.execute("""GRANT SELECT, INSERT ON usage_ledger TO moc_app""")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("""DROP TABLE IF EXISTS usage_ledger""")
    op.execute("""DROP TYPE IF EXISTS usage_kind""")
