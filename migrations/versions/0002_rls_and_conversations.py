"""rls and conversations

Revision ID: 0002
Revises: 9832db3d8ff7
Create Date: 2026-08-16 22:45:10.118728

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '9832db3d8ff7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""CREATE TABLE conversations (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  state jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    op.execute("""CREATE INDEX ix_conversations_tenant_id ON conversations (tenant_id)""")
    op.execute("""ALTER TABLE conversations ENABLE ROW LEVEL SECURITY""")
    op.execute("""ALTER TABLE conversations FORCE ROW LEVEL SECURITY""")
    op.execute("""CREATE POLICY tenant_isolation ON conversations
  USING (tenant_id = current_setting('moc.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('moc.tenant_id', true)::uuid)""")
    op.execute("""DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'moc_app') THEN
    CREATE ROLE moc_app LOGIN PASSWORD 'CHANGE_ME_VIA_ENV';
  END IF;
END $$""")
    op.execute("""GRANT USAGE ON SCHEMA public TO moc_app""")
    op.execute("""GRANT SELECT, INSERT, UPDATE, DELETE ON conversations TO moc_app""")
    op.execute("""GRANT SELECT ON tenants TO moc_app""")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS conversations;")
