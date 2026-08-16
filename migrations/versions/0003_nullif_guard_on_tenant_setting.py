"""nullif guard on tenant setting

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-16 22:52:03.902739

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""DROP POLICY tenant_isolation ON conversations""")
    op.execute("""CREATE POLICY tenant_isolation ON conversations
  USING (tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid)""")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("""DROP POLICY tenant_isolation ON conversations""")
