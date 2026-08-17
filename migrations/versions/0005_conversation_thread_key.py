"""conversation thread key

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-17

A conversation had no way to be found. `conversations` held state and a tenant
and nothing that identifies *whose* thread it is, so the inbound worker could
create rows and never read one back — every message would start a fresh
conversation, slots would never carry across turns, and multi-turn cases would
fail in a way that looks like a model problem rather than a schema one.

The key is (tenant_id, channel, sender_ref): one thread per customer per
channel. `channel_account_id` rides along because a tenant can connect two
WhatsApp numbers, and which one the customer wrote to is part of the audit
trail even though it is not part of the identity of the thread.

Nullable, then indexed. Existing rows predate the concept and there are none in
production; making the columns NOT NULL would still be the wrong shape, because
a conversation opened by an agent from the inbox has no inbound sender.

One statement per op.execute(): asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""ALTER TABLE conversations ADD COLUMN channel text""")
    op.execute("""ALTER TABLE conversations ADD COLUMN sender_ref text""")
    op.execute("""ALTER TABLE conversations ADD COLUMN channel_account_id uuid""")
    # Unique per tenant so a redelivery or a race cannot open a second thread
    # for a customer already mid-conversation — two threads means two script
    # cursors, and the customer gets answered from whichever one won.
    op.execute(
        """CREATE UNIQUE INDEX uq_conversations_thread
  ON conversations (tenant_id, channel, sender_ref)
  WHERE channel IS NOT NULL AND sender_ref IS NOT NULL"""
    )
    op.execute(
        """ALTER TABLE conversations
  ADD COLUMN last_inbound_at timestamptz"""
    )


def downgrade() -> None:
    op.execute("""DROP INDEX IF EXISTS uq_conversations_thread""")
    op.execute("""ALTER TABLE conversations DROP COLUMN IF EXISTS last_inbound_at""")
    op.execute("""ALTER TABLE conversations DROP COLUMN IF EXISTS channel_account_id""")
    op.execute("""ALTER TABLE conversations DROP COLUMN IF EXISTS sender_ref""")
    op.execute("""ALTER TABLE conversations DROP COLUMN IF EXISTS channel""")
