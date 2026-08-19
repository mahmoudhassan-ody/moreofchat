"""contacts, messages and handoffs — the agent inbox

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-19

Design §9. The script engine already decides *when* to hand off; these are the
tables the human side needs.

**`contacts`** exists because a person is not a channel address. Someone who
asks on WhatsApp and follows up on Instagram is one customer with one history,
and an agent who sees only the channel the handoff fired on will ask them
something they already answered — which reads, to the customer, as the company
not keeping records. `conversations.contact_id` is what joins the two threads.

**`messages`** is the thread an agent reads. Conversations held only the
script cursor until now, which is all a bot needs and none of what a human
needs. `author` separates customer, bot and agent: the distinction is what
makes the thread legible, and it is also what keeps an agent's own words from
being replayed as customer input.

**`handoffs`** carries `resume_state` — the script cursor as it stood when the
human took over. Return-to-bot writes it back. Without it the conversation
would restart at the script's entry node and ask again for the slots the
customer already gave, which is the exact frustration that caused the handoff.

The partial unique index is the invariant that matters here: one open handoff
per conversation, so two agents are never both told they own it.

All three are tenant-scoped, with the nullif-guarded predicate from CLAUDE.md.
One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid"
_TABLES = ("contacts", "messages", "handoffs")


def upgrade() -> None:
    op.execute("""CREATE TABLE contacts (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  contact_ref text NOT NULL,
  display_name text,
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    op.execute("CREATE UNIQUE INDEX uq_contacts_ref ON contacts (tenant_id, contact_ref)")

    op.execute("ALTER TABLE conversations ADD COLUMN contact_id uuid REFERENCES contacts(id)")
    op.execute("CREATE INDEX ix_conversations_contact_id ON conversations (contact_id)")

    op.execute("""CREATE TABLE messages (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  channel text NOT NULL,
  author text NOT NULL,
  body text,
  -- clock_timestamp(), not now(). `now()` is transaction start, so every
  -- message written in one transaction shares a timestamp and the thread has
  -- no order — which is how a turn reads back as the reply preceding the
  -- question.
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  -- Total order, independent of clock resolution. Two messages appended in
  -- the same microsecond still have an unambiguous sequence, and the agent's
  -- thread is read in it.
  seq bigserial NOT NULL
)""")
    # `author` is closed on purpose. A fourth value would silently change what
    # `unprocessed_inbound` returns, and that query decides what the bot is
    # asked to answer when a conversation comes back from a human.
    op.execute(
        "ALTER TABLE messages ADD CONSTRAINT ck_messages_author "
        "CHECK (author IN ('customer', 'bot', 'agent'))"
    )
    # The agent's thread view: one contact, every channel, in time order.
    op.execute(
        "CREATE INDEX ix_messages_conversation_seq "
        "ON messages (tenant_id, conversation_id, seq)"
    )

    op.execute("""CREATE TABLE handoffs (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  reason text NOT NULL,
  status text NOT NULL DEFAULT 'open',
  resume_state jsonb NOT NULL,
  opened_at timestamptz NOT NULL DEFAULT now(),
  claimed_at timestamptz,
  claimed_by text,
  returned_at timestamptz
)""")
    # One live handoff per conversation. Partial rather than plain: a
    # conversation may be handed off again later, and the returned ones must
    # not block that.
    op.execute(
        "CREATE UNIQUE INDEX uq_handoffs_open ON handoffs (conversation_id) "
        "WHERE status <> 'returned'"
    )
    op.execute("CREATE INDEX ix_handoffs_tenant_status ON handoffs (tenant_id, status)")

    # bigserial owns a sequence, and moc_app must be able to draw from it.
    op.execute("GRANT USAGE ON SEQUENCE messages_seq_seq TO moc_app")

    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""CREATE POLICY tenant_isolation ON {table}
  USING ({_PREDICATE})
  WITH CHECK ({_PREDICATE})""")
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO moc_app")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS handoffs")
    op.execute("DROP TABLE IF EXISTS messages")
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS contact_id")
    op.execute("DROP TABLE IF EXISTS contacts")
