"""kb documents, chunks and outbox

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-19

Postgres is the source of truth; Qdrant and Meilisearch are derived (design
§7.1). The outbox is what keeps that true: chunks and their sync rows commit
in one transaction, so there is no window in which a chunk exists and nothing
knows to index it. Writing to the search stores directly from the ingest path
would reintroduce exactly the dual-write problem D2 chose the outbox to avoid.

All three tables are tenant-scoped, with the `nullif`-guarded predicate from
CLAUDE.md. `kb_chunks` is the one that matters most: it holds the text a reply
is grounded in, so a missing filter here is one tenant's fees answered from
another tenant's corpus.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.execute("""CREATE TABLE kb_documents (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  doc_id text NOT NULL,
  title text,
  vertical text NOT NULL,
  lang text,
  source_uri text,
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    # Stable per tenant, so re-ingesting the same document updates it rather
    # than accumulating a second copy the retriever would return twice.
    op.execute(
        "CREATE UNIQUE INDEX uq_kb_documents_tenant_doc ON kb_documents (tenant_id, doc_id)"
    )

    op.execute("""CREATE TABLE kb_chunks (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  document_id uuid NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
  chunk_id text NOT NULL,
  ordinal integer NOT NULL,
  content text NOT NULL,
  content_normalized text NOT NULL,
  lang text,
  topic text,
  entity_ref text,
  effective_from date,
  effective_to date,
  embedding_version text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    op.execute(
        "CREATE UNIQUE INDEX uq_kb_chunks_tenant_chunk ON kb_chunks (tenant_id, chunk_id)"
    )
    op.execute("CREATE INDEX ix_kb_chunks_document ON kb_chunks (document_id)")

    op.execute("""CREATE TABLE kb_outbox (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  chunk_id text NOT NULL,
  target text NOT NULL,
  op text NOT NULL,
  point_id uuid NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  attempts integer NOT NULL DEFAULT 0,
  last_error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  processed_at timestamptz
)""")
    # The drain query. Partial, because a drained outbox is the normal state
    # and a full index over mostly-done rows earns nothing.
    op.execute(
        """CREATE INDEX ix_kb_outbox_pending ON kb_outbox (tenant_id, created_at)
  WHERE status = 'pending'"""
    )

    for table in ("kb_documents", "kb_chunks", "kb_outbox"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""CREATE POLICY tenant_isolation ON {table}
  USING      ({_PREDICATE})
  WITH CHECK ({_PREDICATE})"""
        )
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO moc_app")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kb_outbox")
    op.execute("DROP TABLE IF EXISTS kb_chunks")
    op.execute("DROP TABLE IF EXISTS kb_documents")
