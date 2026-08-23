"""kb_documents.content_hash — "unchanged" as an answer, not an estimate

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-22

Demo plan Task 31. A tenant re-uploading a corpus they have not edited is the
normal path — they fix one row in a spreadsheet and export the whole sheet —
and it should cost nothing.

The embedding cache already makes the *vectors* free on a repeat, keyed per
text. This is the level above: with the document's hash on the row, an
unchanged upload is recognised before a single provider call is made, and the
console can say "unchanged" rather than showing a progress bar over work that
achieves nothing.

Over the title and the body together, because `embedding_text` prepends the
title to every chunk — a re-titled document embeds differently and is not
unchanged.

Nullable with no backfill: every document ingested before this column existed
was written under no hash at all, and a computed one would claim those rows
were verified when nothing verified them. NULL means "predates the column",
which makes the first re-upload a full ingest — correct, and cheap once.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0014"
down_revision: Union[str, Sequence[str], None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE kb_documents ADD COLUMN content_hash text")


def downgrade() -> None:
    op.execute("ALTER TABLE kb_documents DROP COLUMN content_hash")
