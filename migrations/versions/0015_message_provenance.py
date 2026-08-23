"""messages.provenance — where each figure in a reply came from

Revision ID: 0015
Revises: 0014
Create Date: 2026-08-22

Demo plan Task 32, and the differentiator the console is built around.

**The data was already computed and thrown away.** Every composed turn runs
`check_numeric_grounding` over the reply and the retrieved passages, deciding
figure by figure whether each has a source — and keeps a pass/fail plus two
lists of numbers. The mapping, which figure came from which chunk, was
discarded at the point it stopped being needed for the gate. A dean asking
"where did 1400 come from?" is asking a question the system answered a
millisecond before it dropped the answer.

So it is stored on the message, next to the words it explains. On the message
rather than in its own table because it is a property of one reply, is written
once, is read whole, and never joins to anything: a `message_figures` table
would be a join, an index and a migration for data with no independent life.

jsonb rather than columns because the shape belongs to the reply — a turn can
state no figures or six, from six different chunks — and because the console
reads it whole. It is documentation of a decision already made, never an input
to one: nothing queries inside it, and if something ever needs to, that is the
moment for real columns.

Nullable, and NULL is meaningful: a scripted reply composed no figures and a
message written before this column existed was never traced. Both are "no
provenance recorded", which is different from "traced and found nothing" —
that is an empty `figures` list.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0015"
down_revision: Union[str, Sequence[str], None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE messages ADD COLUMN provenance jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE messages DROP COLUMN provenance")
