"""messages.timings — where a turn's seconds went, in production

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-25

Demo plan, after the first real vendor traffic. `Orchestrator.handle` has
always produced a per-phase breakdown — extraction, retrieval, composition,
the figure audit — and `moc.evals.runner` has always been the only thing that
read it. The worker discarded it, so the phase breakdown existed exactly where
nothing waits on a reply and nowhere a customer does.

That was found by needing it. Two real Telegram messages on 2026-08-25 took
5.430s and 5.017s end to end, and nothing on the host could say which phase
either second belonged to. §2.5's budget is a product claim about what a
customer waits through on WhatsApp; an instrument that only runs in the test
harness cannot answer "the demo felt slow".

Same shape and the same reasoning as `provenance` in 0015: on the message
rather than its own table, because it is a property of one reply, written
once, read whole, and joining to nothing. jsonb because the phases are not a
fixed list — `intake.requery` exists on the turns that resume a node and on no
others (Task 42f), and a column per phase would need a migration each time a
phase is added.

Nullable, and NULL is meaningful. A customer's inbound message is not a turn
and has no phases; a row written before this column existed was never timed.
Neither is the same as `{}`, which would claim a turn ran and measured
nothing — the distinction §2.4 draws between "measured zero" and "not
measured".

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0018"
down_revision: Union[str, Sequence[str], None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE messages ADD COLUMN timings jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE messages DROP COLUMN timings")
