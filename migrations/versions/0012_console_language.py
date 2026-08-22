"""agents.console_language — the console's language, per person

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-22

Demo plan Task 29.

**On `agents`, not on `tenants`.** Two colleagues in one office disagree about
which language they read a console in, and in an Egyptian office that is the
ordinary case rather than an edge one. A tenant-level column looks identical
until the day it is wrong, and then it is wrong for everybody at once.

**And it is not the reply language.** What the bot answers a customer in is
decided per turn by mirroring the customer (§8.3), and the reply path cannot
see this column — a test asserts no module under `moc/agent/` mentions it. One
setting for both would mean an officer switching their own console to English
silently switching every student's reply to English, and nothing would report
it, because a bot replying in English is still a bot replying. That is the
register/language collapse composition already had to be fixed for.

`NOT NULL DEFAULT 'en'`, matching i18next's `fallbackLng`. A nullable column
would render as "no preference" in one place and as English in another, which
is two behaviours for one state.

The CHECK is the catalogue, in the database. A third language stored here
renders the console in its fallback and nothing says why — the constraint
turns that into a failed write at the point somebody could still fix it.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agents ADD COLUMN console_language text NOT NULL DEFAULT 'en'"
    )
    op.execute(
        "ALTER TABLE agents ADD CONSTRAINT ck_agents_console_language "
        "CHECK (console_language IN ('en', 'ar'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN console_language")
