"""tenant identity — logo, timezone, and RLS on the registry itself

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-22

Demo plan Task 30. Three tenants see this product and each has to see
*themselves*: their name in the header, their logo, their data.

**`tenants` had no row-level security at all**, which was fine while the table
held a slug, a name and a vertical — and stops being fine the moment it holds
a tenant's crest. `moc_app` held SELECT on the whole table, so any authenticated
console session could read every tenant's identity. Found while adding the
logo column, and it is the older bug of the two.

So the registry now carries the same treatment as every other tenant-scoped
table, with the predicate keyed on `id` rather than `tenant_id` because here
the row *is* the tenant:

    USING (id = nullif(current_setting('moc.tenant_id', true), '')::uuid)

FORCE is safe for provisioning and migrations: both run as `postgres`, a
superuser, and superusers bypass RLS whatever FORCE says. What FORCE buys is
that a future non-superuser owner does not quietly get the run of the table.

**`moc_app` gets SELECT and UPDATE, and not INSERT or DELETE.** A console
session may change its own tenant's name and logo; creating and destroying
tenants is provisioning, which happens as an owner and is not something an
authenticated request should be able to reach even in principle.

`logo` is bytea on the row rather than a path on disk. At a cap of 512 KiB
that is small, it inherits the RLS boundary above for free, and it removes an
entire class of question about who can read the filesystem that serves it. If
logos ever become large or numerous this is the wrong shape — but they are one
per tenant and they are read once per console load.

`logo_media_type` is stored because it is *sniffed from the content*, not taken
from the upload's filename. Serving a file back under a type derived from a
name an uploader chose is how a "logo" becomes a script.

`timezone` completes what §19 wants on a tenant. Default `Africa/Cairo`: both
pilots are Egyptian, and a NULL here would mean every timestamp in the console
renders in the server's zone while looking like the tenant's.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0013"
down_revision: Union[str, Sequence[str], None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: Keyed on `id`, not `tenant_id`. Same shape, same nullif guard, same reason:
#: without it an unset tenant raises instead of filtering.
_PREDICATE = "id = nullif(current_setting('moc.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.execute("ALTER TABLE tenants ADD COLUMN logo bytea")
    op.execute("ALTER TABLE tenants ADD COLUMN logo_media_type text")
    op.execute(
        "ALTER TABLE tenants ADD COLUMN timezone text NOT NULL DEFAULT 'Africa/Cairo'"
    )
    # Both columns or neither. A media type with no bytes renders a broken
    # image; bytes with no media type get served as something the browser
    # guesses, which is the failure the sniffing exists to prevent.
    op.execute(
        "ALTER TABLE tenants ADD CONSTRAINT ck_tenants_logo_complete "
        "CHECK ((logo IS NULL) = (logo_media_type IS NULL))"
    )

    op.execute("ALTER TABLE tenants ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenants FORCE ROW LEVEL SECURITY")
    op.execute(f"""CREATE POLICY tenant_isolation ON tenants
  USING ({_PREDICATE})
  WITH CHECK ({_PREDICATE})""")
    # SELECT and UPDATE only. See the module docstring: provisioning is not a
    # thing an authenticated request should be able to reach.
    op.execute("GRANT UPDATE ON tenants TO moc_app")


def downgrade() -> None:
    op.execute("REVOKE UPDATE ON tenants FROM moc_app")
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON tenants")
    op.execute("ALTER TABLE tenants NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenants DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_logo_complete")
    op.execute("ALTER TABLE tenants DROP COLUMN timezone")
    op.execute("ALTER TABLE tenants DROP COLUMN logo_media_type")
    op.execute("ALTER TABLE tenants DROP COLUMN logo")
