"""The `channel_accounts` bootstrap — design §5, §6.1, and P1 Task 21.

**Why this table breaks the RLS rule, and what replaces it.**

CLAUDE.md says every tenant-scoped table gets RLS with `FORCE ROW LEVEL
SECURITY`. This lookup cannot obey that, because it runs *before* a tenant
context exists — establishing which tenant a message belongs to is its entire
job. Under the standard predicate with no tenant set, `nullif` yields NULL,
the comparison yields NULL, and the query returns nothing; setting the tenant
first requires the answer the query was going to give.

So the base table keeps RLS, and a separate `moc_lookup` role reads a view
that exposes only the columns needed to resolve: id, tenant_id, channel,
address, secret_ref. The view is the boundary. The role has no reach past it —
not to the base table, not to the signing secret, not to any other table.

**`test_the_view_exposes_only_the_resolving_columns` is the important one.**
Every other test here checks something a reviewer would think to check. That
one catches the thing nobody reviews: an `ALTER VIEW` months from now that
adds a column for some unrelated convenience, silently widening what an
unauthenticated caller's resolution path can read. It is a privilege
escalation with no other signal — the feature works, the tests pass, and the
role's reach grew.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text

# Exactly what resolution needs, and nothing that is not needed to resolve.
# Ordered, because the assertion below compares the view's declared order —
# an inserted column is as much a change as an appended one.
RESOLVING_COLUMNS = ["id", "tenant_id", "channel", "address", "secret_ref"]

VIEW = "channel_account_lookup"
BASE_TABLE = "channel_accounts"
ROLE = "moc_lookup"


@pytest_asyncio.fixture(loop_scope="session")
async def account(engine, tenant_tables):
    """One tenant with one connected WhatsApp number, committed.

    Committed rather than rolled back because the lookup role connects on its
    own engine and would not see an uncommitted row.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.tenancy.models import Tenant

    tenant_id = uuid.uuid4()
    account_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            # noqa S608: names come from the conftest tuple, not from input.
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="lookup-co", name="Lookup Co", vertical="education"))
        await s.flush()
        await s.execute(
            text(
                "INSERT INTO channel_accounts "
                "(id, tenant_id, channel, address, secret_ref, signing_secret) "
                "VALUES (:id, :tenant_id, 'whatsapp', '+201000000001', "
                "'twilio/lookup-co/wa', 'the-real-signing-secret')"
            ),
            {"id": account_id, "tenant_id": tenant_id},
        )
        await s.commit()

    yield account_id, tenant_id

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            # noqa S608: names come from the conftest tuple, not from input.
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


# ─────────────────────── the column list (the one that matters) ───────────────────────


async def test_the_view_exposes_only_the_resolving_columns(engine):
    """Column-list assertion.

    A later ALTER adding a column to this view is a privilege escalation that
    no other test would catch: resolution keeps working, isolation tests keep
    passing, and the role can suddenly read something it was never reviewed
    for. Adding a column here must be a deliberate edit to this list, made by
    someone who has just read why the list is short.
    """
    async with engine.connect() as conn:
        columns = (
            await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :view ORDER BY ordinal_position"
                ),
                {"view": VIEW},
            )
        ).scalars().all()
    assert list(columns) == RESOLVING_COLUMNS


async def test_the_lookup_role_has_no_privilege_anywhere_else(engine):
    """The other half of the same guarantee, from the grant side.

    The column list bounds what the view shows; this bounds what the role can
    reach at all. A `GRANT ... TO moc_lookup` added elsewhere for convenience
    is the same escalation arriving by a different door.
    """
    async with engine.connect() as conn:
        granted = (
            await conn.execute(
                text(
                    "SELECT c.relname, a.privilege_type "
                    "FROM pg_class c, aclexplode(c.relacl) a "
                    "JOIN pg_roles r ON r.oid = a.grantee "
                    "WHERE r.rolname = :role ORDER BY 1, 2"
                ),
                {"role": ROLE},
            )
        ).all()
    assert [tuple(row) for row in granted] == [(VIEW, "SELECT")]


async def test_the_lookup_role_holds_no_column_level_grants(engine):
    """Column grants do not appear in `relacl`, so they are checked separately
    — otherwise a column-level GRANT on the base table would be invisible to
    the test above."""
    async with engine.connect() as conn:
        granted = (
            await conn.execute(
                text(
                    "SELECT c.relname, att.attname, a.privilege_type "
                    "FROM pg_attribute att "
                    "JOIN pg_class c ON c.oid = att.attrelid, "
                    "aclexplode(att.attacl) a "
                    "JOIN pg_roles r ON r.oid = a.grantee "
                    "WHERE r.rolname = :role"
                ),
                {"role": ROLE},
            )
        ).all()
    assert granted == []


async def test_the_lookup_role_cannot_bypass_row_level_security(engine):
    """A role attribute, not a grant. `BYPASSRLS` or `SUPERUSER` here would
    make every other test in this file decorative."""
    async with engine.connect() as conn:
        attrs = (
            await conn.execute(
                text(
                    "SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole "
                    "FROM pg_roles WHERE rolname = :role"
                ),
                {"role": ROLE},
            )
        ).one()
    assert list(attrs) == [False, False, False, False]


# ─────────────────────── what the role can and cannot do ───────────────────────


async def test_lookup_role_can_resolve_a_channel_account(lookup_engine, account):
    """The one thing it exists for, with no tenant context set."""
    account_id, tenant_id = account
    async with lookup_engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    f"SELECT id, tenant_id, channel, address, secret_ref FROM {VIEW} "  # noqa: S608
                    "WHERE channel = 'whatsapp' AND address = :address"
                ),
                {"address": "+201000000001"},
            )
        ).one()
    assert row.id == account_id
    assert row.tenant_id == tenant_id
    assert row.secret_ref == "twilio/lookup-co/wa"


async def test_lookup_role_cannot_read_the_base_table(lookup_engine, account):
    from sqlalchemy.exc import ProgrammingError

    async with lookup_engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="permission denied"):
            await conn.execute(text(f"SELECT * FROM {BASE_TABLE}"))  # noqa: S608


async def test_lookup_role_cannot_read_secrets_beyond_the_reference(lookup_engine, account):
    """`secret_ref` names a secret; it is not one.

    The signing secret is what an attacker needs to forge inbound traffic for
    a tenant, so the role that runs before authentication must not be able to
    reach it — through the view or around it.
    """
    from sqlalchemy.exc import ProgrammingError

    # A connection each: the first failure aborts its transaction, and the
    # second statement would then be rejected for that reason rather than for
    # the permission check this test is about.
    async with lookup_engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="does not exist"):
            await conn.execute(text(f"SELECT signing_secret FROM {VIEW}"))  # noqa: S608
    async with lookup_engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="permission denied"):
            await conn.execute(text(f"SELECT signing_secret FROM {BASE_TABLE}"))  # noqa: S608


async def test_lookup_role_cannot_read_conversations(lookup_engine, account):
    """Customer message content, one join away from a resolved tenant id."""
    from sqlalchemy.exc import ProgrammingError

    async with lookup_engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="permission denied"):
            await conn.execute(text("SELECT * FROM conversations"))


async def test_lookup_role_cannot_read_tenants(lookup_engine, account):
    from sqlalchemy.exc import ProgrammingError

    async with lookup_engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="permission denied"):
            await conn.execute(text("SELECT * FROM tenants"))


async def test_lookup_role_cannot_write_through_the_view(lookup_engine, account):
    """SELECT only. An INSERT here would let an unauthenticated path attach a
    new address to any tenant it names."""
    from sqlalchemy.exc import ProgrammingError

    async with lookup_engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="permission denied"):
            await conn.execute(
                text(
                    f"INSERT INTO {VIEW} (id, tenant_id, channel, address, secret_ref) "  # noqa: S608
                    "VALUES (gen_random_uuid(), gen_random_uuid(), 'whatsapp', '+2', 'r')"
                )
            )


# ─────────────────────── the base table keeps its RLS ───────────────────────


async def test_the_base_table_still_enforces_rls_for_the_app_role(app_engine, account):
    """The view is the exception, not the table.

    `moc_app` — the role every tenant-scoped query runs as — must still see
    nothing without a tenant set. If this regresses, the bootstrap view has
    become a hole in the table rather than a door beside it.
    """
    _, tenant_id = account
    async with app_engine.connect() as conn:
        unset = (
            await conn.execute(text(f"SELECT count(*) FROM {BASE_TABLE}"))  # noqa: S608
        ).scalar_one()
        assert unset == 0

        await conn.execute(
            text("SELECT set_config('moc.tenant_id', :tenant, false)"),
            {"tenant": str(tenant_id)},
        )
        scoped = (
            await conn.execute(text(f"SELECT count(*) FROM {BASE_TABLE}"))  # noqa: S608
        ).scalar_one()
        assert scoped == 1
