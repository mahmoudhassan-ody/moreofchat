import pytest
from sqlalchemy import text

from moc.tenancy.context import tenant_session


async def test_tenant_cannot_read_other_tenants_rows(app_engine, two_tenants):
    a, b = two_tenants
    async with tenant_session(app_engine, a.id) as s:
        await s.execute(
            text(
                "INSERT INTO conversations (id, tenant_id, state) "
                "VALUES (gen_random_uuid(), :t, '{}'::jsonb)"
            ),
            {"t": a.id},
        )
        await s.commit()

    async with tenant_session(app_engine, b.id) as s:
        rows = (await s.execute(text("SELECT * FROM conversations"))).all()
    assert rows == [], "RLS leak: tenant B can see tenant A's conversations"


async def test_insert_with_wrong_tenant_id_is_rejected(app_engine, two_tenants):
    a, b = two_tenants
    async with tenant_session(app_engine, a.id) as s:
        with pytest.raises(Exception):  # noqa: B017
            await s.execute(
                text(
                    "INSERT INTO conversations (id, tenant_id, state) "
                    "VALUES (gen_random_uuid(), :t, '{}'::jsonb)"
                ),
                {"t": b.id},
            )
            await s.commit()


async def test_query_without_tenant_context_returns_nothing(app_engine, two_tenants):
    """A code path that forgets to set the tenant must fail closed, not open."""
    async with app_engine.connect() as c:
        rows = (await c.execute(text("SELECT * FROM conversations"))).all()
    assert rows == []
