import pytest

from moc.tenancy.models import Tenant


async def test_create_tenant(session):
    t = Tenant(slug="sinai", name="Sinai University", vertical="education")
    session.add(t)
    await session.flush()
    assert t.id is not None
    assert t.created_at is not None


async def test_slug_is_unique(session):
    session.add(Tenant(slug="dup", name="A", vertical="education"))
    await session.flush()
    session.add(Tenant(slug="dup", name="B", vertical="realestate"))
    with pytest.raises(Exception):  # noqa: B017
        await session.flush()
