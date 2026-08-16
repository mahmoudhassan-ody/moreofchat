import os
from pathlib import Path

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine


def pytest_configure(config):
    """Load .env so tests see MOC_* config without a manual export."""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


TEST_DB = "moc_test"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def engine():
    from moc.config import settings

    admin_url = settings.database_url.replace(f"/{settings.pg_database}", "/postgres")
    admin = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as c:
        await c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
        await c.execute(text(f"CREATE DATABASE {TEST_DB}"))
    await admin.dispose()

    test_url = settings.database_url.replace(f"/{settings.pg_database}", f"/{TEST_DB}")
    eng = create_async_engine(test_url)

    # Alembic, not create_all: RLS policies live only in the migrations.
    # Run in a thread: env.py is the async template and calls asyncio.run(),
    # which cannot nest inside the running test loop.
    import asyncio

    from alembic import command
    from alembic.config import Config

    def _migrate() -> None:
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", test_url)
        command.upgrade(cfg, "head")

    await asyncio.to_thread(_migrate)

    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def session(engine):
    """Rolls back after each test, so tests never see each other's rows."""
    async with engine.connect() as conn:
        trans = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as s:
            yield s
        await trans.rollback()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_engine(engine):
    """Engine connecting as moc_app — a non-owner role, so RLS is enforced."""
    from moc.config import settings

    async with engine.begin() as c:
        await c.execute(
            text(f"ALTER ROLE moc_app WITH PASSWORD '{settings.app_password}'")
        )

    eng = create_async_engine(settings.app_database_url(TEST_DB))
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def two_tenants(engine):
    """Two committed tenants, visible to the app role."""
    from moc.tenancy.models import Tenant

    async with AsyncSession(engine, expire_on_commit=False) as s:
        await s.execute(text("DELETE FROM usage_ledger"))
        await s.execute(text("DELETE FROM conversations"))
        await s.execute(text("DELETE FROM tenants"))
        a = Tenant(slug="tenant-a", name="A", vertical="education")
        b = Tenant(slug="tenant-b", name="B", vertical="realestate")
        s.add_all([a, b])
        await s.commit()
        return a, b
