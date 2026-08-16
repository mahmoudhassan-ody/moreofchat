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

    from moc.tenancy.models import Base

    async with eng.begin() as c:
        await c.run_sync(Base.metadata.create_all)

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
