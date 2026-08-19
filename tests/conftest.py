import os
from pathlib import Path

import pytest
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


#: Children before parents. Listed rather than cascaded so that adding a
#: tenant-scoped table without clearing it here fails loudly in a fixture
#: instead of leaking rows between tests. Exported because more than one test
#: module needs committed rows and therefore needs to clean up after itself.
TENANT_SCOPED_TABLES = (
    "channel_accounts",
    "kb_outbox",
    "kb_chunks",
    "kb_documents",
    "usage_ledger",
    "conversations",
    "tenants",
)


@pytest.fixture(scope="session")
def tenant_tables() -> tuple[str, ...]:
    """The delete order for tests that commit rows and must clean up.

    A fixture rather than an importable constant because `tests` is not a
    package; the point is that the list exists once, so a new tenant-scoped
    table is added in one place.
    """
    return TENANT_SCOPED_TABLES


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


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def lookup_engine(engine):
    """Engine connecting as moc_lookup — the pre-tenant bootstrap role.

    Separate from `app_engine` because the point of the role is what it cannot
    reach; sharing an engine would let a test pass while querying as moc_app.
    """
    from moc.config import settings

    async with engine.begin() as c:
        await c.execute(
            text(f"ALTER ROLE moc_lookup WITH PASSWORD '{settings.lookup_password}'")
        )

    eng = create_async_engine(settings.lookup_database_url(TEST_DB))
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def two_tenants(engine):
    """Two committed tenants, visible to the app role."""
    from moc.tenancy.models import Tenant

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in TENANT_SCOPED_TABLES:
            # noqa S608: the table names are the literal tuple above, not input.
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        a = Tenant(slug="tenant-a", name="A", vertical="education")
        b = Tenant(slug="tenant-b", name="B", vertical="realestate")
        s.add_all([a, b])
        await s.commit()
        return a, b
