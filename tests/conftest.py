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
#: Valkey database index for tests, distinct from the Postgres database name
#: above. Shared here rather than per module: two suites now need a real
#: Valkey, and two fixtures would be two places to get the index wrong — which
#: would show up as one suite quietly flushing the other's streams.
VALKEY_TEST_DB = 15


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
    "agent_sessions",
    "agents",
    "inventory_units",
    "handoffs",
    "messages",
    "channel_accounts",
    "kb_outbox",
    "kb_chunks",
    "kb_documents",
    "usage_ledger",
    "conversations",
    "contacts",
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


@pytest_asyncio.fixture(loop_scope="session")
async def valkey():
    """Real Valkey, on a test-only database index.

    `moc.config` is imported inside the fixture, not at module scope.
    Instantiating `Settings()` requires the database passwords, and pytest
    imports every test module during collection — including in jobs that run
    without infra. A module-level import here took the whole `eval-smoke` job
    down with a collection error, which is why `conftest.py` has always done
    it this way.

    **Unreachable Valkey fails in CI and skips locally.** A developer without
    compose up should get a skip; CI must not, because a silent skip is a test
    that stopped running while the run stayed green. That has bitten this
    project twice, and it is the same rule the MOC_PUBLIC_IP guard applies
    from the other direction.
    """
    import os

    import redis.asyncio as redis

    from moc.config import settings

    client = redis.from_url(settings.valkey_url(VALKEY_TEST_DB), decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        message = (
            f"valkey unreachable at {settings.valkey_host}:{settings.valkey_port} — {exc}"
        )
        if os.environ.get("CI"):
            pytest.fail(
                f"{message}. CI brings the stack up with compose, so this is a broken "
                f"run rather than a missing dependency, and skipping it would hide "
                f"every worker test behind a green tick."
            )
        pytest.skip(f"{message}. Start it with: docker compose up -d valkey")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()
