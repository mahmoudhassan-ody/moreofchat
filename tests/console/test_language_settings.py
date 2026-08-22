"""Console language versus reply language — demo plan Task 29.

**They are two settings and they are stored in two different places**, which
is the whole point of the pair of tests here.

An admissions officer works in English while the bot answers students in
Masri. A broker's agent works in Arabic while the bot answers an expatriate
buyer in English. One setting for both would mean an officer switching their
own console to English silently switched every student's reply to English —
and nothing would report it, because a bot replying in English is a bot
replying.

That is the same collapse composition already had to be fixed for, where
register and language were being decided together. It is cheap to keep apart
now and expensive to separate once forty components read one flag.
"""

import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.tenancy.context import tenant_session

PASSWORD = "correct horse battery staple"  # noqa: S105 - a test fixture, not a secret
SOURCE = Path(__file__).parents[2] / "src" / "moc"


@pytest_asyncio.fixture(loop_scope="session")
async def one_tenant(engine, app_engine, lookup_engine, tenant_tables):
    """Two agents in the SAME tenant — the shape the per-user claim is about."""
    from moc.tenancy.agent_auth import AgentDirectory
    from moc.tenancy.models import Tenant

    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="lang-co", name="Lang", vertical="education"))
        await s.commit()

    directory = AgentDirectory(engine=app_engine, lookup=lookup_engine)
    await directory.create_agent(
        tenant_id=tenant_id, email="omar@lang.example", password=PASSWORD, display_name="Omar"
    )
    await directory.create_agent(
        tenant_id=tenant_id, email="nour@lang.example", password=PASSWORD, display_name="Nour"
    )
    yield directory, tenant_id

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


async def test_language_preference_is_per_user_not_per_tenant(one_tenant, app_engine):
    """Two colleagues, one tenant, two languages, at the same time.

    The tenant-level version of this setting looks identical until the day two
    people who work together disagree — which is the ordinary case in an
    Egyptian office, not an edge one.
    """
    directory, tenant_id = one_tenant
    omar = await directory.login(email="omar@lang.example", password=PASSWORD)
    nour = await directory.login(email="nour@lang.example", password=PASSWORD)

    await directory.set_console_language(token=omar.token, language="ar")
    await directory.set_console_language(token=nour.token, language="en")

    assert await directory.console_language(token=omar.token) == "ar"
    assert await directory.console_language(token=nour.token) == "en"

    async with tenant_session(app_engine, tenant_id) as session:
        columns = [
            row[0]
            for row in (
                await session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'tenants'"
                    )
                )
            ).all()
        ]
    assert "console_language" not in columns, "the console's language is not the tenant's"


async def test_console_language_and_reply_language_are_separate_settings():
    """Structural, because the behavioural version cannot fail loudly.

    A reply path that read the console's language would still produce a reply,
    in a real language, graded by a judge that only checks the reply mirrors
    the customer — and it would be wrong only for the customers whose language
    happens to differ from whichever agent last changed a setting.

    So the assertion is that the reply path cannot see it: no module under
    `moc/agent/` — extraction, composition, the script engine, the guards —
    mentions `console_language` at all.
    """
    readers = [
        path.relative_to(SOURCE).as_posix()
        for path in (SOURCE / "agent").rglob("*.py")
        if "console_language" in path.read_text(encoding="utf-8")
    ]
    assert readers == [], f"the reply path can see the console's language: {readers}"


async def test_an_unknown_console_language_is_refused(one_tenant):
    """The catalogue has two languages. A third stored here renders the
    console in its fallback and nothing says why."""
    directory, _ = one_tenant
    issued = await directory.login(email="omar@lang.example", password=PASSWORD)

    with pytest.raises(ValueError, match="not a console language"):
        await directory.set_console_language(token=issued.token, language="fr")


async def test_a_new_agent_starts_in_the_fallback_language(one_tenant):
    """`en`, matching i18next's `fallbackLng`. A NULL here would render as a
    missing preference in one place and as English in another."""
    directory, _ = one_tenant
    issued = await directory.login(email="nour@lang.example", password=PASSWORD)
    assert await directory.console_language(token=issued.token) == "en"


async def test_setting_the_language_takes_no_tenant_from_its_caller(one_tenant):
    """Task 28's rule, applied to every method added afterwards.

    This is the shape a bypass actually takes: not a change to the
    authenticator, but a new method on the same object that quietly accepts
    the tenant because it was convenient at the call site.
    """
    import inspect

    directory, _ = one_tenant
    for name in ("set_console_language", "console_language"):
        parameters = set(inspect.signature(getattr(directory, name)).parameters)
        assert "tenant_id" not in parameters
        assert "token" in parameters
