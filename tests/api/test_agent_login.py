"""The console login surface — demo plan Task 28.

The tenancy tests prove the boundary in isolation. These prove it through the
thing that actually serves requests: a real `build_inbox` app with the real
cookie authenticator, two tenants, and one tenant's open handoff.

**`test_a_token_for_tenant_a_cannot_read_tenant_bs_inbox` is the acceptance
criterion**, and it is only worth anything because it has been proven to fail
when tenant resolution is moved to a header. That sabotage is recorded in the
docstring rather than left as a claim.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.api.auth import build_auth_router, cookie_authenticator
from moc.api.inbox import build_inbox
from moc.config_store import load

COOKIE = load("security/agents")["session"]["cookie"]["name"]
PASSWORD = "correct horse battery staple"  # noqa: S105 - a test fixture, not a secret

RESUME_STATE = {
    "script_id": "education_fees",
    "script_version": 1,
    "node": "fees",
    "slots": {},
    "consecutive_clarifications": 3,
}


class FakePublisher:
    def __init__(self) -> None:
        self.jobs: list = []

    async def publish(self, job) -> None:
        self.jobs.append(job)


class FakeEvents:
    async def publish(self, *, tenant_id, event: dict) -> None: ...

    async def subscribe(self, *, tenant_id):
        return
        yield {}


@pytest_asyncio.fixture(loop_scope="session")
async def console(engine, app_engine, lookup_engine, tenant_tables):
    """Two tenants, an agent in each, and one open handoff belonging to A."""
    from moc.tenancy.agent_auth import AgentDirectory
    from moc.tenancy.models import Tenant

    ids = {key: uuid.uuid4() for key in ("a", "b", "contact", "conversation", "handoff")}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add_all(
            [
                Tenant(id=ids["a"], slug="login-a", name="A", vertical="education"),
                Tenant(id=ids["b"], slug="login-b", name="B", vertical="education"),
            ]
        )
        await s.flush()
        await s.execute(
            text(
                "INSERT INTO contacts (id, tenant_id, contact_ref) "
                "VALUES (:id, :t, '+201000000001')"
            ),
            {"id": ids["contact"], "t": ids["a"]},
        )
        await s.execute(
            text(
                "INSERT INTO conversations "
                "(id, tenant_id, state, channel, sender_ref, contact_id, last_inbound_at) "
                "VALUES (:id, :t, cast(:state as jsonb), 'whatsapp', '+201000000001', "
                ":contact, :at)"
            ),
            {
                "id": ids["conversation"],
                "t": ids["a"],
                "state": json.dumps(RESUME_STATE),
                "contact": ids["contact"],
                "at": datetime.now(UTC) - timedelta(hours=1),
            },
        )
        await s.execute(
            text(
                "INSERT INTO handoffs (id, tenant_id, conversation_id, reason, resume_state) "
                "VALUES (:id, :t, :c, 'three clarifications', cast(:state as jsonb))"
            ),
            {
                "id": ids["handoff"],
                "t": ids["a"],
                "c": ids["conversation"],
                "state": json.dumps(RESUME_STATE),
            },
        )
        await s.commit()

    directory = AgentDirectory(engine=app_engine, lookup=lookup_engine)
    await directory.create_agent(
        tenant_id=ids["a"], email="ali@login-a.example", password=PASSWORD, display_name="Ali"
    )
    await directory.create_agent(
        tenant_id=ids["b"], email="basma@login-b.example", password=PASSWORD,
        display_name="Basma",
    )

    app = build_inbox(
        engine=app_engine,
        publisher=FakePublisher(),
        events=FakeEvents(),
        authenticate=cookie_authenticator(directory=directory),
    )
    app.include_router(build_auth_router(directory=directory))

    yield app, ids, directory

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    )


async def login(http, email: str) -> httpx.Response:
    return await http.post("/auth/login", json={"email": email, "password": PASSWORD})


# ─────────────────────────── the acceptance test ───────────────────────────


async def test_a_token_for_tenant_a_cannot_read_tenant_bs_inbox(console):
    """**Proven by sabotage, 2026-08-22.**

    `cookie_authenticator` was temporarily changed to build the principal from
    `request.headers["x-tenant-id"]` instead of the session row. With that
    change, tenant B's agent reading with `X-Tenant-Id: <A>` received A's open
    handoff and this assertion failed. Reverted; the test stands because it
    has been seen to fail for the reason it exists for.

    A's inbox holds one handoff. B's holds none — not "is filtered to none":
    under RLS the row does not exist for B, which is also why fetching one by
    id is a 404 rather than a 403.
    """
    app, ids, _ = console
    async with client(app) as http:
        await login(http, "ali@login-a.example")
        mine = await http.get("/inbox")
        assert [row["id"] for row in mine.json()] == [str(ids["handoff"])]

    async with client(app) as http:
        await login(http, "basma@login-b.example")
        theirs = await http.get(
            "/inbox",
            # The header a frontend "already sends", carrying A's id. It is
            # read by nothing and changes nothing.
            headers={"X-Tenant-Id": str(ids["a"])},
        )
        assert theirs.json() == []


async def test_the_handoff_itself_is_a_404_for_the_other_tenant(console):
    """403 would confirm the id is real, which is a different leak."""
    app, ids, _ = console
    async with client(app) as http:
        await login(http, "basma@login-b.example")
        response = await http.get(f"/inbox/{ids['handoff']}/thread")
    assert response.status_code == 404


# ─────────────────────────── the cookie ───────────────────────────


async def test_a_request_without_a_cookie_is_refused(console):
    app, _, _ = console
    async with client(app) as http:
        assert (await http.get("/inbox")).status_code == 401


async def test_the_session_cookie_is_httponly_and_samesite(console):
    """httpOnly is what stops an XSS in the console from exfiltrating a live
    session; SameSite is what stops another site from spending it."""
    app, _, _ = console
    async with client(app) as http:
        response = await login(http, "ali@login-a.example")
    header = response.headers["set-cookie"].lower()

    assert "httponly" in header
    assert "samesite=lax" in header
    assert "secure" in header


async def test_the_token_is_not_echoed_in_the_response_body(console):
    """An httpOnly cookie no script can read, handed back in JSON a script
    can, is an httpOnly cookie in name only."""
    app, _, directory = console
    async with client(app) as http:
        response = await login(http, "ali@login-a.example")
        token = http.cookies.get(COOKIE)

    assert token
    assert token not in response.text


async def test_a_bad_password_is_401_with_no_cookie(console):
    app, _, _ = console
    async with client(app) as http:
        response = await http.post(
            "/auth/login",
            json={"email": "ali@login-a.example", "password": "wrong"},
        )
    assert response.status_code == 401
    assert "set-cookie" not in response.headers


async def test_an_unknown_email_answers_exactly_like_a_wrong_password(console):
    """Different answers here enumerate who has an account."""
    app, _, _ = console
    async with client(app) as http:
        wrong = await http.post(
            "/auth/login", json={"email": "ali@login-a.example", "password": "wrong"}
        )
        unknown = await http.post(
            "/auth/login", json={"email": "nobody@nowhere.example", "password": PASSWORD}
        )
    assert wrong.status_code == unknown.status_code
    assert wrong.json() == unknown.json()


async def test_logout_stops_the_next_request_even_with_the_cookie_replayed(console):
    """The point of server-side revocation: a token captured before logout
    must stop working, and the holder of it never cleared any cookie."""
    app, _, _ = console
    async with client(app) as http:
        await login(http, "ali@login-a.example")
        stolen = http.cookies.get(COOKIE)
        assert (await http.get("/inbox")).status_code == 200
        await http.post("/auth/logout")

    async with client(app) as replay:
        replay.cookies.set(COOKIE, stolen, domain="moc.example")
        assert (await replay.get("/inbox")).status_code == 401


async def test_an_expired_cookie_is_refused(console, app_engine):
    app, ids, _ = console
    async with client(app) as http:
        await login(http, "ali@login-a.example")
        assert (await http.get("/inbox")).status_code == 200

        from moc.tenancy.context import tenant_session

        async with tenant_session(app_engine, ids["a"]) as session:
            await session.execute(
                text("UPDATE agent_sessions SET expires_at = :past"),
                {"past": datetime.now(UTC) - timedelta(seconds=1)},
            )
            await session.commit()

        assert (await http.get("/inbox")).status_code == 401
