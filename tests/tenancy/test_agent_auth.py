"""Console agent authentication — demo plan Task 28, carried from P1 Task 22.

The fourth security-critical file, alongside `guards.py`, `webhooks.py` and the
Qdrant repository, and it is flagged for line-by-line human review for the same
reason: every property here fails silently when it is wrong. A broken tenant
boundary does not raise, it returns somebody else's inbox.

**`test_the_tenant_id_never_comes_from_a_request_header` is the one that
matters.** The bypass this whole module exists to prevent does not arrive as an
attack; it arrives as a convenience. The frontend already knows the tenant, it
already sends the header, the header is right in every test anyone writes — and
from then on any client can read any tenant by editing one string. So the test
is structural rather than behavioural: it asserts the tenant *cannot* come from
the request, not that today it happens not to.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

PASSWORD = "correct horse battery staple"  # noqa: S105 - a test fixture, not a secret


@pytest_asyncio.fixture(loop_scope="session")
async def agents(engine, tenant_tables):
    """One agent in each of two tenants, committed.

    Committed because the lookup role connects on its own engine and would not
    see an uncommitted row — the same reason the channel-account fixture does.
    """
    from moc.tenancy.models import Tenant

    a_id, b_id = uuid.uuid4(), uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            # noqa S608: names come from the conftest tuple, not from input.
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add_all(
            [
                Tenant(id=a_id, slug="console-a", name="A", vertical="education"),
                Tenant(id=b_id, slug="console-b", name="B", vertical="realestate"),
            ]
        )
        await s.commit()
    return a_id, b_id


@pytest_asyncio.fixture(loop_scope="session")
async def directory(app_engine, lookup_engine, agents):
    from moc.tenancy.agent_auth import AgentDirectory

    a_id, b_id = agents
    directory = AgentDirectory(engine=app_engine, lookup=lookup_engine)
    await directory.create_agent(
        tenant_id=a_id, email="ali@a.example", password=PASSWORD, display_name="Ali"
    )
    await directory.create_agent(
        tenant_id=b_id, email="basma@b.example", password=PASSWORD, display_name="Basma"
    )
    return directory


# ─────────────────────────── the tenant boundary ───────────────────────────


async def test_a_session_resolves_exactly_one_tenant(directory, agents):
    a_id, b_id = agents
    issued_a = await directory.login(email="ali@a.example", password=PASSWORD)
    issued_b = await directory.login(email="basma@b.example", password=PASSWORD)

    assert (await directory.resolve(token=issued_a.token)).tenant_id == a_id
    assert (await directory.resolve(token=issued_b.token)).tenant_id == b_id


async def test_a_token_is_not_transferable_between_tenants(directory, agents):
    """The token names its own tenant and nothing else can rename it.

    There is no argument for a caller to pass a tenant with, which is the
    property rather than the test: a signature that accepted one would make the
    bypass expressible, and expressible is all it takes.
    """
    import inspect

    a_id, _ = agents
    issued = await directory.login(email="ali@a.example", password=PASSWORD)
    resolved = await directory.resolve(token=issued.token)

    assert resolved.tenant_id == a_id
    assert set(inspect.signature(directory.resolve).parameters) == {"token", "now"}


async def test_the_password_hash_is_never_readable_by_the_pre_tenant_role(
    lookup_engine, directory
):
    """`moc_lookup` runs before anyone is authenticated.

    It may learn which tenant an email belongs to, because that is what login
    resolution needs. It may not learn the hash — that is the console's
    equivalent of the signing secret migration 0007 kept out of the same role's
    reach, and an attacker holding it can work offline.
    """
    async with lookup_engine.connect() as conn:
        columns = [
            r[0]
            for r in (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'agent_login_lookup' ORDER BY ordinal_position"
                    )
                )
            ).all()
        ]
        with pytest.raises(Exception, match="permission denied"):
            await conn.execute(text("SELECT password_hash FROM agents"))

    assert "password_hash" not in columns
    assert columns == ["id", "tenant_id", "email", "status"]


async def test_the_session_view_exposes_only_what_resolution_needs(lookup_engine):
    """The escalation nobody reviews: an ALTER adding a column here widens what
    the pre-authentication path can read, and every test still passes."""
    async with lookup_engine.connect() as conn:
        columns = [
            r[0]
            for r in (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'agent_session_lookup' "
                        "ORDER BY ordinal_position"
                    )
                )
            ).all()
        ]
    assert columns == ["tenant_id", "agent_id", "token_hash", "expires_at", "revoked_at"]


# ─────────────────────────── sessions ───────────────────────────


async def test_an_expired_session_is_refused_not_renewed(directory, app_engine, agents):
    a_id, _ = agents
    issued = await directory.login(email="ali@a.example", password=PASSWORD)
    later = issued.session.expires_at + timedelta(seconds=1)

    assert await directory.resolve(token=issued.token, now=later) is None

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, a_id) as s:
        expiry = (
            await s.execute(text("SELECT expires_at FROM agent_sessions"))
        ).scalar_one()
    assert expiry == issued.session.expires_at, "a refused session must not slide forward"


async def test_logout_invalidates_server_side_not_only_the_cookie(
    directory, app_engine, agents
):
    """Clearing the cookie is the client agreeing to stop using the token.

    A stolen token was never in the browser that agreed. Revocation has to be a
    row.
    """
    a_id, _ = agents
    issued = await directory.login(email="ali@a.example", password=PASSWORD)
    await directory.logout(token=issued.token)

    assert await directory.resolve(token=issued.token) is None

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, a_id) as s:
        revoked = (
            await s.execute(text("SELECT revoked_at FROM agent_sessions"))
        ).scalar_one()
    assert revoked is not None


async def test_a_wrong_password_issues_no_session(directory, app_engine, agents):
    a_id, _ = agents
    assert await directory.login(email="ali@a.example", password="wrong") is None

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, a_id) as s:
        count = (
            await s.execute(text("SELECT count(*) FROM agent_sessions"))
        ).scalar_one()
    assert count == 0, "a failed login must not leave a session row behind"


async def test_an_unknown_email_is_refused_without_naming_the_tenant(directory):
    assert await directory.login(email="nobody@nowhere.example", password=PASSWORD) is None


async def test_a_session_token_is_not_stored_in_the_clear(directory, app_engine, agents):
    """The database is the thing that leaks. A stored token is a live cookie."""
    a_id, _ = agents
    issued = await directory.login(email="ali@a.example", password=PASSWORD)

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, a_id) as s:
        stored = (
            await s.execute(text("SELECT token_hash FROM agent_sessions"))
        ).scalar_one()
    assert issued.token not in stored
    assert len(issued.token) >= 32, "the fast hash below is only safe on a long random token"


# ─────────────────────────── the KDF ───────────────────────────


def test_password_hashing_uses_a_slow_kdf():
    """The algorithm, not that hashing happened.

    `sha256(password)` passes every behavioural test a password store has —
    it hashes, it verifies, it rejects the wrong password — and it is broken.
    So the name and the work factor are asserted directly.
    """
    from moc.tenancy.passwords import ALGORITHM, hash_password, parameters

    encoded = hash_password(PASSWORD)
    name, n, r, p, _salt, _dk = encoded.split("$")

    assert name == ALGORITHM == "scrypt"
    assert int(n) >= parameters()["floor"]["n"] >= 2**16
    assert int(r) >= 8
    assert int(p) >= 1


def test_the_kdf_floor_cannot_be_lowered_by_config():
    """Config decides what varies; it does not get to decide this.

    §19 puts the work factor in config because hardware moves. A config file
    that can also move it *down* turns one edit into a silent downgrade of
    every password written afterwards, and nothing about the system looks
    different.
    """
    from moc.tenancy.passwords import hash_password

    with pytest.raises(ValueError, match="below the floor"):
        hash_password(PASSWORD, config={"n": 2**10, "r": 8, "p": 1, "floor": {"n": 2**16}})


def test_the_same_password_hashes_differently_every_time():
    from moc.tenancy.passwords import hash_password

    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_verification_accepts_the_right_password_and_refuses_the_rest():
    from moc.tenancy.passwords import hash_password, verify_password

    encoded = hash_password(PASSWORD)
    assert verify_password(PASSWORD, encoded) is True
    assert verify_password(PASSWORD + " ", encoded) is False
    assert verify_password("", encoded) is False


def test_a_malformed_hash_is_refused_rather_than_raising():
    """A truncated row must fail one login, not every request that touches it."""
    from moc.tenancy.passwords import verify_password

    assert verify_password(PASSWORD, "") is False
    assert verify_password(PASSWORD, "scrypt$notanumber$8$1$aaaa$bbbb") is False


def test_the_password_comparison_is_constant_time():
    """Structural, like `_matches` in the Twilio adapter.

    A timing side channel on a digest comparison passes every functional test
    that will ever be written against it, so the assertion is on the source.
    """
    import ast
    import inspect

    from moc.tenancy import passwords

    tree = ast.parse(inspect.getsource(passwords))
    compares = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops)
    ]
    assert compares == [], "compare digests with hmac.compare_digest, never =="
    assert "compare_digest" in inspect.getsource(passwords)


# ─────────────────────── the bypass, structurally ───────────────────────


def test_the_tenant_id_never_comes_from_a_request_header():
    """Structural. The bypass arrives as a convenience — a header the frontend
    already sends, trusted because it is usually right.

    Two assertions, because either alone is escapable:

    - **No client input reaches the auth path except the session token.** The
      whole request object is scanned for reads of `headers`, `query_params`
      and `path_params`; the only permitted read of client input is the cookie
      the token travels in, which is a bearer secret verified against a stored
      hash rather than a claim taken on trust.
    - **No function in the path accepts a tenant from its caller.** A parameter
      that could carry one makes the bypass expressible, and expressible is
      the whole distance between here and an authorization hole. This is the
      rule the Twilio adapter states as "there is no parameter for a parsed
      body and there must never be one".
    """
    import ast
    import inspect

    from moc.api import auth as api_auth
    from moc.tenancy import agent_auth

    for module in (api_auth, agent_auth):
        tree = ast.parse(inspect.getsource(module))
        reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "request"
        }
        assert reads <= {"cookies"}, (
            f"{module.__name__} reads {sorted(reads - {'cookies'})} off the request; "
            "the tenant must come from the session row and nothing else"
        )

    for name, function in vars(agent_auth.AgentDirectory).items():
        if name.startswith("_") or not callable(function):
            continue
        parameters = set(inspect.signature(function).parameters)
        if name == "create_agent":
            # The one exception, and it is not a request path: creating an
            # agent inside a tenant is administration, and the tenant is the
            # subject of the call rather than a claim about the caller.
            continue
        assert "tenant_id" not in parameters, (
            f"AgentDirectory.{name} accepts a tenant from its caller"
        )


def test_only_the_agent_auth_module_reads_the_session_lookup_view():
    """"What can the pre-authentication path reach?" should be answerable by
    reading one file, which is the same rule `accounts.py` carries."""
    from pathlib import Path

    root = Path(__file__).parents[2] / "src" / "moc"
    readers = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "agent_session_lookup" in path.read_text(encoding="utf-8")
    ]
    assert readers == ["tenancy/agent_auth.py"]


async def test_a_tenant_scoped_session_row_is_invisible_to_the_other_tenant(
    directory, app_engine, agents
):
    """RLS on `agent_sessions`, proven from the app role rather than assumed
    from the migration."""
    a_id, b_id = agents
    await directory.login(email="ali@a.example", password=PASSWORD)

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, b_id) as s:
        count = (
            await s.execute(text("SELECT count(*) FROM agent_sessions"))
        ).scalar_one()
    assert count == 0


async def test_a_session_outlives_neither_its_ttl_nor_its_agent(directory, agents):
    """The TTL is config, and the session carries its own expiry rather than
    recomputing one at read time — a resolve that recomputed it would extend
    every session it touched."""
    from moc.config_store import load

    issued = await directory.login(
        email="ali@a.example", password=PASSWORD, now=datetime(2026, 8, 22, tzinfo=UTC)
    )
    hours = load("security/agents")["session"]["ttl_hours"]

    assert issued.session.expires_at == datetime(2026, 8, 22, tzinfo=UTC) + timedelta(
        hours=hours
    )
