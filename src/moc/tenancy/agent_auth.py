"""Console agent sessions — demo plan Task 28, carried from P1 Task 22.

**Flagged for line-by-line human review**, with `guards.py`, `webhooks.py`, the
Qdrant repository and `passwords.py`. This is the file that decides which
tenant's data a request can see, and every way of getting that wrong returns
somebody else's inbox rather than an error.

**The tenant comes from the session row and from nowhere else.** Not from a
header, not from a query parameter, not from a claim inside the cookie. That is
the single property this module exists to hold, and it is held structurally
rather than by discipline: no method here takes a tenant from its caller, so a
caller has nothing to pass. `AgentDirectory.resolve` accepts a token and a
clock, and there is no third argument for a bypass to arrive in. The rule is
the one the Twilio adapter states about raw bytes — the bug is unrepresentable
if the parameter does not exist.

The reason it needs saying is that the bypass is not an attack. It is a
convenience: the frontend already knows the tenant, it already sends the
header, and the header is correct in every test anybody writes. From then on
any client reads any tenant by editing one string.

**Two reads happen before a tenant exists, and both go through `moc_lookup`.**
Resolving which tenant an email belongs to, and which tenant a cookie belongs
to, are the questions whose answers the tenant context is made of — under RLS
with nothing set they return nothing. Migration 0011 gives each its own narrow
view, and this module is the only thing that opens them, so "what can the
pre-authentication path reach?" stays answerable by reading one file.

**The password hash is not in either view.** Login resolves the tenant through
`moc_lookup`, then verifies the password on a `moc_app` session with that
tenant set, under RLS. The hash therefore never enters the pre-authentication
path — the same rule migration 0007 applies to `signing_secret`, for the same
reason: it is exactly the value an attacker wants to take away.

**A token is hashed fast and a password is hashed slow, deliberately.** The
token is 256 bits of `os.urandom` and has no preimage to guess, so SHA-256 is
sufficient and scrypt on every authenticated request would be a self-inflicted
denial of service. A password is whatever a human chose. Treating them alike in
either direction is a mistake — one is slow for no benefit, the other is fast
for no defence.
"""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from moc.tenancy.context import tenant_session
from moc.tenancy.passwords import hash_password, verify_password

_ACTIVE = "active"

#: Resolution, through `moc_lookup`. Expiry and revocation are in the predicate
#: rather than checked afterwards: a caller that fetched the row and then
#: forgot the check would authenticate an expired session, and forgetting is
#: what callers do.
_RESOLVE_SESSION = text(
    "SELECT tenant_id, agent_id, expires_at FROM agent_session_lookup "
    "WHERE token_hash = :token_hash AND revoked_at IS NULL AND expires_at > :now"
)

#: Login's pre-tenant half. No `password_hash` — see the module docstring.
_RESOLVE_LOGIN = text(
    "SELECT id, tenant_id, status FROM agent_login_lookup WHERE lower(email) = lower(:email)"
)


@dataclass(frozen=True)
class Session:
    """An authenticated request's whole authority.

    `tenant_id` is the only thing downstream needs and the only thing that
    decides what exists for this request — every query the console makes runs
    under `tenant_session(engine, session.tenant_id)` and names no tenant of
    its own.
    """

    tenant_id: uuid.UUID
    agent_id: uuid.UUID
    expires_at: datetime


@dataclass(frozen=True)
class Issued:
    """A new session, and the one moment its token exists outside a browser.

    The token is returned here and stored nowhere: `agent_sessions` holds its
    SHA-256. A caller that logs this object logs a live credential, which is
    why it is a distinct type rather than a tuple somebody destructures.
    """

    token: str
    session: Session


class AgentDirectory:
    """Console agents and their sessions.

    Takes two engines because the two halves of authentication genuinely run as
    different roles: `engine` connects as `moc_app` and sees rows only under a
    tenant, `lookup` connects as `moc_lookup` and sees the two resolving views
    and nothing else. Passing one engine for both would either put the
    pre-tenant reads under RLS — where they return nothing — or put every
    authenticated query on a role that does not need RLS at all, and the second
    mistake is silent.
    """

    def __init__(self, *, engine: Any, lookup: Any) -> None:
        self._engine = engine
        self._lookup = lookup

    # ─────────────────────────── administration ───────────────────────────

    async def create_agent(
        self,
        *,
        tenant_id: uuid.UUID,
        email: str,
        password: str,
        display_name: str,
    ) -> uuid.UUID:
        """Add an agent to a tenant.

        The one method here that takes a `tenant_id`, and it is not a request
        path: the tenant is the *subject* of the call — "put this person in
        that tenant" — rather than a claim about who is calling. There is no
        HTTP route to this yet, deliberately; console user management is its
        own screen with its own authorization, and exposing it early would be
        the most direct possible way to undo everything above.
        """
        agent_id = uuid.uuid4()
        async with tenant_session(self._engine, tenant_id) as session:
            await session.execute(
                text(
                    "INSERT INTO agents (id, tenant_id, email, display_name, password_hash) "
                    "VALUES (:id, :tenant_id, :email, :display_name, :password_hash)"
                ),
                {
                    "id": agent_id,
                    "tenant_id": tenant_id,
                    "email": email,
                    "display_name": display_name,
                    "password_hash": hash_password(password),
                },
            )
            await session.commit()
        return agent_id

    # ─────────────────────────── the request path ───────────────────────────

    async def login(
        self, *, email: str, password: str, now: datetime | None = None
    ) -> Issued | None:
        """A session, or None. Never an exception, and never a reason.

        **One `None` for every failure.** Unknown address, wrong password,
        suspended account — the caller cannot tell them apart, because the
        difference is precisely what enumerates who has an account here. The
        cost is a worse error message for a legitimate typo, and the console
        says "check your email and password" for all three.
        """
        moment = now or datetime.now(UTC)
        async with self._lookup.connect() as connection:
            row = (
                await connection.execute(_RESOLVE_LOGIN, {"email": email})
            ).one_or_none()
        if row is None or row.status != _ACTIVE:
            return None

        # The tenant is now known — from the row, not from anything presented —
        # so the hash can be read under RLS rather than through the
        # pre-authentication role.
        async with tenant_session(self._engine, row.tenant_id) as session:
            stored = (
                await session.execute(
                    text("SELECT password_hash FROM agents WHERE id = :id"),
                    {"id": row.id},
                )
            ).scalar_one_or_none()
            if stored is None or not verify_password(password, stored):
                return None
            issued = await self._open(
                session, tenant_id=row.tenant_id, agent_id=row.id, now=moment
            )
            await session.commit()
        return issued

    async def resolve(self, *, token: str, now: datetime | None = None) -> Session | None:
        """Which tenant this token belongs to, or None.

        **There is no tenant parameter and there must never be one.** Adding
        one — for a "faster path", for a test, for a frontend that already
        knows — is the entire bypass this module was left unimplemented to
        avoid, and it would look like a convenience in the diff that
        introduced it.

        An expired or revoked token is None rather than a refreshed session.
        Renewing on read would make the sessions most worth expiring the ones
        that never do.
        """
        moment = now or datetime.now(UTC)
        if not token:
            return None
        async with self._lookup.connect() as connection:
            row = (
                await connection.execute(
                    _RESOLVE_SESSION,
                    {"token_hash": _token_hash(token), "now": moment},
                )
            ).one_or_none()
        if row is None:
            return None
        return Session(
            tenant_id=row.tenant_id, agent_id=row.agent_id, expires_at=row.expires_at
        )

    async def logout(self, *, token: str, now: datetime | None = None) -> None:
        """Revoke server-side. Clearing the cookie is not logging out.

        Idempotent, and silent on a token that resolves to nothing: a logout
        that reported whether the token was live would answer, for anyone who
        can call it, the one question a stolen-token holder wants answered.
        """
        moment = now or datetime.now(UTC)
        session_row = await self.resolve(token=token, now=moment)
        if session_row is None:
            return
        async with tenant_session(self._engine, session_row.tenant_id) as session:
            await session.execute(
                text(
                    "UPDATE agent_sessions SET revoked_at = :now "
                    "WHERE token_hash = :token_hash AND revoked_at IS NULL"
                ),
                {"now": moment, "token_hash": _token_hash(token)},
            )
            await session.commit()

    # ─────────────────────────── internals ───────────────────────────

    async def _open(
        self, session: Any, *, tenant_id: uuid.UUID, agent_id: uuid.UUID, now: datetime
    ) -> Issued:
        settings = _config()["session"]
        token = secrets.token_urlsafe(int(settings["token_bytes"]))
        expires_at = now + timedelta(hours=int(settings["ttl_hours"]))
        await session.execute(
            text(
                "INSERT INTO agent_sessions "
                "(id, tenant_id, agent_id, token_hash, issued_at, expires_at) "
                "VALUES (:id, :tenant_id, :agent_id, :token_hash, :now, :expires_at)"
            ),
            {
                "id": uuid.uuid4(),
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "token_hash": _token_hash(token),
                "now": now,
                "expires_at": expires_at,
            },
        )
        return Issued(
            token=token,
            session=Session(
                tenant_id=tenant_id, agent_id=agent_id, expires_at=expires_at
            ),
        )


def _token_hash(token: str) -> str:
    """SHA-256, and fast on purpose — see the module docstring."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _config() -> dict[str, Any]:
    from moc.config_store import load

    return load("security/agents")


__all__ = ["AgentDirectory", "Issued", "Session"]
