"""agents and agent_sessions — console authentication

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-22

Demo plan Task 28, carried since P1 Task 22 where `AgentAuthenticator` was left
a seam with no default. The seam existed because the obvious implementation —
read the tenant from a header the frontend already sends — is an authorization
bypass wearing the shape of a feature.

**Two tables, and both are tenant-scoped with the standard predicate.** An
agent belongs to exactly one tenant and so does their session; there is no
cross-tenant console user, and if there is ever a need for one it will be a new
table rather than a nullable column here.

**Two lookup views, for the same reason migration 0007 has one.** Both reads
that authentication needs run *before* a tenant context exists — resolving
which tenant an email belongs to, and which tenant a session cookie belongs to
— and under the standard predicate with nothing set they return no rows. So
they go through `moc_lookup`, the role whose entire privilege in this database
is SELECT on a small number of deliberately narrow views.

What each view does NOT expose is the point:

- `agent_login_lookup` has **no `password_hash`**. The pre-authentication role
  may learn that an email belongs to a tenant, because that is what resolution
  needs; it may not learn the value an attacker can take away and grind
  offline. This is `signing_secret`'s rule from 0007, applied to the console's
  equivalent secret. Verification happens afterwards, on a `moc_app` session
  with the resolved tenant set, under RLS.
- `agent_session_lookup` exposes the token *hash*, never a token — there are no
  tokens in this database to expose. A row here is enough to tell whether a
  presented token is live and whose it is, and useless for producing one.

**The unique index on `lower(email)` is global, not per tenant.** Login
resolves the tenant *from* the email, so an address claimed by two tenants
would make resolution a guess — and a guess in an authentication path is an
account takeover with extra steps. The cost is that one person working for two
tenants needs two addresses, which is the correct trade.

`token_hash` is SHA-256 and deliberately not a slow KDF: it covers 256 bits of
`os.urandom`, where a password covers whatever a human chose. Running scrypt on
every authenticated request would be a self-inflicted denial of service, and
buys nothing against a preimage nobody can guess.

One statement per op.execute() — asyncpg rejects multi-command strings.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0011"
down_revision: Union[str, Sequence[str], None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PREDICATE = "tenant_id = nullif(current_setting('moc.tenant_id', true), '')::uuid"
_TABLES = ("agents", "agent_sessions")

#: Each view's whole surface, as a list so the migration and the tests read the
#: same shape and so adding one is visibly a change to a security object.
_LOGIN_COLUMNS = ("id", "tenant_id", "email", "status")
_SESSION_COLUMNS = ("tenant_id", "agent_id", "token_hash", "expires_at", "revoked_at")


def upgrade() -> None:
    op.execute("""CREATE TABLE agents (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  email text NOT NULL,
  display_name text NOT NULL,
  -- `scrypt$n$r$p$salt$derived`. Self-describing so a row written under
  -- weaker parameters still verifies after the work factor is raised.
  password_hash text NOT NULL,
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now()
)""")
    # Global, not per tenant. See the module docstring: the email resolves the
    # tenant, so two tenants claiming one address makes resolution a guess.
    op.execute("CREATE UNIQUE INDEX uq_agents_email ON agents (lower(email))")
    op.execute(
        "ALTER TABLE agents ADD CONSTRAINT ck_agents_status "
        "CHECK (status IN ('active', 'suspended'))"
    )

    op.execute("""CREATE TABLE agent_sessions (
  id uuid PRIMARY KEY,
  tenant_id uuid NOT NULL REFERENCES tenants(id),
  agent_id uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  -- SHA-256 of the token. The token itself is returned once, to the browser,
  -- and never stored: a database dump must not be a drawer full of live
  -- cookies.
  token_hash text NOT NULL,
  issued_at timestamptz NOT NULL DEFAULT now(),
  -- Written at issue, never recomputed at read time. A resolve that derived
  -- the expiry from `now()` would extend every session it touched, and the
  -- sessions it touched most are the ones most worth expiring.
  expires_at timestamptz NOT NULL,
  -- Logout is a row, not a cleared cookie. Clearing the cookie is the client
  -- agreeing to stop using the token, and a stolen token was never in the
  -- browser that agreed.
  revoked_at timestamptz
)""")
    op.execute(
        "CREATE UNIQUE INDEX uq_agent_sessions_token ON agent_sessions (token_hash)"
    )
    op.execute(
        "CREATE INDEX ix_agent_sessions_agent ON agent_sessions (tenant_id, agent_id)"
    )

    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""CREATE POLICY tenant_isolation ON {table}
  USING ({_PREDICATE})
  WITH CHECK ({_PREDICATE})""")
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO moc_app")

    # ─────────────────────── the bootstrap doors ───────────────────────

    op.execute(
        "CREATE VIEW agent_login_lookup AS SELECT {} FROM agents".format(
            ", ".join(_LOGIN_COLUMNS)
        )
    )
    op.execute(
        "CREATE VIEW agent_session_lookup AS SELECT {} FROM agent_sessions".format(
            ", ".join(_SESSION_COLUMNS)
        )
    )
    for view in ("agent_login_lookup", "agent_session_lookup"):
        # Stated rather than left to the default, as in 0007: flipping this to
        # true runs the view as the caller, `moc_lookup` has no privilege on
        # the base table, and authentication stops working. Fail-closed, and
        # worth naming so the choice is not mistaken for an oversight.
        op.execute(f"ALTER VIEW {view} SET (security_invoker = false)")
        op.execute("GRANT SELECT ON {} TO moc_lookup".format(view))

    op.execute(
        "COMMENT ON VIEW agent_login_lookup IS "
        "'Pre-tenant bootstrap read for moc_lookup. password_hash is absent "
        "deliberately - adding it hands every console password to the role "
        "that runs before anyone is authenticated. See migration 0011.'"
    )
    op.execute(
        "COMMENT ON VIEW agent_session_lookup IS "
        "'Pre-tenant bootstrap read for moc_lookup. Adding a column here widens "
        "what an unauthenticated request can read. See migration 0011.'"
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS agent_session_lookup")
    op.execute("DROP VIEW IF EXISTS agent_login_lookup")
    op.execute("DROP TABLE IF EXISTS agent_sessions")
    op.execute("DROP TABLE IF EXISTS agents")
