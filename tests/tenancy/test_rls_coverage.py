"""Every tenant-scoped table, checked from the catalogue rather than by memory.

CLAUDE.md states the rule once and every migration has to remember it: RLS
enabled, **FORCE** so the owner does not bypass it, and the nullif-guarded
predicate on both USING and WITH CHECK. Until now each table proved it in its
own test file, which means the guarantee held for the tables somebody wrote a
test for.

That is the wrong shape for this rule. A new tenant-scoped table arriving
without RLS does not fail: it works, for every tenant, all the time — the rows
come back, the queries are fast, and nothing distinguishes it from a table that
is protected. It is discovered when one tenant reads another's.

So this asks Postgres. Any table holding a `tenant_id` is in scope by
construction, present and future, and a table added next month is covered
because it exists rather than because somebody added a line here.

`tenants` is the one exception and it is scoped, not unscoped: its predicate is
on `id`, because the tenant *is* the row. Listed by name so the exception is a
decision rather than a hole.
"""

import pytest
from sqlalchemy import text

#: The predicate from CLAUDE.md, as Postgres renders it back. The nullif guard
#: is the load-bearing half: `current_setting(..., true)` returns the empty
#: string when unset, and casting that to uuid *raises* instead of filtering —
#: so an unset tenant would produce an error rather than an empty result, and
#: somebody would eventually "fix" it by dropping the cast.
_GUARD = "NULLIF(current_setting('moc.tenant_id'::text, true), ''::text)"

#: Scoped on `id` rather than `tenant_id`, because the tenant is the row.
_BY_ID = {"tenants"}


@pytest.fixture
def predicate_for():
    def column(table: str) -> str:
        return "id" if table in _BY_ID else "tenant_id"

    return column


async def _scoped_tables(session) -> list[str]:
    rows = (
        await session.execute(
            text(
                "SELECT c.relname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "JOIN information_schema.columns col "
                "  ON col.table_name = c.relname AND col.table_schema = n.nspname "
                "WHERE n.nspname = 'public' AND c.relkind = 'r' "
                "  AND col.column_name IN ('tenant_id', 'id') "
                "  AND (col.column_name = 'tenant_id' OR c.relname = 'tenants') "
                "GROUP BY c.relname ORDER BY c.relname"
            )
        )
    ).scalars().all()
    return list(rows)


async def test_every_tenant_scoped_table_forces_row_level_security(session):
    tables = await _scoped_tables(session)
    # The catalogue query is the whole test: one that silently matched nothing
    # would report perfect coverage of no tables. Named tables rather than a
    # count alone, because a count passes while the query drifts onto the
    # wrong relkind.
    assert {"tenants", "conversations", "handoffs", "inventory_units", "sales_teams"} <= set(
        tables
    ), f"the catalogue query is not finding tenant-scoped tables: {tables}"
    assert len(tables) > 10, f"only {len(tables)} tables found; the query is wrong"

    unprotected = []
    for table in tables:
        enabled, forced = (
            await session.execute(
                text(
                    # Not `:name::regclass`: SQLAlchemy's bind-parameter
                    # parser leaves the second colon in the statement, and
                    # Postgres rejects it. The same trap the inventory scope
                    # hit.
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE relname = :name AND relnamespace = 'public'::regnamespace"
                ),
                {"name": table},
            )
        ).one()
        if not (enabled and forced):
            unprotected.append((table, enabled, forced))

    assert unprotected == [], (
        "tenant-scoped tables without ENABLE + FORCE row level security. Without "
        "FORCE the owner bypasses the policy, and the owner is what migrations "
        f"and any psql session run as: {unprotected}"
    )


async def test_every_policy_uses_the_nullif_guarded_predicate(session, predicate_for):
    """Both USING and WITH CHECK, on the right column.

    A policy with USING and no WITH CHECK reads correctly and *writes* into
    another tenant, which is the direction nobody checks: the insert succeeds,
    the row is invisible to its author, and it surfaces in the other tenant's
    console.
    """
    tables = await _scoped_tables(session)
    wrong = []
    for table in tables:
        rows = (
            await session.execute(
                text(
                    "SELECT policyname, qual, with_check FROM pg_policies "
                    "WHERE schemaname = 'public' AND tablename = :name"
                ),
                {"name": table},
            )
        ).all()
        if not rows:
            wrong.append((table, "no policy at all"))
            continue
        expected = f"({predicate_for(table)} = ({_GUARD})::uuid)"
        for policy in rows:
            if policy.qual != expected:
                wrong.append((table, f"USING {policy.qual}"))
            if policy.with_check != expected:
                wrong.append((table, f"WITH CHECK {policy.with_check}"))

    assert wrong == [], (
        "policies that are not the predicate CLAUDE.md pins. Each of these "
        f"filters something other than what every other table filters: {wrong}"
    )


async def test_the_truncation_list_matches_the_catalogue(session):
    """`TENANT_SCOPED_TABLES` in conftest is what every fixture empties between
    tests, and unlike the two assertions above it is hand-maintained.

    A tenant-scoped table missing from it does not fail either. It leaves rows
    behind, and the next test that counts them reads its own writes plus
    somebody else's — which passes in file order and fails alone, or the other
    way round, depending on which test wrote first. That is the shape found in
    `test_both_retrievals_are_metered` (Task 42f): `assert rows == 2` passed
    when the file ran in order and failed when the test ran by itself.

    Three ledger-count assertions across three files are correct today *only*
    because `usage_ledger` is on that list — tests/agent/test_orchestrator.py,
    tests/evals/test_runner_inventory.py and tests/tenancy/test_settings.py.
    None of them says so, and none of them would notice.

    Same argument as the two tests above, one layer out: ask the catalogue, so
    a table added next month is covered because it exists.
    """
    from tests.conftest import TENANT_SCOPED_TABLES

    catalogue = set(await _scoped_tables(session))
    listed = set(TENANT_SCOPED_TABLES)
    assert catalogue - listed == set(), (
        f"tenant-scoped and never truncated between tests: "
        f"{sorted(catalogue - listed)} — any count assertion on these is "
        f"order-dependent, and will pass or fail on which test ran first"
    )
    assert listed - catalogue == set(), (
        f"truncated but not tenant-scoped: {sorted(listed - catalogue)} — "
        f"either the table lost its tenant_id or the name is stale"
    )
