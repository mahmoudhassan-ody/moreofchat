from decimal import Decimal

import pytest
from asyncpg.exceptions import InsufficientPrivilegeError
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from moc.tenancy.context import tenant_session
from moc.tenancy.metering import UsageKind, record_usage

SELECT_LEDGER = text(
    "SELECT tenant_id, kind, channel, quantity, model, provider, "
    "input_tokens, output_tokens, cached_tokens, provider_cost_usd, degraded "
    "FROM usage_ledger"
)


async def test_record_message_usage(app_engine, two_tenants):
    a, _ = two_tenants
    async with tenant_session(app_engine, a.id) as s:
        await record_usage(
            s,
            kind=UsageKind.message_out,
            channel="whatsapp",
            quantity=1,
            provider_cost_usd=Decimal("0.0085"),
        )
        await s.commit()

    # A fresh session: set_config is transaction-local, so the read needs its
    # own tenant context.
    async with tenant_session(app_engine, a.id) as s:
        rows = (await s.execute(SELECT_LEDGER)).all()

    assert len(rows) == 1
    assert rows[0].tenant_id == a.id
    assert rows[0].kind == UsageKind.message_out
    assert rows[0].channel == "whatsapp"
    assert rows[0].quantity == 1
    assert rows[0].provider_cost_usd == Decimal("0.0085")
    assert rows[0].degraded is False


async def test_records_llm_call_with_tokens_and_degraded_flag(app_engine, two_tenants):
    """Week 1 exit criterion: provider, tokens, and the failover flag land in the row."""
    a, _ = two_tenants
    async with tenant_session(app_engine, a.id) as s:
        await record_usage(
            s,
            kind=UsageKind.llm_call,
            model="claude-opus-5",
            provider="bedrock",
            input_tokens=1200,
            output_tokens=340,
            cached_tokens=900,
            provider_cost_usd=Decimal("0.021500"),
            degraded=True,
        )
        await s.commit()

    async with tenant_session(app_engine, a.id) as s:
        row = (await s.execute(SELECT_LEDGER)).one()

    assert row.kind == UsageKind.llm_call
    assert row.model == "claude-opus-5"
    assert row.provider == "bedrock"
    assert row.input_tokens == 1200
    assert row.output_tokens == 340
    assert row.cached_tokens == 900
    assert row.degraded is True


async def test_usage_is_tenant_scoped(app_engine, two_tenants):
    a, b = two_tenants
    async with tenant_session(app_engine, a.id) as s:
        await record_usage(s, kind=UsageKind.message_out, channel="whatsapp", quantity=1)
        await s.commit()

    async with tenant_session(app_engine, b.id) as s:
        rows = (await s.execute(SELECT_LEDGER)).all()
    assert rows == [], "RLS leak: tenant B can see tenant A's usage"


async def test_tenant_survives_commit_within_one_session(app_engine, two_tenants):
    """Regression: set_config is transaction-local, so commit drops the tenant.

    Without the after_begin hook in tenant_session, the read below runs in a
    fresh transaction with moc.tenant_id unset, RLS filters every row, and the
    caller silently sees an empty ledger instead of the row it just wrote.
    """
    a, _ = two_tenants
    async with tenant_session(app_engine, a.id) as s:
        await record_usage(s, kind=UsageKind.message_out, channel="whatsapp", quantity=1)
        await s.commit()

        rows = (await s.execute(SELECT_LEDGER)).all()

    assert len(rows) == 1, "tenant context was lost on commit"
    assert rows[0].tenant_id == a.id


async def test_record_without_tenant_context_fails_closed(app_engine, two_tenants):
    """A code path that forgets the tenant must not write an unattributed row.

    tenant_id resolves to NULL, so the policy's WITH CHECK is NULL — not true —
    and RLS rejects the insert before the NOT NULL constraint is ever reached.
    The guard is the policy itself, which is the layer we want holding.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    async with AsyncSession(app_engine) as s:
        with pytest.raises(ProgrammingError) as exc:
            await record_usage(s, kind=UsageKind.message_in, channel="whatsapp", quantity=1)
            await s.commit()
    assert isinstance(exc.value.orig.__cause__, InsufficientPrivilegeError)
    assert "row-level security" in str(exc.value)


async def test_ledger_is_force_rls_with_the_mandated_predicate(engine):
    """Guards the reason the isolation tests pass.

    A permissive policy, or RLS the owner can bypass, would still let the
    tenant-scoping test above go green for the wrong reason.
    """
    async with engine.connect() as c:
        rls, force = (
            await c.execute(
                text(
                    "SELECT relrowsecurity, relforcerowsecurity "
                    "FROM pg_class WHERE relname = 'usage_ledger'"
                )
            )
        ).one()
        qual, with_check = (
            await c.execute(
                text(
                    "SELECT qual, with_check FROM pg_policies "
                    "WHERE tablename = 'usage_ledger' AND policyname = 'tenant_isolation'"
                )
            )
        ).one()

    assert rls is True
    assert force is True, "owner would bypass the policy without FORCE ROW LEVEL SECURITY"
    for expr in (qual, with_check):
        assert "tenant_id =" in expr
        assert "NULLIF" in expr.upper(), "unset tenant must filter, not raise"
        assert "moc.tenant_id" in expr


async def test_ledger_is_append_only_for_the_app_role(engine):
    """No UPDATE or DELETE grant: invoice history must not be rewritable."""
    async with engine.connect() as c:
        grants = (
            await c.execute(
                text(
                    "SELECT privilege_type FROM information_schema.role_table_grants "
                    "WHERE table_name = 'usage_ledger' AND grantee = 'moc_app'"
                )
            )
        ).scalars().all()
    assert sorted(grants) == ["INSERT", "SELECT"]


async def test_usage_kind_covers_the_metered_events():
    assert [k.value for k in UsageKind] == [
        "message_in",
        "message_out",
        "llm_call",
        "embedding_call",
        "tool_call",
    ]


# ───────── the column that nothing filled ─────────


async def _cost_of(app_engine, tenant, **usage):
    async with tenant_session(app_engine, tenant.id) as s:
        await record_usage(s, **usage)
        await s.commit()
    async with tenant_session(app_engine, tenant.id) as s:
        return (
            await s.execute(text("SELECT provider_cost_usd FROM usage_ledger"))
        ).scalar()


async def test_a_priced_call_lands_with_its_cost(app_engine, two_tenants):
    """`provider_cost_usd` has existed since migration 0004 and every caller
    left it null, so "what did that run cost" was an estimate reconstructed
    from code paths. Priced inside `record_usage` rather than at each call
    site: four callers would be four chances to forget, and the one that
    forgot would be invisible."""
    a, _ = two_tenants
    cost = await _cost_of(
        app_engine,
        a,
        kind=UsageKind.llm_call,
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        input_tokens=281,
        output_tokens=82,
    )
    assert cost == Decimal("0.000691")


async def test_an_unpriced_model_lands_null_rather_than_zero(app_engine, two_tenants):
    """Every judge call in the suite runs on an OpenAI chat model and nothing
    here carries a confirmed rate for one. A zero would make the run total look
    complete while understating its largest line item."""
    a, _ = two_tenants
    cost = await _cost_of(
        app_engine,
        a,
        kind=UsageKind.llm_call,
        model="gpt-5.6-sol",
        provider="openai",
        input_tokens=1450,
        output_tokens=90,
    )
    assert cost is None


async def test_an_explicit_cost_wins_over_the_table(app_engine, two_tenants):
    """A caller holding the provider's own reported figure — a batch discount,
    a negotiated rate — must not have it recomputed from a list price."""
    a, _ = two_tenants
    cost = await _cost_of(
        app_engine,
        a,
        kind=UsageKind.llm_call,
        model="claude-sonnet-5",
        provider="anthropic",
        input_tokens=1_000_000,
        provider_cost_usd=Decimal("0.5"),
    )
    assert cost == Decimal("0.5")


async def test_a_row_with_no_model_is_not_priced(app_engine, two_tenants):
    """message_in and message_out name no model and cost nothing here — the
    channel bills those, and inventing a zero would put them in the same column
    as provider spend."""
    a, _ = two_tenants
    cost = await _cost_of(app_engine, a, kind=UsageKind.message_in, channel="whatsapp")
    assert cost is None
