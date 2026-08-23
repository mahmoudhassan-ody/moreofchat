"""Analytics — demo plan Task 34, the last of Track A.

What a buyer asks: how many did the bot answer, how many needed a person, what
did it cost, and what did people ask about. Three of those had no answer before
the ledger; the fourth still does not have a good one and this says so.

**`test_containment_rate_is_shown_and_never_gated` is the one with a design in
it.** Containment is the share of conversations the bot handled without a
human, and it is the most tempting number in the product to put a target on.
The moment it has one, every handoff is a regression — and the way to move it
is to answer where the honest behaviour is to hand off. §19.3 exists to stop
the bot guessing; a containment gate would pay it to.

**`test_an_unpriced_model_shows_unknown_not_zero`.** A model with no rate in
the price table contributes NULL, and a sum over NULLs is a smaller number that
looks complete. The report says how much of itself it could not price.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture(loop_scope="session")
async def traffic(engine, app_engine, tenant_tables):
    """Two tenants. A has three conversations, one handed off, and some spend."""
    from moc.tenancy.context import tenant_session
    from moc.tenancy.models import Tenant

    ids = {"a": uuid.uuid4(), "b": uuid.uuid4()}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add_all(
            [
                Tenant(id=ids["a"], slug="an-a", name="A", vertical="education"),
                Tenant(id=ids["b"], slug="an-b", name="B", vertical="education"),
            ]
        )
        await s.commit()

    conversations = [uuid.uuid4() for _ in range(3)]
    async with tenant_session(app_engine, ids["a"]) as session:
        for index, conversation_id in enumerate(conversations):
            await session.execute(
                text(
                    "INSERT INTO conversations "
                    "(id, tenant_id, state, channel, sender_ref, last_inbound_at) "
                    "VALUES (:id, :t, cast('{}' as jsonb), 'whatsapp', :ref, :at)"
                ),
                {"id": conversation_id, "t": ids["a"], "ref": f"+2010000000{index}",
                 "at": NOW - timedelta(hours=index)},
            )
        # One of the three needed a person.
        await session.execute(
            text(
                "INSERT INTO handoffs (id, tenant_id, conversation_id, reason, resume_state) "
                "VALUES (:id, :t, :c, 'three clarifications', cast('{}' as jsonb))"
            ),
            {"id": uuid.uuid4(), "t": ids["a"], "c": conversations[0]},
        )
        await session.commit()
    yield ids, conversations, app_engine

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


async def meter(app_engine, tenant_id, **kwargs):
    from moc.tenancy.context import tenant_session
    from moc.tenancy.metering import UsageKind, record_usage

    async with tenant_session(app_engine, tenant_id) as session:
        await record_usage(session, kind=UsageKind.llm_call, **kwargs)
        await session.commit()


# ─────────────────────────── containment ───────────────────────────


async def test_containment_rate_is_shown_and_never_gated(traffic):
    """Two out of three conversations never needed a person.

    Reported, and deliberately not in `gates.yaml`. A target on this number
    pays the bot to answer where the honest behaviour is to hand off, which is
    the exact pressure §19.3 exists to remove.
    """
    from moc.tenancy.analytics import AnalyticsStore

    ids, _, app_engine = traffic
    report = await AnalyticsStore(engine=app_engine).report(tenant_id=ids["a"])

    assert report.conversations == 3
    assert report.handed_off == 1
    assert report.containment_rate == 2 / 3


async def test_containment_is_absent_from_the_gate_configuration():
    """Structural, because the pressure arrives as a reasonable request.

    "Can we track containment against a target" is a sensible-sounding
    sentence, and the config file is where it would land.
    """
    from moc.config_store import load

    gates = load("evals/gates")
    assert "containment_rate" not in gates["hard_gates"]
    assert "containment_rate" not in gates["soft_gates"]
    assert "containment_rate" in gates["tracked"]


async def test_a_tenant_with_no_traffic_reports_none_not_a_perfect_score(traffic):
    """Zero conversations is not 100% containment. A rate over an empty
    denominator is a number nobody measured, and on this particular metric it
    is the most flattering one available."""
    from moc.tenancy.analytics import AnalyticsStore

    ids, _, app_engine = traffic
    report = await AnalyticsStore(engine=app_engine).report(tenant_id=ids["b"])

    assert report.conversations == 0
    assert report.containment_rate is None


# ─────────────────────────── cost ───────────────────────────


async def test_cost_per_conversation_comes_from_the_ledger_not_an_estimate(traffic):
    """The number a buyer asks for, as a query.

    Before the ledger this was arrived at by reading code paths and guessing
    token counts, which is how a billing column comes to exist and never hold
    a value.
    """
    from moc.tenancy.analytics import AnalyticsStore

    ids, _, app_engine = traffic
    await meter(app_engine, ids["a"], model="claude-sonnet-5", provider="anthropic",
                input_tokens=1000, output_tokens=200,
                provider_cost_usd=Decimal("0.0040"))
    await meter(app_engine, ids["a"], model="claude-sonnet-5", provider="anthropic",
                input_tokens=500, output_tokens=100,
                provider_cost_usd=Decimal("0.0020"))

    report = await AnalyticsStore(engine=app_engine).report(tenant_id=ids["a"])

    assert report.cost_usd == Decimal("0.0060")
    assert report.cost_per_conversation == Decimal("0.0060") / 3
    assert report.unpriced_calls == 0
    assert report.cost_is_complete is True


async def test_an_unpriced_model_shows_unknown_not_zero(traffic):
    """A model with no rate contributes NULL, and a sum over NULLs is a
    smaller number that looks complete.

    So the report carries what it could not price. `cost_is_complete` is False
    the moment one row is unpriced, and the screen says "at least" rather than
    a total it cannot stand behind.
    """
    from moc.tenancy.analytics import AnalyticsStore

    ids, _, app_engine = traffic
    await meter(app_engine, ids["a"], model="some-new-model", provider="anthropic",
                input_tokens=900, output_tokens=100, provider_cost_usd=None)

    report = await AnalyticsStore(engine=app_engine).report(tenant_id=ids["a"])

    assert report.unpriced_calls == 1
    assert report.cost_is_complete is False
    assert "some-new-model" in report.unpriced_models


async def test_the_cost_breakdown_names_the_models(traffic):
    """"What did it cost" is one number; "what is costing it" is the one that
    changes a decision."""
    from moc.tenancy.analytics import AnalyticsStore

    ids, _, app_engine = traffic
    # Metered here rather than relying on an earlier test. `traffic` rebuilds
    # per test, so a breakdown assertion leaning on another test's rows passes
    # only in the order they happen to run — and reports a KeyError that reads
    # like a missing model rather than a missing fixture.
    await meter(app_engine, ids["a"], model="claude-sonnet-5", provider="anthropic",
                input_tokens=1000, output_tokens=200,
                provider_cost_usd=Decimal("0.0040"))
    await meter(app_engine, ids["a"], model="claude-sonnet-5", provider="anthropic",
                input_tokens=500, output_tokens=100,
                provider_cost_usd=Decimal("0.0020"))
    await meter(app_engine, ids["a"], model="some-new-model", provider="anthropic",
                input_tokens=900, output_tokens=100, provider_cost_usd=None)
    report = await AnalyticsStore(engine=app_engine).report(tenant_id=ids["a"])

    by_model = {row.model: row for row in report.by_model}
    assert by_model["claude-sonnet-5"].calls == 2
    assert by_model["claude-sonnet-5"].cost_usd == Decimal("0.0060")
    # The unpriced one is listed with no cost rather than dropped: a breakdown
    # that hid it would make the total look like the whole.
    assert by_model["some-new-model"].cost_usd is None


# ─────────────────────────── what they asked about ───────────────────────────


async def test_the_report_says_why_the_bot_handed_off(traffic):
    """The most actionable thing on the screen. "Three clarifications" fifty
    times is a script that cannot route a question the corpus can answer."""
    from moc.tenancy.analytics import AnalyticsStore

    ids, _, app_engine = traffic
    report = await AnalyticsStore(engine=app_engine).report(tenant_id=ids["a"])

    assert report.handoff_reasons == [("three clarifications", 1)]


# ─────────────────────────── isolation ───────────────────────────


async def test_analytics_are_tenant_scoped(traffic):
    """B has no conversations and no spend, and A's do not leak into it —
    under RLS they do not exist for B rather than being filtered out."""
    from moc.tenancy.analytics import AnalyticsStore

    ids, _, app_engine = traffic
    store = AnalyticsStore(engine=app_engine)

    mine = await store.report(tenant_id=ids["a"])
    theirs = await store.report(tenant_id=ids["b"])

    assert mine.conversations == 3
    assert theirs.conversations == 0
    assert theirs.cost_usd == Decimal("0")
    assert theirs.handoff_reasons == []


async def test_the_report_takes_no_tenant_from_anything_but_its_caller(traffic):
    """Task 28's rule, applied to the last screen in the track.

    Every method takes the tenant it was given and opens a session under it;
    there is no query here that names a tenant in SQL, because a WHERE clause
    is a filter somebody can forget and RLS is not.
    """
    import ast
    import inspect

    from moc.tenancy import analytics

    # The SQL, not the prose. This module's docstring explains at length why a
    # WHERE clause is the wrong tool here, and a substring scan reads its own
    # rationale as a violation — the third time that trap has caught me, after
    # the CSS scanner and the preview module. Every `text(...)` argument is a
    # query; nothing else in the file is.
    tree = ast.parse(inspect.getsource(analytics))
    queries = [
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "text"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    ]
    assert queries, "no SQL found — the scanner is looking at the wrong thing"
    for query in queries:
        assert "tenant_id" not in query, f"a WHERE clause where RLS should be: {query}"
    assert "tenant_session" in inspect.getsource(analytics)
