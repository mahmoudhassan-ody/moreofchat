"""Tenant settings — demo plan Task 33.

Two of these matter more than the rest.

**`test_a_script_cannot_lower_the_confidence_gate`.** §19.3 through the UI. The
console must not offer what the engine already refuses — and *not offer* means
the control does not exist, not that it is greyed out. A disabled slider is a
worse lie than no slider, because it implies the setting is there and you are
not allowed it; the truth is that the platform's floor is not a permission
level, it is a property of the system.

The floor may be raised and never lowered. That is the same shape as the KDF
work factor in Task 28, for the same reason: a tenant may make the system more
cautious, and a change that makes it less cautious produces no error, no log
and no failing test — just more answers, some of them wrong.

**`test_synonyms_are_editable_per_tenant`.** Every broker names areas
differently. This is the screen that stops that being an engineering ticket,
and it has to be per tenant: the Meilisearch index is shared across every
tenant in a vertical, so a synonym written into the index settings is one
tenant's vocabulary changing another tenant's ranking.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.tenancy.context import tenant_session


@pytest_asyncio.fixture(loop_scope="session")
async def two(engine, tenant_tables):
    from moc.tenancy.models import Tenant

    ids = {"a": uuid.uuid4(), "b": uuid.uuid4()}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add_all(
            [
                Tenant(id=ids["a"], slug="set-a", name="A", vertical="education"),
                Tenant(id=ids["b"], slug="set-b", name="B", vertical="realestate"),
            ]
        )
        await s.commit()
    yield ids
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


# ─────────────────────── the gate the console must not offer ───────────────────────


async def test_a_script_cannot_lower_the_confidence_gate_from_its_own_document(
    app_engine, two
):
    """The engine reads the threshold from the platform tier and nowhere else.

    §19.3 makes it a product guarantee rather than a per-script preference, and
    `ScriptEngine.__init__` already takes it from `defaults` — a script that
    sets one is not overridden, it is *unread*. This pins that, because the
    line that reads it is one word away from reading the script instead and
    nothing about the change would look wrong.
    """
    from moc.agent.script_engine import ScriptEngine
    from moc.config_store import load

    platform = load("agent/defaults")["confidence_threshold"]
    sneaky = {**DRAFT, "settings": {**DRAFT["settings"], "confidence_threshold": 0.01}}
    engine = ScriptEngine(script=sneaky, defaults=load("agent/defaults"))

    assert engine._confidence_threshold == platform


async def test_the_console_is_offered_no_control_for_the_confidence_gate():
    """Not disabled — absent.

    A greyed-out slider implies the setting exists and you are not allowed it.
    The truth is that the floor is a property of the system, not a permission
    level, and the console renders exactly what `bounds()` declares — so a
    setting that is not there cannot be drawn.
    """
    from moc.tenancy.settings import bounds

    assert "confidence_threshold" not in bounds()


async def test_a_script_cannot_lower_the_confidence_gate(app_engine, two):
    """§19.3 through the UI. Raised, never lowered.

    Refused rather than clamped: silently substituting the floor would leave a
    settings screen saying one thing and the system doing another, and the next
    person to read the screen would believe it.
    """
    from moc.tenancy.settings import BelowFloor, SettingsStore

    store = SettingsStore(engine=app_engine)

    # Up is fine — a tenant whose corpus separates cleanly may be stricter.
    await store.put(tenant_id=two["a"], changes={"min_score": 0.45}, agent_id="ali")
    assert (await store.effective(tenant_id=two["a"]))["min_score"] == 0.45

    with pytest.raises(BelowFloor, match="min_score"):
        await store.put(tenant_id=two["a"], changes={"min_score": -0.1}, agent_id="ali")

    # And the refusal changed nothing.
    assert (await store.effective(tenant_id=two["a"]))["min_score"] == 0.45


async def test_the_floor_comes_from_the_platform_config_not_a_literal(app_engine, two):
    """The floor is §19's platform tier. A literal here would be a second
    answer to "how permissive may retrieval be", and the two would diverge the
    first time somebody tuned one."""
    from moc.config_store import load
    from moc.tenancy.settings import bounds

    assert bounds()["min_score"]["min"] == load("retrieval/lexical")["fusion"]["min_score"]


async def test_the_editable_settings_declare_their_own_bounds(app_engine, two):
    """What the console renders comes from here, so a setting the engine
    refuses simply never appears on the screen.

    A frontend holding its own list would show a control for a setting the
    backend has withdrawn — which is the disabled-slider failure arriving from
    the other direction.
    """
    from moc.tenancy.settings import bounds

    declared = bounds()
    assert "min_score" in declared
    for name, bound in declared.items():
        assert {"min", "max", "kind"} <= set(bound), name


async def test_a_tenant_that_has_set_nothing_gets_the_platform_defaults(app_engine, two):
    """Not zeros, and not an empty dict. An unconfigured tenant runs on the
    platform tier, which is what §19's layering means."""
    from moc.config_store import load
    from moc.tenancy.settings import SettingsStore

    effective = await SettingsStore(engine=app_engine).effective(tenant_id=two["b"])
    assert effective["min_score"] == load("retrieval/lexical")["fusion"]["min_score"]


# ─────────────────────────── synonyms ───────────────────────────


async def test_synonyms_are_editable_per_tenant(app_engine, two):
    """Every broker names areas differently.

    This is the demo moment: a broker adds their own name for an area and
    watches the bot understand it, without an engineering ticket.
    """
    from moc.tenancy.settings import SettingsStore

    store = SettingsStore(engine=app_engine)
    await store.put(
        tenant_id=two["b"],
        changes={"synonyms": {"التجمع": ["التجمع الخامس", "القاهرة الجديدة"]}},
        agent_id="basma",
    )

    effective = await store.effective(tenant_id=two["b"])
    assert effective["synonyms"]["التجمع"] == ["التجمع الخامس", "القاهرة الجديدة"]


async def test_one_tenants_synonyms_are_invisible_to_another(app_engine, two):
    """The Meilisearch index is shared across every tenant in a vertical, so a
    synonym written into the index settings is one tenant's vocabulary
    changing another tenant's ranking. These live on the tenant row and are
    applied to the query, which is the only place they can be scoped."""
    from moc.tenancy.settings import SettingsStore

    store = SettingsStore(engine=app_engine)
    await store.put(tenant_id=two["a"], changes={"synonyms": {"منحة": ["خصم"]}},
                    agent_id="ali")

    assert (await store.effective(tenant_id=two["b"]))["synonyms"] == {}

    async with tenant_session(app_engine, two["b"]) as session:
        rows = (
            await session.execute(text("SELECT count(*) FROM tenant_settings"))
        ).scalar_one()
    assert rows == 0, "tenant B can see tenant A's settings row"


async def test_tenant_synonyms_expand_the_query_they_are_scoped_to(app_engine, two):
    """The mechanism, not just the storage.

    Index-level synonyms are shared; these are applied to the query before it
    reaches the index, which is the only scoping the shared index allows. It
    is deliberately not identical to Meilisearch's own synonym handling — the
    expansion is appended, and `matching_strategy: frequency` drops what
    matches nothing.
    """
    from moc.retrieval.lexical import expand_query

    expanded = expand_query("عندكم شقق في التجمع؟", {"التجمع": ["التجمع الخامس"]})

    assert "التجمع الخامس" in expanded
    assert "شقق" in expanded, "the customer's own words survive the expansion"
    assert expand_query("عندكم شقق؟", {}) == "عندكم شقق؟"


# ─────────────────────────── the audit log ───────────────────────────


async def test_settings_changes_are_written_to_the_audit_log(app_engine, two):
    """Who changed what, and when.

    A settings screen with no audit is a screen where "the bot got worse
    yesterday" has no answer — which is the same problem §19.4 solves for eval
    runs, arriving through the console instead of through a deploy.
    """
    from moc.tenancy.settings import SettingsStore

    await SettingsStore(engine=app_engine).put(
        tenant_id=two["a"], changes={"min_score": 0.4}, agent_id="ali"
    )

    async with tenant_session(app_engine, two["a"]) as session:
        row = (
            await session.execute(
                text(
                    "SELECT setting, old_value, new_value, agent_id "
                    "FROM settings_audit ORDER BY changed_at DESC"
                )
            )
        ).first()
    assert row.setting == "min_score"
    assert row.agent_id == "ali"
    assert row.new_value == "0.4"
    # The old value too. "Someone raised it" is a different fact from "someone
    # raised it from the floor", and only the second says whether the change
    # mattered.
    assert row.old_value is not None


async def test_a_refused_change_writes_no_audit_row(app_engine, two):
    """An audit of things that did not happen is an audit nobody trusts."""
    from moc.tenancy.settings import BelowFloor, SettingsStore

    with pytest.raises(BelowFloor):
        await SettingsStore(engine=app_engine).put(
            tenant_id=two["a"], changes={"min_score": -1.0}, agent_id="ali"
        )

    async with tenant_session(app_engine, two["a"]) as session:
        count = (
            await session.execute(text("SELECT count(*) FROM settings_audit"))
        ).scalar_one()
    assert count == 0


# ─────────────────────────── scripts ───────────────────────────

#: The real script shape — `nodes` is a mapping and `entry` names one of them.
#: Copied from `config/scripts/education/fees.yaml` rather than invented: a
#: fixture with a shape the engine does not read tests a format nothing uses.
DRAFT = {
    "version": 1,
    "script_id": "education_fees",
    "entry": "fees",
    "settings": {"max_consecutive_clarifications": 3},
    "nodes": {"fees": {"intents": ["fees"], "slots": ["faculty"]}},
}


async def test_a_script_can_be_edited_and_previewed_before_publishing(app_engine, two):
    """A draft exists and is unreachable by a customer.

    Preview is not a rendering of the YAML — it is the engine built from the
    draft, answering what it would do. A preview that only pretty-printed the
    document would be a text editor with a border round it.
    """
    from moc.agent.scripts import ScriptStore

    store = ScriptStore(engine=app_engine)
    draft = await store.save_draft(
        tenant_id=two["a"], script_id="education_fees", body=DRAFT
    )

    assert draft.version == 1
    assert draft.published is False
    assert await store.current(tenant_id=two["a"], script_id="education_fees") is None

    preview = await store.preview(tenant_id=two["a"], script_id="education_fees")
    assert preview["nodes"] == ["fees"]
    assert preview["entry_node"] == "fees"
    assert preview["max_consecutive_clarifications"] == 3


async def test_publishing_pins_a_version_and_in_flight_conversations_keep_theirs(
    app_engine, two
):
    """A customer mid-conversation is not moved to a script they never started.

    The state already carries `script_version`; this is what makes it mean
    something. Publishing v2 while somebody is three turns into v1 must not
    change the questions they are being asked — a slot they already filled
    disappearing mid-flow is the frustration that causes the handoff the
    script exists to avoid.
    """
    from moc.agent.scripts import ScriptStore

    store = ScriptStore(engine=app_engine)
    await store.save_draft(tenant_id=two["a"], script_id="education_fees", body=DRAFT)
    first = await store.publish(
        tenant_id=two["a"], script_id="education_fees", agent_id="ali"
    )
    assert first.version == 1

    edited = {
        **DRAFT,
        "nodes": {
            "fees": {"intents": ["fees"], "slots": ["faculty"]},
            "housing": {"intents": ["housing"], "slots": []},
        },
    }
    await store.save_draft(tenant_id=two["a"], script_id="education_fees", body=edited)
    second = await store.publish(
        tenant_id=two["a"], script_id="education_fees", agent_id="ali"
    )
    assert second.version == 2

    # New conversations get v2.
    assert (await store.current(tenant_id=two["a"], script_id="education_fees")).version == 2
    # One already holding v1 still loads v1, node for node.
    pinned = await store.at_version(
        tenant_id=two["a"], script_id="education_fees", version=1
    )
    assert sorted(pinned.body["nodes"]) == ["fees"]


async def test_publishing_is_audited_like_any_other_setting(app_engine, two):
    """"The bot got worse yesterday" is most often a script change."""
    from moc.agent.scripts import ScriptStore

    store = ScriptStore(engine=app_engine)
    await store.save_draft(tenant_id=two["a"], script_id="education_fees", body=DRAFT)
    await store.publish(tenant_id=two["a"], script_id="education_fees", agent_id="nour")

    async with tenant_session(app_engine, two["a"]) as session:
        row = (
            await session.execute(
                text(
                    "SELECT setting, new_value, agent_id FROM settings_audit "
                    "ORDER BY changed_at DESC"
                )
            )
        ).first()
    assert row.setting == "script:education_fees"
    assert row.agent_id == "nour"
    assert row.new_value == "2" or row.new_value == "1"


async def test_a_script_with_no_draft_cannot_be_published(app_engine, two):
    """Nothing to publish is a refusal, not a silent no-op that reports
    success to a screen."""
    from moc.agent.scripts import NoDraft, ScriptStore

    with pytest.raises(NoDraft):
        await ScriptStore(engine=app_engine).publish(
            tenant_id=two["a"], script_id="never_drafted", agent_id="ali"
        )
