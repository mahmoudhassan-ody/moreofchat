"""Agent inbox and handoff — design §9, P1 Task 22.

The human side of a handoff. The script engine already decides *when* to hand
off (§9, three failed clarifications, or an explicit request); this is what
happens next: the conversation appears in an agent's inbox, the agent replies,
the customer receives it on the channel they wrote from, and the bot resumes
where it left off.

**`test_agent_reply_goes_out_through_the_same_provider_adapter` is the one
that matters.** An agent reply is a message to a customer on a messaging
platform, which means it is subject to the same rate limit and the same
24-hour service window as a bot reply (§6.2). A second send path would be a
second token bucket — and two buckets each allowing the full rate is the same
as no limit — and a second place for the window rule to be got wrong. So the
inbox publishes an `OutboundJob` onto the same stream the orchestrator uses,
and the existing sender worker delivers it. The inbox never touches a provider.
"""

import ast
import inspect
import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.handoff import HandoffStore, MessageLog
from moc.api.inbox import AgentPrincipal, build_inbox
from moc.config_store import load

QUEUES = load("workers/queues")
SCRIPT = "scripts/education/fees"

RESUME_STATE = {
    "script_id": "education_fees",
    "script_version": 1,
    "node": "fees",
    "slots": {"faculty": "pharmacy"},
    "consecutive_clarifications": 3,
}


class FakePublisher:
    """Stands in for the outbound stream, recording what was published."""

    def __init__(self) -> None:
        self.jobs: list = []

    async def publish(self, job) -> None:
        self.jobs.append(job)


class FakeEvents:
    """In-memory inbox event bus, one queue per subscriber."""

    def __init__(self) -> None:
        self.published: list[tuple[uuid.UUID, dict]] = []

    async def publish(self, *, tenant_id, event: dict) -> None:
        self.published.append((tenant_id, event))

    async def subscribe(self, *, tenant_id):
        for owner, event in self.published:
            if owner == tenant_id:
                yield event


def principal(tenant_id, agent: str = "agent-1"):
    async def resolve(request) -> AgentPrincipal:
        return AgentPrincipal(tenant_id=tenant_id, agent_id=agent)

    return resolve


@pytest_asyncio.fixture(loop_scope="session")
async def seeded(engine, tenant_tables):
    """One tenant, one contact who wrote on two channels, one open handoff."""
    from moc.tenancy.models import Tenant

    ids = {key: uuid.uuid4() for key in ("tenant", "other", "contact", "wa", "ig")}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=ids["tenant"], slug="inbox-co", name="Inbox", vertical="education"))
        s.add(Tenant(id=ids["other"], slug="other-co", name="Other", vertical="education"))
        await s.flush()
        await s.execute(
            text(
                "INSERT INTO contacts (id, tenant_id, contact_ref, display_name) "
                "VALUES (:id, :t, '+201012345678', 'Mona')"
            ),
            {"id": ids["contact"], "t": ids["tenant"]},
        )
        now = datetime.now(UTC)
        for key, channel, sender in (
            ("wa", "whatsapp", "+201012345678"),
            ("ig", "instagram", "mona.h"),
        ):
            await s.execute(
                text(
                    "INSERT INTO conversations "
                    "(id, tenant_id, state, channel, sender_ref, contact_id, last_inbound_at) "
                    "VALUES (:id, :t, cast(:state as jsonb), :channel, :sender, :contact, :at)"
                ),
                {
                    "id": ids[key],
                    "t": ids["tenant"],
                    "state": json.dumps(RESUME_STATE),
                    "channel": channel,
                    "sender": sender,
                    "contact": ids["contact"],
                    "at": now - timedelta(hours=1),
                },
            )
        await s.commit()

    yield ids

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def tenant_db(app_engine, seeded):
    """A session as moc_app with the inbox tenant set — what the API uses."""
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, seeded["tenant"]) as session:
        yield session, seeded


# ─────────────────────────── the handoff record ───────────────────────────


async def test_handoff_records_reason_and_timestamps(tenant_db):
    """The reason is why a human was needed, and it is the only thing that
    tells an agent what went wrong before they read the thread."""
    session, ids = tenant_db
    store = HandoffStore(session=session)

    opened = await store.open(
        conversation_id=ids["wa"],
        reason="three consecutive clarifications",
        resume_state=RESUME_STATE,
    )
    assert opened.reason == "three consecutive clarifications"
    assert opened.opened_at is not None
    assert opened.claimed_at is None
    assert opened.returned_at is None

    claimed = await store.claim(handoff_id=opened.id, agent_id="agent-7")
    assert claimed.claimed_by == "agent-7"
    assert claimed.claimed_at >= claimed.opened_at

    returned = await store.return_to_bot(handoff_id=opened.id)
    assert returned.returned_at >= returned.claimed_at


async def test_one_open_handoff_per_conversation(tenant_db):
    """Two agents must not each be told they own the same conversation."""
    session, ids = tenant_db
    store = HandoffStore(session=session)
    await store.open(conversation_id=ids["wa"], reason="first", resume_state=RESUME_STATE)
    with pytest.raises(Exception, match="uq_handoffs_open|already"):
        await store.open(
            conversation_id=ids["wa"], reason="second", resume_state=RESUME_STATE
        )


# ─────────────────────── one contact, many channels ───────────────────────


async def test_conversation_history_spans_channels_for_one_contact(tenant_db):
    """A contact who wrote on WhatsApp and Instagram is one thread to the
    agent, not two.

    An agent who sees only the channel the handoff fired on will ask the
    customer something they already answered somewhere else — which reads, to
    the customer, as the company not keeping records.
    """
    session, ids = tenant_db
    log = MessageLog(session=session)
    await log.append(
        conversation_id=ids["wa"], channel="whatsapp", author="customer", body="مصاريف الصيدلة؟"
    )
    await log.append(
        conversation_id=ids["wa"], channel="whatsapp", author="bot", body="أي فرع؟"
    )
    await log.append(
        conversation_id=ids["ig"], channel="instagram", author="customer", body="العريش"
    )

    thread = await log.history_for_contact(contact_id=ids["contact"])
    assert [m.channel for m in thread] == ["whatsapp", "whatsapp", "instagram"]
    assert [m.author for m in thread] == ["customer", "bot", "customer"]


async def test_a_thread_written_in_one_transaction_keeps_its_order(tenant_db):
    """Found by the test above, and worth its own assertion.

    `now()` is transaction start in Postgres, so a turn's inbound and outbound
    messages — written together, as they must be — shared a timestamp and the
    thread had no defined order. It read back with the reply before the
    question about as often as not. `created_at` is `clock_timestamp()` now,
    and `seq` gives a total order that does not depend on clock resolution.
    """
    session, ids = tenant_db
    log = MessageLog(session=session)
    bodies = [f"m{n}" for n in range(8)]
    for body in bodies:
        await log.append(
            conversation_id=ids["wa"], channel="whatsapp", author="customer", body=body
        )

    thread = await log.history_for_contact(contact_id=ids["contact"])
    assert [m.body for m in thread] == bodies
    assert [m.seq for m in thread] == sorted(m.seq for m in thread)


# ────────────────── the send path (the one that matters) ──────────────────


async def test_agent_reply_goes_out_through_the_same_provider_adapter(
    app_engine, seeded
):
    """Not a second send path.

    The inbox publishes the same `OutboundJob` the orchestrator publishes,
    onto the same stream, and the existing sender worker delivers it through
    the channel adapter. Two paths would mean two token buckets — and two
    buckets each allowing the full rate is the same as no limit — and two
    places for the 24-hour window to be got wrong.
    """
    from moc.workers.outbound import OutboundJob

    publisher = FakePublisher()
    app = build_inbox(
        engine=app_engine,
        publisher=publisher,
        events=FakeEvents(),
        authenticate=principal(seeded["tenant"]),
    )

    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, seeded["tenant"]) as session:
        opened = await HandoffStore(session=session).open(
            conversation_id=seeded["wa"], reason="asked for a human", resume_state=RESUME_STATE
        )
        await session.commit()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        response = await client.post(
            f"/inbox/{opened.id}/reply", json={"text": "أهلاً، أنا منى من القبول"}
        )

    assert response.status_code == 200
    assert len(publisher.jobs) == 1
    job = publisher.jobs[0]
    assert isinstance(job, OutboundJob), "the agent reply is an ordinary outbound job"
    assert job.channel == "whatsapp", "the channel the customer wrote from"
    assert job.to == "+201012345678"
    assert job.text == "أهلاً، أنا منى من القبول"
    # The window is a property of the conversation, and the sender applies it.
    # Omitting it here would make every agent reply look freeform-eligible.
    assert job.last_inbound_at is not None


async def test_the_agent_reply_reaches_the_channel_adapter_through_the_sender(valkey):
    """End to end over the real stream and the real sender worker.

    The assertion above proves the shape of what the inbox publishes; this
    proves that shape is the one the existing worker already consumes, rather
    than a lookalike that happens to have the same field names.
    """
    from moc.workers.outbound import OutboundJob, OutboundWorker

    sent: list[dict] = []

    class RecordingProvider:
        async def send(self, **kwargs):
            sent.append(kwargs)
            return None

    stream = QUEUES["outbound"]["stream"]
    await valkey.delete(stream)
    job = OutboundJob(
        tenant_id=str(uuid.uuid4()),
        channel="whatsapp",
        to="+201012345678",
        text="أهلاً، أنا منى من القبول",
        last_inbound_at=datetime.now(UTC).isoformat(),
    )
    await valkey.xadd(stream, {"payload": job.to_json()})

    worker = OutboundWorker(
        client=valkey, providers={"whatsapp": RecordingProvider()}, config=QUEUES
    )
    assert await worker.run_once() == 1
    assert len(sent) == 1
    assert sent[0]["text"] == "أهلاً، أنا منى من القبول"
    assert sent[0]["last_inbound_at"] is not None, "the window travels with the job"


def test_the_inbox_never_imports_a_messaging_provider():
    """Structural, because the second send path arrives as a convenience.

    Someone with a `TwilioWhatsApp` in scope and an agent waiting will call it
    directly, and nothing at runtime will complain — the message goes out, and
    the rate limit silently covers half the traffic.
    """
    import moc.api.inbox as module

    tree = ast.parse(inspect.getsource(module))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden = {
        name for name in imported if "twilio" in name or name.endswith("provider")
    }
    assert not forbidden, f"the inbox reaches a provider directly via {forbidden}"


# ─────────────────────────── return to bot ───────────────────────────


async def test_return_to_bot_restores_the_script_cursor(tenant_db):
    """The customer's conversation resumes where it stopped.

    Restarting at the script's entry node would ask again for slots the
    customer already gave — the exact frustration that caused the handoff.
    """
    session, ids = tenant_db
    store = HandoffStore(session=session)
    opened = await store.open(
        conversation_id=ids["wa"], reason="three clarifications", resume_state=RESUME_STATE
    )

    # The agent's turn does not run the script, so nothing advances the cursor
    # while a human holds the conversation.
    await session.execute(
        text("UPDATE conversations SET state = cast(:state as jsonb) WHERE id = :id"),
        {
            "state": json.dumps({"script_id": "education_fees", "script_version": 1}),
            "id": ids["wa"],
        },
    )

    await store.return_to_bot(handoff_id=opened.id)
    restored = (
        await session.execute(
            text("SELECT state FROM conversations WHERE id = :id"), {"id": ids["wa"]}
        )
    ).scalar_one()
    assert restored["node"] == "fees"
    assert restored["slots"] == {"faculty": "pharmacy"}


async def test_return_to_bot_does_not_replay_the_agent_turn(tenant_db):
    """The agent's message was already delivered.

    It is recorded so the thread reads correctly, but it is not customer input
    and must never reach the orchestrator — replaying it would answer the
    agent's own words back to the customer.
    """
    session, ids = tenant_db
    store, log = HandoffStore(session=session), MessageLog(session=session)
    opened = await store.open(
        conversation_id=ids["wa"], reason="asked for a human", resume_state=RESUME_STATE
    )
    await log.append(
        conversation_id=ids["wa"], channel="whatsapp", author="agent", body="أهلاً"
    )
    await store.return_to_bot(handoff_id=opened.id)

    pending = await log.unprocessed_inbound(conversation_id=ids["wa"])
    assert pending == [], "an agent message is not queued as customer input"

    restored = (
        await session.execute(
            text("SELECT state FROM conversations WHERE id = :id"), {"id": ids["wa"]}
        )
    ).scalar_one()
    assert restored["consecutive_clarifications"] == 3, (
        "the cursor is restored verbatim; the agent turn did not advance it"
    )


# ─────────────────────────── tenancy ───────────────────────────


async def test_inbox_is_tenant_scoped(app_engine, seeded):
    """RLS does the work; the endpoint must not add its own filter and must
    not be reachable without a tenant."""
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, seeded["tenant"]) as session:
        await HandoffStore(session=session).open(
            conversation_id=seeded["wa"], reason="mine", resume_state=RESUME_STATE
        )
        await session.commit()

    app = build_inbox(
        engine=app_engine,
        publisher=FakePublisher(),
        events=FakeEvents(),
        authenticate=principal(seeded["other"]),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        response = await client.get("/inbox")

    assert response.status_code == 200
    assert response.json() == []


async def test_another_tenants_handoff_cannot_be_replied_to(app_engine, seeded):
    """The stronger half: not merely absent from a listing, unreachable by id."""
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, seeded["tenant"]) as session:
        opened = await HandoffStore(session=session).open(
            conversation_id=seeded["wa"], reason="mine", resume_state=RESUME_STATE
        )
        await session.commit()

    publisher = FakePublisher()
    app = build_inbox(
        engine=app_engine,
        publisher=publisher,
        events=FakeEvents(),
        authenticate=principal(seeded["other"]),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        response = await client.post(f"/inbox/{opened.id}/reply", json={"text": "hello"})

    assert response.status_code == 404
    assert publisher.jobs == [], "nothing may be sent for another tenant's conversation"


async def test_sse_stream_only_emits_events_for_the_agents_own_tenant(app_engine, seeded):
    """One process serves every tenant's agents. A stream that leaked would
    show one company the volume, timing and content of another's escalations."""
    events = FakeEvents()
    await events.publish(tenant_id=seeded["tenant"], event={"type": "opened", "id": "mine"})
    await events.publish(tenant_id=seeded["other"], event={"type": "opened", "id": "theirs"})

    app = build_inbox(
        engine=app_engine,
        publisher=FakePublisher(),
        events=events,
        authenticate=principal(seeded["tenant"]),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        async with client.stream("GET", "/inbox/events") as response:
            assert response.status_code == 200
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert "mine" in body
    assert "theirs" not in body


# ─────────────────────── taking over, and provenance ───────────────────────


async def test_taking_over_stops_the_bot_replying(tenant_db):
    """A live handoff suspends the bot.

    Without this the customer's next message is answered by the bot *over* the
    human who just took the conversation, and both replies go out. It is the
    most visible failure on this screen: an agent types a considered answer and
    the bot argues with them in front of the customer.

    "Live" is any handoff that has not been returned, which is exactly the
    condition the partial unique index already encodes. An open-but-unclaimed
    handoff exists because the bot could not help — letting it answer again
    would repeat the failure that raised the handoff.
    """
    from moc.agent.handoff import HandoffStore

    session, ids = tenant_db
    store = HandoffStore(session=session)

    assert await store.live_for_conversation(conversation_id=ids["wa"]) is None

    opened = await store.open(
        conversation_id=ids["wa"], reason="three clarifications",
        resume_state=RESUME_STATE,
    )
    live = await store.live_for_conversation(conversation_id=ids["wa"])
    assert live is not None and live.id == opened.id

    # Claimed is still live — the human is mid-conversation, which is the
    # state this is most about.
    await store.claim(handoff_id=opened.id, agent_id="agent-7")
    assert await store.live_for_conversation(conversation_id=ids["wa"]) is not None

    # Returned gives it back.
    await store.return_to_bot(handoff_id=opened.id)
    assert await store.live_for_conversation(conversation_id=ids["wa"]) is None


async def test_the_thread_carries_where_each_figure_came_from(tenant_db):
    """The differentiator, at the boundary where it used to be discarded."""
    session, ids = tenant_db
    log = MessageLog(session=session)

    written = await log.append(
        conversation_id=ids["wa"],
        channel="whatsapp",
        author="bot",
        body="رسوم الساعة 1400 جنيه.",
        provenance={
            "figures": [
                {"value": 1400, "raw": "1400", "grounded": True, "source": "chunk",
                 "chunkId": "sinai_fee_hour_ar", "title": "رسوم الساعة",
                 "asOf": "2026-01-01", "excerpt": "رسوم الساعة المعتمدة 1400 جنيه."}
            ],
            "gates": {"numeric_grounding": True, "figure_audit": True,
                      "figure_audit_degraded": False},
        },
    )

    assert written.provenance["figures"][0]["chunkId"] == "sinai_fee_hour_ar"

    thread = await log.history_for_contact(contact_id=ids["contact"])
    stored = next(m for m in thread if m.id == written.id)
    assert stored.provenance["figures"][0]["value"] == 1400
    assert stored.provenance["figures"][0]["excerpt"]


async def test_a_customer_turn_carries_no_provenance(tenant_db):
    """None, not an empty object. A customer did not state a grounded figure,
    and an empty source pane on their message reads as evidence that went
    missing."""
    session, ids = tenant_db
    written = await MessageLog(session=session).append(
        conversation_id=ids["wa"], channel="whatsapp", author="customer",
        body="كام الرسوم؟",
    )
    assert written.provenance is None


async def test_the_routed_sales_team_reaches_the_agent(app_engine, seeded):
    """§11.2's routing, carried across the API boundary — demo plan Task 38.

    The team is chosen when the handoff opens and written to the row. If the
    listing drops it, the lead is routed to nobody in the only sense that
    matters: whoever picks the conversation up is whoever happened to be
    looking, and the routing exists solely in a column nobody reads.

    This is the same failure the source pane was built to end, one screen over.
    """
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, seeded["tenant"]) as session:
        await HandoffStore(session=session).open(
            conversation_id=seeded["wa"],
            reason="lead",
            resume_state=RESUME_STATE,
            team="villas",
            lead_qualified=True,
            lead_score=4,
        )
        await session.commit()

    app = build_inbox(
        engine=app_engine,
        publisher=FakePublisher(),
        events=FakeEvents(),
        authenticate=principal(seeded["tenant"]),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        rows = (await client.get("/inbox")).json()

    assert [row["team"] for row in rows] == ["villas"]
    assert rows[0]["lead_qualified"] is True


async def test_a_handoff_that_is_not_a_lead_carries_no_team(app_engine, seeded):
    """The negative control, and the reason these columns are nullable. A bot
    that ran out of clarifications is not somebody wanting to buy a villa, and
    a zero score would put it in the denominator of a KPI it does not belong
    to."""
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, seeded["tenant"]) as session:
        await HandoffStore(session=session).open(
            conversation_id=seeded["wa"], reason="three clarifications",
            resume_state=RESUME_STATE,
        )
        await session.commit()

    app = build_inbox(
        engine=app_engine,
        publisher=FakePublisher(),
        events=FakeEvents(),
        authenticate=principal(seeded["tenant"]),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        rows = (await client.get("/inbox")).json()

    assert rows[0]["team"] is None
    assert rows[0]["lead_qualified"] is None
