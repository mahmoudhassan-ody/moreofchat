"""The wire from a signed webhook to a sent reply — design §3, §6.

Everything before this was green in isolation: the webhook enqueued into a
Protocol, the orchestrator took a state and returned a result, the adapter
posted to a mock transport. None of it proved they connect, and "each piece
works" is exactly the shape of a system that does not.

So these run against **real Valkey** — the same server compose starts, on a
separate database index. A fake queue would re-prove the Protocol rather than
the streams semantics the design actually depends on: consumer groups,
pending entries, explicit acks. The two providers stay fakes, because spending
money to prove a pipe is connected is not a trade worth making.

Retrieval is still a seam (P1). The null retriever returns nothing, the
confidence gate routes the turn to the script's fallback node, and the
customer gets a scripted clarification. That is a correct turn — and it proves
the whole path, because the reply still has to travel back out through the
outbound stream and reach the vendor.
"""

import hashlib
import hmac
import json
from base64 import b64encode
from urllib.parse import urlencode
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text as sql
from sqlalchemy import text as sql_text

from moc.agent.orchestrator import Orchestrator, Retrieval
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import ConversationState, TurnInput
from moc.api.webhooks import build_app
from moc.channels.base import Channel, ChannelAccount
from moc.channels.twilio_wa import TwilioWhatsApp
from moc.channels.valkey import ValkeyEventLog, ValkeyInboundQueue
from moc.config_store import load
from moc.llm.fake import FakeProvider
from moc.llm.router import Router
from moc.workers.inbound import InboundWorker
from moc.workers.outbound import OutboundJob, OutboundWorker

QUEUES = load("workers/queues")
WHATSAPP = load("channels/whatsapp")
SCRIPT = "scripts/education/fees"

SECRET = "auth-token-for-this-account"
SECRET_REF = "twilio/test/wa"
PATH = "/webhooks/twilio/whatsapp"
URL = f"https://moc.example{PATH}"
BUSINESS_NUMBER = "+201555000111"
CUSTOMER = "+201012345678"

#: A test-only database index. Flushed between tests, never the dev index.


class NullRetriever:
    """P1's seam, honestly empty.

    Returns nothing at zero confidence, which is what an un-ingested tenant
    actually looks like. §7.5 then routes the turn to the script's fallback
    rather than to composition — the correct behaviour, and the one that keeps
    a fee out of a reply nobody could source.
    """

    async def search(self, *, query: str) -> Retrieval:
        return Retrieval(passages=(), confidence=0.0)


class FixedExtractor:
    """Intent and slots, as Haiku would supply them (§3.1).

    The faculty slot is filled deliberately. Without it the engine asks for the
    slot and the turn never reaches the confidence gate — correct behaviour,
    and documented precedence in `script_engine`, but it would mean this file
    claimed to exercise the empty-retrieval path while exercising the
    missing-slot one.
    """

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput:
        return TurnInput(intent="fees", slots={"faculty": "engineering"})


@pytest_asyncio.fixture(loop_scope="session")
async def tenant(engine, tenant_tables):
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.tenancy.models import Tenant

    async with AsyncSession(engine, expire_on_commit=False) as s:
        # The shared ordering from conftest rather than a local list: this
        # fixture already went stale once when new tenant-scoped tables
        # landed, and a stale list here reads as a foreign-key error in an
        # unrelated test.
        for table in tenant_tables:
            await s.execute(sql(f"DELETE FROM {table}"))  # noqa: S608
        row = Tenant(slug="pipeline", name="Pipeline", vertical="education")
        s.add(row)
        await s.commit()
        return row


@pytest.fixture
def account(tenant) -> ChannelAccount:
    return ChannelAccount(
        id=uuid4(),
        tenant_id=tenant.id,
        channel=Channel.whatsapp,
        account_ref=BUSINESS_NUMBER,
        secret_ref=SECRET_REF,
    )


class FakeSecrets:
    """`secret_ref` -> secret, without a secret store.

    What matters here is that the secret arrives by reference rather than
    riding on the account the bootstrap lookup returned (Task 21).
    """

    def for_ref(self, secret_ref: str) -> str:
        return {SECRET_REF: SECRET}[secret_ref]


class Registry:
    def __init__(self, account: ChannelAccount) -> None:
        self._account = account

    async def resolve(self, *, channel: Channel, account_ref: str):
        return self._account if account_ref == self._account.account_ref else None


def form(**overrides) -> bytes:
    return urlencode(
        {
            "MessageSid": "SM-pipeline-1",
            "AccountSid": "AC0",
            "From": f"{WHATSAPP['address_prefix']}{CUSTOMER}",
            "To": f"{WHATSAPP['address_prefix']}{BUSINESS_NUMBER}",
            "Body": "كام رسوم الساعة؟",
            "NumMedia": "0",
            **overrides,
        }
    ).encode()


def sign(raw: bytes) -> str:
    from urllib.parse import parse_qsl

    canonical = URL + "".join(k + v for k, v in sorted(parse_qsl(raw.decode(), True)))
    return b64encode(hmac.new(SECRET.encode(), canonical.encode(), hashlib.sha1).digest()).decode()


def orchestrator() -> Orchestrator:
    provider = FakeProvider("anthropic", text="unused — retrieval is empty")
    router = Router(
        config=load("llm/routing"),
        providers={"anthropic": provider, "openai": FakeProvider("openai")},
    )
    return Orchestrator(
        engine=ScriptEngine.from_config(SCRIPT),
        router=router,
        retriever=NullRetriever(),
        extractor=FixedExtractor(),
    )


def sender(handler) -> TwilioWhatsApp:
    return TwilioWhatsApp(
        account_sid="AC0",
        auth_token=SECRET,
        sender=BUSINESS_NUMBER,
        config=WHATSAPP,
        transport=httpx.MockTransport(handler),
    )


async def deliver(app, raw: bytes, **headers):
    headers.setdefault(WHATSAPP["signature_header"], sign(raw))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        return await client.post(
            PATH,
            content=raw,
            headers={"content-type": WHATSAPP["form_content_type"], **headers},
        )


# ─────────────────────────── the whole wire ───────────────────────────


async def test_a_signed_webhook_produces_a_sent_reply(valkey, app_engine, account):
    """Week 2's exit criterion, driven end to end.

    Webhook -> Valkey stream -> inbound worker -> orchestrator -> outbound
    stream -> sender -> vendor. Real streams, real consumer groups, real acks;
    fakes only at the two provider edges.
    """
    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets())

    response = await deliver(app, form())
    assert response.status_code == 200

    inbound = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    assert await inbound.run_once() == 1

    sent = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(dict(x.split("=", 1) for x in request.content.decode().split("&")))
        return httpx.Response(201, json={"sid": "SM-out", "status": "queued"})

    outbound = OutboundWorker(client=valkey, provider=sender(capture), config=QUEUES)
    assert await outbound.run_once() == 1

    assert len(sent) == 1, "exactly one reply reached the vendor"
    assert sent[0]["To"].endswith(CUSTOMER.replace("+", "%2B"))
    assert sent[0]["Body"], "the customer got words, not an empty message"


async def test_the_reply_is_the_scripted_fallback_when_retrieval_is_empty(
    valkey, app_engine, account
):
    """An un-ingested tenant answers safely rather than not at all (§7.5).

    Empty retrieval means no grounding, so the turn cannot reach composition.
    What the customer must not get is silence or an error — they get the
    script's clarification, in Masri, with a next step.
    """
    from moc.agent.state import Register

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets())
    await deliver(app, form())

    inbound = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    await inbound.run_once()

    raw = await valkey.xrange(QUEUES["outbound"]["stream"])
    assert len(raw) == 1
    job = OutboundJob.from_json(json.loads(raw[0][1]["payload"]))
    expected = load("agent/replies")["replies"]["low_confidence"][Register.masri]
    assert job.text == expected


async def test_the_turn_is_metered_under_the_right_tenant(valkey, app_engine, account):
    """The worker opens a tenant session; the ledger proves which one."""
    from moc.tenancy.context import tenant_session

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets()
    )
    await deliver(app, form())

    inbound = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    await inbound.run_once()

    async with tenant_session(app_engine, account.tenant_id) as s:
        kinds = (await s.execute(sql("SELECT kind FROM usage_ledger"))).scalars().all()
    assert "message_in" in kinds
    assert "message_out" in kinds


# ─────────────────────────── stream semantics ───────────────────────────


async def test_a_processed_entry_is_acked_and_not_reprocessed(valkey, app_engine, account):
    """At-least-once needs the ack, or every restart replays the backlog."""
    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets()
    )
    await deliver(app, form())

    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    assert await worker.run_once() == 1
    assert await worker.run_once() == 0

    pending = await valkey.xpending(QUEUES["inbound"]["stream"], QUEUES["inbound"]["group"])
    assert pending["pending"] == 0


async def test_a_failing_turn_leaves_the_entry_pending_rather_than_acking_it(
    valkey, app_engine, account
):
    """A crashed turn must be reclaimable.

    Acking before the work succeeds is how a customer's message disappears
    into a worker that died three seconds later — invisible, unreproducible,
    and indistinguishable from the customer never having written.
    """

    class BrokenOrchestrator:
        async def handle(self, **kwargs):
            raise RuntimeError("turn blew up")

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets()
    )
    await deliver(app, form())

    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=BrokenOrchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    assert await worker.run_once() == 0

    pending = await valkey.xpending(QUEUES["inbound"]["stream"], QUEUES["inbound"]["group"])
    assert pending["pending"] == 1


async def test_a_poisoned_entry_is_dead_lettered_rather_than_retried_forever(
    valkey, app_engine, account
):
    """One bad message must not become a worker that processes nothing else."""

    class BrokenOrchestrator:
        async def handle(self, **kwargs):
            raise RuntimeError("always fails")

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets()
    )
    await deliver(app, form())

    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=BrokenOrchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    for _ in range(QUEUES["inbound"]["max_attempts"] + 1):
        await worker.run_once()

    dead = await valkey.xrange(QUEUES["inbound"]["dead_letter_stream"])
    assert len(dead) == 1
    pending = await valkey.xpending(QUEUES["inbound"]["stream"], QUEUES["inbound"]["group"])
    assert pending["pending"] == 0, "a dead-lettered entry must be acked, or it blocks forever"


# ─────────────────────────── idempotency, for real ───────────────────────────


async def test_a_vendor_redelivery_produces_one_turn_not_two(valkey, app_engine, account):
    """The §6.2 guarantee, through the real key store rather than a fake set."""
    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets())

    first = await deliver(app, form())
    second = await deliver(app, form())
    assert (first.status_code, second.status_code) == (200, 200)

    entries = await valkey.xrange(QUEUES["inbound"]["stream"])
    assert len(entries) == 1


async def test_a_released_claim_lets_the_retry_through(valkey):
    """`release` must actually clear the key, not just satisfy the Protocol."""
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    assert await events.claim("SM-x") is True
    assert await events.claim("SM-x") is False
    await events.release("SM-x")
    assert await events.claim("SM-x") is True


async def test_a_claim_expires_so_the_key_store_does_not_grow_without_bound(valkey):
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    await events.claim("SM-ttl")
    ttl = await valkey.ttl(f"{QUEUES['idempotency']['key_prefix']}SM-ttl")
    assert 0 < ttl <= QUEUES["idempotency"]["ttl_seconds"]


# ─────────────────────────── conversation state ───────────────────────────


async def test_the_thread_is_reused_across_turns(valkey, app_engine, account):
    """One customer, one thread. Two rows would mean two script cursors.

    Slots live on the conversation, so a second row is a customer who has to
    say which faculty twice — the failure edu-0004 exists to catch, arriving
    through the schema instead of the model.
    """
    from moc.tenancy.context import tenant_session

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets())
    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )

    await deliver(app, form(MessageSid="SM-turn-1"))
    await worker.run_once()
    await deliver(app, form(MessageSid="SM-turn-2"))
    await worker.run_once()

    async with tenant_session(app_engine, account.tenant_id) as s:
        rows = (
            await s.execute(
                sql("SELECT state FROM conversations WHERE sender_ref = :s"),
                {"s": CUSTOMER},
            )
        ).scalars().all()

    assert len(rows) == 1, "a second turn opened a second thread"
    assert rows[0]["consecutive_clarifications"] == 2, "state did not carry across turns"


# ─────────────────────────── outbound policy ───────────────────────────


async def test_outbound_dead_letters_after_the_configured_attempts(valkey):
    """A permanently rejected send is an alert, not an infinite retry."""

    def rejects(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "invalid number"})

    job = OutboundJob(
        tenant_id=str(uuid4()), channel=Channel.whatsapp, to=CUSTOMER, text="أهلا"
    )
    await ValkeyInboundQueue(client=valkey, config=QUEUES).publish_raw(
        QUEUES["outbound"]["stream"], job.to_json()
    )

    worker = OutboundWorker(client=valkey, provider=sender(rejects), config=QUEUES)
    for _ in range(QUEUES["outbound"]["max_attempts"] + 1):
        await worker.run_once()

    dead = await valkey.xrange(QUEUES["outbound"]["dead_letter_stream"])
    assert len(dead) == 1


async def test_the_rate_limiter_is_shared_across_sender_processes(valkey):
    """The bucket lives in Valkey, not in the process (§6.2).

    Two senders each holding their own bucket each allow the full rate, which
    is the same as having no limit — and the limit exists because Meta's is
    real and being throttled by them is worse than throttling ourselves.
    """
    limit = QUEUES["rate_limit"]
    tenant = str(uuid4())
    one = OutboundWorker(client=valkey, provider=sender(lambda r: None), config=QUEUES)
    two = OutboundWorker(client=valkey, provider=sender(lambda r: None), config=QUEUES)

    taken = 0
    for index in range(limit["capacity"] + 5):
        worker = one if index % 2 == 0 else two
        if await worker.take_token(tenant):
            taken += 1

    assert taken == limit["capacity"], (
        "the two processes drew from separate buckets — each would allow the full rate"
    )


# ─────────────────── the thread an agent will read (§9) ───────────────────
#
# The inbox tables existed and were tested before anything populated them, so
# a real handoff would have shown an agent an empty thread — the acceptance
# criterion held in tests and not in production. These close that.


async def test_a_turn_records_both_sides_of_the_conversation(valkey, app_engine, account):
    """The customer's words and the bot's reply, in order.

    Written inside the turn's transaction, not after it. A thread that can
    disagree with the conversation state is one an agent reads while the bot
    believes something else — and the disagreement appears exactly when a turn
    half-fails, which is when a human is most likely to be looking.
    """
    from moc.agent.handoff import MessageLog
    from moc.tenancy.context import tenant_session

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets()
    )
    await deliver(app, form())

    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    await worker.run_once()

    async with tenant_session(app_engine, account.tenant_id) as session:
        contact_id = (
            await session.execute(
                sql_text("SELECT contact_id FROM conversations WHERE sender_ref = :s"),
                {"s": CUSTOMER},
            )
        ).scalar_one()
        assert contact_id is not None, "the turn must attach the thread to a contact"
        thread = await MessageLog(session=session).history_for_contact(contact_id=contact_id)

    assert [m.author for m in thread] == ["customer", "bot"]
    assert thread[0].body == "كام رسوم الساعة؟"
    assert thread[1].body, "the bot's reply is part of the thread the agent reads"
    assert all(m.channel == "whatsapp" for m in thread)


async def test_a_contact_is_created_once_and_reused_across_turns(
    valkey, app_engine, account
):
    """One customer, one contact row.

    A contact per message would make the agent's thread one message long and
    defeat the join the inbox reads through.
    """
    from moc.tenancy.context import tenant_session

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets()
    )
    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    for n in range(3):
        await deliver(app, form(MessageSid=f"SM-contact-{n}", Body=f"سؤال {n}"))
        await worker.run_once()

    async with tenant_session(app_engine, account.tenant_id) as session:
        contacts = (
            await session.execute(sql_text("SELECT count(*) FROM contacts"))
        ).scalar_one()
        messages = (
            await session.execute(sql_text("SELECT count(*) FROM messages"))
        ).scalar_one()

    assert contacts == 1
    assert messages == 6, "three turns, both sides of each"


async def test_a_handed_off_conversation_shows_the_agent_what_happened(
    valkey, app_engine, account
):
    """P1's acceptance criterion, end to end.

    A real conversation runs through the webhook and the worker, a handoff is
    opened on it, and the inbox thread returns the messages that actually
    happened — rather than an empty list, which is what it returned while the
    tables existed and nothing wrote to them.
    """
    import httpx

    from moc.agent.handoff import HandoffStore
    from moc.api.inbox import AgentPrincipal, build_inbox
    from moc.tenancy.context import tenant_session

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets()
    )
    await deliver(app, form(MessageSid="SM-handoff-1"))
    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
    )
    await worker.run_once()

    async with tenant_session(app_engine, account.tenant_id) as session:
        row = (
            await session.execute(
                sql_text("SELECT id, state FROM conversations WHERE sender_ref = :s"),
                {"s": CUSTOMER},
            )
        ).one()
        opened = await HandoffStore(session=session).open(
            conversation_id=row.id, reason="customer asked for a human", resume_state=row.state
        )
        await session.commit()

    class Publisher:
        def __init__(self) -> None:
            self.jobs: list = []

        async def publish(self, job) -> None:
            self.jobs.append(job)

    class Events:
        async def publish(self, *, tenant_id, event) -> None:
            return None

        async def subscribe(self, *, tenant_id):
            return
            yield  # pragma: no cover

    async def authenticate(request) -> AgentPrincipal:
        return AgentPrincipal(tenant_id=account.tenant_id, agent_id="agent-1")

    inbox = build_inbox(
        engine=app_engine,
        publisher=Publisher(),
        events=Events(),
        authenticate=authenticate,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=inbox), base_url="https://moc.example"
    ) as client:
        listed = await client.get("/inbox")
        thread = await client.get(f"/inbox/{opened.id}/thread")

    assert [item["reason"] for item in listed.json()] == ["customer asked for a human"]
    bodies = [message["body"] for message in thread.json()]
    assert "كام رسوم الساعة؟" in bodies, "the agent must see what the customer actually wrote"
    assert len(bodies) == 2


async def test_a_live_handoff_suspends_the_bot(valkey, app_engine, account):
    """An agent has taken the conversation. The bot must not answer over them.

    Without this the customer's next message produces a bot reply *and* the
    agent's, both delivered — an officer types a considered answer and the bot
    argues with them in front of a student. It is the most visible failure the
    inbox screen can produce, and nothing in the pipeline prevented it.

    The customer's words are still recorded: the agent has to see what was said
    while they were reading, and a message dropped because a human was
    attached is a message nobody ever answers.
    """
    from sqlalchemy import text as sql

    from moc.agent.conversations import ConversationStore
    from moc.agent.handoff import ContactStore, HandoffStore
    from moc.agent.script_engine import ScriptEngine
    from moc.tenancy.context import tenant_session

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets())

    # One ordinary turn, so the conversation exists.
    assert (await deliver(app, form())).status_code == 200
    inbound = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=orchestrator(),
        script=SCRIPT, config=QUEUES,
    )
    assert await inbound.run_once() == 1

    # A human takes it.
    async with tenant_session(app_engine, account.tenant_id) as session:
        store = ConversationStore(session=session, engine=ScriptEngine.from_config(SCRIPT))
        await ContactStore(session=session).resolve(contact_ref=CUSTOMER)
        conversation_id = await store.find(channel="whatsapp", sender_ref=CUSTOMER)
        assert conversation_id is not None
        await HandoffStore(session=session).open(
            conversation_id=conversation_id,
            reason="agent took over",
            resume_state={"script_id": "education_fees", "script_version": 1,
                          "node": "fees", "slots": {}, "consecutive_clarifications": 0},
        )
        await session.commit()

    class Refuses:
        """The orchestrator must not be reached at all. A double that answered
        would let this test pass on a worker that called it and discarded the
        reply — which still bills the tenant for a turn nobody sees."""

        async def handle(self, **kwargs):
            raise AssertionError("the bot answered over a human")

    # A distinct MessageSid: the webhook dedupes on it, and reusing the first
    # one would have this test pass by never delivering a second message.
    assert (
        await deliver(app, form(MessageSid="SM-pipeline-2", Body="وكمان سؤال تاني"))
    ).status_code == 200
    suspended = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=Refuses(),
        script=SCRIPT, config=QUEUES,
    )
    assert await suspended.run_once() == 1, "the entry must still be acked, not retried"

    async with tenant_session(app_engine, account.tenant_id) as session:
        bodies = [
            row[0]
            for row in (
                await session.execute(
                    sql("SELECT body FROM messages WHERE author = 'customer' ORDER BY seq")
                )
            ).all()
        ]
    assert "وكمان سؤال تاني" in bodies, "the agent cannot see what was said"

    # And nothing was queued for the customer.
    outbound = OutboundWorker(client=valkey, provider=sender(lambda r: None), config=QUEUES)
    assert await outbound.run_once() == 0


async def test_a_published_script_is_what_the_bot_runs(valkey, app_engine, account):
    """Publishing pins a version, and something has to read it.

    Before this the worker built one `ScriptEngine.from_config` at construction
    and reused it for every turn and every tenant, so a script edited in the
    console changed nothing the bot did — someone edits a script, asks the
    question, and gets the old answer.

    Worse than nothing changing: `_require_pinned_version` raises when a
    conversation's state names a version the engine is not running, so
    publishing v2 would have broken every in-flight conversation with an
    exception rather than merely ignoring the edit.
    """
    from moc.agent.scripts import ScriptStore
    from moc.config_store import load as load_config
    from moc.tenancy.context import tenant_session

    published = {**load_config(SCRIPT), "version": 2}
    await ScriptStore(engine=app_engine).save_draft(
        tenant_id=account.tenant_id,
        script_id=published["script_id"],
        body=published,
    )
    await ScriptStore(engine=app_engine).publish(
        tenant_id=account.tenant_id, script_id=published["script_id"], agent_id="ali"
    )

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets())
    assert (await deliver(app, form(MessageSid="SM-pinned-1"))).status_code == 200

    worker = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=orchestrator(),
        script=SCRIPT, config=QUEUES, scripts=ScriptStore(engine=app_engine),
    )
    assert await worker.run_once() == 1

    from sqlalchemy import text as sql

    async with tenant_session(app_engine, account.tenant_id) as session:
        state = (
            await session.execute(sql("SELECT state FROM conversations"))
        ).scalar_one()
    assert state["script_version"] == 2, "the published script is not what ran"


async def test_a_conversation_pinned_to_an_older_version_keeps_running(
    valkey, app_engine, account
):
    """The half that publishing exists to protect.

    A customer three turns into version 1 must not be moved onto a script they
    never started — and must not hit `_require_pinned_version`'s exception
    either, which is what an unresolved engine would have given them.
    """
    from sqlalchemy import text as sql

    from moc.agent.scripts import ScriptStore
    from moc.config_store import load as load_config
    from moc.tenancy.context import tenant_session

    scripts = ScriptStore(engine=app_engine)
    body = load_config(SCRIPT)
    for version in (1, 2):
        await scripts.save_draft(
            tenant_id=account.tenant_id,
            script_id=body["script_id"],
            body={**body, "version": version},
        )
        await scripts.publish(
            tenant_id=account.tenant_id, script_id=body["script_id"], agent_id="ali"
        )

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(registry=Registry(account), queue=queue, events=events, secrets=FakeSecrets())
    worker = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=orchestrator(),
        script=SCRIPT, config=QUEUES, scripts=scripts,
    )

    assert (await deliver(app, form(MessageSid="SM-pinned-2"))).status_code == 200
    assert await worker.run_once() == 1

    # Pin the conversation back to version 1, as an in-flight one would be.
    async with tenant_session(app_engine, account.tenant_id) as session:
        await session.execute(
            sql(
                "UPDATE conversations SET state = jsonb_set(state, '{script_version}', '1')"
            )
        )
        await session.commit()

    assert (await deliver(app, form(MessageSid="SM-pinned-3"))).status_code == 200
    assert await worker.run_once() == 1, "a pinned conversation errored instead of running"

    async with tenant_session(app_engine, account.tenant_id) as session:
        state = (
            await session.execute(sql("SELECT state FROM conversations"))
        ).scalar_one()
    assert state["script_version"] == 1, "the customer was moved onto a newer script"
