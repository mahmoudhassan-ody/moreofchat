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
import uuid
from base64 import b64encode
from datetime import UTC, datetime
from pathlib import Path
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


class ByChannel:
    """Single-tenant test wiring: one adapter per channel, tenant ignored.

    Deliberately not in `src`. The production registry keys on the tenant as
    well, and a convenience that ignores it, living in the source tree, is the
    one that ends up wired — after which every tenant's replies go out under
    one name.
    """

    def __init__(self, providers: dict) -> None:
        self._providers = dict(providers)

    async def for_job(self, *, tenant_id: str, channel: str):
        return self._providers.get(channel)


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

    outbound = OutboundWorker(
        client=valkey, providers=ByChannel({"whatsapp": sender(capture)}), config=QUEUES
    )
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

    worker = OutboundWorker(
        client=valkey, providers=ByChannel({"whatsapp": sender(rejects)}), config=QUEUES
    )
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
    one = OutboundWorker(
        client=valkey, providers=ByChannel({"whatsapp": sender(lambda r: None)}), config=QUEUES
    )
    two = OutboundWorker(
        client=valkey, providers=ByChannel({"whatsapp": sender(lambda r: None)}), config=QUEUES
    )

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
    outbound = OutboundWorker(
        client=valkey, providers=ByChannel({"whatsapp": sender(lambda r: None)}), config=QUEUES
    )
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


async def test_a_second_channel_is_not_sent_through_the_first_channels_adapter(valkey):
    """The outbound worker read `job.channel` nowhere.

    One stream, one consumer group, one provider — so the moment a second
    channel exists, whichever sender drew the entry sends it. A Telegram reply
    would go to Twilio with a chat id where a phone number belongs, and Twilio
    would reject it: a customer silently unanswered, and a dead-letter row
    blaming the number.
    """
    from moc.channels.base import OutboundJob, OutboundReceipt

    class Recording:
        def __init__(self, name: str) -> None:
            self.name = name
            self.sent: list[str] = []

        async def send(self, *, to: str, **kwargs) -> OutboundReceipt:
            self.sent.append(to)
            return OutboundReceipt(provider_message_id="x", status="ok")

    whatsapp, telegram = Recording("whatsapp"), Recording("telegram")
    worker = OutboundWorker(
        client=valkey,
        providers=ByChannel({"whatsapp": whatsapp, "telegram": telegram}),
        config=QUEUES,
    )

    for channel, to in (("whatsapp", "+201012345678"), ("telegram", "987654321")):
        await valkey.xadd(
            QUEUES["outbound"]["stream"],
            {"payload": OutboundJob(
                tenant_id=str(uuid.uuid4()), channel=channel, to=to, text="مرحبا"
            ).to_json()},
        )

    assert await worker.run_once() == 2
    assert whatsapp.sent == ["+201012345678"]
    assert telegram.sent == ["987654321"]


async def test_a_job_for_a_channel_with_no_provider_is_dead_lettered(valkey):
    """Not retried forever, and above all not sent through whichever adapter
    happens to be wired. A channel nobody configured is a configuration
    problem for a person to see, and retrying it is how one bad row becomes a
    sender that delivers nothing else."""
    from moc.channels.base import OutboundJob

    worker = OutboundWorker(
        client=valkey, providers=ByChannel({}), config=QUEUES
    )
    await valkey.xadd(
        QUEUES["outbound"]["stream"],
        {"payload": OutboundJob(
            tenant_id=str(uuid.uuid4()), channel="carrier-pigeon", to="x", text="hi"
        ).to_json()},
    )

    assert await worker.run_once() == 0
    dead = await valkey.xrange(QUEUES["outbound"]["dead_letter_stream"])
    assert dead, "an unroutable job vanished instead of being dead-lettered"


# ─────────────────────── Telegram, end to end ───────────────────────

TELEGRAM_SECRET = "telegram-webhook-secret-value"  # noqa: S105 - a test fixture
TELEGRAM_TOKEN = "123456:AAHfake-bot-token"  # noqa: S105 - a test fixture
CHAT = "987654321"


def telegram_update(**overrides) -> bytes:
    return json.dumps(
        {
            "update_id": 5000,
            "message": {
                "message_id": 11,
                "date": 1755859200,
                "chat": {"id": int(CHAT), "type": "private"},
                "text": "كام رسوم الساعة؟",
                **overrides,
            },
        },
        ensure_ascii=False,
    ).encode()


@pytest_asyncio.fixture(loop_scope="session")
async def telegram_account(engine, account):
    """The same tenant, with a Telegram bot connected as well."""
    from sqlalchemy import text as sql
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.channels.base import Channel, ChannelAccount

    account_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        await s.execute(
            sql(
                "INSERT INTO channel_accounts "
                "(id, tenant_id, channel, address, secret_ref, signing_secret) "
                "VALUES (:id, :t, 'telegram', 'sinai_bot', 'telegram/sinai/bot', :secret)"
            ),
            {"id": account_id, "t": account.tenant_id, "secret": TELEGRAM_SECRET},
        )
        await s.commit()
    return ChannelAccount(
        id=account_id,
        tenant_id=account.tenant_id,
        channel=Channel.telegram,
        account_ref="sinai_bot",
        secret_ref="telegram/sinai/bot",
    )


class TwoChannelRegistry:
    """Resolves both of this tenant's accounts, by channel and address.

    A registry that ignored the channel would answer a Telegram lookup with a
    WhatsApp account, and the secret it carries would then verify nothing.
    """

    def __init__(self, *accounts) -> None:
        self._accounts = {(a.channel, a.account_ref): a for a in accounts}

    async def resolve(self, *, channel, account_ref: str):
        return self._accounts.get((channel, account_ref))


class TwoSecrets:
    """Both channels' secrets, by reference.

    Keyed rather than branched: the first version returned a WhatsApp constant
    that did not exist, and every test passed because no test ever asked for
    the WhatsApp branch — a NameError sitting in a fake, waiting for the first
    test that used both channels at once.
    """

    def for_ref(self, secret_ref: str) -> str:
        return {SECRET_REF: SECRET, "telegram/sinai/bot": TELEGRAM_SECRET}[secret_ref]


async def test_a_telegram_message_produces_a_sent_reply(
    valkey, app_engine, account, telegram_account
):
    """Task 35's exit criterion, driven end to end.

    Telegram webhook -> Valkey stream -> inbound worker -> orchestrator ->
    outbound stream -> the Telegram sender -> the Bot API. Real streams, real
    consumer groups, real acks; fakes only at the two provider edges.
    """
    from moc.channels.telegram import TelegramBot

    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=TwoChannelRegistry(account, telegram_account),
        queue=queue,
        events=events,
        secrets=TwoSecrets(),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        response = await http.post(
            "/webhooks/telegram/sinai_bot",
            content=telegram_update(),
            headers={
                "content-type": "application/json",
                "X-Telegram-Bot-Api-Secret-Token": TELEGRAM_SECRET,
            },
        )
    assert response.status_code == 200

    inbound = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=orchestrator(),
        script=SCRIPT, config=QUEUES,
    )
    assert await inbound.run_once() == 1

    sent = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 77}})

    bot = TelegramBot(token=TELEGRAM_TOKEN, transport=httpx.MockTransport(capture))
    outbound = OutboundWorker(
        client=valkey, providers=ByChannel({"telegram": bot}), config=QUEUES
    )
    assert await outbound.run_once() == 1
    await bot.aclose()

    assert len(sent) == 1, "exactly one reply reached the Bot API"
    assert sent[0]["chat_id"] == CHAT
    assert sent[0]["text"], "the customer got words, not an empty message"


async def test_a_telegram_webhook_without_the_secret_is_refused(
    valkey, account, telegram_account
):
    """Anyone can POST to this path — it is a public URL with a guessable
    account reference in it, which is exactly why the header is the thing that
    authenticates."""
    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=TwoChannelRegistry(account, telegram_account),
        queue=queue,
        events=events,
        secrets=TwoSecrets(),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        unsigned = await http.post(
            "/webhooks/telegram/sinai_bot",
            content=telegram_update(),
            headers={"content-type": "application/json"},
        )
        wrong = await http.post(
            "/webhooks/telegram/sinai_bot",
            content=telegram_update(),
            headers={
                "content-type": "application/json",
                "X-Telegram-Bot-Api-Secret-Token": "not-the-secret",
            },
        )
        unknown = await http.post(
            "/webhooks/telegram/some_other_bot",
            content=telegram_update(),
            headers={
                "content-type": "application/json",
                "X-Telegram-Bot-Api-Secret-Token": TELEGRAM_SECRET,
            },
        )

    assert (unsigned.status_code, wrong.status_code, unknown.status_code) == (403, 403, 403)

    # And the WhatsApp half of the same app still works, on the same tenant.
    # `TwoSecrets` claims to serve both channels; without this nothing asks it
    # to, which is how its first version shipped with an undefined name in the
    # branch no test reached.
    assert (await deliver(app, form(MessageSid="SM-two-channel"))).status_code == 200


async def test_an_edit_is_acknowledged_and_not_answered(
    valkey, account, telegram_account
):
    """Telegram delivers edits, channel posts and callback queries down the
    same webhook. Answering a customer's edit of a question they already asked
    is worse than ignoring it, and 200 stops the retries."""
    queue = ValkeyInboundQueue(client=valkey, config=QUEUES)
    events = ValkeyEventLog(client=valkey, config=QUEUES)
    app = build_app(
        registry=TwoChannelRegistry(account, telegram_account),
        queue=queue,
        events=events,
        secrets=TwoSecrets(),
    )
    edit = json.dumps({"update_id": 6, "edited_message": {"message_id": 11}}).encode()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        response = await http.post(
            "/webhooks/telegram/sinai_bot",
            content=edit,
            headers={
                "content-type": "application/json",
                "X-Telegram-Bot-Api-Secret-Token": TELEGRAM_SECRET,
            },
        )
    assert response.status_code == 200

    # Acknowledged, and nothing enqueued. Asserted on the stream rather than
    # by running a worker: "the worker found nothing" is also what a broken
    # consumer group looks like.
    assert await valkey.xlen(QUEUES["inbound"]["stream"]) == 0


# ─────────────────────── Instagram and Messenger, end to end ───────────────────────

META_APP_SECRET = "meta-app-secret"  # noqa: S105 - a test fixture
META_VERIFY = "meta-verify-token"  # noqa: S105 - a test fixture
PAGE_ID = "1122334455"
IG_ID = "5544332211"
FAN = "9988776655"


def meta_body(obj: str, page: str, mid: str = "m_1") -> bytes:
    # A current timestamp, not a frozen one. `received_at` comes from Meta's
    # own value and feeds §6.2's 24-hour window, so a fixture pinned to a date
    # in the past makes every reply legitimately out of window — which is the
    # adapter being right and the test being stale.
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    return json.dumps(
        {
            "object": obj,
            "entry": [
                {
                    "id": page,
                    "time": now_ms // 1000,
                    "messaging": [
                        {
                            "sender": {"id": FAN},
                            "recipient": {"id": page},
                            "timestamp": now_ms,
                            "message": {"mid": mid, "text": "كام رسوم الساعة؟"},
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
    ).encode()


def meta_sign(raw: bytes) -> str:
    return "sha256=" + hmac.new(META_APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()


@pytest_asyncio.fixture(loop_scope="session")
async def meta_accounts(engine, account):
    """The same tenant with a Facebook page and an Instagram account."""
    from sqlalchemy import text as sql
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.channels.base import Channel, ChannelAccount

    rows = {"messenger": (uuid.uuid4(), PAGE_ID), "instagram": (uuid.uuid4(), IG_ID)}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for channel, (row_id, address) in rows.items():
            await s.execute(
                sql(
                    "INSERT INTO channel_accounts "
                    "(id, tenant_id, channel, address, secret_ref, signing_secret) "
                    "VALUES (:id, :t, :ch, :addr, 'meta/app/secret', :secret)"
                ),
                {"id": row_id, "t": account.tenant_id, "ch": channel,
                 "addr": address, "secret": META_APP_SECRET},
            )
        await s.commit()
    return {
        channel: ChannelAccount(
            id=row_id, tenant_id=account.tenant_id, channel=Channel(channel),
            account_ref=address, secret_ref="meta/app/secret",
        )
        for channel, (row_id, address) in rows.items()
    }


class MetaSecrets:
    def for_ref(self, secret_ref: str) -> str:
        return {
            SECRET_REF: SECRET,
            "meta/app/secret": META_APP_SECRET,
            "meta/app/verify_token": META_VERIFY,
        }[secret_ref]


def meta_app(account, meta_accounts, valkey):
    return build_app(
        registry=TwoChannelRegistry(account, *meta_accounts.values()),
        queue=ValkeyInboundQueue(client=valkey, config=QUEUES),
        events=ValkeyEventLog(client=valkey, config=QUEUES),
        secrets=MetaSecrets(),
    )


@pytest.mark.parametrize(
    ("obj", "page", "channel"),
    [("page", PAGE_ID, "messenger"), ("instagram", IG_ID, "instagram")],
)
async def test_a_meta_message_produces_a_sent_reply(
    valkey, app_engine, account, meta_accounts, obj, page, channel
):
    """Task 36's exit criterion, on both surfaces, through one webhook.

    Parametrised rather than duplicated: the whole claim is that one
    integration serves both, and two copies of this test would let it stop
    being true in one of them.
    """
    from moc.channels.meta import MetaMessenger

    app = meta_app(account, meta_accounts, valkey)
    raw = meta_body(obj, page, mid=f"m_{channel}")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        response = await http.post(
            "/webhooks/meta",
            content=raw,
            headers={"content-type": "application/json",
                     "X-Hub-Signature-256": meta_sign(raw)},
        )
    assert response.status_code == 200

    inbound = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=orchestrator(),
        script=SCRIPT, config=QUEUES,
    )
    assert await inbound.run_once() == 1

    sent = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message_id": "m_out", "recipient_id": FAN})

    bot = MetaMessenger(
        page_id=page, access_token="tok", transport=httpx.MockTransport(capture)
    )
    outbound = OutboundWorker(
        client=valkey, providers=ByChannel({channel: bot}), config=QUEUES
    )
    assert await outbound.run_once() == 1
    await bot.aclose()

    assert len(sent) == 1
    assert sent[0]["recipient"] == {"id": FAN}
    assert sent[0]["message"]["text"], "the customer got words, not an empty message"


async def test_the_subscription_handshake_answers_the_challenge(
    valkey, account, meta_accounts
):
    app = meta_app(account, meta_accounts, valkey)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        good = await http.get(
            "/webhooks/meta",
            params={"hub.mode": "subscribe", "hub.verify_token": META_VERIFY,
                    "hub.challenge": "1158201444"},
        )
        bad = await http.get(
            "/webhooks/meta",
            params={"hub.mode": "subscribe", "hub.verify_token": "guess",
                    "hub.challenge": "1158201444"},
        )

    assert (good.status_code, good.text) == (200, "1158201444")
    assert bad.status_code == 403


async def test_an_unsigned_meta_delivery_is_refused(valkey, account, meta_accounts):
    app = meta_app(account, meta_accounts, valkey)
    raw = meta_body("page", PAGE_ID)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        unsigned = await http.post(
            "/webhooks/meta", content=raw, headers={"content-type": "application/json"}
        )
        forged = await http.post(
            "/webhooks/meta",
            content=raw,
            headers={"content-type": "application/json",
                     "X-Hub-Signature-256": "sha256=" + "0" * 64},
        )

    assert (unsigned.status_code, forged.status_code) == (403, 403)
    assert await valkey.xlen(QUEUES["inbound"]["stream"]) == 0


async def test_an_echo_reaches_no_queue(valkey, account, meta_accounts):
    """The loop, at the webhook. An echo enqueued is a turn the bot answers,
    and the answer echoes."""
    app = meta_app(account, meta_accounts, valkey)
    payload = json.loads(meta_body("page", PAGE_ID))
    payload["entry"][0]["messaging"][0]["message"]["is_echo"] = True
    raw = json.dumps(payload).encode()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        response = await http.post(
            "/webhooks/meta",
            content=raw,
            headers={"content-type": "application/json",
                     "X-Hub-Signature-256": meta_sign(raw)},
        )

    assert response.status_code == 200
    assert await valkey.xlen(QUEUES["inbound"]["stream"]) == 0


# ─────────────────────────── email, end to end ───────────────────────────

EMAIL_CREDENTIAL = "sendgrid:parse-password"  # noqa: S105 - a test fixture
SENDGRID_KEY = "SG.pipeline-key"  # noqa: S105 - a test fixture
MAILBOX = "admissions@sinai.edu.eg"
STUDENT = "mariam@example.com"
EMAIL_BOUNDARY = "pipelineBoundary"
EMAIL_CONTENT_TYPE = f'multipart/form-data; boundary="{EMAIL_BOUNDARY}"'


def email_form(**fields: bytes) -> bytes:
    body = b""
    for name, value in fields.items():
        body += (
            f"--{EMAIL_BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        ).encode() + value + b"\r\n"
    return body + f"--{EMAIL_BOUNDARY}--\r\n".encode()


def email_body(
    *,
    message_id: str = "<student-1@example.com>",
    references: str | None = None,
    envelope_from: str = STUDENT,
    spf: bytes = b"pass",
    dkim: bytes = b"{@example.com : pass}",
    extra_headers: str = "",
) -> bytes:
    lines = [
        f"From: Mariam Adel <{STUDENT}>",
        f"To: {MAILBOX}",
        "Subject: Fees",
        f"Message-ID: {message_id}",
    ]
    if references:
        lines.append(f"References: {references}")
    if extra_headers:
        lines.append(extra_headers)
    return email_form(
        headers="\r\n".join(lines).encode(),
        to=MAILBOX.encode(),
        subject=b"Fees",
        text="كام رسوم الساعة؟\r\n\r\nOn Sat, Sinai wrote:\r\n> quoted".encode(),
        envelope=json.dumps({"to": [MAILBOX], "from": envelope_from}).encode(),
        SPF=spf,
        dkim=dkim,
        charsets=b'{"text":"UTF-8","subject":"UTF-8"}',
    )


@pytest_asyncio.fixture(loop_scope="session")
async def email_account(engine, account):
    from sqlalchemy import text as sql
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.channels.base import Channel, ChannelAccount

    row_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        await s.execute(
            sql(
                "INSERT INTO channel_accounts "
                "(id, tenant_id, channel, address, secret_ref, signing_secret) "
                "VALUES (:id, :t, 'email', :addr, 'sendgrid/sinai/parse', :secret)"
            ),
            {"id": row_id, "t": account.tenant_id, "addr": MAILBOX, "secret": EMAIL_CREDENTIAL},
        )
        await s.commit()
    return ChannelAccount(
        id=row_id,
        tenant_id=account.tenant_id,
        channel=Channel.email,
        account_ref=MAILBOX,
        secret_ref="sendgrid/sinai/parse",
    )


class EmailSecrets:
    def for_ref(self, secret_ref: str) -> str:
        return {SECRET_REF: SECRET, "sendgrid/sinai/parse": EMAIL_CREDENTIAL}[secret_ref]


def email_app(account, email_account, valkey):
    return build_app(
        registry=TwoChannelRegistry(account, email_account),
        queue=ValkeyInboundQueue(client=valkey, config=QUEUES),
        events=ValkeyEventLog(client=valkey, config=QUEUES),
        secrets=EmailSecrets(),
    )


def email_credential() -> str:
    return "Basic " + b64encode(EMAIL_CREDENTIAL.encode()).decode()


async def post_email(app, raw: bytes, credential: str | None = None):
    headers = {"content-type": EMAIL_CONTENT_TYPE}
    if credential is not None:
        headers["Authorization"] = credential
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        return await http.post(f"/webhooks/email/{MAILBOX}", content=raw, headers=headers)


async def test_an_email_produces_a_sent_reply(valkey, app_engine, account, email_account):
    """Task 37's exit criterion, driven end to end.

    Inbound Parse -> Valkey stream -> inbound worker -> orchestrator ->
    outbound stream -> the SendGrid sender. Real streams, real consumer groups,
    real acks; a fake only at the vendor edge.
    """
    from moc.channels.sendgrid_email import SendGridEmail

    response = await post_email(
        email_app(account, email_account, valkey), email_body(), email_credential()
    )
    assert response.status_code == 200

    inbound = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=orchestrator(),
        script=SCRIPT, config=QUEUES,
    )
    assert await inbound.run_once() == 1

    sent = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(202, headers={"X-Message-Id": "sg-pipeline-1"})

    mailer = SendGridEmail(
        api_key=SENDGRID_KEY, sender=MAILBOX, transport=httpx.MockTransport(capture)
    )
    outbound = OutboundWorker(client=valkey, providers=ByChannel({"email": mailer}), config=QUEUES)
    assert await outbound.run_once() == 1
    await mailer.aclose()

    assert len(sent) == 1, "exactly one reply reached SendGrid"
    assert sent[0]["personalizations"][0]["to"][0]["email"] == STUDENT
    assert sent[0]["content"][0]["value"], "the customer got words, not an empty message"


async def test_the_reply_lands_in_the_customers_thread(
    valkey, app_engine, account, email_account
):
    """The half of threading that only the wiring can prove.

    The adapter sets `In-Reply-To` from what it is given; whether it is given
    anything depends on the thread reference surviving the queue, the worker
    and the outbound job. A reply with no thread header opens a new
    conversation in the customer's client — every answer a separate email,
    which is what the whole thread-ref field exists to prevent.
    """
    from moc.channels.sendgrid_email import SendGridEmail

    response = await post_email(
        email_app(account, email_account, valkey),
        email_body(references="<root@example.com>\r\n <second@example.com>"),
        email_credential(),
    )
    assert response.status_code == 200

    inbound = InboundWorker(
        client=valkey, engine=app_engine, orchestrator=orchestrator(),
        script=SCRIPT, config=QUEUES,
    )
    assert await inbound.run_once() == 1

    sent = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(202, headers={"X-Message-Id": "sg-pipeline-2"})

    mailer = SendGridEmail(
        api_key=SENDGRID_KEY, sender=MAILBOX, transport=httpx.MockTransport(capture)
    )
    outbound = OutboundWorker(client=valkey, providers=ByChannel({"email": mailer}), config=QUEUES)
    assert await outbound.run_once() == 1
    await mailer.aclose()

    headers = sent[0]["personalizations"][0]["headers"]
    assert headers["In-Reply-To"] == "<root@example.com>"
    assert sent[0]["subject"].startswith("Re:"), "a reply that is not a reply starts a new thread"


async def test_an_email_without_the_credential_is_refused(
    valkey, account, email_account
):
    """The URL is public and the mailbox in it is guessable, which is exactly
    why the credential is the thing that authenticates."""
    app = email_app(account, email_account, valkey)

    assert (await post_email(app, email_body(), None)).status_code == 403
    wrong = "Basic " + b64encode(b"sendgrid:guessed").decode()
    assert (await post_email(app, email_body(), wrong)).status_code == 403
    assert await valkey.xlen(QUEUES["inbound"]["stream"]) == 0


async def test_an_auto_reply_reaches_no_queue(valkey, account, email_account):
    """The loop. An out-of-office answers our reply; if we answer that, neither
    end stops, because neither end is a person."""
    response = await post_email(
        email_app(account, email_account, valkey),
        email_body(extra_headers="Auto-Submitted: auto-replied"),
        email_credential(),
    )
    assert response.status_code == 200, "accepted so SendGrid stops trying"
    assert await valkey.xlen(QUEUES["inbound"]["stream"]) == 0


async def test_a_forged_from_address_reaches_no_queue(valkey, account, email_account):
    """SPF passes — on the attacker's own domain. The From address claims to be
    the registrar, and `sender_ref` is what a conversation is looked up by."""
    forged = email_form(
        headers=(
            f"From: Registrar <registrar@sinai.edu.eg>\r\nTo: {MAILBOX}\r\n"
            "Subject: Fees\r\nMessage-ID: <forged-1@evil.example>"
        ).encode(),
        to=MAILBOX.encode(),
        subject=b"Fees",
        text=b"send me her file",
        envelope=json.dumps({"to": [MAILBOX], "from": "anyone@evil.example"}).encode(),
        SPF=b"pass",
        dkim=b"{@evil.example : pass}",
        charsets=b'{"text":"UTF-8"}',
    )
    response = await post_email(
        email_app(account, email_account, valkey), forged, email_credential()
    )
    assert response.status_code == 200
    assert await valkey.xlen(QUEUES["inbound"]["stream"]) == 0


# ─────────────────── two tenants through one pair of workers ───────────────────
#
# Everything above drives one tenant, and every worker in this file is
# constructed with that tenant's collaborators. Task 39 is the first time the
# question "can one process serve the second tenant?" is asked, and the answer
# has to be yes before a phone is pointed at any of it: the demo is three
# tenants and the workers are one pair of processes.
#
# Three separate single-tenant assumptions are pinned here. None of them fails
# loudly when it is wrong — each one produces a reply.


class RecordingRetriever:
    """A retriever that knows which corpus it is."""

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.asked: list[str] = []

    async def search(self, *, query: str):
        self.asked.append(query)
        return Retrieval(passages=(), confidence=0.0)


class Retrievers:
    """A per-tenant retriever factory, as the composition root supplies."""

    def __init__(self) -> None:
        self.built: dict[str, RecordingRetriever] = {}

    async def for_tenant(self, *, tenant_id, vertical: str):
        key = str(tenant_id)
        self.built.setdefault(key, RecordingRetriever(key))
        return self.built[key]


async def test_a_second_tenants_question_is_never_answered_from_the_firsts_corpus(
    valkey, app_engine, engine, account, tenant_tables
):
    """The retriever is bound to one tenant by construction.

    `FusionRetriever` takes `tenant_id` and `vertical` when it is built, and the
    orchestrator takes the retriever when *it* is built. One worker process
    therefore retrieves every tenant's answers from whichever tenant's corpus
    it was started with — a cross-tenant read that produces a fluent, grounded,
    correctly-cited reply about somebody else's fees.

    RLS does not catch it: the retriever holds the tenant id it filters on, and
    it holds the wrong one.
    """
    second = await _a_second_tenant(engine, "second-tenant")

    retrievers = Retrievers()
    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
        retrievers=retrievers,
    )

    for tenant_id in (account.tenant_id, second):
        await ValkeyInboundQueue(client=valkey, config=QUEUES).publish(
            _message_for(tenant_id, account)
        )
    assert await worker.run_once() == 2

    assert set(retrievers.built) == {str(account.tenant_id), str(second)}, (
        "one retriever served both tenants; the second tenant's question was "
        "answered from the first tenant's corpus"
    )
    # Built is not enough: the orchestrator holds one from construction, and a
    # per-turn retriever it accepts and ignores looks identical from here.
    for owner, retriever in retrievers.built.items():
        assert retriever.asked, f"{owner}'s retriever was built and never asked"


async def test_a_reply_goes_out_over_the_tenants_own_sender(valkey, app_engine, account):
    """One number for every tenant puts Sinai's reply under the broker's name.

    The Twilio adapter says so in its own docstring — the sender comes from the
    tenant's `channel_accounts` row and is never platform-wide — and then the
    worker holds exactly one adapter per channel for the life of the process.
    The customer sees a reply from a business they never wrote to.
    """
    senders = {}

    def capture(owner):
        def handler(request: httpx.Request) -> httpx.Response:
            senders.setdefault(owner, []).append(request.url.path)
            return httpx.Response(201, json={"sid": "SM-out", "status": "queued"})

        return handler

    registry = TenantSenders(
        {
            (str(account.tenant_id), "whatsapp"): sender(capture("first")),
            ("22222222-2222-2222-2222-222222222222", "whatsapp"): sender(capture("second")),
        }
    )
    worker = OutboundWorker(client=valkey, providers=registry, config=QUEUES)

    for tenant in (str(account.tenant_id), "22222222-2222-2222-2222-222222222222"):
        await valkey.xadd(
            QUEUES["outbound"]["stream"],
            {
                "payload": OutboundJob(
                    tenant_id=tenant,
                    channel="whatsapp",
                    to=CUSTOMER,
                    text="أهلا",
                    last_inbound_at=datetime.now(UTC).isoformat(),
                ).to_json()
            },
        )
    assert await worker.run_once() == 2

    assert set(senders) == {"first", "second"}, (
        "both replies went out through one tenant's Twilio account"
    )


async def test_a_tenant_the_worker_cannot_serve_is_refused_rather_than_answered(
    valkey, app_engine, engine, account
):
    """A real-estate tenant has no path through this worker.

    Inventory turns are a different agent with a different result type
    (`InventoryTurn`, no passages, no retrieval confidence). The education
    orchestrator will happily run its own script against a broker's customer
    and produce a fluent reply about credit-hour fees — the worst available
    outcome, because it is indistinguishable from working.

    Dead-lettered on the first attempt: retrying will not change the tenant's
    vertical.
    """
    broker = await _a_second_tenant(engine, "broker-tenant", vertical="realestate")

    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
        retrievers=Retrievers(),
    )
    await ValkeyInboundQueue(client=valkey, config=QUEUES).publish(
        _message_for(broker, account)
    )
    await worker.run_once()

    dead = await valkey.xrange(QUEUES["inbound"]["dead_letter_stream"])
    assert len(dead) == 1, "a vertical this worker cannot serve was answered anyway"
    assert "realestate" in dead[0][1]["reason"]


class TenantSenders:
    """The production registry's shape: one adapter per (tenant, channel)."""

    def __init__(self, senders: dict) -> None:
        self._senders = senders

    async def for_job(self, *, tenant_id: str, channel: str):
        return self._senders.get((tenant_id, channel))


async def _a_second_tenant(engine, slug: str, vertical: str = "education"):
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.tenancy.models import Tenant

    async with AsyncSession(engine, expire_on_commit=False) as s:
        row = Tenant(slug=slug, name=slug, vertical=vertical)
        s.add(row)
        await s.commit()
        return row.id


def _message_for(tenant_id, account):
    from moc.channels.base import Channel, InboundMessage

    return InboundMessage(
        tenant_id=tenant_id,
        channel=Channel.whatsapp,
        channel_account_id=account.id,
        provider_message_id=f"SM-{uuid.uuid4()}",
        sender_ref=f"+2010{uuid.uuid4().int % 10**8:08d}",
        received_at=datetime.now(UTC),
        text="كام رسوم الساعة؟",
    )


async def test_a_worker_blocking_on_an_idle_stream_stays_up(valkey):
    """The read every worker in production actually performs.

    `block=True` had never run. Every test in this file polls with the default
    `block=False`, which returns immediately, so the one call shape a deployed
    worker uses was the one shape nothing exercised — and it raised
    `TimeoutError` within five seconds of starting, because `block_ms` is 5000
    and redis-py 8 defaults `socket_timeout` to exactly 5.

    An idle stream is what a worker sees almost all the time. This asserts the
    boring case: nothing to do, and still alive.
    """
    from moc.workers.streams import consumer_from_config

    consumer = consumer_from_config(
        client=valkey, section=QUEUES["inbound"], consumer="idle-1"
    )

    async def never_called(payload: str) -> None:  # pragma: no cover
        raise AssertionError("the stream was supposed to be empty")

    assert await consumer.run_once(never_called, block=True) == 0


def test_the_socket_timeout_outlasts_the_longest_blocking_read():
    """The coupling, asserted where it can be read.

    Two numbers in different files that must not be equal is the shape of a
    bug that comes back. This one costs a worker that cannot stay up with
    nothing to do.
    """
    from moc.channels.valkey import valkey_client

    client = valkey_client(config=QUEUES)
    configured = client.connection_pool.connection_kwargs["socket_timeout"]
    longest = max(
        section["block_ms"]
        for section in QUEUES.values()
        if isinstance(section, dict) and "block_ms" in section
    )
    assert configured > longest / 1000, (
        f"socket_timeout {configured}s does not outlast a {longest}ms blocking "
        "read; a worker on an idle stream exits rather than waiting"
    )


# ─────────────────────── the broker's turn, through the worker ───────────────────────


async def test_a_broker_tenant_is_answered_from_its_own_inventory(
    valkey, app_engine, engine, account, tenant_tables
):
    """A real-estate tenant has a path through the worker.

    Until now it did not, and the refusal above was the honest version of that:
    the education orchestrator would have answered a broker's customer with a
    credit-hour fee script. A broker's number could not be connected at all.

    Driven through the same worker as every education turn, because "one pair
    of processes serves every tenant" is the claim, and two workers would let
    it stop being true in one of them.
    """
    import json as _json

    from moc.verticals.realestate.agent import KeywordSlotExtractor
    from moc.verticals.realestate.runner import InventoryRunner
    from moc.workers.inbound import Served

    broker = await _a_second_tenant(engine, "broker-with-stock", vertical="realestate")
    await _stock(engine, broker)

    script = "scripts/realestate/search"
    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
        retrievers=Retrievers(),
        runners={
            "realestate": Served(
                runner=InventoryRunner(
                    script=script,
                    extractor=lambda catalogue: KeywordSlotExtractor(catalogue=catalogue),
                ),
                script=script,
            )
        },
    )

    message = _message_for(broker, account)
    message = type(message)(
        **{
            **{f: getattr(message, f) for f in message.__dataclass_fields__},
            "text": "عايز شقة في مدينتي",
        }
    )
    await ValkeyInboundQueue(client=valkey, config=QUEUES).publish(message)
    assert await worker.run_once() == 1

    entries = await valkey.xrange(QUEUES["outbound"]["stream"])
    assert len(entries) == 1, "the broker's customer got no reply"
    job = _json.loads(entries[0][1]["payload"])
    assert job["tenant_id"] == str(broker)
    assert job["text"], "an empty reply is a customer left waiting"
    # The engine that ran is the real-estate one, so the reply is inventory
    # wording rather than the education script's slot question.
    assert "كلية" not in job["text"], (
        "the broker's customer was answered by the education script"
    )

    # The evidence reaches the thread an agent reads — demo plan Task 41b.
    # `InventoryTurn` has carried the unit and the schedule since P1b and the
    # worker discarded both, which is the failure Task 32 fixed one screen over.
    from moc.agent.handoff import ContactStore, MessageLog
    from moc.tenancy.context import tenant_session

    async with tenant_session(app_engine, broker) as session:
        contact_id = await ContactStore(session=session).resolve(
            contact_ref=message.sender_ref
        )
        history = await MessageLog(session=session).history_for_contact(
            contact_id=contact_id
        )
    replies = [row for row in history if row.provenance]
    assert replies, "the broker's reply reached the inbox with no evidence behind it"
    assert {f["source"] for f in replies[0].provenance["figures"]} <= {
        "inventory",
        "calculator",
    }


async def _stock(engine, tenant_id) -> None:
    """Two units, through the real ingestion path."""
    import json as _json
    import tempfile

    from moc.retrieval.inventory import load_units
    from moc.tenancy.context import tenant_session

    rows = [
        {
            "unit_id": "MD-1",
            "as_of": "2026-08-01",
            "property_type": "apartment",
            "compound": "Madinaty",
            "city": "New Cairo",
            "price": 5_500_000,
            "currency": "EGP",
            "availability": "available",
            "bedrooms": 3,
        },
        {
            "unit_id": "MD-2",
            "as_of": "2026-08-01",
            "property_type": "apartment",
            "compound": "Madinaty",
            "city": "New Cairo",
            "price": 6_100_000,
            "currency": "EGP",
            "availability": "available",
            "bedrooms": 3,
        },
    ]
    path = Path(tempfile.mkdtemp()) / "units.jsonl"
    path.write_text(
        "\n".join(_json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    async with tenant_session(engine, tenant_id) as session:
        await load_units(session=session, path=path)
        await session.commit()


# ─────────────────────── "seen, typing", on the turn path ───────────────────────


class RecordingIndicators:
    """A per-tenant typing indicator factory, as the composition root supplies."""

    def __init__(self, *, order: list[str], works: bool = True) -> None:
        self.order = order
        self.works = works
        self.asked: list[tuple[str, str]] = []

    async def typing_for(self, *, tenant_id, channel: str):
        return self

    async def typing(self, *, message_id: str) -> bool:
        self.asked.append((message_id, "typing"))
        self.order.append("typing")
        return self.works


class RecordingOrchestrator:
    """The real turn, with a mark left when it starts."""

    def __init__(self, *, order: list[str]) -> None:
        self._inner = orchestrator()
        self._order = order

    async def handle(self, **kwargs):
        self._order.append("turn")
        return await self._inner.handle(**kwargs)


async def test_the_customer_sees_typing_before_the_turn_runs(valkey, app_engine, account):
    """§2.5's mitigation, in the only place it can work.

    Not from the webhook: that handler has a millisecond budget and a vendor
    call inside it is the slow work it is forbidden from reaching. Not from the
    outbound queue either — the indicator would then queue behind replies and
    wait on the token bucket, and a courtesy that arrives after the reply it
    was meant to precede is worse than none.

    So it fires at the top of the turn, from the worker that already holds the
    inbound message id.
    """
    order: list[str] = []
    indicators = RecordingIndicators(order=order)
    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=RecordingOrchestrator(order=order),
        script=SCRIPT,
        config=QUEUES,
        indicators=indicators,
    )

    message = _message_for(account.tenant_id, account)
    await ValkeyInboundQueue(client=valkey, config=QUEUES).publish(message)
    assert await worker.run_once() == 1

    assert order == ["typing", "turn"], (
        "the indicator did not precede the turn; it exists to cover the wait"
    )
    assert indicators.asked == [(message.provider_message_id, "typing")], (
        "the indicator must name the inbound message — Twilio keys it on the SID"
    )


async def test_a_broken_indicator_does_not_cost_the_customer_their_answer(
    valkey, app_engine, account
):
    """A courtesy on the turn path. Losing the answer to save the hint is the
    one outcome worse than no hint."""
    order: list[str] = []

    class Explodes(RecordingIndicators):
        async def typing(self, *, message_id: str) -> bool:
            raise RuntimeError("the vendor is having a day")

    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
        indicators=Explodes(order=order),
    )
    await ValkeyInboundQueue(client=valkey, config=QUEUES).publish(
        _message_for(account.tenant_id, account)
    )
    assert await worker.run_once() == 1
    assert await valkey.xlen(QUEUES["outbound"]["stream"]) == 1, "the reply was lost"


async def test_a_conversation_a_human_has_taken_gets_no_typing_indicator(
    valkey, app_engine, account
):
    """"Typing" from a bot while an officer is reading is a lie, and this is
    exactly the case the vendor research flagged: the indicator also marks the
    message read, so a customer would see a blue tick and then wait on a person
    who has not started."""

    from moc.agent.conversations import ConversationStore
    from moc.agent.handoff import HandoffStore
    from moc.agent.script_engine import ScriptEngine
    from moc.tenancy.context import tenant_session

    order: list[str] = []
    indicators = RecordingIndicators(order=order)
    worker = InboundWorker(
        client=valkey,
        engine=app_engine,
        orchestrator=orchestrator(),
        script=SCRIPT,
        config=QUEUES,
        indicators=indicators,
    )

    first = _message_for(account.tenant_id, account)
    await ValkeyInboundQueue(client=valkey, config=QUEUES).publish(first)
    assert await worker.run_once() == 1

    async with tenant_session(app_engine, account.tenant_id) as session:
        store = ConversationStore(session=session, engine=ScriptEngine.from_config(SCRIPT))
        conversation_id = await store.find(channel="whatsapp", sender_ref=first.sender_ref)
        await HandoffStore(session=session).open(
            conversation_id=conversation_id, reason="asked for a person", resume_state={}
        )
        await session.commit()

    order.clear()
    follow_up = _message_for(account.tenant_id, account)
    follow_up = type(follow_up)(
        **{
            **{f: getattr(follow_up, f) for f in follow_up.__dataclass_fields__},
            "sender_ref": first.sender_ref,
        }
    )
    await ValkeyInboundQueue(client=valkey, config=QUEUES).publish(follow_up)
    assert await worker.run_once() == 1

    assert order == [], "the bot told a customer it was typing while a human had the thread"


async def test_a_dead_letter_says_where_it_died_and_not_only_what_it_raised(valkey):
    """`repr(exc)` names the exception and not the line.

    The first rehearsal produced `PermissionError(13, 'Permission denied')` with
    no path, no frame and no clue, on a turn where a customer got silence — and
    finding it meant re-running the whole path by hand. This row is the one
    place that question has to be answerable, because the process whose log
    would have held the frames is by then a container that has restarted.
    """
    from moc.workers.streams import TerminalFailure, consumer_from_config

    consumer = consumer_from_config(
        client=valkey, section=QUEUES["inbound"], consumer="buried-1"
    )

    async def dies(payload: str) -> None:
        raise TerminalFailure("nothing to retry here")

    await valkey.xadd(QUEUES["inbound"]["stream"], {"payload": "{}"})
    await consumer.run_once(dies)

    dead = await valkey.xrange(QUEUES["inbound"]["dead_letter_stream"])
    assert len(dead) == 1
    frames = dead[0][1]["traceback"]
    assert "test_pipeline.py" in frames, "the row does not say where it died"
    assert "raise TerminalFailure" in frames
