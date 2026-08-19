"""The inbound webhook — design §6.1, §6.2, and the ACK budget in §11.

`webhooks.py` is flagged for line-by-line human review. The handler does four
things and must do nothing else: verify, resolve the tenant, enqueue, 200.

Every test here is about something the handler must *not* do. A webhook that
answers correctly but slowly gets retried by Twilio and delivers the customer's
message twice; a webhook that verifies leniently is not a webhook at all.
"""

import ast
import hashlib
import hmac
from base64 import b64encode
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

import httpx
import pytest

from moc.api.webhooks import build_app
from moc.channels.base import Channel, ChannelAccount
from moc.config_store import load

WHATSAPP = load("channels/whatsapp")
SECRET = "auth-token-for-this-account"
SECRET_REF = "twilio/test/wa"
PATH = "/webhooks/twilio/whatsapp"
URL = f"https://moc.example{PATH}"

FORM = {
    "MessageSid": "SM1234567890abcdef",
    "AccountSid": "AC0000000000000000",
    "From": "whatsapp:+201012345678",
    "To": "whatsapp:+201555000111",
    "Body": "كام رسوم الساعة؟",
    "NumMedia": "0",
}

ACCOUNT = ChannelAccount(
    id=uuid4(),
    tenant_id=uuid4(),
    channel=Channel.whatsapp,
    account_ref="+201555000111",
    secret_ref=SECRET_REF,
)


class FakeSecrets:
    """`secret_ref` -> secret, without a secret store.

    The real resolver reads the process environment (`EnvSecretResolver`);
    what matters to these tests is that the secret arrives by reference rather
    than riding on the account the bootstrap lookup returned.
    """

    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self.secrets = {SECRET_REF: SECRET} if secrets is None else secrets
        self.asked: list[str] = []

    def for_ref(self, secret_ref: str) -> str:
        self.asked.append(secret_ref)
        return self.secrets[secret_ref]


class FakeRegistry:
    """Maps a vendor address to a tenant's channel account.

    A Protocol seam, as retrieval and extraction are in Task 14. The table
    exists in the design (`channel_accounts`, §5) but wiring it needs a
    decision this task should not make silently — see the note in
    `channels/base.py` about resolving a tenant *before* a tenant context
    exists, which is the one lookup RLS cannot scope.
    """

    def __init__(self, accounts: dict[str, ChannelAccount] | None = None) -> None:
        self.accounts = {ACCOUNT.account_ref: ACCOUNT} if accounts is None else accounts
        self.lookups: list[str] = []

    async def resolve(self, *, channel: Channel, account_ref: str) -> ChannelAccount | None:
        self.lookups.append(account_ref)
        return self.accounts.get(account_ref)


class FakeQueue:
    def __init__(self) -> None:
        self.published: list = []

    async def publish(self, message) -> None:
        self.published.append(message)


class FakeEventLog:
    """`claim` returns False for a message id already seen."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    async def claim(self, provider_message_id: str) -> bool:
        if provider_message_id in self.seen:
            return False
        self.seen.add(provider_message_id)
        return True

    async def release(self, provider_message_id: str) -> None:
        self.seen.discard(provider_message_id)


@pytest.fixture
def wiring():
    return FakeRegistry(), FakeQueue(), FakeEventLog()


@pytest.fixture
def client(wiring):
    registry, queue, events = wiring
    app = build_app(registry=registry, queue=queue, events=events, secrets=FakeSecrets())
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    )


def body(**overrides) -> bytes:
    return urlencode({**FORM, **overrides}).encode()


def signature(raw: bytes, *, secret: str = SECRET, url: str = URL) -> str:
    from urllib.parse import parse_qsl

    canonical = url + "".join(k + v for k, v in sorted(parse_qsl(raw.decode(), True)))
    return b64encode(hmac.new(secret.encode(), canonical.encode(), hashlib.sha1).digest()).decode()


async def post(client, raw: bytes, *, sign: bool = True, **headers):
    if sign:
        headers.setdefault(WHATSAPP["signature_header"], signature(raw))
    return await client.post(
        PATH,
        content=raw,
        headers={"content-type": WHATSAPP["form_content_type"], **headers},
    )


# ─────────────────────────── the happy path ───────────────────────────


async def test_a_signed_webhook_is_accepted_and_enqueued(client, wiring):
    _, queue, _ = wiring
    response = await post(client, body())

    assert response.status_code == 200
    assert len(queue.published) == 1
    assert queue.published[0].provider_message_id == FORM["MessageSid"]
    assert queue.published[0].tenant_id == ACCOUNT.tenant_id


async def test_normalizes_to_InboundMessage(client, wiring):
    _, queue, _ = wiring
    await post(client, body())
    message = queue.published[0]
    assert message.channel is Channel.whatsapp
    assert message.sender_ref == "+201012345678"
    assert message.text == FORM["Body"]
    assert message.received_at is not None


# ─────────────────────────── rejection ───────────────────────────


async def test_missing_signature_header_is_rejected_not_skipped(client, wiring):
    _, queue, _ = wiring
    response = await post(client, body(), sign=False)
    assert response.status_code == 403
    assert queue.published == []


async def test_a_forged_signature_is_rejected(client, wiring):
    _, queue, _ = wiring
    raw = body()
    forged = {WHATSAPP["signature_header"]: signature(raw, secret="attacker")}
    response = await post(client, raw, sign=False, **forged)
    assert response.status_code == 403
    assert queue.published == []


async def test_a_tampered_body_is_rejected(client, wiring):
    """Sign one message, deliver another."""
    _, queue, _ = wiring
    header = {WHATSAPP["signature_header"]: signature(body())}
    response = await post(client, body(Body="المصاريف مجانية"), sign=False, **header)
    assert response.status_code == 403
    assert queue.published == []


async def test_unknown_channel_account_is_rejected_before_tenant_resolution(wiring):
    """An address we do not serve gets no work done on its behalf.

    Anyone can POST here. Resolving a tenant, opening a session or verifying
    against some default secret for an unknown address turns an open endpoint
    into a lever — and there is no tenant to charge the work to.
    """
    _, queue, events = wiring
    empty = FakeRegistry(accounts={})
    app = build_app(registry=empty, queue=queue, events=events, secrets=FakeSecrets())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as unknown_client:
        response = await post(unknown_client, body())

    assert response.status_code == 403
    assert queue.published == []
    assert events.seen == set(), "no event was claimed for an address we do not serve"


async def test_an_unparseable_body_is_rejected_not_a_500(client, wiring):
    """Hostile bytes are a rejection, not a stack trace on an open endpoint."""
    _, queue, _ = wiring
    header = {WHATSAPP["signature_header"]: "irrelevant"}
    response = await post(client, b"\xff\xfe not form encoded at all", sign=False, **header)
    assert response.status_code in (400, 403)
    assert queue.published == []


# ─────────────────────────── idempotency ───────────────────────────


async def test_duplicate_provider_message_id_is_idempotent(client, wiring):
    """Twilio retries on a slow ACK, so the same message arrives twice.

    The second delivery must be a 200 — a non-2xx makes Twilio retry again —
    and must not enqueue a second turn. The customer asked once; being answered
    twice reads as a broken bot.
    """
    _, queue, _ = wiring
    first = await post(client, body())
    second = await post(client, body())

    assert (first.status_code, second.status_code) == (200, 200)
    assert len(queue.published) == 1


async def test_the_duplicate_check_happens_after_verification(client, wiring):
    """Otherwise an unsigned request can burn a message id.

    Claiming first would let anyone suppress a real inbound message by posting
    its id ahead of Twilio — no signature required.
    """
    _, queue, events = wiring
    await post(client, body(), sign=False)
    assert events.seen == set()
    assert queue.published == []


# ─────────────────────────── §11: ACK before any work ───────────────────────────

WEBHOOKS = Path(__file__).parents[2] / "src" / "moc" / "api" / "webhooks.py"


def test_webhook_acks_before_any_llm_call():
    """Twilio and Meta retry on a slow ACK, and a retry is a duplicate delivery.

    Asserted structurally rather than by timing: a stopwatch test passes on a
    fast machine and on a cached model call. The handler cannot be slow if it
    cannot reach anything slow, so it imports neither the agent nor the LLM
    package — and the queue is the seam that keeps it that way.
    """
    tree = ast.parse(WEBHOOKS.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = [name for name in imported if name.startswith(("moc.agent", "moc.llm"))]
    assert forbidden == [], (
        f"the webhook handler must not reach the agent or a provider: {forbidden}. "
        f"Enqueue and return; the worker owns the turn."
    )


def test_the_handler_does_no_database_work_beyond_the_event_row():
    """§6: verify, resolve, enqueue, 200. No session, no ORM, no transaction."""
    source = WEBHOOKS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not {"sqlalchemy", "sqlalchemy.ext.asyncio", "moc.tenancy.metering"} & imported


async def test_a_queue_failure_is_not_a_silent_success(client, wiring):
    """If the message did not make it onto the queue, do not tell Twilio it did.

    A 200 here means "delivered" and stops the retries. Losing the message and
    acking it is the one outcome with no recovery path.
    """
    registry, _, events = wiring

    class BrokenQueue:
        async def publish(self, message) -> None:
            raise ConnectionError("valkey unreachable")

    app = build_app(registry=registry, queue=BrokenQueue(), events=events, secrets=FakeSecrets())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as broken:
        response = await post(broken, body())

    assert response.status_code >= 500


async def test_a_failed_enqueue_releases_the_message_id_for_the_retry(wiring):
    """The bug this ordering creates, and the reason `release` exists.

    The id is claimed before the enqueue, so a queue failure leaves it claimed.
    The vendor then retries — correctly, we returned 5xx — and the retry is
    discarded as a duplicate. The message is lost permanently, by the exact
    mechanism that was supposed to protect it.
    """
    registry, queue, events = wiring
    failing = True

    class FlakyQueue:
        async def publish(self, message) -> None:
            if failing:
                raise ConnectionError("valkey unreachable")
            queue.published.append(message)

    app = build_app(registry=registry, queue=FlakyQueue(), events=events, secrets=FakeSecrets())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as flaky:
        first = await post(flaky, body())
        failing = False
        retry = await post(flaky, body())

    assert first.status_code >= 500
    assert retry.status_code == 200
    assert len(queue.published) == 1, "the vendor's retry was swallowed as a duplicate"
