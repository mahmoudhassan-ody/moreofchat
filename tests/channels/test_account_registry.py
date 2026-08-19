"""The Postgres-backed channel-account registry — P1 Task 21.

The registry is the only thing in the codebase that reads through
`moc_lookup`, and it exists so that the reach of the pre-tenant bootstrap is
one reviewable object rather than an ambient capability. The isolation
guarantees themselves are asserted in `tests/tenancy/test_channel_accounts.py`
against the role and the view; this file asserts the code goes through them.
"""

import ast
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.channels.accounts import EnvSecretResolver, SqlChannelAccountRegistry
from moc.channels.base import Channel

ADDRESS = "+201555000222"
SECRET_REF = "twilio/registry-co/wa"


@pytest_asyncio.fixture(loop_scope="session")
async def seeded(engine, tenant_tables):
    from moc.tenancy.models import Tenant

    tenant_id, account_id = uuid.uuid4(), uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            # noqa S608: names come from the conftest tuple, not from input.
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add(Tenant(id=tenant_id, slug="registry-co", name="Registry", vertical="education"))
        await s.flush()
        await s.execute(
            text(
                "INSERT INTO channel_accounts "
                "(id, tenant_id, channel, address, secret_ref, signing_secret) "
                "VALUES (:id, :t, 'whatsapp', :address, :ref, 'never-leaves-the-table')"
            ),
            {"id": account_id, "t": tenant_id, "address": ADDRESS, "ref": SECRET_REF},
        )
        await s.commit()

    yield account_id, tenant_id

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            # noqa S608: names come from the conftest tuple, not from input.
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


async def test_resolves_an_address_to_its_tenant(lookup_engine, seeded):
    account_id, tenant_id = seeded
    registry = SqlChannelAccountRegistry(engine=lookup_engine)
    account = await registry.resolve(channel=Channel.whatsapp, account_ref=ADDRESS)

    assert account is not None
    assert account.id == account_id
    assert account.tenant_id == tenant_id
    assert account.account_ref == ADDRESS
    assert account.secret_ref == SECRET_REF


async def test_an_unknown_address_resolves_to_nothing(lookup_engine, seeded):
    """Not an exception. The webhook turns this into a 403 before it opens a
    session or claims a message id, and an unknown address is an ordinary
    event on a public endpoint rather than an error condition."""
    registry = SqlChannelAccountRegistry(engine=lookup_engine)
    assert await registry.resolve(channel=Channel.whatsapp, account_ref="+20999") is None


async def test_the_channel_is_part_of_the_lookup(lookup_engine, seeded):
    """The same address can exist on two channels. Resolving on address alone
    would let a Telegram message be attributed to a WhatsApp account."""
    registry = SqlChannelAccountRegistry(engine=lookup_engine)
    assert await registry.resolve(channel=Channel.telegram, account_ref=ADDRESS) is None


def _sql_literals() -> list[str]:
    """Every string constant in the module, which is where its SQL lives."""
    import moc.channels.accounts as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_the_registry_reads_the_view_and_never_the_base_table():
    """Structural, because the grants alone would not catch it everywhere: a
    database whose `moc_lookup` was over-granted by hand would let a
    base-table query succeed, and this is the code that must not try.

    Checked against the module's SQL rather than its prose — the docstrings
    name the base table deliberately, to explain what is being avoided.
    """
    # Case-sensitive: SQL keywords are written upper here, and lowercasing
    # the test would match the word "from" in every docstring.
    sql = [text for text in _sql_literals() if "SELECT" in text and "FROM" in text]
    assert sql, "no SQL found — this test would pass vacuously"
    assert all("channel_account_lookup" in text for text in sql)
    assert not any("FROM channel_accounts" in text for text in sql)


def test_the_registry_never_selects_a_secret():
    """`secret_ref` names a secret; `signing_secret` is one."""
    assert not any("signing_secret" in text for text in _sql_literals())


def test_no_other_module_connects_as_the_lookup_role():
    """One reader, reviewable in one place.

    `lookup_database_url` is the capability, and a second caller would be a
    second thing to review whenever the role's reach is questioned.
    """
    import moc

    root = Path(moc.__file__).parent
    callers = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == "lookup_database_url"
    }
    assert callers <= {"channels/accounts.py"}, f"lookup role reached from {callers}"


# ─────────────────────────── secrets ───────────────────────────


def test_the_secret_resolver_turns_a_reference_into_a_secret(monkeypatch):
    monkeypatch.setenv("MOC_SECRET_TWILIO__REGISTRY_CO__WA", "s3cr3t")
    assert EnvSecretResolver().for_ref(SECRET_REF) == "s3cr3t"


def test_an_unresolvable_reference_raises_rather_than_returning_empty():
    """An empty secret would make signature verification compare against ""
    and, on a constant-time comparison, fail every message — an outage that
    reads as a vendor problem. Raising names the cause."""
    with pytest.raises(KeyError, match="twilio/registry-co/wa"):
        EnvSecretResolver().for_ref(SECRET_REF)


# ─────────────────── the acceptance: the webhook path ───────────────────


async def test_a_signed_webhook_resolves_its_tenant_through_the_lookup_role(
    lookup_engine, seeded, monkeypatch
):
    """Task 21's acceptance, end to end and with no fake registry.

    A signed Twilio form reaches the webhook, the tenant is resolved through
    `moc_lookup` against the real view, the signing secret arrives by
    reference, and the message is enqueued for the tenant the address belongs
    to. Nothing here has a tenant context set — establishing one is what the
    lookup produced.
    """
    import hashlib
    import hmac
    from base64 import b64encode
    from urllib.parse import parse_qsl, urlencode

    import httpx

    from moc.api.webhooks import _TWILIO_WHATSAPP_PATH, build_app
    from moc.channels.accounts import EnvSecretResolver
    from moc.config_store import load

    _, tenant_id = seeded
    secret = "the-twilio-auth-token"
    monkeypatch.setenv(
        EnvSecretResolver().variable_for(SECRET_REF), secret
    )

    whatsapp = load("channels/whatsapp")
    url = "https://moc.example" + _TWILIO_WHATSAPP_PATH
    body = urlencode(
        {
            "MessageSid": "SM0000000000000000",
            "AccountSid": "AC0000000000000000",
            "From": whatsapp["address_prefix"] + "+201012345678",
            "To": whatsapp["address_prefix"] + ADDRESS,
            "Body": "كام رسوم الساعة؟",
            "NumMedia": "0",
        }
    ).encode()
    canonical = url + "".join(k + v for k, v in sorted(parse_qsl(body.decode(), True)))
    signature = b64encode(
        hmac.new(secret.encode(), canonical.encode(), hashlib.sha1).digest()
    ).decode()

    published: list = []

    class Queue:
        async def publish(self, message) -> None:
            published.append(message)

    class Events:
        async def claim(self, provider_message_id: str) -> bool:
            return True

        async def release(self, provider_message_id: str) -> None:
            return None

    app = build_app(
        registry=SqlChannelAccountRegistry(engine=lookup_engine),
        queue=Queue(),
        events=Events(),
        secrets=EnvSecretResolver(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        response = await client.post(
            _TWILIO_WHATSAPP_PATH,
            content=body,
            headers={
                "content-type": "application/x-www-form-urlencoded",
                whatsapp["signature_header"]: signature,
            },
        )

    assert response.status_code == 200
    assert len(published) == 1
    assert published[0].tenant_id == tenant_id


async def test_an_unknown_address_is_refused_before_tenant_resolution(lookup_engine, seeded):
    """A public endpoint. An address nobody has connected must be refused
    before a session is opened, a message id claimed, or any work attributed
    to a tenant — because there is no tenant to attribute it to."""
    from urllib.parse import urlencode

    import httpx

    from moc.api.webhooks import _TWILIO_WHATSAPP_PATH, build_app
    from moc.channels.accounts import EnvSecretResolver
    from moc.config_store import load

    whatsapp = load("channels/whatsapp")

    class ExplodingQueue:
        async def publish(self, message) -> None:
            raise AssertionError("an unknown address must never reach the queue")

    class ExplodingEvents:
        async def claim(self, provider_message_id: str) -> bool:
            raise AssertionError("an unknown address must never claim a message id")

        async def release(self, provider_message_id: str) -> None:
            return None

    app = build_app(
        registry=SqlChannelAccountRegistry(engine=lookup_engine),
        queue=ExplodingQueue(),
        events=ExplodingEvents(),
        secrets=EnvSecretResolver(),
    )
    body = urlencode(
        {
            "MessageSid": "SM1",
            "AccountSid": "AC0",
            "From": whatsapp["address_prefix"] + "+201012345678",
            "To": whatsapp["address_prefix"] + "+20111111111",
            "Body": "hello",
            "NumMedia": "0",
        }
    ).encode()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        response = await client.post(
            _TWILIO_WHATSAPP_PATH,
            content=body,
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
    assert response.status_code == 403
