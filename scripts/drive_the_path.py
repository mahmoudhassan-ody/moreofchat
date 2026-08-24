"""Drive the whole path with real processes — demo plan Task 39.

Not a test. `pytest` proves the pieces connect through `ASGITransport`, in one
process, with fakes at both ends. This starts the actual programs:

    a signed POST over TCP
      -> uvicorn running `moc.api.main.webhook_app`
      -> real Valkey, real consumer group
      -> `python -m moc.workers.run inbound` (real Postgres, real model call)
      -> real Valkey again
      -> `python -m moc.workers.run outbound` (real Twilio adapter)
      -> an HTTP request leaving the machine's loopback

Everything is real except the vendor's host, which is pointed at a stub on
127.0.0.1 through `MOC_CONFIG_DIR`. That is the one leg this script cannot
reach: sending a WhatsApp message needs Twilio, a business number and a phone.
`docs/ops/real-phone-path.md` is the procedure for the rest of it.

Run it with the compose stack up and `.env` sourced:

    uv run python scripts/drive_the_path.py

`--compose` instead drives the *containers*: it publishes a message onto the
inbound stream and watches the running `worker-inbound` turn it into a queued
reply. That exercises what only the composed system can get wrong — image
contents, service-name resolution for the four backing stores, environment
propagation — and skips the signature, which `POST /webhooks/...` through Caddy
already exercises by refusing an unsigned body with a 403.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qsl, urlencode

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
TENANT_SLUG = "drive-the-path"
BUSINESS_NUMBER = "+201555000999"
CUSTOMER = "+201012340000"
AUTH_TOKEN = "drive-the-path-auth-token"  # noqa: S105 - a local fixture
ACCOUNT_SID = "AC00000000000000000000000000000000"
SECRET_REF = "twilio/drive/wa"  # noqa: S105 - a reference, not a secret

received: list[dict] = []
indicators: list[dict] = []


class Vendor(BaseHTTPRequestHandler):
    """Stands in for api.twilio.com and messaging.twilio.com, on loopback."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's shape
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length).decode("utf-8")
        if self.headers.get("content-type", "").startswith("application/json"):
            # The typing indicator: JSON, on the messaging host. Recorded
            # separately because it is not a message and must not be counted
            # as the reply.
            indicators.append(json.loads(body))
            payload = json.dumps({"success": True}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        received.append(dict(parse_qsl(body)))
        payload = json.dumps({"sid": "SM-drive", "status": "queued"}).encode()
        self.send_response(201)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # noqa: A002 - silence the default logger
        return


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def say(step: str, detail: str = "") -> None:
    print(f"  {step:<52} {detail}", flush=True)


def config_tree_pointing_at(vendor_port: int) -> Path:
    """A copy of `config/` with Twilio's host swapped for the stub.

    Through `MOC_CONFIG_DIR`, which exists precisely so nothing has to be
    patched at runtime. The adapter, the signature, the form encoding and the
    error handling are all the real ones; only the hostname differs.
    """
    root = Path(tempfile.mkdtemp(prefix="moc-drive-")) / "config"
    shutil.copytree(ROOT / "config", root)
    whatsapp = root / "channels" / "whatsapp.yaml"
    document = yaml.safe_load(whatsapp.read_text(encoding="utf-8"))
    document["api_base"] = f"http://127.0.0.1:{vendor_port}"
    # The indicator too. It is a *different host* in the real config, so a copy
    # that rewrote only the message base left every drive quietly posting to
    # api.twilio.com with fixture credentials — a real external call from a
    # script whose whole point is that nothing leaves the machine.
    document["typing_indicator"]["api_base"] = f"http://127.0.0.1:{vendor_port}"
    whatsapp.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return root


async def seed() -> uuid.UUID:
    """One tenant with one connected WhatsApp number."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings

    await purge()
    engine = create_async_engine(settings.database_url)
    tenant_id = uuid.uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await session.execute(
            text(
                # `default_lang` is spelled out: the column is NOT NULL with
                # only a Python-side ORM default, so any insert that is not a
                # `Tenant(...)` — a migration, an ops script, a psql session —
                # fails on it.
                "INSERT INTO tenants (id, slug, name, vertical, default_lang) "
                "VALUES (:id, :slug, 'Drive The Path', 'education', 'ar')"
            ),
            {"id": tenant_id, "slug": TENANT_SLUG},
        )
        await session.execute(
            text(
                "INSERT INTO channel_accounts "
                "(id, tenant_id, channel, address, secret_ref, signing_secret) "
                "VALUES (:id, :t, 'whatsapp', :addr, :ref, :secret)"
            ),
            {
                "id": uuid.uuid4(),
                "t": tenant_id,
                "addr": BUSINESS_NUMBER,
                "ref": SECRET_REF,
                "secret": AUTH_TOKEN,
            },
        )
        await session.commit()
    await engine.dispose()
    return tenant_id


#: Deleted in dependency order. Not a `TRUNCATE ... CASCADE`: this runs against
#: a real database that may hold real tenants, and a cascade here is how a
#: throwaway driver removes somebody's data.
_PURGE = (
    "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations "
    "WHERE tenant_id IN (SELECT id FROM tenants WHERE slug = :slug))",
    "DELETE FROM handoffs WHERE tenant_id IN (SELECT id FROM tenants WHERE slug = :slug)",
    "DELETE FROM conversations WHERE tenant_id IN (SELECT id FROM tenants WHERE slug = :slug)",
    "DELETE FROM contacts WHERE tenant_id IN (SELECT id FROM tenants WHERE slug = :slug)",
    "DELETE FROM usage_ledger WHERE tenant_id IN (SELECT id FROM tenants WHERE slug = :slug)",
    "DELETE FROM channel_accounts WHERE address = :address",
    "DELETE FROM tenants WHERE slug = :slug",
)


async def purge() -> None:
    """Remove what this script created, in an order the foreign keys accept."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings

    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        for statement in _PURGE:
            await session.execute(
                text(statement), {"slug": TENANT_SLUG, "address": BUSINESS_NUMBER}
            )
        await session.commit()
    await engine.dispose()


async def ensure_search_backends() -> None:
    """Indexes and collections, without which retrieval raises rather than
    returning nothing — and a turn that raises is a stuck stream entry, not a
    scripted fallback."""
    from moc.retrieval.lexical import MeilisearchAdmin, meilisearch_client
    from moc.retrieval.vectors import QdrantAdmin, qdrant_client

    meili = meilisearch_client()
    await MeilisearchAdmin(client=meili).ensure_indexes()
    await meili.aclose()

    qdrant = qdrant_client()
    await QdrantAdmin(client=qdrant).ensure_collections()
    await qdrant.close()


def signed(url: str, fields: dict[str, str], token: str) -> str:
    canonical = url + "".join(k + v for k, v in sorted(fields.items()))
    return base64.b64encode(
        hmac.new(token.encode(), canonical.encode(), hashlib.sha1).digest()
    ).decode()


def wait_for(port: int, seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


async def _stream_state() -> str:
    """Where the message got to, in the queue's own terms.

    A failure here is almost never a stack trace — the worker is alive and
    blocking. What says what happened is whether the entry is unread, pending
    against a consumer, or dead-lettered.
    """
    from moc.channels.valkey import valkey_client
    from moc.config_store import load

    queues = load("workers/queues")
    client = valkey_client()
    lines = []
    for name in ("inbound", "outbound"):
        stream = queues[name]["stream"]
        dead = queues[name]["dead_letter_stream"]
        length = await client.xlen(stream)
        deads = await client.xlen(dead)
        try:
            pending = await client.xpending(stream, queues[name]["group"])
            pending_count = pending["pending"] if isinstance(pending, dict) else pending
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            pending_count = f"no group ({type(exc).__name__})"
        lines.append(f"    {name:<10} entries={length} pending={pending_count} dead={deads}")
        if deads:
            for _, fields in await client.xrange(dead):
                lines.append(f"      dead: {fields.get('reason', '')[:200]}")
    await client.aclose()
    return "\n".join(lines)


def _tails(logs: dict[str, Path], lines: int = 25) -> None:
    """What each process said. The whole point of running them separately is
    that a failure names which one."""
    for name, path in logs.items():
        body = path.read_text(encoding="utf-8").strip().splitlines()[-lines:]
        print(f"\n--- {name} ---")
        print("\n".join(body))


def against_compose() -> int:
    """Prove the containers process a turn.

    Deliberately publishes to the stream rather than posting to the webhook.
    The containers read `.env` at start, and the signing secret for a
    throwaway account is not something to write into somebody's secrets file
    to make a driver work — the webhook leg is proven by the host-process mode
    above and by an unsigned POST through Caddy being refused.
    """
    print("\ndriving the composed system\n")

    tenant_id = asyncio.run(seed())
    say("tenant and channel account seeded", str(tenant_id))

    before = asyncio.run(_outbound_length())
    asyncio.run(_publish_inbound(tenant_id))
    say("message published onto the inbound stream", "")

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        after = asyncio.run(_outbound_length())
        if after > before:
            say("a containerised worker produced a reply", f"outbound {before} -> {after}")
            job = asyncio.run(_last_outbound())
            print(f"\n  the customer would have received:\n\n    {job.get('text', '')}\n")
            # Drop the queued reply before removing the tenant it belongs to.
            # Left behind, `worker-outbound` resolves a sender for a tenant
            # that no longer has a channel account and dead-letters — correct
            # behaviour, and debris this script created.
            asyncio.run(_drop_last_outbound())
            asyncio.run(purge())
            return 0
        time.sleep(1)

    say("no reply was queued", "")
    print("\n  queue state:")
    print(asyncio.run(_stream_state()))
    asyncio.run(purge())
    return 1


async def _outbound_length() -> int:
    from moc.channels.valkey import valkey_client
    from moc.config_store import load

    client = valkey_client()
    length = await client.xlen(load("workers/queues")["outbound"]["stream"])
    await client.aclose()
    return length


async def _last_outbound() -> dict:
    from moc.channels.valkey import valkey_client
    from moc.config_store import load

    client = valkey_client()
    entries = await client.xrevrange(load("workers/queues")["outbound"]["stream"], count=1)
    await client.aclose()
    return json.loads(entries[0][1]["payload"]) if entries else {}


async def _drop_last_outbound() -> None:
    from moc.channels.valkey import valkey_client
    from moc.config_store import load

    client = valkey_client()
    stream = load("workers/queues")["outbound"]["stream"]
    entries = await client.xrevrange(stream, count=1)
    if entries:
        await client.xdel(stream, entries[0][0])
    await client.aclose()


async def _publish_inbound(tenant_id) -> None:
    from datetime import UTC, datetime

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.channels.base import Channel, InboundMessage
    from moc.channels.valkey import ValkeyInboundQueue, valkey_client
    from moc.config import settings
    from moc.config_store import load

    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        account_id = (
            await session.execute(
                text("SELECT id FROM channel_accounts WHERE address = :a"),
                {"a": BUSINESS_NUMBER},
            )
        ).scalar_one()
    await engine.dispose()

    client = valkey_client()
    await ValkeyInboundQueue(client=client, config=load("workers/queues")).publish(
        InboundMessage(
            tenant_id=tenant_id,
            channel=Channel.whatsapp,
            channel_account_id=account_id,
            provider_message_id=f"SM{uuid.uuid4().hex}",
            sender_ref=CUSTOMER,
            received_at=datetime.now(UTC),
            text="كام رسوم الساعة المعتمدة؟",
        )
    )
    await client.aclose()


def main() -> int:
    # Synchronous on purpose. This starts subprocesses, blocks on sockets and
    # polls — all of which are correct here and all of which are mistakes
    # inside a coroutine. The two pieces that genuinely need an event loop get
    # their own.
    print("\ndriving the real path\n")

    vendor_port = free_port()
    api_port = free_port()
    server = HTTPServer(("127.0.0.1", vendor_port), Vendor)
    Thread(target=server.serve_forever, daemon=True).start()
    say("vendor stub listening", f"127.0.0.1:{vendor_port}")

    config_root = config_tree_pointing_at(vendor_port)
    environment = {
        **os.environ,
        "MOC_CONFIG_DIR": str(config_root),
        f"MOC_SECRET_{SECRET_REF.replace('/', '__').upper()}": AUTH_TOKEN,
        f"MOC_SECRET_{SECRET_REF.replace('/', '__').upper()}__SID": ACCOUNT_SID,
        "PYTHONUNBUFFERED": "1",
    }

    tenant_id = asyncio.run(seed())
    say("tenant and channel account seeded", str(tenant_id))
    asyncio.run(ensure_search_backends())
    say("meilisearch indexes and qdrant collections", "ready")

    processes: list[subprocess.Popen] = []
    logs: dict[str, Path] = {}
    try:
        log_dir = Path(tempfile.mkdtemp(prefix="moc-drive-logs-"))
        for name, command in (
            (
                "api",
                [
                    sys.executable, "-m", "uvicorn",
                    "--factory", "moc.api.main:webhook_app",
                    "--host", "127.0.0.1", "--port", str(api_port),
                ],
            ),
            ("worker-inbound", [sys.executable, "-m", "moc.workers.run", "inbound"]),
            ("worker-outbound", [sys.executable, "-m", "moc.workers.run", "outbound"]),
        ):
            logs[name] = log_dir / f"{name}.log"
            handle = logs[name].open("w", encoding="utf-8")
            processes.append(
                subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                    command, cwd=ROOT, env=environment, stdout=handle, stderr=handle
                )
            )
            say(f"{name} started", f"pid {processes[-1].pid}")

        if not wait_for(api_port):
            say("api never listened", str(logs["api"]))
            return 1
        say("api listening", f"127.0.0.1:{api_port}")

        url = f"http://127.0.0.1:{api_port}/webhooks/twilio/whatsapp"
        fields = {
            "MessageSid": f"SM{uuid.uuid4().hex}",
            "AccountSid": ACCOUNT_SID,
            "From": f"whatsapp:{CUSTOMER}",
            "To": f"whatsapp:{BUSINESS_NUMBER}",
            "Body": "كام رسوم الساعة المعتمدة؟",
            "NumMedia": "0",
        }
        response = httpx.post(
            url,
            content=urlencode(fields).encode(),
            headers={
                "content-type": "application/x-www-form-urlencoded",
                "X-Twilio-Signature": signed(url, fields, AUTH_TOKEN),
            },
            timeout=10,
        )
        say("signed webhook POSTed over TCP", f"{response.status_code}")
        if response.status_code != 200:
            _tails(logs)
            return 1

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not received:
            time.sleep(0.5)

        if not received:
            say("no reply reached the vendor", "")
            print("\n  queue state:")
            print(asyncio.run(_stream_state()))
            _tails(logs)
            return 1

        say(
            "the customer was shown a typing indicator",
            indicators[0]["messageId"] if indicators else "NONE — §2.5's mitigation "
            "did not fire, and the customer sees silence for the whole turn",
        )
        reply = received[0]
        say("a reply reached the vendor", reply.get("To", ""))
        print(f"\n  the customer would have received:\n\n    {reply.get('Body', '')}\n")
        return 0
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        server.shutdown()
        asyncio.run(purge())
        say("seeded rows removed", TENANT_SLUG)


if __name__ == "__main__":
    raise SystemExit(against_compose() if "--compose" in sys.argv else main())
