"""Run the demo, end to end, without a person in the loop — Task 42.

    uv run python scripts/rehearse.py

Three tenants, each with their own number, their own data and their own
vertical, driven through the real programs: a signed webhook over TCP, real
Valkey, the real workers, real model calls, and the reply leaving through the
Twilio adapter. The vendor's two hosts are a stub on loopback; everything else
is what runs.

**The questions that are not in the demo are the point.** A rehearsal that asks
only what the corpus answers proves the corpus. `evals/demo/rehearsal.yaml`
carries the adversarial ones beside the rest — negotiation, another
institution's fees, a bare request for a number, a type the catalogue does not
hold, a project this tenant does not sell — and each has an expectation that
fails loudly rather than a note that somebody reads afterwards.

**Every figure in every reply must trace.** Not "the answer looks right": the
provenance the source pane renders is checked on every turn, because a figure
with no source is the one thing §19.3 forbids and it is invisible in a
transcript.

Exit code 0 when every expectation held.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import httpx
import yaml

# `harness` is a sibling script, not a package. Inserted before the import
# rather than in `__main__` because the block below runs at import time and
# needs it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402 - the path bootstrap above must run first
    ROOT,
    Processes,
    Vendor,
    config_tree_pointing_at,
    free_port,
    isolate_queues,
    say,
    wait_for_port,
)

#: This run's own streams, consumer groups and idempotency keys — see
#: `harness.isolate_queues`. Set here, before anything imports `moc`, because
#: `config_store` caches per root and the parent process reads the same queue
#: names its children do.
_TAG = uuid.uuid4().hex[:8]
_VENDOR = Vendor()
_VENDOR.start()
_CONFIG_ROOT = config_tree_pointing_at(_VENDOR.port)
isolate_queues(_CONFIG_ROOT, _TAG)
os.environ["MOC_CONFIG_DIR"] = str(_CONFIG_ROOT)

SCRIPT = ROOT / "evals" / "demo" / "rehearsal.yaml"
AUTH_TOKEN = "rehearsal-auth-token"  # noqa: S105 - a loopback fixture
ACCOUNT_SID = "AC" + "0" * 32
SECRET_REF = "twilio/rehearsal/wa"  # noqa: S105 - a reference, not a secret


@dataclass
class Result:
    tenant: str
    ask: str
    reply: str = ""
    failures: list[str] = field(default_factory=list)
    traceback: str = ""


def signed(url: str, fields: dict[str, str], token: str) -> str:
    canonical = url + "".join(k + v for k, v in sorted(fields.items()))
    return base64.b64encode(
        hmac.new(token.encode(), canonical.encode(), hashlib.sha1).digest()
    ).decode()


# ─────────────────────────── the tenants and their data ───────────────────────────

_PURGE = (
    "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations "
    "WHERE tenant_id IN (SELECT id FROM tenants WHERE slug LIKE 'rehearsal-%'))",
    "DELETE FROM handoffs WHERE tenant_id IN "
    "(SELECT id FROM tenants WHERE slug LIKE 'rehearsal-%')",
    "DELETE FROM conversations WHERE tenant_id IN "
    "(SELECT id FROM tenants WHERE slug LIKE 'rehearsal-%')",
    "DELETE FROM contacts WHERE tenant_id IN "
    "(SELECT id FROM tenants WHERE slug LIKE 'rehearsal-%')",
    "DELETE FROM usage_ledger WHERE tenant_id IN "
    "(SELECT id FROM tenants WHERE slug LIKE 'rehearsal-%')",
    "DELETE FROM inventory_units WHERE tenant_id IN "
    "(SELECT id FROM tenants WHERE slug LIKE 'rehearsal-%')",
    "DELETE FROM channel_accounts WHERE tenant_id IN "
    "(SELECT id FROM tenants WHERE slug LIKE 'rehearsal-%')",
    "DELETE FROM tenants WHERE slug LIKE 'rehearsal-%'",
)


async def purge() -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings

    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        for statement in _PURGE:
            await session.execute(text(statement))
        await session.commit()
    await engine.dispose()


async def seed(tenants: list[dict]) -> dict[str, uuid.UUID]:
    """Three tenants, each with their own number, data and vertical."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings

    await purge()
    engine = create_async_engine(settings.database_url)
    ids: dict[str, uuid.UUID] = {}
    async with AsyncSession(engine) as session:
        for tenant in tenants:
            tenant_id = uuid.uuid4()
            ids[tenant["slug"]] = tenant_id
            await session.execute(
                text(
                    "INSERT INTO tenants (id, slug, name, vertical, default_lang, project) "
                    "VALUES (:id, :slug, :name, :vertical, 'ar', :project)"
                ),
                {
                    "id": tenant_id,
                    "slug": tenant["slug"],
                    "name": tenant["name"],
                    "vertical": tenant["vertical"],
                    "project": tenant.get("project"),
                },
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
                    "addr": tenant["number"],
                    "ref": SECRET_REF,
                    "secret": AUTH_TOKEN,
                },
            )
        await session.commit()
    await engine.dispose()
    return ids


async def stock(tenant_id: uuid.UUID, fixture: str) -> int:
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.config import settings
    from moc.retrieval.inventory import load_units
    from moc.tenancy.context import tenant_session

    engine = create_async_engine(settings.database_url)
    path = ROOT / "evals" / "fixtures" / fixture / "units.jsonl"
    async with tenant_session(engine, tenant_id) as session:
        count = await load_units(session=session, path=path)
        await session.commit()
    await engine.dispose()
    return count


async def ingest(tenant_id: uuid.UUID, fixture: str) -> int:
    """The education corpus into both arms.

    Both, deliberately. A retriever built with no dense arm is §7.3's degraded
    shape, every case still runs, and the recall is merely lower — which is the
    rehearsal proving something other than what it claims to.

    Embeddings come from the content-addressed cache, so re-running this costs
    nothing after the first time.
    """
    from moc.config_store import load
    from moc.llm.anthropic_direct import AnthropicDirect
    from moc.llm.openai_direct import OpenAIDirect
    from moc.llm.router import Router
    from moc.retrieval.chunker import embedding_text
    from moc.retrieval.embedding_cache import EmbeddingCache
    from moc.retrieval.lexical import (
        LexicalDocument,
        MeilisearchAdmin,
        MeilisearchRepository,
        meilisearch_client,
    )
    from moc.retrieval.records import VectorPoint
    from moc.retrieval.vectors import QdrantAdmin, QdrantRepository, qdrant_client

    routing = load("llm/routing")
    records = [
        json.loads(line)
        for line in (ROOT / "evals" / "fixtures" / fixture / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    meili = meilisearch_client()
    await MeilisearchAdmin(client=meili).ensure_indexes()
    await MeilisearchRepository(client=meili).add(
        tenant_id=tenant_id,
        vertical="education",
        documents=[
            LexicalDocument(
                point_id=f"{tenant_id}-{record['chunk_id']}",
                chunk_id=record["chunk_id"],
                content=record["content"],
                title=record["title"],
            )
            for record in records
        ],
    )
    await meili.aclose()

    router = Router(
        config=routing,
        providers={
            "anthropic": AnthropicDirect(
                api_key=os.environ["MOC_ANTHROPIC_API_KEY"], http=routing["http"]
            ),
            "openai": OpenAIDirect(
                api_key=os.environ["MOC_OPENAI_API_KEY"], http=routing["http"]
            ),
        },
    )

    class _Embedder:
        async def embed(self, *, texts):
            return await router.embed(texts=texts)

    embedding = routing["tasks"]["embedding"]["primary"]
    ingested = await EmbeddingCache(
        root=ROOT / ".cache" / "embeddings",
        model=embedding["model"],
        dimensions=embedding["dimensions"],
    ).embed(
        _Embedder(),
        [embedding_text(title=r["title"], content=r["content"]) for r in records],
    )

    qdrant = qdrant_client()
    await QdrantAdmin(client=qdrant).ensure_collections()
    await QdrantRepository(client=qdrant).upsert(
        tenant_id=tenant_id,
        vertical="education",
        points=[
            VectorPoint(
                chunk_id=record["chunk_id"],
                vector=vector,
                payload={"content": record["content"], "title": record["title"]},
            )
            for record, vector in zip(records, ingested.vectors, strict=True)
        ],
    )
    await qdrant.close()
    return len(records)


# ─────────────────────────── the expectations ───────────────────────────


async def dead_letters_since(before: int) -> list[dict]:
    """Turns that died rather than replying, with where they died.

    The first rehearsal waited for a reply that never came, timed out, and then
    matched the *next* turn's reply to this turn — so one dead turn shifted
    every line of the transcript after it and the run read as a series of
    non-sequiturs rather than as one failure. A turn is finished when a reply
    arrives **or** when it is buried, and both are answers to "what happened".
    """
    from moc.channels.valkey import valkey_client
    from moc.config_store import load

    stream = load("workers/queues")["inbound"]["dead_letter_stream"]
    client = valkey_client()
    entries = await client.xrange(stream)
    await client.aclose()
    return [fields for _, fields in entries[before:]]


async def dead_letter_count() -> int:
    from moc.channels.valkey import valkey_client
    from moc.config_store import load

    client = valkey_client()
    count = await client.xlen(load("workers/queues")["inbound"]["dead_letter_stream"])
    await client.aclose()
    return count


async def provenance_for(tenant_id: uuid.UUID, customer: str) -> dict | None:
    """The evidence behind the newest bot reply, as the source pane would show
    it. Read from the thread rather than from the turn, because what the pane
    renders is what was *stored*."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.config import settings
    from moc.tenancy.context import tenant_session

    engine = create_async_engine(settings.app_database_url())
    async with tenant_session(engine, tenant_id) as session:
        row = (
            await session.execute(
                text(
                    "SELECT m.provenance FROM messages m "
                    "JOIN conversations c ON c.id = m.conversation_id "
                    "WHERE c.sender_ref = :ref AND m.author = 'bot' "
                    "ORDER BY m.seq DESC LIMIT 1"
                ),
                {"ref": customer},
            )
        ).scalar_one_or_none()
    await engine.dispose()
    return row


def judge(turn: dict, reply: str, provenance: dict | None) -> list[str]:
    """What went wrong with this turn, in the terms the script asked for."""
    failures: list[str] = []

    if not reply:
        return ["no reply reached the customer"]

    for forbidden in turn.get("must_not_contain", []):
        if forbidden in reply:
            failures.append(f"the reply contains {forbidden!r}")

    figures = (provenance or {}).get("figures", [])
    if turn.get("every_figure_traced"):
        orphans = [f["raw"] for f in figures if not f["grounded"]]
        if orphans:
            failures.append(f"figures with no source: {orphans}")

    if turn.get("traces_to_calculator") and not any(
        f["source"] == "calculator" for f in figures
    ):
        failures.append("no figure traced to the calculator")

    if turn.get("must_state_asof") and not any(
        f.get("asOf") for f in figures
    ):
        failures.append("no figure carried the date its row was snapshotted")

    return failures


# ─────────────────────────── the run ───────────────────────────


def deliver(api_port: int, tenant: dict, ask: str) -> int:
    url = f"http://127.0.0.1:{api_port}/webhooks/twilio/whatsapp"
    fields = {
        "MessageSid": f"SM{uuid.uuid4().hex}",
        "AccountSid": ACCOUNT_SID,
        "From": f"whatsapp:{tenant['customer']}",
        "To": f"whatsapp:{tenant['number']}",
        "Body": ask,
        "NumMedia": "0",
    }
    response = httpx.post(
        url,
        content=urlencode(fields).encode(),
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "X-Twilio-Signature": signed(url, fields, AUTH_TOKEN),
        },
        timeout=15,
    )
    return response.status_code


def main() -> int:
    script = yaml.safe_load(SCRIPT.read_text(encoding="utf-8"))
    tenants = script["tenants"]
    # `--only <slug>` re-runs one tenant. A rehearsal takes minutes and costs
    # model calls, and the question after a failure is always about one of
    # them.
    if "--only" in sys.argv:
        wanted = sys.argv[sys.argv.index("--only") + 1]
        tenants = [tenant for tenant in tenants if tenant["slug"] == wanted]
    print("\nrehearsal\n")

    vendor = _VENDOR
    say("vendor stub listening", f"127.0.0.1:{vendor.port}")
    say("this run's own streams", f"…:{_TAG}")

    ids = asyncio.run(seed(tenants))
    for tenant in tenants:
        tenant_id = ids[tenant["slug"]]
        if tenant.get("inventory"):
            units = asyncio.run(stock(tenant_id, tenant["inventory"]))
            say(f"{tenant['slug']} inventory", f"{units} units")
        if tenant.get("corpus"):
            chunks = asyncio.run(ingest(tenant_id, tenant["corpus"]))
            say(f"{tenant['slug']} corpus", f"{chunks} chunks")

    api_port = free_port()
    reference = SECRET_REF.replace("/", "__").upper()
    processes = Processes(
        environment={
            **os.environ,
            "MOC_CONFIG_DIR": str(_CONFIG_ROOT),
            f"MOC_SECRET_{reference}": AUTH_TOKEN,
            f"MOC_SECRET_{reference}__SID": ACCOUNT_SID,
            "PYTHONUNBUFFERED": "1",
        }
    )

    results: list[Result] = []
    try:
        processes.api("moc.api.main:webhook_app", api_port)
        processes.worker("inbound")
        processes.worker("outbound")
        if not wait_for_port(api_port):
            processes.tails()
            return 1
        say("api and workers up", f"127.0.0.1:{api_port}")

        print()
        for tenant in tenants:
            print(f"  ── {tenant['name']} ({tenant['vertical']}) " + "─" * 24)
            for turn in tenant["turns"]:
                result = Result(tenant=tenant["slug"], ask=turn["ask"])
                dead_before = asyncio.run(dead_letter_count())
                status = deliver(api_port, tenant, turn["ask"])
                if status != 200:
                    result.failures.append(f"the webhook answered {status}")
                    results.append(result)
                    continue

                message = vendor.wait_for_message()
                result.reply = (message or {}).get("Body", "")
                if not result.reply:
                    # Buried rather than answered. Reported here so the turn is
                    # finished either way and the next one is not matched to a
                    # reply that belongs to this one.
                    for buried in asyncio.run(dead_letters_since(dead_before)):
                        result.failures.append(f"the turn died: {buried['reason']}")
                        result.traceback = buried.get("traceback") or (
                            f"<the row carries no traceback; its fields are "
                            f"{sorted(buried)}>"
                        )
                    dead_before = asyncio.run(dead_letter_count())

                provenance = asyncio.run(
                    provenance_for(ids[tenant["slug"]], tenant["customer"])
                )
                result.failures += judge(turn, result.reply, provenance)
                results.append(result)

                mark = "ok  " if not result.failures else "FAIL"
                print(f"  {mark}  {turn['ask']}")
                if result.reply:
                    print(f"        -> {result.reply[:110]}")
                for failure in result.failures:
                    print(f"        !! {failure}")
            print()
    finally:
        processes.stop()
        vendor.stop()

    failed = [result for result in results if result.failures]
    print(f"  {len(results) - len(failed)}/{len(results)} turns held their expectations")
    if failed:
        print("\n  what would have gone wrong in the room:")
        for result in failed:
            print(f"    {result.tenant}: {result.ask}")
            for failure in result.failures:
                print(f"      {failure}")
            if result.traceback:
                print("      " + result.traceback.strip().replace("\n", "\n      "))
        return 1

    asyncio.run(purge())
    say("rehearsal tenants removed", "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
