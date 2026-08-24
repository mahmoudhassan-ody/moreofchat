"""Check every link in the path before a phone is pointed at it — Task 39.

    uv run python scripts/preflight.py

Each check names the one thing that is wrong and how to fix it. Three of them
exist because driving the path for the first time found exactly that fault, and
none of the three had a behavioural signature short of "no reply arrives":

- **The dev database was eight migrations behind.** Every test migrates a fresh
  `moc_test`, so the database the system would actually run against had been at
  revision 0009 since Task 23 while the code moved to 0017.
- **`.env` defined `MOC_LOOKUP_PASSWORD` twice**, with different values. The
  last assignment wins in both `set -a; . .env` and pydantic-settings, and the
  role's password had been set from the other one — so the webhook process
  could not authenticate at all.
- **The role's password had never been set outside the test fixture.** The test
  conftest runs `ALTER ROLE ... WITH PASSWORD` on every session, which is why
  the suite is green about a credential production does not have.

Exit code 0 when everything passes, 1 otherwise. Nothing here writes.
"""

import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: What each channel needs to *send*, beyond the inbound-verification secret
#: named by `channel_accounts.secret_ref`. See `moc.channels.senders`.
_OUTBOUND_SUFFIXES = {
    "whatsapp": ("/sid",),
    "telegram": ("/token",),
    "messenger": ("/token",),
    "instagram": ("/token",),
    "email": ("/apikey",),
}

_OK = "  ok  "
_BAD = " FAIL "

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "", note: str = "") -> bool:
    """`detail` is the fix, and is printed only when the check fails.

    Printing it either way put "run: uv run alembic upgrade head" beside a
    passing migration check, which reads as an instruction. `note` is for the
    value worth seeing when it passes.
    """
    shown = note if passed else detail
    print(f"{_OK if passed else _BAD}{name}{': ' + shown if shown else ''}")
    if not passed:
        failures.append(name)
    return passed


def env_file_has_no_duplicate_keys() -> None:
    """A duplicate assignment is not an error anywhere that reads it.

    The last one wins, silently, and whatever was configured from the first
    stops matching. Values are never printed here — only names and counts.
    """
    path = ROOT / ".env"
    if not path.exists():
        check(".env present", False, "no .env file")
        return
    names = Counter(
        line.split("=", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    duplicated = {name for name, count in names.items() if count > 1}
    check(
        ".env has no duplicate keys",
        not duplicated,
        f"shadowed: {sorted(duplicated)}" if duplicated else "",
    )


async def database() -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.config import settings

    engine = create_async_engine(settings.database_url)
    try:
        async with engine.connect() as connection:
            current = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one_or_none()
        check("postgres reachable", True, note=settings.pg_database)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        check("postgres reachable", False, str(exc)[:120])
        await engine.dispose()
        return
    await engine.dispose()

    head = _head_revision()
    check(
        "database is migrated to head",
        current == head,
        f"at {current}, head is {head} — run: uv run alembic upgrade head",
        note=str(current),
    )


def _head_revision() -> str:
    """The newest revision on disk, without importing alembic's config."""
    revisions: dict[str, str] = {}
    for path in (ROOT / "migrations" / "versions").glob("*.py"):
        body = path.read_text(encoding="utf-8")
        revision = _literal(body, "revision")
        down = _literal(body, "down_revision")
        if revision:
            revisions[revision] = down or ""
    parents = set(revisions.values())
    heads = [r for r in revisions if r not in parents]
    return heads[0] if len(heads) == 1 else f"ambiguous: {sorted(heads)}"


def _literal(body: str, name: str) -> str:
    for line in body.splitlines():
        if line.startswith(f"{name}:") and "=" in line:
            value = line.split("=", 1)[1].strip()
            return value.strip("\"'") if value not in ("None", "") else ""
    return ""


async def roles() -> None:
    """Both application roles, as the processes actually connect.

    `moc_lookup` is the one that matters here: the test conftest sets its
    password on every run, so a suite that is entirely green says nothing about
    whether the deployed webhook process can log in.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.config import settings

    for label, url, probe in (
        ("moc_app connects", settings.app_database_url(), "SELECT 1"),
        (
            "moc_lookup connects and can read the view",
            settings.lookup_database_url(),
            "SELECT count(*) FROM channel_account_lookup",
        ),
    ):
        engine = create_async_engine(url)
        try:
            async with engine.connect() as connection:
                await connection.execute(text(probe))
            check(label, True)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            check(
                label,
                False,
                f"{type(exc).__name__} — set the role's password to match .env",
            )
        await engine.dispose()


async def queue() -> None:
    from moc.channels.valkey import valkey_client
    from moc.config_store import load

    queues = load("workers/queues")
    client = valkey_client()
    try:
        await client.ping()
        check("valkey reachable", True)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        check("valkey reachable", False, str(exc)[:120])
        await client.aclose()
        return

    configured = client.connection_pool.connection_kwargs["socket_timeout"]
    longest = max(
        section["block_ms"]
        for section in queues.values()
        if isinstance(section, dict) and "block_ms" in section
    )
    check(
        "socket timeout outlasts the longest blocking read",
        configured > longest / 1000,
        f"{configured}s vs {longest}ms — a worker on an idle stream would exit",
        note=f"{configured}s > {longest}ms",
    )

    for name in ("inbound", "outbound"):
        dead = await client.xlen(queues[name]["dead_letter_stream"])
        check(
            f"{name} dead-letter stream is empty",
            dead == 0,
            f"{dead} buried — read them before the demo; each one is a customer "
            f"who was not answered",
        )
    await client.aclose()


async def search_backends() -> None:
    from moc.config_store import load
    from moc.retrieval.lexical import meilisearch_client
    from moc.retrieval.vectors import qdrant_client

    wanted = set(load("retrieval/lexical")["meilisearch"]["indexes"].values())
    meili = meilisearch_client()
    try:
        present = {index.uid for index in await meili.get_indexes(limit=100) or []}
        check(
            "meilisearch indexes exist",
            wanted <= present,
            f"missing {sorted(wanted - present)}",
            note=f"{len(present)} present",
        )
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        check("meilisearch indexes exist", False, str(exc)[:120])
    await meili.aclose()

    qdrant = qdrant_client()
    try:
        collections = {c.name for c in (await qdrant.get_collections()).collections}
        check(
            "qdrant collections exist",
            bool(collections),
            "none — run the ingestion path once, or QdrantAdmin.ensure_collections",
            note=", ".join(sorted(collections)),
        )
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        check("qdrant collections exist", False, str(exc)[:120])
    await qdrant.close()


async def channel_secrets() -> None:
    """Every connected account, and whether this host can send on it.

    The inbound secret verifies; a *different* value sends. An account whose
    outbound credential is missing accepts messages and answers none of them —
    the customer's question arrives, the turn runs, the model is paid for, and
    the reply dead-letters.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.channels.accounts import EnvSecretResolver
    from moc.config import settings

    engine = create_async_engine(settings.database_url)
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text("SELECT channel, address, secret_ref FROM channel_accounts ORDER BY address")
            )
        ).all()
    await engine.dispose()

    if not rows:
        check("at least one channel account is connected", False, "none in the table")
        return
    check("at least one channel account is connected", True, note=f"{len(rows)}")

    resolver = EnvSecretResolver()
    for row in rows:
        refs = [row.secret_ref] + [
            row.secret_ref + suffix for suffix in _OUTBOUND_SUFFIXES.get(row.channel, ())
        ]
        missing = [
            resolver.variable_for(ref)
            for ref in refs
            if not os.environ.get(resolver.variable_for(ref))
        ]
        check(
            f"{row.channel} {row.address} has its secrets",
            not missing,
            f"missing {missing}",
        )


async def main() -> int:
    print("\npreflight\n")
    env_file_has_no_duplicate_keys()
    await database()
    await roles()
    await queue()
    await search_backends()
    await channel_secrets()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("every link in the path is configured")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
