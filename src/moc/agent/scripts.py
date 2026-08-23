"""Tenant scripts, drafted and published — demo plan Task 33.

A script is the tenant's own conversation flow, and until now it was a file in
the repository. This is the tier that lets an admissions officer change what
the bot asks without a deploy.

**A draft is a version that has never been reachable by a customer.** Preview
builds the engine from the draft and asks it what it would do — a preview that
pretty-printed the document would be a text editor with a border round it, and
the question a tenant has is "what will it say", not "is my YAML valid".

**Publishing pins.** `ConversationState` has carried `script_version` since it
was written; this is what makes it mean something. A customer three turns into
version 1 keeps version 1 — a slot they already filled disappearing mid-flow is
exactly the frustration that causes the handoff the script exists to avoid.
Nothing deletes a published version for the same reason: a conversation pinned
to it must still be able to load it.

**This lives in `moc.agent`, not in `moc.tenancy`.** It was written under
tenancy — it is a tenant-scoped store using `tenant_session` — and
import-linter refused it: tenancy is the bottom layer, and previewing a script
means building a `ScriptEngine`, which is an agent concept. A script is a thing
the agent runs that happens to be stored per tenant, and the arrow points that
way round. The contract caught it; the alternative was importing the engine
inside the function to hide the dependency from the linter, which would have
hidden it from a reader too.

**The confidence gate is not here.** It is a platform property with a floor
(`moc.tenancy.settings`), not a field on a script, and a script that could
carry one would be a second place to set it — with the tenant tier winning.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from moc.agent.script_engine import ScriptEngine
from moc.tenancy.context import tenant_session

#: The engine's own platform defaults, by the name the engine uses. A second
#: name for the same document would be a second answer to "what does a script
#: inherit when it says nothing".
_DEFAULTS = "agent/defaults"


class NoDraft(ValueError):
    """Publish was asked for and there is nothing drafted to publish."""


@dataclass(frozen=True)
class ScriptVersion:
    script_id: str
    version: int
    body: dict[str, Any]
    published: bool


class ScriptStore:
    def __init__(self, *, engine: Any) -> None:
        self._engine = engine

    async def save_draft(
        self, *, tenant_id: uuid.UUID, script_id: str, body: dict[str, Any]
    ) -> ScriptVersion:
        """Write or replace the unpublished draft for this script.

        One draft at a time, deliberately. Two would need a way to say which
        one publish means, and the answer would be a dropdown on a screen whose
        whole job is to make a change safe to make.
        """
        async with tenant_session(self._engine, tenant_id) as session:
            existing = await self._draft_row(session, script_id)
            if existing is not None:
                await session.execute(
                    text("UPDATE tenant_scripts SET body = cast(:body as jsonb) WHERE id = :id"),
                    {"body": json.dumps(body, ensure_ascii=False), "id": existing.id},
                )
                version = existing.version
            else:
                version = await self._next_version(session, script_id)
                await session.execute(
                    text(
                        "INSERT INTO tenant_scripts "
                        "(id, tenant_id, script_id, version, body) "
                        "VALUES (:id, :tenant_id, :script_id, :version, cast(:body as jsonb))"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "tenant_id": tenant_id,
                        "script_id": script_id,
                        "version": version,
                        "body": json.dumps(body, ensure_ascii=False),
                    },
                )
            await session.commit()
        return ScriptVersion(
            script_id=script_id, version=version, body=body, published=False
        )

    async def preview(self, *, tenant_id: uuid.UUID, script_id: str) -> dict[str, Any]:
        """What the drafted script would do, from the engine rather than the text."""
        from moc.config_store import load

        async with tenant_session(self._engine, tenant_id) as session:
            row = await self._draft_row(session, script_id)
        if row is None:
            raise NoDraft(f"no draft for {script_id!r}")

        body = row.body if isinstance(row.body, dict) else json.loads(row.body)
        engine = ScriptEngine(script=body, defaults=load(_DEFAULTS))
        start = engine.start()
        return {
            # `nodes` is a mapping of name -> node, which is the shape the
            # engine reads. Listing the keys rather than a field inside each
            # one, because the name IS the node id.
            "nodes": sorted(body.get("nodes") or {}),
            "entry_node": start.node,
            # Read back off the engine, not off the document: a script that
            # omits it inherits the platform default, and a preview showing the
            # blank would be showing something other than what would run.
            "max_consecutive_clarifications": engine._max_clarifications,
            "referral": engine.referral("ar"),
        }

    async def publish(
        self, *, tenant_id: uuid.UUID, script_id: str, agent_id: str
    ) -> ScriptVersion:
        async with tenant_session(self._engine, tenant_id) as session:
            row = await self._draft_row(session, script_id)
            if row is None:
                # Refused, not a silent no-op reporting success to a screen.
                raise NoDraft(
                    f"no draft for {script_id!r} — there is nothing to publish, and "
                    "a publish that quietly did nothing would report success for it"
                )
            await session.execute(
                text("UPDATE tenant_scripts SET published_at = now() WHERE id = :id"),
                {"id": row.id},
            )
            # Audited with the settings, in the same table: "the bot got worse
            # yesterday" is most often a script change, and two logs would be
            # two places to look.
            await session.execute(
                text(
                    "INSERT INTO settings_audit "
                    "(id, tenant_id, setting, old_value, new_value, agent_id) "
                    "VALUES (:id, :tenant_id, :setting, :old, :new, :agent_id)"
                ),
                {
                    "id": uuid.uuid4(),
                    "tenant_id": tenant_id,
                    "setting": f"script:{script_id}",
                    "old": str(row.version - 1) if row.version > 1 else None,
                    "new": str(row.version),
                    "agent_id": agent_id,
                },
            )
            await session.commit()
        body = row.body if isinstance(row.body, dict) else json.loads(row.body)
        return ScriptVersion(
            script_id=script_id, version=row.version, body=body, published=True
        )

    async def current(
        self, *, tenant_id: uuid.UUID, script_id: str
    ) -> ScriptVersion | None:
        """The newest published version, or None if none has been published."""
        async with tenant_session(self._engine, tenant_id) as session:
            row = (
                await session.execute(
                    text(
                        "SELECT version, body FROM tenant_scripts "
                        "WHERE script_id = :script_id AND published_at IS NOT NULL "
                        "ORDER BY version DESC LIMIT 1"
                    ),
                    {"script_id": script_id},
                )
            ).first()
        return _version(script_id, row, published=True) if row is not None else None

    async def at_version(
        self, *, tenant_id: uuid.UUID, script_id: str, version: int
    ) -> ScriptVersion | None:
        """What a pinned conversation is still running."""
        async with tenant_session(self._engine, tenant_id) as session:
            row = (
                await session.execute(
                    text(
                        "SELECT version, body, published_at FROM tenant_scripts "
                        "WHERE script_id = :script_id AND version = :version"
                    ),
                    {"script_id": script_id, "version": version},
                )
            ).first()
        if row is None:
            return None
        return _version(script_id, row, published=row.published_at is not None)

    # ─────────────────────────── internals ───────────────────────────

    async def _draft_row(self, session: Any, script_id: str) -> Any:
        return (
            await session.execute(
                text(
                    "SELECT id, version, body FROM tenant_scripts "
                    "WHERE script_id = :script_id AND published_at IS NULL"
                ),
                {"script_id": script_id},
            )
        ).first()

    async def _next_version(self, session: Any, script_id: str) -> int:
        highest = (
            await session.execute(
                text(
                    "SELECT max(version) FROM tenant_scripts WHERE script_id = :script_id"
                ),
                {"script_id": script_id},
            )
        ).scalar_one_or_none()
        return (highest or 0) + 1


def _version(script_id: str, row: Any, *, published: bool) -> ScriptVersion:
    body = row.body if isinstance(row.body, dict) else json.loads(row.body)
    return ScriptVersion(
        script_id=script_id, version=row.version, body=body, published=published
    )


__all__ = ["NoDraft", "ScriptResolver", "ScriptStore", "ScriptVersion"]


class ScriptResolver:
    """Which script a turn runs, for this tenant and this conversation.

    **The missing half of publishing.** Version pinning was written into
    `ConversationState` from the start and `ScriptStore.publish` records the
    version — and until this existed nothing read either. The worker built one
    `ScriptEngine.from_config` at construction and reused it for every turn and
    every tenant, so a script edited in the console changed nothing the bot
    did: someone edits a script, asks the question, and gets the old answer.

    Worse than nothing changing. `ScriptEngine._require_pinned_version` raises
    when a conversation's state names a version the engine is not running, so
    publishing version 2 would have broken every in-flight conversation with an
    exception rather than merely ignoring the edit.

    Three sources, in order:

    - the version a conversation is **pinned** to, so a customer three turns in
      is not moved onto a script they never started;
    - the tenant's **current published** version, for a new conversation;
    - the **config file**, for a tenant who has published nothing — which is
      every tenant until they open the console, and is the only reason the
      platform script is still a file.

    Cached by (tenant, script, version). A published version is immutable —
    publishing again writes a new row — so the cache can never serve something
    stale. A *draft* is mutable and is deliberately not resolvable here: a
    draft is by definition the version no customer has reached.
    """

    def __init__(self, *, store: ScriptStore, fallback: str) -> None:
        self._store = store
        self._fallback = fallback
        self._cache: dict[tuple[uuid.UUID, str, int], ScriptEngine] = {}

    async def current(self, *, tenant_id: uuid.UUID) -> ScriptEngine:
        """What a NEW conversation should run."""

        default = ScriptEngine.from_config(self._fallback)
        published = await self._store.current(
            tenant_id=tenant_id, script_id=default.script_id
        )
        if published is None:
            return default
        return self._engine(tenant_id, published.script_id, published.version, published.body)

    async def at(
        self, *, tenant_id: uuid.UUID, script_id: str, version: int
    ) -> ScriptEngine | None:
        """What a PINNED conversation is still running, or None if it is gone.

        None rather than a fallback to the current version: silently moving a
        conversation onto a newer script is the thing pinning exists to
        prevent, and the caller should decide loudly.
        """
        cached = self._cache.get((tenant_id, script_id, version))
        if cached is not None:
            return cached
        stored = await self._store.at_version(
            tenant_id=tenant_id, script_id=script_id, version=version
        )
        if stored is None:
            return None
        return self._engine(tenant_id, script_id, version, stored.body)

    def _engine(
        self, tenant_id: uuid.UUID, script_id: str, version: int, body: dict[str, Any]
    ) -> ScriptEngine:
        from moc.config_store import load

        engine = ScriptEngine(script=body, defaults=load(_DEFAULTS))
        self._cache[(tenant_id, script_id, version)] = engine
        return engine
