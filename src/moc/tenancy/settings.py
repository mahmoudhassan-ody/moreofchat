"""The tenant tier of §19's config layering — demo plan Task 33.

Platform config is a file; this is what a tenant may change from the console
without a deploy. Two settings so far, and the pair states the rule.

**`min_score` may be raised and never lowered.** A tenant whose corpus
separates cleanly should be able to make retrieval stricter — that is what the
platform config comment has said since fusion was written. The other direction
produces no error, no log and no failing test: just more answers, some of them
wrong, arriving as a quality drift nobody can date. So the floor lives in code
and the value lives in config, exactly as the KDF work factor does.

**Refused, never clamped.** Silently substituting the floor would leave a
settings screen saying one thing and the system doing another, and the next
person to read the screen would believe it.

**`bounds()` is what the console renders from.** A frontend holding its own
list of settings would show a control for something the backend has withdrawn,
which is the disabled-slider failure arriving from the other side: a control
that exists on screen and not in the system. A setting the engine refuses
should simply not appear — a greyed-out slider implies the setting is there and
you are not allowed it, and the truth is that a floor is a property of the
system rather than a permission level.

**`synonyms` is unbounded, and it is here rather than in the search index
because the index is shared.** One Meilisearch index per vertical serves every
tenant in it, so a synonym written into the index settings is one broker's word
for an area changing another broker's ranking. These are applied to the query
instead — see `moc.retrieval.lexical.expand_query` — which is the only scoping
a shared index allows.
"""

import json
import uuid
from typing import Any

from sqlalchemy import text

from moc.tenancy.context import tenant_session

MIN_SCORE = "min_score"
SYNONYMS = "synonyms"

_LEXICAL = "retrieval/lexical"
#: A cosine score. Above 1.0 nothing can ever pass, which is a way of turning
#: the bot off that should not be reachable by dragging a slider.
_MAX_SCORE = 1.0


class BelowFloor(ValueError):
    """A change the engine refuses. Raised, and nothing is written."""


def bounds() -> dict[str, dict[str, Any]]:
    """Every setting the console may offer, and what it may offer for it.

    The floor comes from the platform config rather than from a literal here:
    two answers to "how permissive may retrieval be" would diverge the first
    time somebody tuned one of them.
    """
    from moc.config_store import load

    floor = load(_LEXICAL)["fusion"][MIN_SCORE]
    return {
        MIN_SCORE: {
            "kind": "number",
            "min": floor,
            "max": _MAX_SCORE,
            "description": (
                "How similar a passage must be before the bot will answer from "
                "it. Higher means the bot hands off more often and guesses less."
            ),
        },
        SYNONYMS: {
            "kind": "map",
            # A tenant's own vocabulary cannot be wrong, so there is nothing to
            # bound. Present with nulls rather than absent, so the console can
            # render every declared setting from one shape.
            "min": None,
            "max": None,
            "description": (
                "Your words for things the knowledge base calls something else."
            ),
        },
    }


def defaults() -> dict[str, Any]:
    """What an unconfigured tenant runs on — the platform tier, not zeros."""
    from moc.config_store import load

    return {
        MIN_SCORE: load(_LEXICAL)["fusion"][MIN_SCORE],
        SYNONYMS: {},
    }


class SettingsStore:
    def __init__(self, *, engine: Any) -> None:
        self._engine = engine

    async def effective(self, *, tenant_id: uuid.UUID) -> dict[str, Any]:
        """Platform defaults with this tenant's overrides on top."""
        async with tenant_session(self._engine, tenant_id) as session:
            rows = (
                await session.execute(text("SELECT setting, value FROM tenant_settings"))
            ).all()
        settings = defaults()
        for row in rows:
            settings[row.setting] = json.loads(row.value)
        return settings

    async def put(
        self, *, tenant_id: uuid.UUID, changes: dict[str, Any], agent_id: str
    ) -> dict[str, Any]:
        """Apply `changes`, refusing anything outside its declared bounds.

        **Everything is checked before anything is written.** A partial apply
        would leave the tenant configured half in the state they asked for and
        half in the state they had, with no single row saying which — and the
        audit would record the half that landed as though it were the request.
        """
        declared = bounds()
        for name, value in changes.items():
            _check(name, value, declared)

        current = await self.effective(tenant_id=tenant_id)
        async with tenant_session(self._engine, tenant_id) as session:
            for name, value in changes.items():
                await session.execute(
                    text(
                        "INSERT INTO tenant_settings (id, tenant_id, setting, value) "
                        "VALUES (:id, :tenant_id, :setting, :value) "
                        "ON CONFLICT (tenant_id, setting) DO UPDATE SET "
                        "value = EXCLUDED.value, updated_at = now()"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "tenant_id": tenant_id,
                        "setting": name,
                        "value": json.dumps(value, ensure_ascii=False),
                    },
                )
                await session.execute(
                    text(
                        "INSERT INTO settings_audit "
                        "(id, tenant_id, setting, old_value, new_value, agent_id) "
                        "VALUES (:id, :tenant_id, :setting, :old, :new, :agent_id)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "tenant_id": tenant_id,
                        "setting": name,
                        # The old value too: "someone raised it" and "someone
                        # raised it from the floor" are different facts, and
                        # only the second says whether the change mattered.
                        "old": json.dumps(current.get(name), ensure_ascii=False),
                        "new": json.dumps(value, ensure_ascii=False),
                        "agent_id": agent_id,
                    },
                )
            await session.commit()
        return await self.effective(tenant_id=tenant_id)


def _check(name: str, value: Any, declared: dict[str, dict[str, Any]]) -> None:
    bound = declared.get(name)
    if bound is None:
        raise BelowFloor(
            f"{name!r} is not a setting a tenant may change. The console renders "
            "what bounds() declares, so this can only arrive from a client that "
            "made it up."
        )
    if bound["kind"] != "number":
        return
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BelowFloor(f"{name} must be a number, not {type(value).__name__}")
    if value < bound["min"]:
        raise BelowFloor(
            f"{name} is {value}, below the platform floor of {bound['min']}. It may "
            "be raised and not lowered: a stricter setting costs a handoff, and a "
            "more permissive one costs an answer that should never have been sent — "
            "and only the second is invisible."
        )
    if value > bound["max"]:
        raise BelowFloor(
            f"{name} is {value}, above {bound['max']}. Nothing could pass this gate, "
            "which is a way of switching the bot off by dragging a slider."
        )


__all__ = ["MIN_SCORE", "SYNONYMS", "BelowFloor", "SettingsStore", "bounds", "defaults"]
