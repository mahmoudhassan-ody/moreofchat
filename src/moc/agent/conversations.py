"""Loading and persisting a conversation — design §5.

One thread per customer per channel. `ConversationState` already knows how to
render itself as JSON (Task 12 built `to_json`/`from_json` for exactly this
column), so this module is only the lookup and the upsert.

The upsert is `ON CONFLICT` against the partial unique index from migration
0005 rather than a select-then-insert. Two deliveries of the same customer's
messages can be in flight across two workers, and select-then-insert would let
both find nothing and both insert — two threads, two script cursors, and the
customer answered from whichever one happened to win.

Everything here runs inside the caller's tenant session, so the tenant comes
from the transaction and never from an argument. A worker that forgot to open
one reads nothing rather than reading someone else's conversation.
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.script_engine import ScriptEngine
from moc.agent.state import ConversationState

_LOAD = text(
    """
SELECT state FROM conversations
WHERE channel = :channel AND sender_ref = :sender_ref
"""
)

_UPSERT = text(
    """
INSERT INTO conversations (
  id, tenant_id, state, channel, sender_ref, channel_account_id, last_inbound_at
) VALUES (
  :id,
  nullif(current_setting('moc.tenant_id', true), '')::uuid,
  cast(:state as jsonb), :channel, :sender_ref, :channel_account_id, :last_inbound_at
)
ON CONFLICT (tenant_id, channel, sender_ref)
  WHERE channel IS NOT NULL AND sender_ref IS NOT NULL
DO UPDATE SET
  state = EXCLUDED.state,
  channel_account_id = EXCLUDED.channel_account_id,
  last_inbound_at = EXCLUDED.last_inbound_at
"""
)


class ConversationStore:
    def __init__(self, *, session: AsyncSession, engine: ScriptEngine) -> None:
        self._session = session
        self._engine = engine

    async def load(self, *, channel: str, sender_ref: str) -> ConversationState:
        """The customer's thread, or a fresh one at the script's entry node.

        A first message and a resumed conversation both have to produce a
        usable state, and the difference must not be the caller's problem —
        every call site would get it right until one did not.
        """
        row = (
            await self._session.execute(
                _LOAD, {"channel": channel, "sender_ref": sender_ref}
            )
        ).scalar_one_or_none()
        if not row:
            return self._engine.start()
        return ConversationState.from_json(row)

    async def save(
        self,
        *,
        channel: str,
        sender_ref: str,
        channel_account_id: UUID | None,
        last_inbound_at: datetime | None,
        state: ConversationState,
    ) -> None:
        import json

        await self._session.execute(
            _UPSERT,
            {
                "id": uuid4(),
                "state": json.dumps(state.to_json(), ensure_ascii=False),
                "channel": channel,
                "sender_ref": sender_ref,
                "channel_account_id": channel_account_id,
                "last_inbound_at": last_inbound_at,
            },
        )


__all__ = ["ConversationStore"]
