"""Handoff records and the agent-visible thread — design §9, P1 Task 22.

The script engine decides *when* a conversation needs a human (§9: an explicit
request, or three clarifications that got nowhere). This module is what happens
after: the record an agent picks up, the thread they read, and the cursor the
bot resumes from.

**`resume_state` is the point of the handoff row.** It holds the script cursor
exactly as it stood when the human took over. Return-to-bot writes it back
verbatim rather than recomputing it, because the alternative — restarting at
the script's entry node — asks the customer again for the slots they already
gave, which is the frustration that caused the handoff in the first place.

**An agent's message is recorded but never replayed.** It is in the thread so
the conversation reads correctly, and it is excluded from what the bot is
asked to answer, because it is not customer input. Feeding it back would have
the bot answer the agent's own words to the customer.

Every query here runs under RLS on a session whose tenant is already set. None
of them names `tenant_id`: the filter is the policy's job, and a hand-written
one here would be a second place for it to be wrong, and the place nobody
checks once the policy exists.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

#: Who wrote a message. Closed, and mirrored by a CHECK in migration 0008 —
#: `unprocessed_inbound` decides what the bot is asked to answer, and a fourth
#: author value would change that answer silently.
CUSTOMER, BOT, AGENT = "customer", "bot", "agent"

_OPEN = text("""
INSERT INTO handoffs (id, tenant_id, conversation_id, reason, resume_state)
SELECT :id, c.tenant_id, c.id, :reason, cast(:resume_state as jsonb)
FROM conversations c WHERE c.id = :conversation_id
RETURNING id, conversation_id, reason, status, resume_state,
          opened_at, claimed_at, claimed_by, returned_at
""")

_CLAIM = text("""
UPDATE handoffs
SET claimed_at = now(), claimed_by = :agent_id, status = 'claimed'
WHERE id = :id AND status <> 'returned'
RETURNING id, conversation_id, reason, status, resume_state,
          opened_at, claimed_at, claimed_by, returned_at
""")

_RETURN = text("""
UPDATE handoffs SET returned_at = now(), status = 'returned'
WHERE id = :id AND status <> 'returned'
RETURNING id, conversation_id, reason, status, resume_state,
          opened_at, claimed_at, claimed_by, returned_at
""")

_RESTORE_CURSOR = text(
    "UPDATE conversations SET state = cast(:state as jsonb) WHERE id = :conversation_id"
)

_LIST_OPEN = text("""
SELECT h.id, h.conversation_id, h.reason, h.status, h.resume_state,
       h.opened_at, h.claimed_at, h.claimed_by, h.returned_at,
       c.channel, c.sender_ref, c.contact_id
FROM handoffs h JOIN conversations c ON c.id = h.conversation_id
WHERE h.status <> 'returned'
ORDER BY h.opened_at
""")

_GET = text("""
SELECT h.id, h.conversation_id, h.reason, h.status, h.resume_state,
       h.opened_at, h.claimed_at, h.claimed_by, h.returned_at,
       c.channel, c.sender_ref, c.contact_id, c.last_inbound_at
FROM handoffs h JOIN conversations c ON c.id = h.conversation_id
WHERE h.id = :id
""")


@dataclass(frozen=True)
class Handoff:
    id: uuid.UUID
    conversation_id: uuid.UUID
    reason: str
    status: str
    resume_state: dict[str, Any]
    opened_at: datetime
    claimed_at: datetime | None = None
    claimed_by: str | None = None
    returned_at: datetime | None = None
    #: Joined from the conversation, because every one of these is something
    #: the inbox has to show or send to and none of them is worth a second
    #: round trip.
    channel: str | None = None
    sender_ref: str | None = None
    contact_id: uuid.UUID | None = None
    last_inbound_at: datetime | None = None


@dataclass(frozen=True)
class Message:
    id: uuid.UUID
    conversation_id: uuid.UUID
    channel: str
    author: str
    body: str | None
    created_at: datetime
    #: Total order within a tenant. `created_at` alone is not enough: it
    #: defaults to `clock_timestamp()` but two appends can still land in the
    #: same microsecond, and a thread with an ambiguous order is one an agent
    #: can read backwards.
    seq: int


def _handoff(row: Any) -> Handoff:
    fields = {
        name: getattr(row, name, None)
        for name in (
            "id", "conversation_id", "reason", "status", "resume_state",
            "opened_at", "claimed_at", "claimed_by", "returned_at",
            "channel", "sender_ref", "contact_id", "last_inbound_at",
        )
    }
    return Handoff(**fields)


class HandoffStore:
    def __init__(self, *, session: AsyncSession) -> None:
        self._session = session

    async def open(
        self, *, conversation_id: uuid.UUID, reason: str, resume_state: dict[str, Any]
    ) -> Handoff:
        """Record that a human is needed, and freeze the cursor to come back to.

        `tenant_id` comes from the conversation row rather than from the
        caller: the caller could pass a different one, and under RLS the
        `WITH CHECK` would reject it — but only at write time, and with an
        error that reads as a database problem rather than as a bug in the
        call.
        """
        import json

        row = (
            await self._session.execute(
                _OPEN,
                {
                    "id": uuid.uuid4(),
                    "conversation_id": conversation_id,
                    "reason": reason,
                    "resume_state": json.dumps(resume_state),
                },
            )
        ).one()
        return _handoff(row)

    async def claim(self, *, handoff_id: uuid.UUID, agent_id: str) -> Handoff:
        row = (
            await self._session.execute(_CLAIM, {"id": handoff_id, "agent_id": agent_id})
        ).one()
        return _handoff(row)

    async def get(self, *, handoff_id: uuid.UUID) -> Handoff | None:
        row = (await self._session.execute(_GET, {"id": handoff_id})).one_or_none()
        return _handoff(row) if row is not None else None

    async def open_handoffs(self) -> list[Handoff]:
        rows = (await self._session.execute(_LIST_OPEN)).all()
        return [_handoff(row) for row in rows]

    async def return_to_bot(self, *, handoff_id: uuid.UUID) -> Handoff:
        """Give the conversation back, at the node it was taken from.

        The cursor is written back verbatim. Anything the agent said sits in
        `messages` and is deliberately not part of it — a human turn is not a
        script turn, and treating it as one would advance a state machine on
        input it was never designed to read.
        """
        import json

        row = (await self._session.execute(_RETURN, {"id": handoff_id})).one()
        handoff = _handoff(row)
        await self._session.execute(
            _RESTORE_CURSOR,
            {
                "state": json.dumps(handoff.resume_state),
                "conversation_id": handoff.conversation_id,
            },
        )
        return handoff


#: Upsert on (tenant_id, contact_ref). `DO UPDATE` rather than `DO NOTHING`
#: because `RETURNING` yields no row for a conflict that did nothing, and a
#: second round trip to fetch the id would be a race with a concurrent turn
#: for the same customer.
_RESOLVE_CONTACT = text("""
INSERT INTO contacts (id, tenant_id, contact_ref, display_name)
VALUES (
  :id,
  nullif(current_setting('moc.tenant_id', true), '')::uuid,
  :contact_ref,
  :display_name
)
ON CONFLICT (tenant_id, contact_ref) DO UPDATE
  SET display_name = coalesce(contacts.display_name, EXCLUDED.display_name)
RETURNING id
""")


class ContactStore:
    """The person behind an address.

    One contact per `contact_ref`, which for now is the channel address the
    customer wrote from. That means a customer who uses WhatsApp and Instagram
    starts as two contacts, and the schema is what lets them later become one.

    Deliberately not merged automatically: knowing that a phone number and an
    Instagram handle are the same person requires evidence this pipeline does
    not have, and guessing would join two strangers' conversations into one
    thread and show each of them the other's messages. Merging is an operator
    action, and the `contact_id` on `conversations` is where it lands.
    """

    def __init__(self, *, session: AsyncSession) -> None:
        self._session = session

    async def resolve(
        self, *, contact_ref: str, display_name: str | None = None
    ) -> uuid.UUID:
        return (
            await self._session.execute(
                _RESOLVE_CONTACT,
                {
                    "id": uuid.uuid4(),
                    "contact_ref": contact_ref,
                    "display_name": display_name,
                },
            )
        ).scalar_one()


_APPEND = text("""
INSERT INTO messages (id, tenant_id, conversation_id, channel, author, body)
SELECT :id, c.tenant_id, c.id, :channel, :author, :body
FROM conversations c WHERE c.id = :conversation_id
RETURNING id, conversation_id, channel, author, body, created_at, seq
""")

#: Every conversation this contact holds, as one thread in insertion order.
#: The join through `contact_id` is what makes WhatsApp and Instagram one
#: history instead of two.
_HISTORY = text("""
SELECT m.id, m.conversation_id, m.channel, m.author, m.body, m.created_at, m.seq
FROM messages m JOIN conversations c ON c.id = m.conversation_id
WHERE c.contact_id = :contact_id
ORDER BY m.seq
""")

_UNPROCESSED = text("""
SELECT m.id, m.conversation_id, m.channel, m.author, m.body, m.created_at, m.seq
FROM messages m
WHERE m.conversation_id = :conversation_id
  AND m.author = 'customer'
  AND m.created_at > coalesce(
    (SELECT max(h.returned_at) FROM handoffs h
     WHERE h.conversation_id = m.conversation_id), '-infinity'::timestamptz)
ORDER BY m.created_at
""")


class MessageLog:
    """The thread an agent reads, and the bot's view of what is still unanswered."""

    def __init__(self, *, session: AsyncSession) -> None:
        self._session = session

    async def append(
        self,
        *,
        conversation_id: uuid.UUID,
        channel: str,
        author: str,
        body: str | None,
    ) -> Message:
        if author not in (CUSTOMER, BOT, AGENT):
            raise ValueError(f"unknown author {author!r}")
        row = (
            await self._session.execute(
                _APPEND,
                {
                    "id": uuid.uuid4(),
                    "conversation_id": conversation_id,
                    "channel": channel,
                    "author": author,
                    "body": body,
                },
            )
        ).one()
        return Message(**row._mapping)

    async def history_for_contact(self, *, contact_id: uuid.UUID) -> list[Message]:
        rows = (await self._session.execute(_HISTORY, {"contact_id": contact_id})).all()
        return [Message(**row._mapping) for row in rows]

    async def unprocessed_inbound(self, *, conversation_id: uuid.UUID) -> list[Message]:
        """Customer messages since the conversation came back from a human.

        Only `customer`. An agent's message and the bot's own replies are in
        the thread for reading, never for answering — replaying either would
        have the bot respond to itself or to the agent.
        """
        rows = (
            await self._session.execute(
                _UNPROCESSED, {"conversation_id": conversation_id}
            )
        ).all()
        return [Message(**row._mapping) for row in rows]


__all__ = [
    "AGENT",
    "BOT",
    "CUSTOMER",
    "ContactStore",
    "Handoff",
    "HandoffStore",
    "Message",
    "MessageLog",
]
