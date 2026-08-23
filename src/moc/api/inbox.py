"""The agent inbox — design §9, P1 Task 22.

Handed-off conversations, the thread behind each one, the agent's reply, and
return-to-bot.

**One send path.** The single decision this module exists to make correctly:
an agent's reply is published as an ordinary `OutboundJob` onto the same
stream the orchestrator publishes to, and the existing sender worker delivers
it. This module never touches a channel adapter, and a test asserts it never
imports one.

Two paths would mean two token buckets, and two buckets each allowing the full
rate is the same as having no limit — the limit exists because Meta's is real.
It would also mean two places to get §6.2's 24-hour service window right, and
the agent path is the one where getting it wrong is most likely: an agent
replies to a conversation that has been sitting in a queue, which is exactly
when the window has expired.

**Tenancy is RLS, not a WHERE clause.** Every request opens a session with the
authenticated agent's tenant set, and no query here names `tenant_id`. A
handoff belonging to another tenant is not filtered out — it does not exist,
which is why fetching one by id returns 404 rather than 403. 403 would confirm
the id is real.

**Authentication is a seam.** `AgentAuthenticator` is a Protocol with no
default: there is no agent login yet, and a header-derived tenant id would be
an authorization bypass wearing the shape of a feature. Wiring it is P2; what
this module guarantees is that whatever produces the principal is the only
thing that decides which tenant's data a request sees.

SSE rather than WebSocket (§9 notes): cheaper to operate, and agents do not
need presence yet.
"""

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from moc.agent.handoff import AGENT, HandoffStore, MessageLog
from moc.channels.base import OutboundJob
from moc.tenancy.context import tenant_session

_NOT_FOUND = 404


@dataclass(frozen=True)
class AgentPrincipal:
    """Who is asking, and therefore which tenant's data exists for them."""

    tenant_id: uuid.UUID
    agent_id: str


class OutboundPublisher(Protocol):
    """Hands a reply to the sender worker. Deliberately not a provider.

    The narrowest possible port: if this could send, someone would make it
    send, and the rate limit would then cover only the messages that still
    went the long way round.
    """

    async def publish(self, job: OutboundJob) -> None: ...


class InboxEvents(Protocol):
    """Fan-out for the SSE stream, scoped by tenant at the source."""

    async def publish(self, *, tenant_id: uuid.UUID, event: dict[str, Any]) -> None: ...

    def subscribe(self, *, tenant_id: uuid.UUID) -> AsyncIterator[dict[str, Any]]: ...


AgentAuthenticator = Callable[[Request], Awaitable[AgentPrincipal]]


def build_inbox(
    *,
    engine: Any,
    publisher: OutboundPublisher,
    events: InboxEvents,
    authenticate: AgentAuthenticator,
) -> FastAPI:
    """Assemble the inbox from its collaborators.

    Constructed rather than module-global for the same reason the webhook is:
    it makes visible, at a glance, that this app holds a publisher and not a
    provider.
    """
    app = FastAPI()

    @app.get("/inbox")
    async def list_inbox(request: Request) -> list[dict[str, Any]]:
        principal = await authenticate(request)
        async with tenant_session(engine, principal.tenant_id) as session:
            handoffs = await HandoffStore(session=session).open_handoffs()
        return [
            {
                "id": str(handoff.id),
                "conversation_id": str(handoff.conversation_id),
                "reason": handoff.reason,
                "status": handoff.status,
                "channel": handoff.channel,
                "sender_ref": handoff.sender_ref,
                "opened_at": handoff.opened_at.isoformat(),
                "claimed_by": handoff.claimed_by,
            }
            for handoff in handoffs
        ]

    @app.get("/inbox/{handoff_id}/thread")
    async def thread(handoff_id: uuid.UUID, request: Request) -> list[dict[str, Any]]:
        """Every channel this contact has used, as one conversation."""
        principal = await authenticate(request)
        async with tenant_session(engine, principal.tenant_id) as session:
            handoff = await _require(session, handoff_id)
            messages = await MessageLog(session=session).history_for_contact(
                contact_id=handoff.contact_id
            )
        return [
            {
                "channel": message.channel,
                "author": message.author,
                "body": message.body,
                "created_at": message.created_at.isoformat(),
                # Where each figure came from, and which gates passed. Null on
                # a customer turn and on a scripted reply — the console shows
                # a source pane only where there is one.
                "provenance": message.provenance,
            }
            for message in messages
        ]

    @app.post("/inbox/{handoff_id}/claim")
    async def claim(handoff_id: uuid.UUID, request: Request) -> dict[str, Any]:
        principal = await authenticate(request)
        async with tenant_session(engine, principal.tenant_id) as session:
            await _require(session, handoff_id)
            handoff = await HandoffStore(session=session).claim(
                handoff_id=handoff_id, agent_id=principal.agent_id
            )
            await session.commit()
        return {"id": str(handoff.id), "claimed_by": handoff.claimed_by}

    @app.post("/inbox/{handoff_id}/reply")
    async def reply(
        handoff_id: uuid.UUID, request: Request, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Record the agent's words, then hand them to the sender.

        Recorded before publishing, and in the same transaction: a reply the
        customer received but the thread does not show is a conversation no
        agent can pick up correctly, and it is the version a dispute turns on.
        """
        principal = await authenticate(request)
        body = payload.get("text")
        async with tenant_session(engine, principal.tenant_id) as session:
            handoff = await _require(session, handoff_id)
            await MessageLog(session=session).append(
                conversation_id=handoff.conversation_id,
                channel=handoff.channel or "",
                author=AGENT,
                body=body,
            )
            await session.commit()

        # The same job the orchestrator publishes, onto the same stream. The
        # window travels with it rather than being resolved here, so the
        # sender applies one rule to every outbound message regardless of who
        # wrote it (§6.2).
        await publisher.publish(
            OutboundJob(
                tenant_id=str(principal.tenant_id),
                channel=handoff.channel or "",
                to=handoff.sender_ref or "",
                text=body,
                last_inbound_at=(
                    handoff.last_inbound_at.isoformat() if handoff.last_inbound_at else None
                ),
            )
        )
        await events.publish(
            tenant_id=principal.tenant_id,
            event={"type": "replied", "handoff_id": str(handoff_id)},
        )
        return {"status": "queued"}

    @app.post("/inbox/{handoff_id}/return")
    async def return_to_bot(handoff_id: uuid.UUID, request: Request) -> dict[str, Any]:
        principal = await authenticate(request)
        async with tenant_session(engine, principal.tenant_id) as session:
            await _require(session, handoff_id)
            handoff = await HandoffStore(session=session).return_to_bot(
                handoff_id=handoff_id
            )
            await session.commit()
        await events.publish(
            tenant_id=principal.tenant_id,
            event={"type": "returned", "handoff_id": str(handoff_id)},
        )
        return {"status": "returned", "node": handoff.resume_state.get("node")}

    @app.get("/inbox/events")
    async def stream(request: Request) -> StreamingResponse:
        """SSE, scoped at the subscription rather than filtered afterwards.

        One process serves every tenant's agents. A stream that leaked would
        show one company the volume, timing and content of another's
        escalations — which is competitive intelligence, not just a bug.
        """
        principal = await authenticate(request)

        async def emit() -> AsyncIterator[bytes]:
            async for event in events.subscribe(tenant_id=principal.tenant_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

        return StreamingResponse(emit(), media_type="text/event-stream")

    async def _require(session: Any, handoff_id: uuid.UUID) -> Any:
        """404 for another tenant's handoff, not 403.

        Under RLS the row does not exist for this session, and that is the
        honest status. 403 would confirm the id is real, which is the one bit
        an enumerating caller wants.
        """
        handoff = await HandoffStore(session=session).get(handoff_id=handoff_id)
        if handoff is None:
            raise HTTPException(status_code=_NOT_FOUND)
        return handoff

    return app


__all__ = [
    "AgentAuthenticator",
    "AgentPrincipal",
    "InboxEvents",
    "OutboundPublisher",
    "build_inbox",
]
