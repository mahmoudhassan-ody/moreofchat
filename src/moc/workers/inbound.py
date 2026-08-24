"""The agent worker — design §3.

Consumes the inbound stream, runs one turn, and hands the reply to the outbound
stream. This is where the webhook's promise is kept: the handler returned 200
in milliseconds because this process does the slow part.

Everything the turn needs that the webhook deliberately did not touch happens
here — the tenant session, the conversation row, the ledger writes — inside one
transaction, so a turn that fails leaves no half-written state and its stream
entry stays pending for another attempt.

The reply goes onto a second stream rather than being sent inline. Sending is a
vendor call with its own rate limits and its own failure modes; doing it here
would mean a Twilio 429 fails a turn that was otherwise complete, and the retry
would re-run the model.
"""

from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from moc.agent.conversations import ConversationStore
from moc.agent.handoff import BOT, CUSTOMER, ContactStore, HandoffStore, MessageLog
from moc.agent.script_engine import ScriptEngine
from moc.agent.scripts import ScriptResolver
from moc.channels.base import InboundMessage, OutboundJob
from moc.channels.valkey import decode
from moc.tenancy.context import tenant_session
from moc.workers.streams import TerminalFailure, consumer_from_config

_CONSUMER = "inbound-1"
_EDUCATION = "education"


class TurnRunner(Protocol):
    """What this worker needs from an orchestrator, and nothing more."""

    async def handle(self, **kwargs: Any) -> Any: ...


class RetrieverFactory(Protocol):
    """(tenant, vertical) -> the retriever that reads *their* corpus.

    A factory rather than an instance, because `FusionRetriever` is built with
    a tenant id and a vertical. One held for the life of the process answers
    every tenant from whichever corpus it was started with.
    """

    async def for_tenant(self, *, tenant_id: Any, vertical: str) -> Any: ...


class WrongVertical(TerminalFailure):
    """This worker has no turn handler for the tenant's vertical.

    Terminal on the first attempt: a retry will not change what the tenant
    sells. Dead-lettered rather than answered, because the education
    orchestrator will run its own script against a broker's customer perfectly
    happily and produce a fluent reply about credit-hour fees — which is the
    worst available outcome, being indistinguishable from working.
    """


class InboundWorker:
    def __init__(
        self,
        *,
        client: Any,
        engine: AsyncEngine,
        orchestrator: TurnRunner,
        script: str,
        config: dict[str, Any],
        scripts: Any = None,
        retrievers: RetrieverFactory | None = None,
        vertical: str = _EDUCATION,
        consumer: str = _CONSUMER,
    ) -> None:
        self._client = client
        self._engine = engine
        self._orchestrator = orchestrator
        self._script = script
        self._engine_for_script = ScriptEngine.from_config(script)
        #: Optional so every existing caller keeps working, and passed by the
        #: production wiring. Without it the worker runs the config file for
        #: every tenant, which is what it did before Task 33's publish had
        #: anything reading it.
        self._scripts = (
            ScriptResolver(store=scripts, fallback=script) if scripts is not None else None
        )
        #: Per turn, from the tenant the message belongs to. Optional so the
        #: existing single-tenant tests keep working; the composition root
        #: always supplies one, and a test asserts it.
        self._retrievers = retrievers
        #: What this process can actually answer. Real estate is a different
        #: agent with a different result type, and there is no path from here
        #: to it — so a broker's message is refused rather than run through the
        #: education script.
        self._vertical = vertical
        self._outbound_stream = config["outbound"]["stream"]
        self._consumer = consumer_from_config(
            client=client, section=config["inbound"], consumer=consumer
        )

    async def run_once(self, *, block: bool = False) -> int:
        return await self._consumer.run_once(self._handle, block=block)

    async def _handle(self, payload: str) -> None:
        message = decode(payload)
        # One transaction for the whole turn: the conversation row, every
        # ledger row, and the state update commit together or not at all. A
        # turn that half-committed would bill a customer for a reply whose
        # state never advanced, and the retry would bill them again.
        async with tenant_session(self._engine, message.tenant_id) as session:
            # Read before anything else in the turn. A vertical this process
            # cannot serve must cost no model call and write no conversation
            # row — and must not reach the orchestrator, which would answer it.
            vertical = (
                await session.execute(text("SELECT vertical FROM tenants"))
            ).scalar_one_or_none()
            if vertical != self._vertical:
                raise WrongVertical(
                    f"tenant {message.tenant_id} is {vertical!r} and this worker "
                    f"serves {self._vertical!r}. Answering anyway would run the "
                    f"{self._vertical} script against their customer and produce a "
                    "fluent reply about the wrong business."
                )

            retriever = None
            if self._retrievers is not None:
                # Which corpus this turn reads. See the orchestrator: a
                # process-lifetime retriever is a cross-tenant read that
                # arrives as a correct-looking answer.
                retriever = await self._retrievers.for_tenant(
                    tenant_id=message.tenant_id, vertical=vertical
                )

            # The engine a NEW conversation would start on — the tenant's
            # published script, or the config file if they have published
            # none. Used by `load` only to build a fresh state.
            engine = self._engine_for_script
            if self._scripts is not None:
                engine = await self._scripts.current(tenant_id=message.tenant_id)

            store = ConversationStore(session=session, engine=engine)
            state = await store.load(
                channel=str(message.channel), sender_ref=message.sender_ref
            )

            # An in-flight conversation runs the version it started on. Without
            # this, publishing a new version does not merely fail to take
            # effect — `_require_pinned_version` raises, and every customer
            # mid-conversation gets an error the moment a script is published.
            if self._scripts is not None and state.script_version != engine.version:
                pinned = await self._scripts.at(
                    tenant_id=message.tenant_id,
                    script_id=state.script_id,
                    version=state.script_version,
                )
                if pinned is not None:
                    engine = pinned
                    store = ConversationStore(session=session, engine=engine)

            # **A human has the conversation: the bot does not speak.** Checked
            # before the turn rather than after, so a suspended message costs
            # no provider call — and asked of the conversation rather than the
            # state, because a takeover is a fact about who is answering and
            # not a node in the script.
            #
            # The customer's words are still recorded. The agent has to see
            # what was said while they were reading, and a message dropped
            # because a human was attached is a message nobody ever answers.
            existing = await store.find(
                channel=str(message.channel), sender_ref=message.sender_ref
            )
            if existing is not None and await HandoffStore(
                session=session
            ).live_for_conversation(conversation_id=existing):
                await MessageLog(session=session).append(
                    conversation_id=existing,
                    channel=str(message.channel),
                    author=CUSTOMER,
                    body=message.text,
                )
                await store.touch(conversation_id=existing, last_inbound_at=message.received_at)
                await session.commit()
                return

            result = await self._orchestrator.handle(
                session=session,
                state=state,
                text=message.text or "",
                channel=str(message.channel),
                # The engine resolved above, not the one the orchestrator was
                # built with. Which script runs is a property of the turn:
                # tenant scripts are versioned and a conversation is pinned to
                # the version it started on, so a process-lifetime engine would
                # raise on every in-flight conversation the moment a tenant
                # published.
                engine=engine,
                retriever=retriever,
            )
            # The person behind the address (§9). Resolved before the upsert
            # so a first message arrives with its contact already attached —
            # a conversation that exists for a while without one is a thread
            # the inbox cannot join, and backfilling it later is guesswork.
            contact_id = await ContactStore(session=session).resolve(
                contact_ref=message.sender_ref
            )
            conversation_id = await store.save(
                channel=str(message.channel),
                sender_ref=message.sender_ref,
                channel_account_id=message.channel_account_id,
                last_inbound_at=message.received_at,
                state=result.state,
                contact_id=contact_id,
            )
            # Both sides, inside the turn's transaction. A thread that can
            # disagree with the conversation state is one an agent reads while
            # the bot believes something else — and they diverge exactly when
            # a turn half-fails, which is when a human is most likely looking.
            log = MessageLog(session=session)
            await log.append(
                conversation_id=conversation_id,
                channel=str(message.channel),
                author=CUSTOMER,
                body=message.text,
            )
            if result.reply:
                await log.append(
                    conversation_id=conversation_id,
                    channel=str(message.channel),
                    author=BOT,
                    body=result.reply,
                    # Written with the words, in the same insert. A reply whose
                    # sources went missing shows a customer-visible figure with
                    # no evidence behind it, which is the one thing the source
                    # pane exists to rule out.
                    provenance=result.provenance,
                )
            await session.commit()

        await self._enqueue_reply(message, result.reply)

    async def _enqueue_reply(self, message: InboundMessage, reply: str) -> None:
        """Hand the words to the sender.

        Published *after* the commit, deliberately. Publishing first risks a
        reply going out for a turn whose state was rolled back — the customer
        would be answered from a conversation that, as far as the database is
        concerned, never advanced. The other ordering risks a committed turn
        whose reply is never sent, which the pending-entry retry recovers.
        """
        if not reply:
            return
        job = OutboundJob(
            tenant_id=str(message.tenant_id),
            channel=str(message.channel),
            to=message.sender_ref,
            text=reply,
            # A reply to a message we just received is always inside the
            # 24-hour window (§6.2) — the inbound message is what opened it.
            last_inbound_at=message.received_at.isoformat(),
            # Email's threading, carried back out. §6.1 gives the inbound
            # message a thread reference and nothing read it, so every reply
            # started a new thread in the customer's client — one answer per
            # email, none of them attached to the question.
            thread_ref=message.thread_ref,
        )
        await self._client.xadd(self._outbound_stream, {"payload": job.to_json()})


__all__ = ["InboundWorker"]
