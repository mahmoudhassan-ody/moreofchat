"""One turn, start to finish — design §3.1, §7.3, §7.5, §19.3.

The script engine decides what kind of turn this is, retrieval supplies the
facts and the confidence, the router composes the words, `guards` checks them,
and the ledger records what it cost. This module is the ordering of those
steps and nothing else — every value it uses is config, and every decision it
makes is delegated.

Three orderings here are correctness, not style:

1. **Redaction is first, and the raw text is then discarded.** §7.3 requires it
   before the completion call *and* before the embedding call, and the
   embedding call is the one that gets missed: it happens earlier, it receives
   the customer's query verbatim, and it does not look like "sending the
   message to an LLM". Rather than remember to redact twice, this holds one
   `Redaction` and never rebinds the original string — a rule about ordering is
   only as good as the next person to edit the function, while having nothing
   to leak needs no vigilance.

2. **Confidence comes from retrieval, not from the extractor.** A model's
   report of its own certainty is not evidence that the knowledge base contains
   the fee (§7.5). The extractor supplies intent and slots; the fused retrieval
   score is what the gate reads.

3. **The grounding gate runs after composition and can discard the reply.**
   This is the §19.3 invariant made operational: the model produced a fluent,
   correctly-registered Arabic sentence, and if a figure in it traces to
   nothing the sentence does not get sent. Not repaired, not hedged — replaced
   by a handoff, because a wrong fee is a commercial incident and a slow
   correct answer is not.

**No error ever reaches the customer** (§2.6). Every failure path — providers
down, empty completion, ungrounded figure — ends in a scripted reply from
`agent/replies`. The router raises when everything is unavailable precisely so
this layer, which knows the conversation, can decide what to say instead.

Retrieval and extraction are Protocols. The real implementations land in P1
(fusion) and with the Haiku prompts; keeping them as seams is what lets the
whole turn be tested without a network, and what stops this module from
growing opinions about Qdrant.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.guards import GroundingResult, Redaction, check_numeric_grounding, redact
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import Action, ConversationState, Decision, Register, TurnInput
from moc.config_store import load
from moc.llm.base import AllProvidersUnavailable, Completion, Message, Role, Task
from moc.llm.router import Router
from moc.tenancy.metering import UsageKind, record_usage

_REPLIES = "agent/replies"

# Keys in agent/replies. Named constants rather than inline strings so a
# renamed key is one edit and a missing one is a KeyError at the call site.
_HANDOFF = "handoff"
_CLARIFY = "clarify"
_LOW_CONFIDENCE = "low_confidence"
_GROUNDING_FAILED = "grounding_failed"
_PROVIDER_UNAVAILABLE = "provider_unavailable"


@dataclass(frozen=True)
class Retrieval:
    """What the retrieval layer returns for one query.

    `confidence` is the fused score (§7.5) — the gate's input. `script_constants`
    are figures the script itself is entitled to state, which is the other half
    of grounding: a fee held in a script node is as legitimate a source as a
    retrieved chunk, and a calculator tool's output arrives the same way.
    """

    passages: Sequence[str] = ()
    confidence: float = 0.0
    script_constants: Sequence[float | str] = ()


class Retriever(Protocol):
    """P1 seam. `fusion.py` implements this over Meilisearch + Qdrant."""

    async def search(self, *, query: str) -> Retrieval: ...


class Extractor(Protocol):
    """The Haiku slot-extraction call (§3.1), as a seam.

    Deliberately not implemented here. It is a prompt plus a JSON parse, both
    of which are versioned artefacts the eval suite grades; wiring a
    placeholder in now would make the first real version look like a change to
    the orchestrator.
    """

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput: ...


@dataclass(frozen=True)
class TurnResult:
    """Everything the caller needs to send the message and persist the thread.

    `grounding` is kept even when it passed: the eval harness reports what the
    gate saw, and a turn that passed with an empty source set is a different
    animal from one that matched three figures.
    """

    reply: str
    action: Action
    register: Register
    state: ConversationState
    degraded: bool = False
    redacted: tuple[str, ...] = ()
    passages: tuple[str, ...] = ()
    grounding: GroundingResult | None = None
    completions: list[Completion] = field(default_factory=list)
    #: Every provider was down or breaker-open. Distinct from `degraded`,
    #: which means the failover answered — that is a working turn on the
    #: fallback path. This one means no model answered at all.
    #:
    #: Explicit rather than inferred from `degraded and not completions`,
    #: because the eval runner has to tell an outage from a quality failure:
    #: folding an outage into the failure rate corrupts the baseline, and the
    #: corruption survives into every comparison made against it afterwards.
    provider_unavailable: bool = False


class Orchestrator:
    def __init__(
        self,
        *,
        engine: ScriptEngine,
        router: Router,
        retriever: Retriever,
        extractor: Extractor,
    ) -> None:
        self._engine = engine
        self._router = router
        self._retriever = retriever
        self._extractor = extractor

    async def handle(
        self,
        *,
        session: AsyncSession,
        state: ConversationState,
        text: str,
        channel: str,
    ) -> TurnResult:
        """Run one inbound message to a reply.

        Takes no tenant id. Attribution comes from the transaction's
        `moc.tenant_id`, so a caller that forgot to open a tenant session
        writes nothing rather than billing someone at random — and a caller
        cannot bill the wrong tenant if it is never holding an id to pass.
        """
        redaction = redact(text)
        # `text` is not read again below. Everything downstream takes
        # `redaction.text`, which is the point of §7.3.
        await record_usage(session, kind=UsageKind.message_in, channel=channel)

        turn = await self._extractor.extract(text=redaction.text, state=state)
        retrieval = await self._retriever.search(query=redaction.text)
        # The extractor's own confidence, if it reported one, is discarded here.
        turn = replace(turn, confidence=retrieval.confidence)

        decision = self._engine.advance(state, turn)
        result = await self._resolve(session, decision, retrieval, redaction, channel)

        await record_usage(session, kind=UsageKind.message_out, channel=channel)
        return result

    # ─────────────────────────── the turn's branches ───────────────────────────

    async def _resolve(
        self,
        session: AsyncSession,
        decision: Decision,
        retrieval: Retrieval,
        redaction: Redaction,
        channel: str,
    ) -> TurnResult:
        if decision.action is not Action.answer:
            return self._scripted(decision, redaction, retrieval, self._non_answer_key(decision))

        try:
            completion = await self._compose(session, decision, retrieval, channel)
        except AllProvidersUnavailable:
            # §2.6: the customer gets a sentence and a human, not an error. The
            # router raised rather than inventing text precisely so this
            # decision is made where the conversation is known.
            return self._scripted(
                decision,
                redaction,
                retrieval,
                _PROVIDER_UNAVAILABLE,
                degraded=True,
                provider_unavailable=True,
            )

        if not completion.text.strip():
            # Measured on claude-sonnet-5: a thinking budget consumed before any
            # text leaves stop_reason max_tokens and an empty body. Sending it
            # is sending silence, which a customer reads as being ignored.
            return self._scripted(
                decision, redaction, retrieval, _PROVIDER_UNAVAILABLE, degraded=True
            )

        grounding = check_numeric_grounding(
            completion.text, list(retrieval.passages), retrieval.script_constants
        )
        if not grounding.passed:
            # §19.3. The reply is discarded whole rather than edited: a sentence
            # containing one ungrounded figure is a sentence built around a fact
            # nobody can source, and stripping the number leaves a claim without
            # its evidence.
            return self._scripted(
                decision,
                redaction,
                retrieval,
                _GROUNDING_FAILED,
                degraded=completion.degraded,
                grounding=grounding,
                completions=[completion],
            )

        return TurnResult(
            reply=completion.text,
            action=decision.action,
            register=decision.register,
            state=decision.state,
            degraded=completion.degraded,
            redacted=redaction.found,
            passages=tuple(retrieval.passages),
            grounding=grounding,
            completions=[completion],
        )

    async def _compose(
        self,
        session: AsyncSession,
        decision: Decision,
        retrieval: Retrieval,
        channel: str,
    ) -> Completion:
        """Ask the model for words, having already decided the facts.

        Passages go in `cache_blocks`, not the message body: they are the
        stable prefix of the prompt and the volatile part is the customer's
        question, which is the ordering both providers' caches reward.
        """
        completion = await self._router.complete(
            task=Task.answer_composition,
            messages=[Message(role=Role.user, content=decision.reason or decision.node)],
            system=None,
            cache_blocks=list(retrieval.passages),
        )
        # Metered here rather than through the router's usage sink so the row
        # commits inside the caller's transaction — the router has no session
        # and no tenant, and a ledger write outside this transaction would
        # survive a turn that rolled back.
        await record_usage(
            session,
            kind=UsageKind.llm_call,
            channel=channel,
            model=completion.model,
            provider=completion.provider,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cached_tokens=completion.cached_tokens,
            degraded=completion.degraded,
        )
        return completion

    # ─────────────────────────── scripted replies ───────────────────────────

    @staticmethod
    def _non_answer_key(decision: Decision) -> str:
        """Which scripted reply a non-answer turn gets.

        The low-confidence case is separated from an ordinary clarification on
        purpose: they are both `clarify` to the engine, but to the customer one
        is "which faculty?" and the other is "I could not find this" — and
        conflating them produces a bot that asks the same question twice.
        """
        if decision.action is Action.handoff:
            return _HANDOFF
        if decision.reason.startswith("retrieval confidence"):
            return _LOW_CONFIDENCE
        return _CLARIFY

    def _scripted(
        self,
        decision: Decision,
        redaction: Redaction,
        retrieval: Retrieval,
        key: str,
        *,
        degraded: bool = False,
        grounding: GroundingResult | None = None,
        completions: list[Completion] | None = None,
        provider_unavailable: bool = False,
    ) -> TurnResult:
        """A reply nobody composed, for a turn the model must not answer.

        Failure keys resolve to `handoff`: an outage and an ungrounded figure
        both mean a human takes this turn, and reporting them as `answer` would
        let a case assert an answer that never happened.
        """
        failed = key in (_PROVIDER_UNAVAILABLE, _GROUNDING_FAILED)
        return TurnResult(
            reply=self._reply_text(key, decision),
            action=Action.handoff if failed else decision.action,
            register=decision.register,
            state=decision.state,
            degraded=degraded,
            redacted=redaction.found,
            passages=tuple(retrieval.passages),
            grounding=grounding,
            completions=completions or [],
            provider_unavailable=provider_unavailable,
        )

    @staticmethod
    def _reply_text(key: str, decision: Decision) -> str:
        document = load(_REPLIES)
        if key is _CLARIFY and decision.ask_for_slot:
            slot = document["ask_for_slot"].get(decision.ask_for_slot)
            if slot:
                return _in_register(slot, decision.register)
        return _in_register(document["replies"][key], decision.register)


def _in_register(entry: dict[str, Any], register: Register) -> str:
    """Pick the register's wording, falling back to Masri.

    The fallback is not a shrug. A key with no entry for the node's register is
    almost always an apology or an outage message, and those are conversation
    rather than regulation — Masri is the right register for them, not a
    degraded one.
    """
    return entry.get(str(register)) or entry[str(Register.masri)]


__all__ = ["Extractor", "Orchestrator", "Retrieval", "Retriever", "TurnResult"]
