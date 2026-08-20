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
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.composition import render_composition
from moc.agent.guards import GroundingResult, Redaction, check_numeric_grounding, redact
from moc.agent.replies import Voice, refusal
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import Action, ConversationState, Decision, Register, TurnInput
from moc.arabic.script import reply_language
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
_REFUSE = "refuse"
_GROUNDING_FAILED = "grounding_failed"

#: Naming one missing slot of three is worse than naming none — it reads as
#: the whole answer, and the customer supplies one thing and waits.
_AT_LEAST_TWO = 2
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
    #: None when no arm supplied a calibrated score (§7.3 degraded, or a
    #: lexical-only deployment). Not zero — nobody looked with an instrument
    #: that reads, which is a different claim from "looked and found nothing".
    confidence: float | None = None
    script_constants: Sequence[float | str] = ()
    #: What each passage is *about*, in the corpus's own words. Only the
    #: fallback clarification reads this (edu-0009): a body does not say which
    #: question it answers, so without titles the only honest reply to an
    #: unroutable message was "what exactly do you need?".
    titles: Sequence[str] = ()


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
    #: What those passages were about. Carried so the harness can tell a
    #: clarification that had nothing to offer from one that had options and
    #: did not use them — the reply text says the same thing either way.
    titles: tuple[str, ...] = ()
    #: §3.1's figures the script itself may state. Carried alongside
    #: `passages` because together they are the full source set the delivered
    #: reply is graded against — without them a scripted fee reads as an
    #: orphan.
    script_constants: tuple[str, ...] = ()
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
        # The extractor's own confidence, if it reported one, is discarded
        # here. `grounded` is the §7.5 gate's real input: a script constant
        # counts, because §3.1 lets the script state figures the corpus does
        # not carry.
        turn = replace(
            turn,
            confidence=retrieval.confidence,
            grounded=bool(retrieval.passages or retrieval.script_constants),
        )

        decision = self._engine.advance(state, turn)
        # The model's reading wins; the heuristic is the fallback. Franco with
        # no digit substitution — `fe manh fe kantara?` — defeats the pattern
        # rule, and the extractor has already read the sentence on a model
        # that handles Egyptian Arabic. The fallback still matters: it is the
        # only signal on a turn whose extraction failed.
        lang = turn.language or reply_language(redaction.text)
        result = await self._resolve(
            session, decision, retrieval, redaction, channel, lang
        )

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
        lang: str | None,
    ) -> TurnResult:
        if decision.action is not Action.answer:
            return self._scripted(
                decision, redaction, retrieval, self._non_answer_key(decision), lang=lang
            )

        try:
            completion = await self._compose(
                session, decision, retrieval, channel, redaction.text, lang
            )
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
            titles=tuple(retrieval.titles),
            script_constants=tuple(str(c) for c in retrieval.script_constants),
            grounding=grounding,
            completions=[completion],
        )

    async def _compose(
        self,
        session: AsyncSession,
        decision: Decision,
        retrieval: Retrieval,
        channel: str,
        message: str,
        lang: str | None,
    ) -> Completion:
        """Ask the model for words, having already decided the facts.

        Passages go in `cache_blocks`, not the message body: they are the
        stable prefix of the prompt and the volatile part is the customer's
        question, which is the ordering both providers' caches reward.

        The customer's question travels too, which it did not: this used to
        send `system=None` and `decision.reason or decision.node` as the whole
        body, so the model saw the retrieved passages and the string "fees".
        It answered the retrieval — in the retrieval's language, formatted as
        a document — and four Arabic education cases came back as English
        markdown because of it.
        """
        completion = await self._router.complete(
            task=Task.answer_composition,
            messages=[Message(role=Role.user, content=message)],
            system=render_composition(
                message=message,
                register=decision.register,
                channel=channel,
                passages=retrieval.passages,
                lang=lang,
                # Where this script sends a turn it cannot answer. edu-0001's
                # reply was truthful, grounded and a dead end.
                referral=self._engine.referral(lang),
            ),
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
        if decision.action is Action.refuse:
            # edu-0017. This branch did not exist, so a `refuse` node emitted
            # the clarification text — "could you tell me more about what you
            # need?" to a question asked perfectly clearly. The real-estate
            # agent's equivalent has handled refuse since it was written.
            return _REFUSE
        if decision.gate_closed:
            return _LOW_CONFIDENCE
        return _CLARIFY

    def _scripted(
        self,
        decision: Decision,
        redaction: Redaction,
        retrieval: Retrieval,
        key: str,
        *,
        lang: str | None = None,
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
            reply=self._reply_text(
                key, decision, redaction.text, lang, titles=tuple(retrieval.titles)
            ),
            action=Action.handoff if failed else decision.action,
            register=decision.register,
            state=decision.state,
            degraded=degraded,
            redacted=redaction.found,
            passages=tuple(retrieval.passages),
            titles=tuple(retrieval.titles),
            script_constants=tuple(str(c) for c in retrieval.script_constants),
            grounding=grounding,
            completions=completions or [],
            provider_unavailable=provider_unavailable,
        )

    @staticmethod
    def _reply_text(
        key: str,
        decision: Decision,
        message: str,
        lang: str | None = None,
        *,
        titles: tuple[str, ...] = (),
    ) -> str:
        # Register is the node's policy; language mirrors the customer. Both,
        # because passing only the register is F6: it renders cleanly and
        # answers an English question in Arabic.
        voice = Voice(decision.register, lang or reply_language(message))
        document = load(_REPLIES)
        if key is _REFUSE:
            return refusal(document["replies"], decision.node, voice)
        if key is _CLARIFY:
            # Named slots first. A node with missing slots knows exactly what
            # it needs, and offering a menu instead would replace an
            # answerable question with a browse.
            asked = _ask_for(document, decision, voice) or _offer(document, titles, voice)
            if asked:
                return asked
        return voice.say(document["replies"][key])


def _ask_for(document: dict, decision: Decision, voice: Voice) -> str | None:
    """Name what is missing, or say nothing and let the generic reply stand.

    Every slot the engine knows about, not the one the script happened to
    name. `admission_thresholds` needs a branch, a certificate and a faculty;
    it asked for one, by a key with no entry, so every such turn fell through
    to "what exactly do you need?" — which the judge failed four times in one
    run, with the same sentence of reasoning each time.

    One missing slot keeps its own question: a list of one reads worse than
    the sentence written for it.
    """
    missing = decision.missing_slots or (
        (decision.ask_for_slot,) if decision.ask_for_slot else ()
    )
    if len(missing) == 1:
        entry = document["ask_for_slot"].get(missing[0])
        return voice.say(entry) if entry else None

    plural = document["ask_for_slots"]
    nouns = [plural["nouns"][slot] for slot in missing if slot in plural["nouns"]]
    if len(nouns) < _AT_LEAST_TWO:
        return None
    joined = voice.say(plural["join"]).join(voice.say(noun) for noun in nouns)
    return voice.say(plural["template"]).replace("{items}", joined)


def _offer(document: dict, titles: tuple[str, ...], voice: Voice) -> str | None:
    """Name the meanings the question might have had, or say nothing.

    edu-0009. The fallback node has no missing slots — the message routed
    nowhere, which is a different problem from an incomplete question — so
    `_ask_for` cannot reach it and the generic "ممكن توضّحلي أكتر" was all it
    could say, to a customer who had already asked clearly.

    The options are the retrieved titles, unedited and in whatever language the
    tenant wrote them; only the sentence around them mirrors the customer. That
    keeps the clarification grounded in the same sense a reply is — an option
    the KB cannot answer is never offered, because it was never retrieved — and
    leaves no per-tenant list to drift out of date.
    """
    options = document["clarify_options"]
    offered = list(titles)[: options["max"]]
    if len(offered) < options["min"]:
        # A list of one is a guess with a question mark on it, not a choice.
        return None
    bullet = options["bullet"]
    return voice.say(options["template"]).replace(
        "{items}", "\n".join(f"{bullet}{title}" for title in offered)
    )


__all__ = ["Extractor", "Orchestrator", "Retrieval", "Retriever", "TurnResult"]
