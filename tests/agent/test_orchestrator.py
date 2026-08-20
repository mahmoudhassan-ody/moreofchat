"""The turn, end to end — design §3.1, §7.3, §7.5, §19.3.

Script engine decides, retrieval grounds, the router composes, guards check,
the ledger records. These tests drive that whole path with fakes at both
edges: no network, no vendor, and every collaborator's arguments recorded, so
"the raw message reached the embedding call" is an assertion rather than a
hope.

The load-bearing one is `test_the_grounding_gate_is_load_bearing`, which
neutralizes the gate and proves the orphan figure would otherwise ship. A gate
nobody has watched fail is a gate nobody knows is wired up.
"""

import json
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy import text as sql

from moc.agent.guards import check_numeric_grounding
from moc.agent.orchestrator import Orchestrator, Retrieval, TurnResult
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import Action, ConversationState, Register, TurnInput
from moc.config_store import load
from moc.llm.base import ProviderUnavailable, Task
from moc.llm.fake import FakeProvider
from moc.llm.router import Router
from moc.tenancy.context import tenant_session

SCRIPT = "scripts/education/fees"
CHANNEL = "whatsapp"

# One retrieved passage and one grounded answer built from it. 2026 is a year,
# which the extractor drops — a figure that qualifies a fee is not a fee.
PASSAGE = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه للعام الدراسي 2026"
GROUNDED_REPLY = "رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه."
ORPHAN_REPLY = "رسوم الساعة المعتمدة لكلية الهندسة 1750 جنيه."

NATIONAL_ID = "٢٩٨٠١٢٣٤٥٦٧٨٩٠"


class FakeExtractor:
    """Stands in for the Haiku slot-extraction call (§3.1).

    Records the text it was given, which is how the redaction tests prove the
    raw message never got this far.
    """

    def __init__(self, turn: TurnInput) -> None:
        self.turn = turn
        self.seen: list[str] = []

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput:
        self.seen.append(text)
        return self.turn


class FakeRetriever:
    """Retrieval with a real embedding call in it.

    The embedding is not decoration. §7.3's whole point is that this call sits
    *earlier* than composition and receives the customer's query text, so a
    test that stubbed it out would prove nothing about the failure it guards.
    """

    def __init__(self, router: Router, retrieval: Retrieval) -> None:
        self._router = router
        self._retrieval = retrieval
        self.seen: list[str] = []

    async def search(self, *, query: str) -> Retrieval:
        self.seen.append(query)
        await self._router.embed(texts=[query])
        return self._retrieval


def build(
    *,
    reply: str = GROUNDED_REPLY,
    confidence: float = 0.9,
    intent: str | None = "fees",
    slots: dict | None = None,
    passages: tuple[str, ...] = (PASSAGE,),
    titles: tuple[str, ...] = (),
    constants: tuple[str, ...] = (),
    audit: dict | None = None,
    fail_with: Exception | None = None,
    fail_primary: Exception | None = None,
) -> tuple[Orchestrator, FakeProvider, FakeProvider, FakeExtractor, FakeRetriever]:
    # §19.3's claim-level audit is a second completion in the same turn, on a
    # different model. Keyed by model rather than mocked away, so the wiring —
    # which task, which material — is exercised rather than assumed.
    anthropic = FakeProvider(
        "anthropic",
        text=reply,
        text_by_model={AUDIT_MODEL: json.dumps(audit or {"figures": []})},
        fail_with=fail_with or fail_primary,
        input_tokens=120,
        output_tokens=40,
        cached_tokens=900,
    )
    # The outage under test is a completion outage. §7.3 makes a dead
    # embedding endpoint a different incident — search degrades to lexical
    # rather than the turn failing — and conflating them here would mean the
    # failover tests were really testing retrieval.
    openai = FakeProvider(
        "openai",
        text=reply,
        fail_with=fail_with,
        fail_kinds=("complete",),
        embedding_dimensions=1024,
    )
    router = Router(
        config=load("llm/routing"), providers={"anthropic": anthropic, "openai": openai}
    )
    extractor = FakeExtractor(
        TurnInput(intent=intent, slots={"faculty": "engineering"} if slots is None else slots)
    )
    retriever = FakeRetriever(
        router,
        Retrieval(
            passages=passages,
            titles=titles,
            confidence=confidence,
            script_constants=constants,
        ),
    )
    orchestrator = Orchestrator(
        engine=ScriptEngine.from_config(SCRIPT),
        router=router,
        retriever=retriever,
        extractor=extractor,
    )
    return orchestrator, anthropic, openai, extractor, retriever


@pytest_asyncio.fixture(loop_scope="session")
async def turn_session(session):
    """A session with a tenant set, rolled back afterwards.

    Every turn writes to the ledger, and the ledger is tenant-scoped — so a
    turn with no tenant context does not run at all. That is the intended
    behaviour, asserted directly in
    `test_a_turn_without_a_tenant_context_writes_nothing`; here it is simply a
    precondition, so these tests can be about the turn rather than about RLS.

    Built on the conftest `session` fixture, whose outer transaction rolls
    back — which is also what makes the transaction-local `set_config` hold
    for the whole test.
    """
    from moc.tenancy.models import Tenant

    tenant = Tenant(slug="turn-tenant", name="Turn", vertical="education")
    session.add(tenant)
    await session.flush()
    await session.execute(
        sql("SELECT set_config('moc.tenant_id', :t, true)"), {"t": str(tenant.id)}
    )
    return session


def start_state() -> ConversationState:
    return ScriptEngine.from_config(SCRIPT).start()


def _models_for(task: str) -> set[str]:
    """Every model a task can land on, primary and failover.

    Both, because a degraded turn runs on the other vendor and a helper that
    knew only the primary would report "the composition never happened" for
    exactly the turns failover exists to serve.
    """
    entry = load("llm/routing")["tasks"][task]
    return {
        entry[role]["model"] for role in ("primary", "failover") if entry.get(role)
    }


#: A turn now makes two completion calls on different models, and every test
#: that reached for "the completion" started reading the figure audit instead.
#: Selecting by task is what keeps each of them about what it says it is.
COMPOSITION_MODELS = _models_for("answer_composition")
AUDIT_MODELS = _models_for("figure_audit")
AUDIT_MODEL = load("llm/routing")["tasks"]["figure_audit"]["primary"]["model"]


def composition_calls(provider: FakeProvider) -> list[dict]:
    return [
        c
        for c in provider.calls
        if c["kind"] == "complete" and c["model"] in COMPOSITION_MODELS
    ]


def audit_calls(provider: FakeProvider) -> list[dict]:
    return [
        c for c in provider.calls if c["kind"] == "complete" and c["model"] in AUDIT_MODELS
    ]


def language_directive(provider: FakeProvider) -> str:
    """The LANGUAGE line of the last composition prompt, and only that.

    Asserting on the whole prompt is vacuous: its own explanation of the rule
    contains both language names — "an Arabic question about an English
    document gets an Arabic answer" — so `"Arabic" in system` is true however
    the turn resolved. Two tests passed under sabotage before this existed.
    """
    system = composition_calls(provider)[-1]["system"] or ""
    return next(
        line for line in system.splitlines() if line.startswith("Write the whole reply")
    )


# ─────────────────────────── the happy path ───────────────────────────


async def test_full_turn_produces_a_grounded_arabic_answer(turn_session):
    orchestrator, anthropic, *_ = build()
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )

    assert isinstance(result, TurnResult)
    assert result.action is Action.answer
    assert result.register is Register.msa
    assert result.reply == GROUNDED_REPLY
    assert result.grounding is not None and result.grounding.passed
    assert composition_calls(anthropic), "the answer must come from the model, not a template"


async def test_retrieved_passages_are_sent_as_cache_blocks(turn_session):
    """§2.6: passages are the stable prefix, so they belong in the cached block.

    Also the mechanism by which the model has the fee at all — a composition
    call without passages is a composition call inventing figures.
    """
    orchestrator, anthropic, *_ = build()
    await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )
    assert PASSAGE in composition_calls(anthropic)[0]["cache_blocks"]


# ─────────────────────────── §19.3: the grounding gate ───────────────────────────


async def test_a_figure_absent_from_retrieval_is_never_emitted(turn_session):
    """edu-0005, end to end: the KB has no fee for what was asked.

    F1, the failure that costs a tenant money. The model produced a fluent,
    confident, correctly-registered Arabic sentence containing a fee nobody
    published — and the only thing standing between that sentence and a
    student's phone is this gate.
    """
    orchestrator, *_ = build(reply=ORPHAN_REPLY)
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )

    assert "1750" not in result.reply
    assert result.action is Action.handoff
    assert result.grounding is not None and not result.grounding.passed
    assert result.grounding.orphan_numbers == [1750]


async def test_the_grounding_gate_is_load_bearing(turn_session, monkeypatch):
    """Sabotage, kept permanently rather than run once by hand.

    Neutralize the gate and the orphan figure ships. That is the acceptance
    criterion for this task, and as a test it also fails if someone later makes
    the orchestrator stop consulting the gate — which the test above, on its
    own, would not catch.
    """
    orchestrator, *_ = build(reply=ORPHAN_REPLY)
    monkeypatch.setattr(
        "moc.agent.orchestrator.check_numeric_grounding",
        lambda reply, passages, constants: check_numeric_grounding(reply, [reply], constants),
    )
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )
    assert "1750" in result.reply, (
        "with the gate neutralized the orphan figure must reach the customer — "
        "if it does not, this test is no longer proving the gate does the work"
    )


async def test_a_hedged_but_grounded_figure_is_also_withheld(turn_session):
    """§19.3's second gate. "Roughly 1400" turns a fixed fee into an opening
    position, which is how a tenant ends up honouring a number they never set."""
    orchestrator, *_ = build(reply="رسوم الساعة حوالي 1400 جنيه.")
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )
    assert result.action is Action.handoff
    assert result.grounding.hedged_numbers == [1400]


# ─────────────────────────── §7.5: the confidence gate ───────────────────────────


async def test_low_confidence_routes_to_handoff_not_composition(turn_session):
    """Below threshold the turn does not reach answer composition at all.

    The route is the script's fallback node, which §7.5 names alongside handoff
    — the engine keeps the conversation rather than escalating on one weak
    retrieval. What is non-negotiable, and what this asserts, is that no
    composition call happens: a model asked to answer without grounding will.
    """
    threshold = load("agent/defaults")["confidence_threshold"]
    orchestrator, anthropic, openai, *_ = build(confidence=threshold - 0.1)
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )

    assert result.action is not Action.answer
    assert composition_calls(anthropic) == []
    assert composition_calls(openai) == []
    assert result.reply, "a scripted reply, never an empty message"


# ─────────────────────────── §2.6: never an error to a customer ───────────────────────────


async def test_breaker_open_degrades_to_the_scripted_reply_not_an_error(turn_session):
    """Both providers down. A WhatsApp user gets a sentence, not a stack trace."""
    orchestrator, *_ = build(fail_with=ProviderUnavailable("provider down"))
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )

    assert result.reply
    assert result.action is Action.handoff
    assert result.degraded is True


async def test_an_empty_completion_is_a_failed_turn_not_an_empty_message(turn_session):
    """Measured on claude-sonnet-5: a thinking budget exhausted by reasoning
    returns stop_reason max_tokens and no text. Sending that is sending
    silence, and the customer reads silence as being ignored."""
    orchestrator, *_ = build(reply="   ")
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )
    assert result.reply.strip()
    assert result.action is Action.handoff


# ─────────────────────────── §7.3: redaction ───────────────────────────


async def test_redacts_before_the_embedding_call_not_only_the_completion_call(turn_session):
    """The call people miss, asserted against the call itself.

    The embedding provider is the first thing outside Egypt to see the turn.
    This walks the identifier all the way to `embed()` and asserts it is not
    there, rather than asserting that some redaction function was invoked.
    """
    orchestrator, anthropic, openai, extractor, retriever = build()
    await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text=f"الرقم القومي {NATIONAL_ID} وعايز اعرف الرسوم",
        channel=CHANNEL,
    )

    embeds = [c for c in openai.calls if c["kind"] == "embed"]
    assert embeds, "retrieval must have embedded the query, or this proves nothing"
    for call in embeds:
        assert NATIONAL_ID not in " ".join(call["texts"])

    for call in composition_calls(anthropic):
        assert NATIONAL_ID not in str(call["messages"])
    assert all(NATIONAL_ID not in seen for seen in extractor.seen)
    assert all(NATIONAL_ID not in seen for seen in retriever.seen)


async def test_the_turn_reports_what_was_redacted_without_repeating_it(turn_session):
    """§11.2: log redacted forms only. The audit trail is the label."""
    orchestrator, *_ = build()
    result = await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text=f"الرقم القومي {NATIONAL_ID}",
        channel=CHANNEL,
    )
    assert result.redacted == ("national_id",)


# ─────────────────────────── metering ───────────────────────────


async def test_every_turn_writes_a_usage_ledger_row(app_engine, two_tenants):
    tenant, _ = two_tenants
    orchestrator, *_ = build()
    async with tenant_session(app_engine, tenant.id) as s:
        await orchestrator.handle(
            session=s, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
        )
        await s.commit()
        kinds = (await s.execute(sql("SELECT kind FROM usage_ledger"))).scalars().all()

    assert "message_in" in kinds
    assert "message_out" in kinds
    assert "llm_call" in kinds


async def test_a_degraded_turn_still_meters_both_message_rows(app_engine, two_tenants):
    """The provider outage must not also be a billing outage.

    Inbound and outbound messages are billable whether or not a model answered
    them, and a turn that silently stops metering when providers fail is a turn
    the tenant is not charged for during exactly the incident they will ask
    about afterwards.
    """
    tenant, _ = two_tenants
    orchestrator, *_ = build(fail_with=ProviderUnavailable("down"))
    async with tenant_session(app_engine, tenant.id) as s:
        await orchestrator.handle(
            session=s, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
        )
        await s.commit()
        kinds = (await s.execute(sql("SELECT kind FROM usage_ledger"))).scalars().all()

    assert "message_in" in kinds
    assert "message_out" in kinds


async def test_a_failover_turn_is_logged_degraded_on_the_ledger(app_engine, two_tenants):
    """Week 2's exit criterion, and what the `degraded` column exists for.

    The primary is down, the failover answers, and the customer gets a real
    reply — so nothing about the turn looks unusual from outside. The ledger
    row is the only record that it came from the fallback path, which is what
    makes a later register or grounding regression attributable to the switch
    rather than blamed on a prompt change (§2.6).
    """
    tenant, _ = two_tenants
    orchestrator, anthropic, openai, *_ = build(
        fail_primary=ProviderUnavailable("primary down")
    )
    async with tenant_session(app_engine, tenant.id) as s:
        result = await orchestrator.handle(
            session=s, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
        )
        await s.commit()
        rows = (
            await s.execute(
                sql("SELECT provider, degraded FROM usage_ledger WHERE kind = 'llm_call'")
            )
        ).all()

    assert result.action is Action.answer, "the customer still gets an answer"
    assert result.reply == GROUNDED_REPLY
    assert result.degraded is True
    # Both provider calls the turn made — composition and the §19.3 audit —
    # and both on the failover, both flagged. A row per call, because "which
    # calls degraded" is the question the flag exists to answer, and one row
    # standing for two would hide a turn where only half failed over.
    assert rows == [("openai", True), ("openai", True)]
    assert composition_calls(anthropic), "the primary must have been tried first"
    assert composition_calls(openai)


async def test_a_primary_only_outage_still_grounds_the_answer(turn_session):
    """The gate does not relax because the fallback model wrote the reply.

    A degraded turn is the one most likely to produce an ungrounded figure —
    different model, different prompt-following — so it is the last place to
    trust the output more.
    """
    orchestrator, *_ = build(
        reply=ORPHAN_REPLY, fail_primary=ProviderUnavailable("primary down")
    )
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )
    assert "1750" not in result.reply
    assert result.action is Action.handoff


async def test_turn_is_tenant_scoped_end_to_end(app_engine, two_tenants):
    """The whole path under the app role, not just the DB layer.

    `app_engine` connects as moc_app, a non-owner — the table owner bypasses
    RLS, so a turn tested through the owner engine proves nothing about tenant
    isolation.
    """
    tenant_a, tenant_b = two_tenants
    orchestrator, *_ = build()

    async with tenant_session(app_engine, tenant_a.id) as s:
        await orchestrator.handle(
            session=s, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
        )
        await s.commit()

    async with tenant_session(app_engine, tenant_a.id) as s:
        mine = (await s.execute(sql("SELECT count(*) FROM usage_ledger"))).scalar_one()
    async with tenant_session(app_engine, tenant_b.id) as s:
        theirs = (await s.execute(sql("SELECT count(*) FROM usage_ledger"))).scalar_one()

    assert mine > 0
    assert theirs == 0, "tenant B can see tenant A's turn — RLS is not holding"


async def test_a_turn_without_a_tenant_context_writes_nothing(app_engine, two_tenants):
    """Fails closed. A turn that forgot to open a tenant session must not bill
    someone at random, and must not quietly succeed either."""
    from sqlalchemy.ext.asyncio import AsyncSession

    orchestrator, *_ = build()
    async with AsyncSession(app_engine) as s:
        with pytest.raises(Exception) as caught:
            await orchestrator.handle(
                session=s, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
            )
    assert "42501" in str(caught.value) or "policy" in str(caught.value).lower()


# ─────────────────────────── state ───────────────────────────


async def test_the_returned_state_is_the_one_to_persist(turn_session):
    """Design §5: the conversation moves to the node the engine chose."""
    orchestrator, *_ = build()
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )
    assert result.state.node == "fees"
    assert result.state.slots == {"faculty": "engineering"}
    assert result.state.script_version == load(SCRIPT)["version"]


async def test_composition_uses_the_configured_task_budget(turn_session):
    """No token ceiling literal in the orchestrator — §19."""
    orchestrator, anthropic, *_ = build()
    await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام رسوم الساعة؟", channel=CHANNEL
    )
    expected = load("llm/routing")["tasks"][Task.answer_composition]["max_tokens"]
    assert composition_calls(anthropic)[0]["max_tokens"] == expected


def test_conversation_state_uuid_helper_is_not_needed_here():
    """Guard against drift: the orchestrator takes no tenant id argument.

    Attribution comes from the transaction's `moc.tenant_id`, never from a
    caller-supplied value — a caller cannot bill the wrong tenant if it is not
    holding a tenant id to pass.
    """
    import inspect

    signature = inspect.signature(Orchestrator.handle)
    assert not any(
        parameter.annotation is UUID for parameter in signature.parameters.values()
    )


async def test_the_composition_call_carries_the_prompt_and_the_question(turn_session):
    """`_compose` sent `system=None` and the node name as the message body, so
    the model answered the retrieval rather than the customer. Both travel
    now: the prompt as the system block, the passages still as cache blocks.
    """
    orchestrator, anthropic, *_ = build()

    await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="رسوم كلية الهندسة كام؟",
        channel="whatsapp",
    )

    call = composition_calls(anthropic)[-1]
    system = call["system"] or ""
    assert system, "the composition prompt travels as the system block"
    assert "رسوم كلية الهندسة كام؟" in system, "and it carries the question"
    assert "Arabic" in language_directive(anthropic), "the language they wrote in"
    assert "emoji" in system.lower(), "and the channel's formatting rules"


async def test_an_english_question_is_composed_in_english(turn_session):
    """F6 in the composition path. The passages are Arabic here; the customer
    is not."""
    orchestrator, anthropic, *_ = build()

    await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="How much are the engineering fees?",
        channel="whatsapp",
    )
    assert "English" in language_directive(anthropic)


async def test_the_passages_still_travel_as_cache_blocks_not_in_the_prompt(
    turn_session,
):
    """The prompt is the volatile part now; the passages are still the stable
    prefix, which is the ordering both providers' caches reward."""
    orchestrator, anthropic, *_ = build()

    await orchestrator.handle(
        session=turn_session, state=start_state(), text="كام؟", channel="whatsapp"
    )
    call = composition_calls(anthropic)[-1]
    assert PASSAGE in call["cache_blocks"]
    assert PASSAGE not in (call["system"] or "")


# ────────────────── a clarification names what it needs ──────────────────


async def test_a_refuse_node_does_not_emit_the_clarify_text(turn_session):
    """edu-0017: `career_advice` is `action: refuse` and the customer got
    "could you tell me more about what you need?" — for a question they had
    asked perfectly clearly.

    `_non_answer_key` branched on handoff and on the confidence gate and fell
    through to clarify for everything else, so refuse had no branch at all.
    The real-estate agent's equivalent has handled it since it was written.
    """
    from moc.config_store import load

    orchestrator, *_ = build(intent="career_advice")
    result = await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="إيه أحسن كلية أدخلها عشان ألاقي شغل بسرعة؟",
        channel=CHANNEL,
    )

    assert result.action is Action.refuse
    assert (
        result.reply
        == load("agent/replies")["replies"]["refuse"]["career_advice"]["masri"]
    )


async def test_a_clarification_names_every_slot_it_is_missing(turn_session):
    """edu-0004, edu-0007 and edu-0008 all failed the judge on the same
    sentence: "طلب التوضيح عام، ولم يحدد أن المطلوب هو معرفة الفرع ونوع
    الشهادة" — a clarification that does not name the missing thing makes the
    customer guess twice.

    `admission_thresholds` needs three slots and asked for one, by a key with
    no entry, so every one of them fell through to the generic sentence.
    """
    orchestrator, *_ = build(intent="admission_thresholds", slots={"faculty": "pharmacy"})
    result = await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="الحد الأدنى للقبول في الصيدلة كام؟",
        channel=CHANNEL,
    )

    assert result.action is Action.clarify
    reply = result.reply
    assert "الفرع" in reply, "the branch is missing and must be named"
    assert "الشهادة" in reply, "so is the certificate type"
    assert "بالظبط" not in reply, "not the generic fallback"


async def test_one_missing_slot_keeps_its_own_question(turn_session):
    """A single missing slot has a better sentence than a list of one."""
    from moc.config_store import load

    orchestrator, *_ = build(intent="fees", slots={})
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="المصاريف كام؟", channel=CHANNEL
    )
    assert result.reply == load("agent/replies")["ask_for_slot"]["faculty"]["masri"]


async def test_a_clarification_with_no_known_slot_still_falls_back(turn_session):
    """The fallback node knows nothing about what is missing. The generic
    sentence is the honest reply there — and the fix for a case landing on it
    is a node, not better wording."""
    orchestrator, *_ = build(intent=None)
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="؟؟", channel=CHANNEL
    )
    assert result.action is Action.clarify
    assert result.reply


async def test_the_extractor_s_language_decides_the_reply_language(turn_session):
    """Option C: the model has read the message and says which language to
    answer in. The heuristic is the fallback, not the primary.

    `ayzeen nesta3lem` is franco that BOTH heuristics miss — one substituted
    digit is below the two-hit floor and neither token is a function word — so
    it is the sentence that separates the two options. If this assertion could
    be satisfied by the fallback it would not be testing option C at all.
    """
    from moc.arabic.script import reply_language

    text = "ayzeen nesta3lem"
    assert reply_language(text) == "en", (
        "the discriminator: the heuristic reads this as English, so a passing "
        "assertion below can only come from the extractor"
    )

    orchestrator, anthropic, _, extractor, _ = build()
    extractor.turn = TurnInput(
        intent="fees", slots={"faculty": "engineering"}, language="ar"
    )

    await orchestrator.handle(
        session=turn_session, state=start_state(), text=text, channel=CHANNEL
    )
    assert "Arabic" in language_directive(anthropic)


async def test_the_extractor_can_override_the_heuristic_in_either_direction(
    turn_session,
):
    """Not just "franco is Arabic". A customer writing Arabic script inside an
    otherwise English conversation is the other direction, and the model is
    the only thing that can see it."""
    orchestrator, anthropic, _, extractor, _ = build()
    extractor.turn = TurnInput(
        intent="fees", slots={"faculty": "engineering"}, language="en"
    )

    await orchestrator.handle(
        session=turn_session, state=start_state(), text="المصاريف كام؟", channel=CHANNEL
    )
    assert "English" in language_directive(anthropic)


async def test_without_an_extractor_language_the_heuristic_still_decides(turn_session):
    """The fallback path, which is also the only path when extraction failed
    and a scripted reply is going out."""
    orchestrator, anthropic, _, extractor, _ = build()
    extractor.turn = TurnInput(intent="fees", slots={"faculty": "engineering"})

    await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="3ayez a3raf el masareef",
        channel=CHANNEL,
    )
    assert "Arabic" in language_directive(anthropic)


async def test_the_fallback_is_load_bearing_when_extraction_reports_nothing(
    turn_session,
):
    """Option A on its own, with option C removed.

    A turn whose extraction failed still has to send something, and what it
    sends is a scripted reply in some language. `fe manh fe kantara?` carries
    no digit substitution, so the pattern rule alone would answer it in
    English — which is exactly what shipped.
    """
    from moc.agent.replies import Voice
    from moc.arabic.script import is_franco

    assert is_franco("fe manh fe kantara?"), "the function-word rule, alone"

    orchestrator, _, _, extractor, _ = build(intent=None)
    extractor.turn = TurnInput(intent=None, slots={})  # no language reported

    result = await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="fe manh fe kantara?",
        channel=CHANNEL,
    )
    assert result.action is Action.clarify
    assert result.reply == Voice(result.register, "ar").say(
        load("agent/replies")["replies"]["clarify"]
    )


# ─────────── edu-0009: the clarification that offers the options ───────────


async def test_the_fallback_clarification_offers_what_was_retrieved(turn_session):
    """edu-0009. "المواعيد إيه؟" could be branch hours, bus times or an
    application deadline, and the reply was "ممكن توضّحلي أكتر عايز تعرف إيه
    بالظبط؟" — asking a customer who had already asked clearly to ask again.
    The judge scored it helpfulness 1 and said so: "طلب توضيحًا عامًا من غير
    ما يسمّي الفروع أو المواعيد المحتملة".

    The fallback node has no slots to name, so the "name the missing thing"
    fix cannot reach it. What it does have is the retrieved set, and in a Q&A
    corpus each title is one of the meanings the question might have had.
    Offering those is both a real clarification and a grounded one: an option
    the KB cannot answer is never offered, because it was never retrieved.
    """
    orchestrator, *_ = build(
        intent=None,
        titles=("ما هي مواعيد عمل الفروع؟", "ما آخر موعد للتقديم؟"),
    )
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="المواعيد إيه؟", channel=CHANNEL
    )

    assert result.action is Action.clarify
    assert "مواعيد عمل الفروع" in result.reply
    assert "آخر موعد للتقديم" in result.reply
    # Against the string itself, not a word from it. The generic reply and this
    # one are both clarifications written in the same voice, so any word worth
    # matching on is a word they might come to share — which is a test that
    # passes on wording rather than on behaviour.
    assert result.reply != load("agent/replies")["replies"]["clarify"]["masri"]


async def test_one_retrieved_topic_is_not_a_choice(turn_session):
    """A list of one is not options, it is a guess with a question mark. If
    retrieval is that certain the fallback should not be reaching for a menu —
    the generic clarification is the honest reply."""
    orchestrator, *_ = build(intent=None, titles=("ما هي مواعيد عمل الفروع؟",))
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="المواعيد إيه؟", channel=CHANNEL
    )
    assert result.reply == load("agent/replies")["replies"]["clarify"]["masri"]


async def test_the_offered_options_are_capped(turn_session):
    """Retrieval returns `final_k` passages and a wall of them is not a
    question anyone answers. The cap is config, not a literal here."""
    from moc.config_store import load as _load

    cap = _load("agent/replies")["clarify_options"]["max"]
    orchestrator, *_ = build(
        intent=None, titles=tuple(f"سؤال رقم {n}" for n in range(cap + 3))
    )
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="المواعيد إيه؟", channel=CHANNEL
    )
    assert result.reply.count("سؤال رقم") == cap


async def test_a_named_missing_slot_still_wins_over_the_options(turn_session):
    """Two clarifications compete only in principle: a node with missing slots
    knows exactly what it needs, and offering a menu instead would replace an
    answerable question with a browse."""
    orchestrator, *_ = build(
        intent="admission_thresholds",
        slots={"faculty": "pharmacy"},
        titles=("ما هي مواعيد عمل الفروع؟", "ما آخر موعد للتقديم؟"),
    )
    result = await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="الحد الأدنى للقبول في الصيدلة كام؟",
        channel=CHANNEL,
    )
    assert "الفرع" in result.reply
    assert "مواعيد عمل الفروع" not in result.reply


async def test_an_english_customer_gets_the_english_option_sentence(turn_session):
    """F6. The options are the corpus's own words and stay as written; the
    sentence around them mirrors the customer."""
    orchestrator, *_ = build(
        intent=None, titles=("Branch working hours?", "Application deadline?")
    )
    result = await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="what times do you mean?",
        channel=CHANNEL,
    )
    assert "Branch working hours?" in result.reply
    assert result.reply.startswith(
        load("agent/replies")["clarify_options"]["template"]["english"].split("{")[0]
    )


# ───────── edu-0017: a refusal says what THIS script can offer ─────────


def test_every_refuse_node_has_its_own_refusal():
    """One shared `refuse` string served both verticals, and it was written for
    one of them.

    edu-0017 asks which faculty leads to a job — career advice, correctly
    refused — and the customer was told "أقدر أقولك السعر الحالي وخطة السداد
    المتاحة": a price and a payment plan, offered to a student. The judge
    caught it as an ungrounded offer, which it also is.

    The refusal's second half is what makes it a limit rather than a wall, and
    what is knowable instead is a property of the node doing the refusing. So
    the entry is per node, and this test is why a new refuse node cannot
    quietly inherit another vertical's answer.
    """
    from moc.config_store import load as _load

    refusals = _load("agent/replies")["replies"]["refuse"]
    nodes = [
        (script, name)
        for script in ("scripts/education/fees", "scripts/realestate/search")
        for name, node in _load(script)["nodes"].items()
        if node.get("action") == "refuse"
    ]
    assert nodes, "no refuse node found — this assertion would pass vacuously"
    missing = [f"{s}:{n}" for s, n in nodes if n not in refusals]
    assert not missing, (
        f"these refuse nodes fall back to another node's offer: {missing}"
    )


async def test_the_education_refusal_offers_something_a_student_can_use(
    turn_session,
):
    """edu-0017's forbidden claims rule out a ranking and an employment rate.
    What is left, and what the case note asks for, is the programme list."""
    from moc.config_store import load as _load

    orchestrator, *_ = build(intent="career_advice")
    result = await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="إيه أحسن كلية أدخلها عشان ألاقي شغل بسرعة؟",
        channel=CHANNEL,
    )

    assert result.action is Action.refuse
    assert result.reply == _load("agent/replies")["replies"]["refuse"]["career_advice"]["masri"]
    assert "السعر" not in result.reply, "the real-estate offer, to a student"


async def test_the_composition_prompt_carries_the_script_s_referral(turn_session):
    """The wiring, not the wording. `render_composition` takes a referral and
    the engine has one; nothing connected them until edu-0001."""
    orchestrator, anthropic, *_ = build()
    await orchestrator.handle(
        session=turn_session,
        state=start_state(),
        text="المصاريف كام لكلية الهندسة؟",
        channel=CHANNEL,
    )
    referral = ScriptEngine.from_config(SCRIPT).referral("ar")
    assert any(
        referral in (call.get("system") or "") for call in composition_calls(anthropic)
    ), (
        "the composition call did not carry the script's referral"
    )


# ───────── §19.3's second figure gate, on the turn path ─────────


async def test_a_relabelled_figure_is_not_sent(turn_session):
    """edu-0002's shape. The figure IS in the retrieved set, so the
    deterministic gate passes it; what the material never says is that this
    number is what the reply calls it.

    The outcome is the same as an orphan figure — the composition is discarded
    whole and a human takes the turn — because the customer-visible difference
    between "a fee we invented" and "a fee for something else" is nothing.
    """
    from moc.config_store import load as _load

    orchestrator, *_ = build(
        reply="رسوم التقديم 500 جنيه مصري",
        passages=("رسوم تغيير المسار 500 جنيه مصري",),
        audit={"figures": [{"figure": "500", "claim": "رسوم التقديم", "span": None}]},
    )
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="رسوم التقديم كام؟",
        channel=CHANNEL,
    )

    assert result.action is Action.handoff
    assert result.reply == _load("agent/replies")["replies"]["grounding_failed"]["msa"]
    assert "500" not in result.reply


async def test_a_correctly_labelled_figure_is_sent(turn_session):
    """The control. A gate that refuses everything is not a gate."""
    orchestrator, *_ = build(
        reply="رسوم التقديم 2000 جنيه مصري",
        passages=("رسوم التقديم 2000 جنيه مصري",),
        audit={
            "figures": [
                {
                    "figure": "2000",
                    "claim": "رسوم التقديم",
                    "span": "رسوم التقديم 2000 جنيه مصري",
                }
            ]
        },
    )
    result = await orchestrator.handle(
        session=turn_session, state=start_state(), text="رسوم التقديم كام؟",
        channel=CHANNEL,
    )
    assert result.action is Action.answer
    assert "2000" in result.reply


async def test_the_audit_sees_the_script_constants_too(turn_session):
    """§3.1: a figure the script states is as sourced as a retrieved one, and
    more directly. Auditing against passages alone would refuse every
    calculator result the real-estate agent produces."""
    orchestrator, anthropic, *_ = build(
        reply="القسط 302343 جنيه",
        passages=(),
        constants=("302343",),
        audit={"figures": []},
    )
    await orchestrator.handle(
        session=turn_session, state=start_state(), text="القسط كام؟", channel=CHANNEL,
    )
    audited = audit_calls(anthropic)
    assert audited, "the audit did not run on a composed reply stating a figure"
    prompt = audited[0]["messages"][0].content
    # Before the reply, not anywhere in the prompt. The reply states the same
    # figure, so "302343 is in the prompt" is true whether or not the constant
    # was passed as material — the first version of this assertion survived
    # deleting the constants entirely.
    material = prompt.split("القسط 302343 جنيه")[0]
    assert "302343" in material, "the calculator output was not offered as material"


async def test_the_audit_call_reaches_the_ledger(two_tenants, app_engine):
    """A provider call nobody meters is a cost that appears in no report.

    The audit is one extra completion per composed figure turn, on the
    customer-facing path — roughly 12 of 19 turns in the education suite. At
    $0.00069 each it is small and it is not nothing, and the reason to record
    it is not the money: an unmetered call is invisible when someone asks why
    a tenant's bill moved.
    """
    tenant, _ = two_tenants
    orchestrator, *_ = build(
        reply="رسوم التقديم 2000 جنيه مصري",
        passages=("رسوم التقديم 2000 جنيه مصري",),
        audit={
            "figures": [
                {
                    "figure": "2000",
                    "claim": "رسوم التقديم",
                    "span": "رسوم التقديم 2000 جنيه مصري",
                }
            ]
        },
    )
    async with tenant_session(app_engine, tenant.id) as s:
        await orchestrator.handle(
            session=s, state=start_state(), text="رسوم التقديم كام؟", channel=CHANNEL
        )
        await s.commit()
        models = [
            row[0]
            for row in (
                await s.execute(
                    sql("SELECT model FROM usage_ledger WHERE kind = 'llm_call'")
                )
            ).all()
        ]

    assert AUDIT_MODEL in models, "the audit call was never metered"
