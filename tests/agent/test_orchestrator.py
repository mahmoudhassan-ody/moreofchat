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
        self._turn = turn
        self.seen: list[str] = []

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput:
        self.seen.append(text)
        return self._turn


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
    fail_with: Exception | None = None,
) -> tuple[Orchestrator, FakeProvider, FakeProvider, FakeExtractor, FakeRetriever]:
    anthropic = FakeProvider(
        "anthropic",
        text=reply,
        fail_with=fail_with,
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
    retriever = FakeRetriever(router, Retrieval(passages=passages, confidence=confidence))
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


def composition_calls(provider: FakeProvider) -> list[dict]:
    return [c for c in provider.calls if c["kind"] == "complete"]


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
