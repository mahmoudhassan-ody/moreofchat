"""Script engine v1 — design §3.1 (script-first) and §7.5 (confidence gate).

Cases are derived from the shipped worked examples, so a change that breaks
edu-0004's slot retention fails here before it fails the eval suite.

No network: the engine is a deterministic state machine. The LLM fills slots
and phrases replies, and neither is needed to test flow logic — TurnInput is
what the orchestrator will hand over in Task 14.
"""

import ast
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from moc import config_store
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import Action, ConversationState, Register, TurnInput
from moc.tenancy.context import tenant_session

SCRIPT = "scripts/education/fees"
ENGINE_MODULE = Path(__file__).parents[2] / "src" / "moc" / "agent" / "script_engine.py"

HIGH = 0.9  # comfortably above the configured confidence threshold


@pytest.fixture
def script() -> ScriptEngine:
    return ScriptEngine.from_config(SCRIPT)


@pytest.fixture
def defaults() -> dict:
    return config_store.load("agent/defaults")


def turn(intent=None, slots=None, confidence=HIGH, grounded=True, **kwargs) -> TurnInput:
    return TurnInput(
        intent=intent,
        slots=slots or {},
        confidence=confidence,
        grounded=grounded,
        **kwargs,
    )


# ─────────────────── edu-0004: the multi-turn slot sequence ───────────────────


def test_clarify_when_a_required_slot_is_missing(script):
    """edu-0004 turn 1: 'عايز أعرف المصاريف' with no faculty -> clarify.

    forbidden_claims on that turn is 'any fee figure at all', so the decision
    must not be `answer` — a turn that reaches composition can quote a fee.
    """
    decision = script.advance(script.start(), turn(intent="fees"))
    assert decision.action is Action.clarify
    assert decision.ask_for_slot == "faculty"
    assert decision.missing_slots == ("faculty",)
    assert decision.state.slots == {}


def test_advances_when_the_slot_arrives(script):
    """edu-0004 turn 2: 'صيدلة' -> answer, slots {faculty: pharmacy}."""
    state = script.advance(script.start(), turn(intent="fees")).state
    decision = script.advance(state, turn(intent="fees", slots={"faculty": "pharmacy"}))
    assert decision.action is Action.answer
    assert decision.state.slots == {"faculty": "pharmacy"}
    assert decision.register is Register.msa


def test_slots_survive_across_turns(script):
    """edu-0004 turn 3: 'وده بيتدفع على كام قسط؟' — faculty must still be pharmacy.

    This is F5. The customer never repeats the faculty, and 'ده' refers back
    two turns; losing the slot here is what drives abandonment.
    """
    state = script.advance(script.start(), turn(intent="fees")).state
    state = script.advance(state, turn(intent="fees", slots={"faculty": "pharmacy"})).state
    decision = script.advance(state, turn(intent="instalments"))

    assert decision.action is Action.answer
    assert decision.state.slots["faculty"] == "pharmacy"


def test_a_new_value_for_a_held_slot_replaces_it(script):
    """The customer switching faculty mid-thread is a correction, not a second value."""
    state = script.advance(script.start(), turn(intent="fees", slots={"faculty": "pharmacy"})).state
    decision = script.advance(state, turn(intent="fees", slots={"faculty": "engineering"}))
    assert decision.state.slots["faculty"] == "engineering"


# ─────────────────────── edu-0011: register is node policy ───────────────────────


def test_register_follows_the_node_not_the_input(script):
    """edu-0011: casual Masri input, MSA output, because transfer regulations
    are official. Register is node policy, never mirrored from the customer."""
    decision = script.advance(script.start(), turn(intent="transfer_rules"))
    assert decision.action is Action.answer
    assert decision.register is Register.msa


def test_a_conversational_node_stays_masri(script):
    """The same rule in the other direction — clarification is small talk."""
    decision = script.advance(script.start(), turn(intent="fees"))
    assert decision.register is Register.masri


# ─────────────────────────── handoff ───────────────────────────


def test_routes_to_handoff_on_the_handoff_node(script):
    decision = script.advance(script.start(), turn(intent="accessibility"))
    assert decision.action is Action.handoff


def test_explicit_handoff_request_is_honoured_from_any_node(script):
    decision = script.advance(script.start(), turn(intent="fees", explicit_handoff_request=True))
    assert decision.action is Action.handoff


def test_three_consecutive_clarifications_escalate_to_handoff(script, defaults):
    """Threshold from config. Design §9 lists this as a handoff trigger."""
    limit = defaults["max_consecutive_clarifications"]
    state = script.start()
    for _ in range(limit):
        decision = script.advance(state, turn(intent="fees"))
        state = decision.state
    assert decision.action is Action.clarify

    decision = script.advance(state, turn(intent="fees"))
    assert decision.action is Action.handoff
    assert "clarification" in decision.reason


def test_a_successful_answer_resets_the_clarification_counter(script):
    state = script.advance(script.start(), turn(intent="fees")).state
    assert state.consecutive_clarifications == 1
    state = script.advance(state, turn(intent="fees", slots={"faculty": "pharmacy"})).state
    assert state.consecutive_clarifications == 0


# ────────────────── §19.3 invariant: the confidence gate ──────────────────


def test_below_threshold_confidence_cannot_reach_answer_composition(script, defaults):
    """The script-first invariant. Not configurable — §19.3.

    §7.5: below the threshold the turn does not reach answer composition. The
    *threshold* is tenant-tunable; that a below-threshold turn cannot compose a
    figure is a product guarantee.
    """
    low = defaults["confidence_threshold"] - 0.01
    decision = script.advance(
        script.start(), turn(intent="fees", slots={"faculty": "pharmacy"}, confidence=low)
    )
    assert decision.action is not Action.answer
    assert decision.grounding_required is False


def test_a_turn_with_nothing_retrieved_cannot_reach_answer_composition(script):
    """§19.3's invariant, now carried by presence instead of a threshold.

    This is what the confidence threshold was really enforcing: not "the
    passages are similar enough" but "there are passages at all". Similarity
    turned out to carry no separating signal; presence does, and it is the
    condition composition actually depends on — with nothing retrieved, every
    figure in the reply would be the model's own.
    """
    decision = script.advance(
        script.start(),
        turn(intent="fees", slots={"faculty": "pharmacy"}, grounded=False),
    )
    assert decision.action is not Action.answer
    assert decision.grounding_required is False


def test_a_correctly_retrieved_low_cosine_turn_still_answers(script):
    """edu-0015's real number, driven through the gate that reads it.

    0.218 is the measured cosine for "fe manh fe kantara?" retrieving
    `sinai_discount_tuition_ar` — the right chunk, first, from a corpus that
    never says منحة. The threshold used to be 0.55, which was set before there
    was anything to measure and would refuse six of the ten education cases
    that must answer, this one included.

    The measurement is in `config/retrieval/lexical.yaml`: across all 17
    cases, similarity does not separate must-answer from must-not, and it
    overlaps the wrong way round.
    """
    decision = script.advance(
        script.start(),
        turn(intent="fees", slots={"faculty": "pharmacy"}, confidence=0.218),
    )
    assert decision.action is Action.answer


def test_absent_confidence_does_not_close_the_gate(script):
    """§7.3. `None` is "no arm supplied a calibrated score", not a low one.

    Meilisearch ranks without scoring, so a lexical-only turn — an embedding
    outage, or a deployment with no dense arm — has no number to threshold.
    Treating that as below-threshold would turn a degraded turn into a refused
    one, which is precisely the failure §7.3 says to avoid: the correct
    behaviour is that search degrades, not that the product stops answering.

    Nothing is lost by allowing it through here. The figures in the composed
    reply still have to survive `check_numeric_grounding`, which is the guard
    that actually prevents an invented fee.
    """
    decision = script.advance(
        script.start(),
        turn(intent="fees", slots={"faculty": "pharmacy"}, confidence=None),
    )
    assert decision.action is Action.answer


def test_confidence_exactly_at_the_threshold_is_allowed(script, defaults):
    """A boundary worth pinning: >= passes, so the threshold means what it reads."""
    decision = script.advance(
        script.start(),
        turn(
            intent="fees",
            slots={"faculty": "pharmacy"},
            confidence=defaults["confidence_threshold"],
        ),
    )
    assert decision.action is Action.answer


def test_the_gate_cannot_be_disabled_by_the_script():
    """§19.3: fixed enforcement, configurable parameter.

    Demonstrated on presence rather than on the threshold. The threshold now
    sits at the platform floor, so a script setting it to 0.0 no longer
    differs from the default and could not show anything — but the rule it was
    protecting is unchanged, and it is the ungrounded turn that must not reach
    composition however the script is written.
    """
    opted_out = dict(config_store.load(SCRIPT))
    opted_out["settings"] = {**opted_out["settings"], "confidence_threshold": 0.0}
    engine = ScriptEngine(script=opted_out, defaults=config_store.load("agent/defaults"))
    decision = engine.advance(
        engine.start(),
        turn(intent="fees", slots={"faculty": "pharmacy"}, grounded=False),
    )
    assert decision.action is not Action.answer


def test_the_threshold_is_read_from_the_platform_not_the_script():
    """The other half of §19.3: a tenant may not raise it either, because the
    parameter is platform-tier config and the script is tenant-authored."""
    stricter = dict(config_store.load(SCRIPT))
    stricter["settings"] = {**stricter["settings"], "confidence_threshold": 0.99}
    engine = ScriptEngine(script=stricter, defaults=config_store.load("agent/defaults"))
    decision = engine.advance(
        engine.start(),
        turn(intent="fees", slots={"faculty": "pharmacy"}, confidence=0.3),
    )
    assert decision.action is Action.answer, "the script's 0.99 must be ignored"


# ─────────────────────────── fallback ───────────────────────────


def test_unknown_intent_hits_the_fallback_node_not_an_exception(script):
    decision = script.advance(script.start(), turn(intent="what_is_the_wifi_password"))
    assert decision.action is Action.clarify
    assert decision.node == "fallback"


def test_no_intent_at_all_also_falls_back(script):
    assert script.advance(script.start(), turn(intent=None)).node == "fallback"


# ─────────────────── the engine composes nothing (§3.1) ───────────────────


def test_decision_carries_no_reply_text(script):
    """It returns an Action plus constrained context. Composition is Task 14."""
    decision = script.advance(script.start(), turn(intent="fees"))
    assert not hasattr(decision, "text")
    assert not hasattr(decision, "reply")


def test_the_engine_module_imports_no_provider_and_no_router():
    """Structural guard for 'never calls a provider'.

    A comment saying so is not enforcement; an import is how it would start.
    """
    tree = ast.parse(ENGINE_MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = {"moc.llm", "moc.llm.router", "moc.llm.base", "httpx", "anthropic", "openai"}
    assert imported & banned == set(), f"script engine reaches for a provider: {imported & banned}"


# ─────────────────────────── versioning and state ───────────────────────────


def test_state_pins_the_script_version(script):
    """Design §5: script_versions are immutable and conversations pin one."""
    state = script.start()
    assert state.script_version == config_store.load(SCRIPT)["version"]
    assert state.script_id == config_store.load(SCRIPT)["script_id"]


def test_an_in_flight_conversation_keeps_its_pinned_version(script):
    """Publishing v2 must not move a conversation already running on v1."""
    state = script.advance(script.start(), turn(intent="fees")).state

    republished = dict(config_store.load(SCRIPT))
    republished["version"] = state.script_version + 1
    v2 = ScriptEngine(script=republished, defaults=config_store.load("agent/defaults"))

    with pytest.raises(ValueError, match="version"):
        v2.advance(state, turn(intent="fees", slots={"faculty": "pharmacy"}))


def test_state_round_trips_through_json(script):
    state = script.advance(script.start(), turn(intent="fees", slots={"faculty": "pharmacy"})).state
    assert ConversationState.from_json(json.loads(json.dumps(state.to_json()))) == state


async def test_state_persists_to_the_conversations_jsonb_column(script, app_engine, two_tenants):
    """Design §5: conversations.state holds the script cursor and slots.

    Through app_engine, so the write goes via the RLS-enforced role rather than
    the owner.
    """
    tenant, _ = two_tenants
    state = script.advance(script.start(), turn(intent="fees", slots={"faculty": "pharmacy"})).state

    async with tenant_session(app_engine, tenant.id) as s:
        await s.execute(
            text(
                "INSERT INTO conversations (id, tenant_id, state) "
                "VALUES (gen_random_uuid(), :t, CAST(:state AS jsonb))"
            ),
            {"t": tenant.id, "state": json.dumps(state.to_json())},
        )
        await s.commit()

    async with tenant_session(app_engine, tenant.id) as s:
        stored = (await s.execute(text("SELECT state FROM conversations"))).scalar_one()

    assert ConversationState.from_json(stored) == state
    assert stored["slots"]["faculty"] == "pharmacy"


async def test_edu_0004_runs_end_to_end_against_the_fake_provider(script):
    """The whole three-turn case, with slot extraction coming from a provider.

    The engine still imports nothing from moc.llm — the caller turns a provider
    response into a TurnInput. This test exists to prove that seam holds, since
    it is the shape Task 14's orchestrator will use.
    """
    from moc.llm.base import Message
    from moc.llm.fake import FakeProvider

    # Canned extractions, one per inbound message, as Haiku would return (§3.1).
    extractions = [
        json.dumps({"intent": "fees", "slots": {}}),
        json.dumps({"intent": "fees", "slots": {"faculty": "pharmacy"}}),
        json.dumps({"intent": "instalments", "slots": {}}),
    ]
    inbound = [
        "عايز أعرف المصاريف",
        "صيدلة",
        "وده بيتدفع على كام قسط؟",
    ]

    state, actions = script.start(), []
    for message, canned in zip(inbound, extractions, strict=True):
        extractor = FakeProvider("anthropic", text=canned)
        completion = await extractor.complete(
            model="m",
            messages=[Message(role="user", content=message)],
            system=None,
            cache_blocks=[],
            max_tokens=64,
        )
        parsed = json.loads(completion.text)
        decision = script.advance(
            state,
            TurnInput(
                intent=parsed["intent"],
                slots=parsed["slots"],
                confidence=HIGH,
                # Built by hand here; the orchestrator is what normally sets
                # this from the retrieval result.
                grounded=True,
            ),
        )
        state, _ = decision.state, actions.append(decision.action)

    assert actions == [Action.clarify, Action.answer, Action.answer]
    assert state.slots == {"faculty": "pharmacy"}, "F5: the faculty survived all three turns"


def test_action_values_match_the_eval_schema():
    """check_action compares the engine's output to a case's expected_action.

    Two enums, one contract — this fails if either drifts.
    """
    from moc.evals.schema import Action as EvalAction

    assert {a.value for a in Action} == {a.value for a in EvalAction}


def test_a_gate_refusal_is_flagged_not_inferred_from_its_wording(script):
    """The customer-visible reply is chosen from this flag.

    It used to be selected by `decision.reason.startswith("retrieval
    confidence")`, so when the ungrounded branch grew its own wording the
    customer silently started getting "which faculty?" instead of "I can't
    find confirmed information about that" — a bot asking a question it had
    already asked. A diagnostic string is not an interface.
    """
    missing_slot = script.advance(script.start(), turn(intent="fees"))
    assert missing_slot.action is Action.clarify
    assert missing_slot.gate_closed is False

    ungrounded = script.advance(
        script.start(),
        turn(intent="fees", slots={"faculty": "pharmacy"}, grounded=False),
    )
    assert ungrounded.action is Action.clarify
    assert ungrounded.gate_closed is True


# ─────────────── requires_any_slot: browse needs a narrowing slot ───────────────

REALESTATE = "scripts/realestate/search"


def realestate() -> ScriptEngine:
    return ScriptEngine.from_config(REALESTATE)


def test_a_lookup_naming_nothing_narrowing_asks_rather_than_answers():
    """re-0018 turn 1: "بدور على حاجة للبيع" — looking for something to buy.

    `requires_slots: [property_type]` was dropped on 2026-08-19 because a
    customer who named no type cannot be substituted against. Correct, but the
    replacement was nothing at all, and the turn then answered with one
    arbitrary studio out of 305 units. `listing_kind: sale` narrows to
    everything the broker sells.
    """
    engine = realestate()
    decision = engine.advance(
        engine.start(), turn(intent="inventory_lookup", slots={"listing_kind": "sale"})
    )
    assert decision.action is Action.clarify


def test_any_one_narrowing_slot_is_enough():
    """re-0023 names only a compound and must be answered — the browse case
    the requirement was dropped for. Type is one option among several, not the
    one that counts."""
    engine = realestate()
    for slot, value in (
        ("compound", "Mivida"),
        ("city", "New Cairo"),
        ("property_type", "villa"),
        ("bedrooms", 3),
        ("budget_max", 5_000_000),
    ):
        decision = engine.advance(
            engine.start(), turn(intent="inventory_lookup", slots={slot: value})
        )
        assert decision.action is Action.answer, f"{slot} alone should answer"


def test_a_narrowing_slot_held_from_an_earlier_turn_still_counts():
    """re-0018 turn 3 names cities after two clarifications; the budget from
    turn 2 is held, and held slots narrow exactly as freshly stated ones do."""
    engine = realestate()
    state = engine.start()
    state = ConversationState(
        script_id=state.script_id,
        script_version=state.script_version,
        slots={"budget_max": 10_000_000},
    )
    decision = engine.advance(state, turn(intent="inventory_lookup", slots={}))
    assert decision.action is Action.answer


def test_the_narrowing_slots_are_config_not_a_literal():
    """§19. Which slots narrow a search is a vertical's decision, and a broker
    with one compound would list different ones."""
    node = config_store.load(REALESTATE)["nodes"]["inventory_lookup"]
    assert node["requires_any_slot"] == [
        "city",
        "compound",
        "property_type",
        "bedrooms",
        "budget_max",
        "near_price",
    ]
    import inspect

    from moc.agent import script_engine

    source = inspect.getsource(script_engine)
    for slot in ("compound", "budget_max"):
        assert f'"{slot}"' not in source, f"{slot} is named in the engine"


def test_every_state_field_survives_a_turn():
    """The engine rebuilt `ConversationState` field by field, so a field added
    later was dropped on every turn — silently, back to its default. Asserted
    structurally rather than by listing fields, which is the same mistake."""
    import dataclasses
    import inspect

    from moc.agent import script_engine

    source = inspect.getsource(script_engine)
    assert source.count("ConversationState(") == 1, (
        "only `start` may construct one; everywhere else must use `replace`, "
        "or the next field added to the state is dropped the same way"
    )
    engine = ScriptEngine.from_config(SCRIPT)
    start = engine.start()
    seeded = dataclasses.replace(start, quoted_unit_id="SOME-UNIT-1")

    decision = engine.advance(seeded, turn(intent="fees", slots={"faculty": "dentistry"}))
    assert decision.state.quoted_unit_id == "SOME-UNIT-1"

    clarified = engine.advance(seeded, turn(intent=None))
    assert clarified.state.quoted_unit_id == "SOME-UNIT-1"


def test_a_cleared_slot_is_dropped_before_the_node_is_chosen():
    """re-0001 turn 3. The held city must be gone by the time
    `requires_any_slot` and the connector see the state — clearing it after
    the search has been built would change nothing."""
    engine = realestate()
    state = engine.start().with_slots({"city": "New Cairo", "property_type": "villa"})

    decision = engine.advance(
        state, turn(intent="inventory_lookup", slots={}, cleared=("city", "compound"))
    )
    assert "city" not in decision.state.slots
    assert decision.state.slots["property_type"] == "villa", "only what was named"
    assert decision.action is Action.answer, "property_type still narrows"


def test_clearing_every_narrowing_slot_asks_rather_than_listing_everything():
    """Dropping the last filter leaves 305 units. That is a clarification."""
    engine = realestate()
    state = engine.start().with_slots({"city": "New Cairo"})
    decision = engine.advance(
        state, turn(intent="inventory_lookup", slots={}, cleared=("city",))
    )
    assert decision.action is Action.clarify


def test_clearing_a_slot_that_was_never_held_is_not_an_error():
    engine = realestate()
    decision = engine.advance(
        engine.start(),
        turn(intent="inventory_lookup", slots={"compound": "Mivida"}, cleared=("city",)),
    )
    assert decision.action is Action.answer


def test_clearing_a_slot_continues_the_turn_it_is_clearing_from():
    """re-0001 turn 3. "In any other location" carries no intent of its own —
    it is the previous request minus a filter, and the model reports
    `intent: null` for exactly that reason.

    Falling back on it asks the customer to clarify a question they narrowed
    one word ago. Clearing a slot IS a continuation; there is nothing else it
    could be.
    """
    engine = realestate()
    state = engine.advance(
        engine.start(),
        turn(intent="inventory_lookup", slots={"city": "New Cairo", "property_type": "villa"}),
    ).state

    decision = engine.advance(state, turn(intent=None, slots={}, cleared=("city",)))
    assert decision.node == "inventory_lookup"
    assert decision.action is Action.answer


def test_a_bare_null_intent_still_falls_back():
    """Only a clearing turn continues. A message the model could not read at
    all is a fallback, or every unreadable turn would re-run the last search.
    """
    engine = realestate()
    state = engine.advance(
        engine.start(), turn(intent="inventory_lookup", slots={"city": "New Cairo"})
    ).state

    decision = engine.advance(state, turn(intent=None, slots={}))
    assert decision.node == "fallback"
