"""Script engine v1 — the deterministic half of a turn (design §3.1).

The script owns the flow and the facts. This module decides *what kind of turn
this is* — answer, clarify, handoff, refuse — which slots are still missing,
and which register applies. It never composes text and never calls a provider;
a test asserts the second by inspecting this module's imports, because a
comment saying so is documentation rather than enforcement.

Order of precedence in `advance`, which is deliberate:

1. **Explicit handoff request** — a customer asking for a human gets one,
   from any node.
2. **Clarification loop limit** — three failed attempts means the bot has not
   understood, and a fourth rephrasing annoys rather than helps (§9).
3. **Missing required slot** — clarify rather than guess. edu-0004 turn 1
   forbids *any* fee figure while the faculty is unknown.
4. **Confidence gate** — §7.5. Below threshold the turn cannot reach answer
   composition, whatever the node says.

The gate sits after the slot check on purpose: a turn missing a slot should
ask for the slot, not hand off for low confidence on a question nobody has
finished asking yet.
"""

from typing import Any

from moc.agent.state import Action, ConversationState, Decision, Register, TurnInput
from moc.config_store import load

_DEFAULTS = "agent/defaults"
_FALLBACK_NODE = "fallback"


class ScriptEngine:
    def __init__(self, *, script: dict[str, Any], defaults: dict[str, Any]) -> None:
        self._script = script
        self._defaults = defaults
        settings = script.get("settings") or {}
        self._max_clarifications = settings.get(
            "max_consecutive_clarifications", defaults["max_consecutive_clarifications"]
        )
        # Read from platform defaults only. §19.3: the threshold is tunable but
        # the gate is a product guarantee, so a script cannot lower its own bar.
        self._confidence_threshold = defaults["confidence_threshold"]
        self._by_intent = {
            intent: name
            for name, node in script["nodes"].items()
            for intent in (node.get("intents") or [])
        }

    @classmethod
    def from_config(cls, name: str) -> ScriptEngine:
        return cls(script=load(name), defaults=load(_DEFAULTS))

    def start(self) -> ConversationState:
        return ConversationState(
            script_id=self._script["script_id"],
            script_version=self._script["version"],
            node=self._script.get("entry"),
        )

    def advance(self, state: ConversationState, turn: TurnInput) -> Decision:
        self._require_pinned_version(state)
        state = state.with_slots(turn.slots)
        node_name = self._by_intent.get(turn.intent or "", _FALLBACK_NODE)
        node = self._script["nodes"][node_name]

        if turn.explicit_handoff_request:
            return self._handoff(state, node_name, node, "customer asked for a human")

        if state.consecutive_clarifications >= self._max_clarifications:
            return self._handoff(
                state,
                node_name,
                node,
                f"{state.consecutive_clarifications} consecutive clarifications",
            )

        action = Action(node.get("action", Action.clarify))
        if action is Action.handoff:
            return self._handoff(state, node_name, node, "script node is a handoff node")

        missing = tuple(s for s in node.get("requires_slots") or [] if s not in state.slots)
        if missing:
            return self._clarify(state, node_name, node, missing)

        uncertain = (
            turn.confidence is not None and turn.confidence < self._confidence_threshold
        )
        if action is Action.answer and uncertain:
            # §7.5 and §19.3. The turn does not reach answer composition; the
            # script's fallback node handles it. A hallucinated tuition figure
            # is a commercial incident, so this is not a tunable behaviour.
            fallback = self._script["nodes"][_FALLBACK_NODE]
            return self._clarify(
                state,
                _FALLBACK_NODE,
                fallback,
                (),
                reason=(
                    f"retrieval confidence {turn.confidence} below threshold "
                    f"{self._confidence_threshold}"
                ),
            )

        if action is Action.clarify:
            return self._clarify(state, node_name, node, missing)

        return Decision(
            action=action,
            register=Register(node["register"]),
            node=node_name,
            # A completed answer means the loop broke; the counter starts over.
            state=_reset_clarifications(state, node_name),
            grounding_required=node.get("grounding") == "required",
        )

    # ─────────────────────────── outcomes ───────────────────────────

    def _clarify(
        self,
        state: ConversationState,
        node_name: str,
        node: dict[str, Any],
        missing: tuple[str, ...],
        reason: str = "",
    ) -> Decision:
        clarify = node.get("clarify") or {}
        register = clarify.get("register") or node.get("register", Register.masri)
        return Decision(
            action=Action.clarify,
            register=Register(register),
            node=node_name,
            state=_count_clarification(state, node_name),
            missing_slots=missing,
            ask_for_slot=clarify.get("ask_for_slot") if missing else None,
            reason=reason,
        )

    def _handoff(
        self, state: ConversationState, node_name: str, node: dict[str, Any], reason: str
    ) -> Decision:
        return Decision(
            action=Action.handoff,
            register=Register(node.get("register", Register.masri)),
            node=node_name,
            state=_reset_clarifications(state, node_name),
            reason=reason,
        )

    def _require_pinned_version(self, state: ConversationState) -> None:
        """Design §5: conversations pin a script version, and versions are immutable.

        Refusing here means a mid-flight conversation cannot silently jump to a
        republished flow — it finishes on the version it started, or the caller
        makes an explicit migration decision.
        """
        if state.script_version != self._script["version"]:
            raise ValueError(
                f"conversation is pinned to {state.script_id} version "
                f"{state.script_version}, engine is running version "
                f"{self._script['version']}"
            )


def _count_clarification(state: ConversationState, node: str) -> ConversationState:
    return ConversationState(
        script_id=state.script_id,
        script_version=state.script_version,
        node=node,
        slots=state.slots,
        consecutive_clarifications=state.consecutive_clarifications + 1,
    )


def _reset_clarifications(state: ConversationState, node: str) -> ConversationState:
    return ConversationState(
        script_id=state.script_id,
        script_version=state.script_version,
        node=node,
        slots=state.slots,
        consecutive_clarifications=0,
    )
