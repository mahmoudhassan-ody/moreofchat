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

from dataclasses import replace
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

    def referral(self, lang: str | None) -> str | None:
        """Where this script sends a turn it cannot answer, in the customer's
        language.

        edu-0001: a reply that says the material does not hold what was asked
        is truthful, grounded and a dead end. Which office to name is tenant
        data, so it is configured and quoted verbatim — a model asked to name
        one will name a plausible one, and a plausible office is the same class
        of failure as a plausible fee.

        None when the script configures none, so composition keeps the
        instruction it had rather than rendering a placeholder at a customer.
        """
        entry = (self._script.get("settings") or {}).get("referral") or {}
        return entry.get(lang or "ar") or entry.get("ar")

    @property
    def version(self) -> int:
        """Which version of the script this engine is running.

        Public because the worker has to compare it against a conversation's
        pinned version — `_require_pinned_version` raises when they differ, so
        the caller needs to be able to ask before it finds out by exception.
        """
        return self._script["version"]

    @property
    def script_id(self) -> str:
        return self._script["script_id"]

    def start(self) -> ConversationState:
        return ConversationState(
            script_id=self._script["script_id"],
            script_version=self._script["version"],
            node=self._script.get("entry"),
        )

    def _stale(self, state: ConversationState, turn: TurnInput) -> tuple[str, ...]:
        """Held slots this turn made obsolete without saying so.

        A slot may declare that it `narrows` another — one value sits inside
        the other's. When a turn names the wider value and not the narrower
        one, the held narrower value is a filter inside something the customer
        has left, and keeping it intersects two places into none. That reads as
        "we have nothing there" about the place they just named.

        Distinct from `clear_slots`, which is the customer saying *this no
        longer applies* — "somewhere else", with no replacement. Here they
        named a replacement, just one slot up.

        Only when this message does not also name the narrower slot: a message
        naming both means both, and the narrower one is not stale at all.

        Which slot narrows which is read from the script, so the engine never
        names a vertical's slots and a vertical that adds a pair does not
        change this file.
        """
        stale = []
        for name, spec in (self._script.get("slots") or {}).items():
            wider = (spec or {}).get("narrows")
            if wider is None:
                continue
            if wider in turn.slots and name not in turn.slots and name in state.slots:
                stale.append(name)
        return tuple(stale)

    def advance(self, state: ConversationState, turn: TurnInput) -> Decision:
        self._require_pinned_version(state)
        # Read against the state as it stood when the question was asked, so
        # "did this turn answer it?" is decided before the answer is merged in.
        resumed = self._resumed(state, turn)
        # Cleared before the node is chosen, so `requires_any_slot` and the
        # connector both see the state the customer actually left behind.
        state = state.with_slots(turn.slots, turn.cleared + self._stale(state, turn))
        node_name = self._by_intent.get(turn.intent or "", _FALLBACK_NODE)
        if turn.intent is None and turn.cleared and state.node in self._script["nodes"]:
            # "In any other location" carries no intent of its own — it is the
            # previous request minus a filter, which is why the model reports
            # null. Falling back on it asks the customer to clarify a question
            # they narrowed one word ago. Only a clearing turn continues: an
            # unreadable message must still fall back, or every one of them
            # would silently re-run the last search.
            node_name = state.node
        elif resumed is not None:
            node_name = resumed
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

        # "At least one of these", where `requires_slots` is "all of these".
        # A browse node cannot demand a specific slot — re-0023 names only a
        # compound and must be answered — but demanding nothing is not the
        # alternative: "بدور على حاجة للبيع" answered with one arbitrary studio
        # out of 305 units, which is a worse turn than asking.
        narrowing = node.get("requires_any_slot") or []
        if narrowing and not any(s in state.slots for s in narrowing):
            return self._clarify(state, node_name, node, tuple(narrowing))

        # §7.5, as two separate questions. Presence is the one that carries:
        # with nothing retrieved, composition would have to invent every
        # figure. The threshold is kept because it is configurable, but it
        # sits at the floor — measurement showed similarity does not separate
        # the turns that must answer from the turns that must not.
        uncertain = (
            turn.confidence is not None and turn.confidence < self._confidence_threshold
        )
        if action is Action.answer and (not turn.grounded or uncertain):
            # §7.5 and §19.3. The turn does not reach answer composition; the
            # script's fallback node handles it. A hallucinated tuition figure
            # is a commercial incident, so this is not a tunable behaviour.
            fallback = self._script["nodes"][_FALLBACK_NODE]
            return self._clarify(
                state,
                _FALLBACK_NODE,
                fallback,
                (),
                gate_closed=True,
                reason=(
                    "retrieval returned nothing to ground an answer in"
                    if not turn.grounded
                    else (
                        f"retrieval confidence {turn.confidence} below threshold "
                        f"{self._confidence_threshold}"
                    )
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

    def _resumed(self, state: ConversationState, turn: TurnInput) -> str | None:
        """The node that asked, when this turn is the answer to it.

        edu-0007 turn 2 is the case: the clarification asked which branch, the
        customer said `العريش`, and a bare slot value carries no intent because
        there is no question in it to read one from. So it fell to `fallback`
        and got the generic "ممكن توضّحلي أكتر". The slot survived — retention
        never dropped — but the node did not, which left the clarification with
        nothing missing to name and put §19's "name the missing thing" fix out
        of reach of the one turn that needed it. The customer answered the
        question and was asked to start over.

        The discriminator is deliberately narrow: this turn filled a slot the
        held node was *still waiting for*. Continuing on any slot-bearing turn
        would make a topic change re-run the question the customer just left,
        and repeating a slot the node already holds answers nothing. Both are
        read against the pre-merge state, because after the merge every slot
        looks held.
        """
        if turn.intent is not None or not turn.slots:
            return None
        node = self._script["nodes"].get(state.node or "")
        if node is None:
            return None
        pending = (
            set(node.get("requires_slots") or []) | set(node.get("requires_any_slot") or [])
        ) - set(state.slots)
        return state.node if pending & set(turn.slots) else None

    # ─────────────────────────── outcomes ───────────────────────────

    def _clarify(
        self,
        state: ConversationState,
        node_name: str,
        node: dict[str, Any],
        missing: tuple[str, ...],
        reason: str = "",
        gate_closed: bool = False,
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
            gate_closed=gate_closed,
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


# `replace`, never a field-by-field rebuild. Both of these used to list every
# field of ConversationState, so `quoted_unit_id` — added later — was dropped on
# every turn that passed through them. Nothing raised: the field simply reset to
# its default, and a follow-up question stopped knowing which unit it was about.
def _count_clarification(state: ConversationState, node: str) -> ConversationState:
    return replace(
        state, node=node, consecutive_clarifications=state.consecutive_clarifications + 1
    )


def _reset_clarifications(state: ConversationState, node: str) -> ConversationState:
    return replace(state, node=node, consecutive_clarifications=0)
