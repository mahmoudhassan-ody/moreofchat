"""Conversation state and the turn contract — design §3.1 and §5.

`ConversationState` is what lives in `conversations.state` jsonb: the script
cursor and the slots. It is a plain value object with explicit JSON mapping
rather than a pickled model, because the column is queried by humans debugging
a thread and by SQL in reports, and both need a shape that stays legible.

`TurnInput` is the seam between the LLM and the state machine. Intent, slots
and confidence come from elsewhere — Haiku in production (§3.1), a test literal
here — which is what keeps flow logic testable without a network.
"""

import enum
from dataclasses import dataclass, field, replace
from typing import Any


class Action(enum.StrEnum):
    """What the orchestrator should do with this turn.

    Values match `moc.evals.schema.Action` exactly, so a case's
    `expected_action` compares directly against what the engine emitted. The
    enums are separate because the eval package must not become a runtime
    dependency; a test pins the two together.
    """

    answer = "answer"
    clarify = "clarify"
    handoff = "handoff"
    refuse = "refuse"


class Register(enum.StrEnum):
    """§8.2. Per-node policy, never inferred from the customer's variety."""

    masri = "masri"
    msa = "msa"
    english = "english"


@dataclass(frozen=True)
class TurnInput:
    """What the orchestrator extracted from one inbound message.

    `confidence` is the fused retrieval score (§7.5), not the model's own
    reported certainty — a model's self-assessment is not evidence that the
    knowledge base contains the fee.
    """

    intent: str | None
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    explicit_handoff_request: bool = False


@dataclass(frozen=True)
class ConversationState:
    script_id: str
    script_version: int
    node: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    consecutive_clarifications: int = 0

    def with_slots(self, new: dict[str, Any]) -> ConversationState:
        """Merge newly extracted slots over the held ones.

        Replace, not accumulate: a customer naming a second faculty is
        correcting themselves, not asking about both.
        """
        return replace(self, slots={**self.slots, **new})

    def to_json(self) -> dict[str, Any]:
        return {
            "script_id": self.script_id,
            "script_version": self.script_version,
            "node": self.node,
            "slots": self.slots,
            "consecutive_clarifications": self.consecutive_clarifications,
        }

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> ConversationState:
        return cls(
            script_id=record["script_id"],
            script_version=record["script_version"],
            node=record.get("node"),
            slots=record.get("slots") or {},
            consecutive_clarifications=record.get("consecutive_clarifications", 0),
        )


@dataclass(frozen=True)
class Decision:
    """The engine's output: an action and the constrained context for it.

    There is no reply text here, and that is the point (§3.1). The engine
    decides *what kind of turn this is* and *what may be said*; composing the
    words is the orchestrator's job, against retrieved passages.
    """

    action: Action
    register: Register
    node: str
    state: ConversationState
    missing_slots: tuple[str, ...] = ()
    ask_for_slot: str | None = None
    grounding_required: bool = False
    reason: str = ""
