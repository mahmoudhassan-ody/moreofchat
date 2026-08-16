"""Eval case schema — mirrors eval-harness-spec.md §3 and §3.2.

Every model sets `extra="forbid"`. A case file is a test suite written in YAML,
and the failure mode of a permissive schema is silent: `expected_slot` instead
of `expected_slots` parses fine, asserts nothing, and the case reports a pass it
never earned. Forbidding extras turns that typo into a load error.

Cases are append-only (§3). Adding a field here is safe; renaming or removing
one invalidates every case file already written against it.
"""

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Vertical(enum.StrEnum):
    education = "education"
    realestate = "realestate"


class Source(enum.StrEnum):
    real_conversation = "real_conversation"
    synthetic = "synthetic"


class Category(enum.StrEnum):
    """Spec §4.1 and §4.2. Four categories are shared across both verticals."""

    # education
    factual_retrieval = "factual_retrieval"
    ambiguous = "ambiguous"
    code_switching = "code_switching"
    register_sensitive = "register_sensitive"
    # real estate
    inventory_lookup = "inventory_lookup"
    payment_plan_math = "payment_plan_math"
    staleness = "staleness"
    sold_or_reserved = "sold_or_reserved"
    # both
    franco_or_misspelled = "franco_or_misspelled"
    multi_turn_slots = "multi_turn_slots"
    adversarial_figures = "adversarial_figures"
    out_of_scope = "out_of_scope"


class Channel(enum.StrEnum):
    whatsapp = "whatsapp"
    instagram = "instagram"
    messenger = "messenger"
    telegram = "telegram"
    email = "email"


class InputLang(enum.StrEnum):
    masri = "masri"
    msa = "msa"
    english = "english"
    mixed = "mixed"
    franco = "franco"


class Action(enum.StrEnum):
    answer = "answer"
    clarify = "clarify"
    handoff = "handoff"
    refuse = "refuse"


class Register(enum.StrEnum):
    masri = "masri"
    msa = "msa"
    english = "english"


class GroundingMode(enum.StrEnum):
    """§3.2. Education grounds on chunks; real estate on a live inventory table."""

    documents = "documents"
    inventory = "inventory"
    hybrid = "hybrid"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedFact(Strict):
    """Atomic by rule (§3.1): one claim per entry, so partial credit means something."""

    id: str
    claim: str
    required: bool = True
    # Absent for facts the judge grades without a specific chunk to trace to.
    source_chunk: str | None = None


class ExpectedToolCall(Strict):
    """§5.1: asserted by code, not judged.

    `args_contain` is a subset check — the orchestrator may pass more arguments
    than a case pins, and pinning all of them would make every case brittle to
    an unrelated signature change.
    """

    name: str
    args_contain: dict[str, Any] = Field(default_factory=dict)


class ExpectedComputation(Strict):
    """§3.2: the model never does arithmetic.

    Every figure in the reply must string-match a value in this tool's output.
    A figure that is merely close is a failure — that is F1 with extra steps.
    """

    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    must_match_fixture: bool = True


class Turn(Strict):
    user: str
    expected_action: Action
    expected_register: Register | None = None
    expected_lang: str | None = None
    expected_facts: list[ExpectedFact] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    # Slot values may be scalars or lists — a customer can hold two areas at
    # once, and a scalar-only type would coerce that to a single value.
    expected_slots: dict[str, Any] | None = None

    # §3.2 — structured-inventory grounding.
    # Defaults are the permissive end on purpose: a turn that does not opt in
    # is not graded on disclosure or tool calls, rather than failing for a
    # dimension its author never considered.
    expected_asof_disclosure: bool = False
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    expected_computation: ExpectedComputation | None = None


class EvalCase(Strict):
    id: str
    vertical: Vertical
    source: Source
    category: Category
    tenant_fixture: str
    channel: Channel
    input_lang: InputLang
    turns: list[Turn] = Field(default_factory=list)

    # §3.1: cases without gold_chunks are excluded from recall metrics rather
    # than counted as failures, so an empty list is meaningful, not a stub.
    gold_chunks: list[str] = Field(default_factory=list)
    notes: str | None = None

    grounding_mode: GroundingMode = GroundingMode.documents
    inventory_fixture: str | None = None
