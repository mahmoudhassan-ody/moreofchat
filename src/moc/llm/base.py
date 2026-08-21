"""The LLMProvider port — design §2.4 and §2.6.

One port, two adapters. That is what keeps the §2.4 migration table honest: if
a university's procurement forces Claude into Bedrock `eu-central-1`, the work
is a new file implementing this Protocol, not an audit of every call site. The
guard in tests/llm/test_no_endpoints_outside_llm.py holds the other half of
that promise — no endpoint string may exist outside this package.

Nothing here performs I/O. Adapters land in Task 13; the fake in `fake.py` is
what Tasks 12-15 test against, so flow logic never waits on a vendor.

**Error taxonomy is load-bearing.** `ProviderUnavailable` means "try someone
else" — rate limits, 5xx, timeouts. Everything else propagates untouched,
because a 400 is our bug and failing over would hide it behind a second
provider while doubling the cost of every malformed call.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Vector = list[float]


class Reasoning(enum.StrEnum):
    """The one control that must mean the same thing on both providers.

    Failover already changes the model; it must not also change *how* the turn
    is produced, or a degraded turn becomes unattributable — you could no
    longer tell a register regression caused by the fallback model from one
    caused by it suddenly reasoning when the primary did not.

    Translation is each adapter's job, and the two APIs spell it differently:
    Anthropic takes `thinking: {type: disabled | adaptive}`, OpenAI takes
    `reasoning_effort: none | <effort>`.
    """

    #: Deliberately "none", not "off": YAML 1.1 parses a bare `off` as the
    #: boolean False, so a config author writing `reasoning: off` would silently
    #: get False and the first live call would raise. "none" has no such
    #: reinterpretation, and it matches OpenAI's own `reasoning_effort` value.
    none = "none"
    auto = "auto"


class Task(enum.StrEnum):
    """Routing is per task, not per tenant (§2.6).

    Per-tenant provider choice doubles the eval matrix and turns every prompt
    into two prompts to maintain.
    """

    answer_composition = "answer_composition"
    slot_extraction = "slot_extraction"
    #: §19.3's second figure gate — is this number labelled the way the
    #: material labels it? Routed separately from extraction because it is on
    #: the customer-facing path and its latency is a product cost, not a
    #: background one.
    figure_audit = "figure_audit"
    query_rewriting = "query_rewriting"
    eval_grading = "eval_grading"
    embedding = "embedding"


class Role(enum.StrEnum):
    user = "user"
    assistant = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    # True when this came from the failover provider. §2.6 requires the turn to
    # be logged as degraded so a quality regression is attributable to the
    # fallback path rather than blamed on a prompt change.
    degraded: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class Embedding:
    """Vectors plus what they cost, the same shape `Completion` already had.

    Embedding was the one provider call that returned no usage, which is
    exactly why embedding spend was invisible in the ledger: `embedding_call`
    existed as a `UsageKind` and nothing could write a row worth reading,
    because there was no token count to price.

    `input_tokens` covers every text in the batch — providers bill an embedding
    request whole, and splitting it per text would invent a division the
    invoice does not make.
    """

    vectors: list[Vector]
    provider: str
    model: str
    input_tokens: int = 0


@dataclass(frozen=True)
class UsageEvent:
    """What the router hands its usage sink after every completion.

    Deliberately not a database write. The router has no tenant and no session;
    the caller owns metering, which keeps this package free of a dependency on
    the tenancy layer and keeps the ledger write inside the caller's
    transaction.
    """

    task: Task
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    degraded: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        cache_blocks: Sequence[str],
        max_tokens: int,
        reasoning: str = Reasoning.auto,
        # Provider-native and passed through verbatim: the two vendors do not
        # share a scale, and claude-haiku-4-5 rejects the parameter outright.
        # Config sets a value valid for that model, or leaves it unset.
        effort: str | None = None,
        temperature: float | None = None,
    ) -> Completion: ...

    async def embed(
        self, *, model: str, texts: Sequence[str], dimensions: int
    ) -> Embedding: ...


class LLMError(Exception):
    """Base for everything this package raises."""


class ProviderUnavailable(LLMError):
    """Transient and someone else's fault: rate limit, 5xx, timeout.

    The only error that earns a failover. Adapters translate into this
    deliberately narrowly (Task 13) — a 400 must not become one.
    """


class AllProvidersUnavailable(ProviderUnavailable):
    """Every candidate for the task is failing or breaker-open.

    The router raises rather than inventing a reply. Scripted fallback is the
    orchestrator's decision (Task 14), because only it knows the conversation.
    """


class ProviderRequestError(LLMError):
    """The provider rejected the request: 4xx that is not a rate limit.

    Deliberately *not* a `ProviderUnavailable`, so it never triggers failover
    and never gets retried. A 400 is our bug; sending the same malformed body
    to a second provider hides it behind a fallback and doubles the cost of
    every broken call. A 401 is a missing key, which failover cannot fix either.
    """


class NoFailoverConfigured(LLMError):
    """A failover was demanded for a task that must not have one.

    Embeddings, in practice (§7.3): vectors only compare within one model and
    dimension, so a fallback provider returns a different space and retrieval
    quality collapses without anything raising.
    """


class UnknownTask(LLMError):
    """The task, or a provider it names, is absent from routing config.

    A config error, never a silent default. Falling back to some default model
    would mean an unreviewed model answering customers.
    """
