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


class Task(enum.StrEnum):
    """Routing is per task, not per tenant (§2.6).

    Per-tenant provider choice doubles the eval matrix and turns every prompt
    into two prompts to maintain.
    """

    answer_composition = "answer_composition"
    slot_extraction = "slot_extraction"
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
    ) -> Completion: ...

    async def embed(
        self, *, model: str, texts: Sequence[str], dimensions: int
    ) -> list[Vector]: ...


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
