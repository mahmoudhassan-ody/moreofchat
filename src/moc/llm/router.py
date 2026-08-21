"""Task-level routing with cross-provider failover — design §2.6.

The router picks a provider and model for a task, fails over on transient
provider errors, and trips a per-provider circuit breaker so a vendor incident
costs one round of timeouts rather than every request until someone notices.

Three things it deliberately does **not** do:

- **Compose a fallback reply.** When everything is down it raises. Only the
  orchestrator knows the conversation and the script, so scripted fallback is
  its call (Task 14). A router that invented text would be composing customer
  replies from a layer with no access to the tenant's script.
- **Write to the usage ledger.** It emits a `UsageEvent` to an optional sink.
  The caller owns the tenant and the transaction; keeping the write there means
  the ledger row commits with the rest of the turn rather than separately.
- **Decide whether failing over mid-conversation is acceptable.** §2.6 says to
  prefer a scripted fallback for a turn when the two providers differ visibly
  in register, and to switch provider at conversation boundaries. The router
  has no conversation, so that judgement also belongs to the orchestrator.

Every value — models, thresholds, the reset window, per-task token ceilings —
comes from `llm/routing`. Nothing lexical or numeric is written here (§19).
"""

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from moc.llm.base import (
    AllProvidersUnavailable,
    Completion,
    Embedding,
    LLMProvider,
    Message,
    NoFailoverConfigured,
    ProviderUnavailable,
    Reasoning,
    Task,
    UnknownTask,
    UsageEvent,
)

UsageSink = Callable[[UsageEvent], Awaitable[None]]


@dataclass
class _Breaker:
    """One provider's failure state.

    Consecutive failures only: a success resets the count, because five
    failures spread across an hour of healthy traffic is not an outage and
    should not open a circuit.
    """

    threshold: int
    reset_seconds: float
    failures: int = 0
    opened_at: float | None = None

    def is_open(self, now: float) -> bool:
        if self.opened_at is None:
            return False
        if now - self.opened_at >= self.reset_seconds:
            # Half-open: let one call through. It either succeeds and closes the
            # circuit, or fails and re-opens it.
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = now


class Router:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        providers: dict[str, LLMProvider],
        clock: Callable[[], float] = time.monotonic,
        usage_sink: UsageSink | None = None,
    ) -> None:
        self._config = config
        self._providers = providers
        self._clock = clock
        self._usage_sink = usage_sink
        breaker = config["breaker"]
        self._breakers = {
            name: _Breaker(
                threshold=breaker["failure_threshold"],
                reset_seconds=breaker["reset_seconds"],
            )
            for name in providers
        }
        self._require_every_configured_provider()

    def _require_every_configured_provider(self) -> None:
        """Fail at construction, not on first use.

        A failover provider that was never wired in is discovered during the
        outage that first needs it otherwise — the worst moment to learn about
        a wiring error, and one a passing test suite would not have caught.
        """
        for name, spec in self._config["tasks"].items():
            for role in ("primary", "failover"):
                candidate = spec.get(role)
                if candidate and candidate["provider"] not in self._providers:
                    raise UnknownTask(
                        f"task {name!r} routes its {role} to provider "
                        f"{candidate['provider']!r}, which was not supplied to the router"
                    )

    # ─────────────────────────── completion ───────────────────────────

    async def complete(
        self,
        *,
        task: Task | str,
        messages: Sequence[Message],
        system: str | None = None,
        cache_blocks: Sequence[str] = (),
        exclude_provider: str | None = None,
    ) -> Completion:
        """Run `task` against its primary, falling back to its failover.

        Raises `AllProvidersUnavailable` when every candidate is failing or
        breaker-open. Any error that is not a `ProviderUnavailable` propagates
        untouched — a malformed request is our bug, and failing over would hide
        it while doubling the cost of every broken call.

        `exclude_provider` removes a provider from consideration entirely, for
        the eval judge (§5.2): a provider must never grade its own output, and
        expressing that as "prefer the other one" would let failover launder a
        violation into a verdict that looks exactly like a valid one. Excluding
        it means there is no ordering of candidates that reaches it. When every
        remaining candidate is down the judge gets an exception, which is the
        correct outcome — a grade from the answering provider is worse than no
        grade, because a missing grade is visible.
        """
        spec = self._task_spec(task)
        candidates = [spec["primary"]]
        if spec.get("failover"):
            candidates.append(spec["failover"])
        if exclude_provider is not None:
            candidates = [c for c in candidates if c["provider"] != exclude_provider]
            if not candidates:
                raise AllProvidersUnavailable(
                    f"{task} has no candidate outside provider {exclude_provider!r}"
                )

        max_tokens = spec["max_tokens"]
        last_error: Exception | None = None

        for index, candidate in enumerate(candidates):
            provider = self._provider(candidate["provider"], task)
            breaker = self._breakers[provider.name]
            if breaker.is_open(self._clock()):
                last_error = AllProvidersUnavailable(
                    f"breaker open for {provider.name}"
                )
                continue
            try:
                completion = await provider.complete(
                    model=candidate["model"],
                    messages=messages,
                    system=system,
                    cache_blocks=cache_blocks,
                    max_tokens=max_tokens,
                    # Per candidate, not per task: the primary and the failover
                    # are different models with different controls, and §2.6
                    # wants the degraded turn produced the same way regardless.
                    reasoning=candidate.get("reasoning", Reasoning.auto),
                    effort=candidate.get("effort"),
                    temperature=candidate.get("temperature"),
                )
            except ProviderUnavailable as exc:
                breaker.record_failure(self._clock())
                last_error = exc
                continue

            breaker.record_success()
            # degraded means "not the first candidate we were willing to use" —
            # §2.6 logs the turn so a quality regression is attributable to the
            # fallback path. Measured after exclusion on purpose: a judge that
            # deliberately routed around the answering provider made a routing
            # decision, not an incident, and flagging it degraded would fill the
            # ledger with outages that never happened.
            result = (
                completion if index == 0 else _mark_degraded(completion)
            )
            await self._emit(task, result)
            return result

        raise AllProvidersUnavailable(
            f"no provider available for {task}: {last_error}"
        ) from last_error

    # ─────────────────────────── embeddings ───────────────────────────

    async def embed(
        self,
        *,
        texts: Sequence[str],
        task: Task | str = Task.embedding,
        force_failover: bool = False,
    ) -> Embedding:
        """Embed `texts`. Never fails over — §7.3.

        A second embedding provider returns vectors from a different space.
        Retrieval quality would collapse and nothing would raise, which is the
        worst failure mode available: silent, gradual, and invisible in every
        metric except the ones a customer feels.
        """
        spec = self._task_spec(task)
        if force_failover or spec.get("failover"):
            raise NoFailoverConfigured(
                f"{task} has no failover by design (§7.3): vectors are comparable "
                f"only within one model and dimension"
            )
        candidate = spec["primary"]
        provider = self._provider(candidate["provider"], task)
        return await provider.embed(
            model=candidate["model"],
            texts=texts,
            dimensions=candidate["dimensions"],
        )

    # ─────────────────────────── internals ───────────────────────────

    def _task_spec(self, task: Task | str) -> dict[str, Any]:
        try:
            return self._config["tasks"][str(task)]
        except KeyError:
            raise UnknownTask(
                f"{task!s} is not in llm/routing. A missing task is a config error, "
                f"never a default model — an unreviewed model must not answer customers."
            ) from None

    def _provider(self, name: str, task: Task | str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise UnknownTask(
                f"{task!s} routes to provider {name!r}, which was not supplied to the router"
            ) from None

    async def _emit(self, task: Task | str, completion: Completion) -> None:
        if self._usage_sink is None:
            return
        await self._usage_sink(
            UsageEvent(
                task=task,
                provider=completion.provider,
                model=completion.model,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cached_tokens=completion.cached_tokens,
                degraded=completion.degraded,
            )
        )


def _mark_degraded(completion: Completion) -> Completion:
    return Completion(
        text=completion.text,
        provider=completion.provider,
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cached_tokens=completion.cached_tokens,
        degraded=True,
        stop_reason=completion.stop_reason,
    )


__all__ = ["AllProvidersUnavailable", "Router", "UsageSink"]
