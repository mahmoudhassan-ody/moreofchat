"""A scriptable LLMProvider for tests — no network, ever.

This is what Tasks 12-15 test against. Flow logic, failover behaviour and the
script engine are all testable without spending money or depending on a vendor
being up, and a CI run that needs neither key is a CI run that works on a fork.

Three levers: canned responses, injectable failures, and a call log. The call
log is the interesting one — most bugs in this layer are "the wrong model was
asked" or "the system prompt was dropped", and both are invisible unless the
arguments are recorded.
"""

from collections.abc import Sequence
from typing import Any

from moc.llm.base import Completion, Embedding, Message, Reasoning


class FakeProvider:
    """Implements the LLMProvider protocol. Deterministic by construction."""

    def __init__(
        self,
        name: str,
        *,
        text: str = "",
        fail_with: Exception | None = None,
        fail_times: int | None = None,
        fail_kinds: tuple[str, ...] = ("complete", "embed"),
        embedding_dimensions: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        stop_reason: str | None = None,
        text_by_model: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.text = text
        #: Per-model overrides, for a turn that makes more than one completion
        #: call. Keyed by model because that is what a provider is actually
        #: told — a task is a routing concept and never reaches this layer, so
        #: keying on one here would test a mapping the provider cannot see.
        self.text_by_model = text_by_model or {}
        #: Raised instead of answering. Set to a ProviderUnavailable to exercise
        #: failover; set to anything else to prove failover does *not* happen.
        self.fail_with = fail_with
        #: How many calls fail before recovering. None means "until changed",
        #: which is what a breaker test needs.
        self.fail_times = fail_times
        #: Which call kinds `fail_with` applies to. A completion outage and an
        #: embedding outage are different incidents — §7.3 has a dead embedding
        #: endpoint degrading search to Meilisearch rather than failing the turn
        #: — so a test that means one must be able to say so.
        self.fail_kinds = fail_kinds
        self.embedding_dimensions = embedding_dimensions
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens
        self.cache_write_tokens = cache_write_tokens
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def _maybe_fail(self, kind: str) -> None:
        if self.fail_with is None or kind not in self.fail_kinds:
            return
        if self.fail_times is None:
            raise self.fail_with
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_with

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        system: str | None,
        cache_blocks: Sequence[str],
        max_tokens: int,
        reasoning: str = Reasoning.auto,
        effort: str | None = None,
        temperature: float | None = None,
    ) -> Completion:
        # Recorded before the failure check: a call that failed still happened,
        # and the breaker tests count exactly that.
        self.calls.append(
            {
                "kind": "complete",
                "model": model,
                "messages": list(messages),
                "system": system,
                "cache_blocks": list(cache_blocks),
                "max_tokens": max_tokens,
                "reasoning": reasoning,
                "effort": effort,
                "temperature": temperature,
            }
        )
        self._maybe_fail("complete")
        return Completion(
            text=self.text_by_model.get(model, self.text),
            provider=self.name,
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            cache_write_tokens=self.cache_write_tokens,
            stop_reason=self.stop_reason,
        )

    async def embed(
        self, *, model: str, texts: Sequence[str], dimensions: int
    ) -> Embedding:
        self.calls.append(
            {"kind": "embed", "model": model, "texts": list(texts), "dimensions": dimensions}
        )
        self._maybe_fail("embed")
        # Distinct but deterministic vectors: identical ones would let a bug
        # that returns the same embedding for every chunk pass unnoticed.
        return Embedding(
            vectors=[[float(index)] * dimensions for index, _ in enumerate(texts)],
            provider=self.name,
            model=model,
            # Proportional to the input, so a test asserting the ledger saw
            # the batch cannot pass on a hard-coded constant.
            input_tokens=sum(len(text) for text in texts),
        )
