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

from moc.llm.base import Completion, Message, Vector


class FakeProvider:
    """Implements the LLMProvider protocol. Deterministic by construction."""

    def __init__(
        self,
        name: str,
        *,
        text: str = "",
        fail_with: Exception | None = None,
        fail_times: int | None = None,
        embedding_dimensions: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        stop_reason: str | None = None,
    ) -> None:
        self.name = name
        self.text = text
        #: Raised instead of answering. Set to a ProviderUnavailable to exercise
        #: failover; set to anything else to prove failover does *not* happen.
        self.fail_with = fail_with
        #: How many calls fail before recovering. None means "until changed",
        #: which is what a breaker test needs.
        self.fail_times = fail_times
        self.embedding_dimensions = embedding_dimensions
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def _maybe_fail(self) -> None:
        if self.fail_with is None:
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
            }
        )
        self._maybe_fail()
        return Completion(
            text=self.text,
            provider=self.name,
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            stop_reason=self.stop_reason,
        )

    async def embed(
        self, *, model: str, texts: Sequence[str], dimensions: int
    ) -> list[Vector]:
        self.calls.append(
            {"kind": "embed", "model": model, "texts": list(texts), "dimensions": dimensions}
        )
        self._maybe_fail()
        # Distinct but deterministic vectors: identical ones would let a bug
        # that returns the same embedding for every chunk pass unnoticed.
        return [[float(index)] * dimensions for index, _ in enumerate(texts)]
