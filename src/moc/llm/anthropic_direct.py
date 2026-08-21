"""Anthropic Messages API adapter — design §2 (direct first-party API).

No SDK. The adapter speaks HTTP directly, which keeps the dependency surface at
httpx and lets the unit tests assert the exact bytes we send rather than mock a
client object. The import-linter contract forbidding `anthropic` outside this
package is satisfied trivially: nothing imports it at all.

Response shapes here were captured from a real call on 2026-08-17, not recalled.

**Token accounting is normalized, and the reason matters.** Anthropic reports
`input_tokens` already excluding cache reads, and reports the cache read
separately as `cache_read_input_tokens`. OpenAI folds cached tokens *into*
`prompt_tokens`. Passing both through raw would make a cached OpenAI turn look
far more expensive than an identical Anthropic turn in `usage_ledger`, and
cross-provider cost comparison is what that table exists for. The convention
across both adapters: `input_tokens` means tokens billed at the full rate,
`cached_tokens` means tokens served from cache.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from moc.llm.base import Completion, Embedding, Message, Reasoning
from moc.llm.http import Sleep, build_client, request_with_retries

# The one endpoint. Design §2.4: a region change (Bedrock eu-central-1, Vertex
# EU) is a new adapter file next to this one, never an edit here — the auth,
# the request envelope and the caching semantics all differ.
BASE_URL = "https://api.anthropic.com"
MESSAGES_PATH = "/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicDirect:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        http: dict[str, Any],
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._http = http
        self._sleep = sleep
        self._client = build_client(
            base_url=BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            http=http,
            transport=transport,
        )

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
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": str(m.role), "content": m.content} for m in messages],
            "thinking": {"type": _thinking_type(reasoning)},
        }
        blocks = _system_blocks(system, cache_blocks)
        if blocks:
            payload["system"] = blocks
        if effort is not None:
            # Omitted entirely when unset: claude-haiku-4-5 answers a request
            # carrying this key with a 400, so a default here would break every
            # slot-extraction turn.
            payload["output_config"] = {"effort": effort}
        if temperature is not None:
            # Omitted when unset, for effort's reason from the other
            # direction: claude-sonnet-5 answers a request carrying this key
            # with `temperature is deprecated for this model` and a 400, while
            # claude-haiku-4-5 accepts it and returns 200. Measured
            # 2026-08-20. Which models take it is config's problem, not this
            # adapter's.
            payload["temperature"] = temperature

        body = await request_with_retries(
            self._client,
            provider=self.name,
            path=MESSAGES_PATH,
            payload=payload,
            http=self._http,
            sleep=self._sleep,
        )
        return _to_completion(body, model)

    async def embed(
        self, *, model: str, texts: Sequence[str], dimensions: int
    ) -> Embedding:
        raise NotImplementedError(
            "Anthropic exposes no embedding API. §7.3 pins embeddings to one "
            "provider and one dimension count anyway — a second embedding space "
            "would collapse retrieval quality without erroring."
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _thinking_type(reasoning: str) -> str:
    """Map the neutral control onto Anthropic's spelling.

    claude-sonnet-5 thinks by default, so "off" has to be sent explicitly —
    omitting the key is not the same thing as disabling it.
    """
    match Reasoning(reasoning):
        case Reasoning.none:
            return "disabled"
        case Reasoning.auto:
            return "adaptive"


def _system_blocks(system: str | None, cache_blocks: Sequence[str]) -> list[dict[str, Any]]:
    """Cacheable blocks first, marked; the volatile persona last, unmarked.

    Order is the whole mechanism. A cache hit requires an identical prefix, so
    anything that changes per turn must sit *after* everything cached — put the
    persona first and every block behind it misses (design §10).
    """
    # Anthropic allows at most four cache_control breakpoints and 400s beyond
    # that. Fusion returns five passages on a normal grounded turn, so the
    # limit is reached by ordinary traffic rather than by an unusual request —
    # this surfaced as four eval cases erroring out before composition ran.
    #
    # The earliest blocks get the breakpoints because the prefix is what a
    # cache hit requires; the rest are sent uncached. A cache miss costs
    # money, a 400 costs the answer.
    limit = _cache_breakpoints()
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": block}
        | ({"cache_control": {"type": "ephemeral"}} if index < limit else {})
        for index, block in enumerate(cache_blocks)
    ]
    if system:
        blocks.append({"type": "text", "text": system})
    return blocks


def _cache_breakpoints() -> int:
    from moc.config_store import load

    return load("llm/routing")["anthropic_cache_breakpoints"]


def _to_completion(body: dict[str, Any], model: str) -> Completion:
    usage = body.get("usage") or {}
    text = "".join(
        block.get("text", "")
        for block in body.get("content") or []
        if block.get("type") == "text"
    )
    return Completion(
        text=text,
        provider=AnthropicDirect.name,
        model=body.get("model") or model,
        # Already excludes cache reads — verified against a live call.
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=usage.get("cache_read_input_tokens", 0),
        stop_reason=body.get("stop_reason"),
    )
