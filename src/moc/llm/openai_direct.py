"""OpenAI Chat Completions and Embeddings adapter — design §2.

No SDK, same as the Anthropic adapter: httpx only, so unit tests assert the
bytes on the wire and the import contract has nothing to forbid.

Two behaviours here were learned from live calls on 2026-08-17 rather than
assumed, and both would have shipped as bugs otherwise:

1. **`max_completion_tokens`, not `max_tokens`.** The gpt-5.x family rejects
   `max_tokens` outright with a 400 (`unsupported_parameter`). Under the error
   taxonomy that is a `ProviderRequestError`, so it would *not* have failed
   over — every answer-composition turn would simply have failed.
2. **Cached tokens are counted inside `prompt_tokens`.** Anthropic reports them
   outside `input_tokens`. Both adapters therefore normalize to the same
   convention: `input_tokens` is what is billed at the full rate and
   `cached_tokens` is what came from cache, so `usage_ledger` rows are
   comparable across providers.

Caching is automatic here — there is no `cache_control` to send. OpenAI matches
on prompt prefix, so cache blocks are simply placed first in the system message.
Design §2.6 warns not to assume cache-hit economics carry across providers;
they are measured separately per provider in the live smoke tests.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from moc.llm.base import Completion, Message, Vector
from moc.llm.http import Sleep, build_client, request_with_retries

BASE_URL = "https://api.openai.com"
CHAT_PATH = "/v1/chat/completions"
EMBEDDINGS_PATH = "/v1/embeddings"


class OpenAIDirect:
    name = "openai"

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
                "authorization": f"Bearer {api_key}",
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
    ) -> Completion:
        wire: list[dict[str, str]] = []
        preamble = _system_message(system, cache_blocks)
        if preamble:
            wire.append({"role": "system", "content": preamble})
        wire.extend({"role": str(m.role), "content": m.content} for m in messages)

        body = await request_with_retries(
            self._client,
            provider=self.name,
            path=CHAT_PATH,
            payload={
                "model": model,
                # Not max_tokens — see the module docstring.
                "max_completion_tokens": max_tokens,
                "messages": wire,
            },
            http=self._http,
            sleep=self._sleep,
        )
        return _to_completion(body, model)

    async def embed(
        self, *, model: str, texts: Sequence[str], dimensions: int
    ) -> list[Vector]:
        """Matryoshka truncation to `dimensions` (§7.3: 1024, from config).

        The provider does the truncation, so the returned vector is already the
        configured width — the alternative, slicing client-side, produces a
        differently-normalized vector and quietly degrades similarity.
        """
        body = await request_with_retries(
            self._client,
            provider=self.name,
            path=EMBEDDINGS_PATH,
            payload={"model": model, "input": list(texts), "dimensions": dimensions},
            http=self._http,
            sleep=self._sleep,
        )
        # Sort by index rather than trusting order: a reordered response would
        # silently attach every chunk to the wrong vector.
        rows = sorted(body.get("data") or [], key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in rows]

    async def aclose(self) -> None:
        await self._client.aclose()


def _system_message(system: str | None, cache_blocks: Sequence[str]) -> str:
    """Cacheable blocks first — caching is prefix-matched, so order decides hits."""
    return "\n\n".join([*cache_blocks, *( [system] if system else [] )])


def _to_completion(body: dict[str, Any], model: str) -> Completion:
    usage = body.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    choices = body.get("choices") or [{}]
    message = choices[0].get("message") or {}
    return Completion(
        text=message.get("content") or "",
        provider=OpenAIDirect.name,
        model=body.get("model") or model,
        # prompt_tokens includes cached; subtract so both adapters agree that
        # input_tokens means "billed at full rate".
        input_tokens=usage.get("prompt_tokens", 0) - cached,
        output_tokens=usage.get("completion_tokens", 0),
        cached_tokens=cached,
        stop_reason=choices[0].get("finish_reason"),
    )
