"""Shared HTTP behaviour for the direct adapters.

Both providers need the same three things and disagree about everything else,
so this holds the agreement: how a status code becomes an exception, when to
retry, and how long to wait.

**Retries sit below the breaker** (router.py). A retry absorbs a blip; the
breaker absorbs an outage. Retrying forever against a provider that is down
means every request pays three timeouts before failing over, which is slower
than not retrying at all.

**Only transient faults retry.** 429 and 5xx and connection errors, never a
4xx — resending a malformed body just buys the same 400 twice.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from moc.llm.base import ProviderRequestError, ProviderUnavailable

Sleep = Callable[[float], Awaitable[None]]

_CLIENT_ERROR = 400
_RATE_LIMITED = 429
_SERVER_ERROR = 500


def build_client(
    *,
    base_url: str,
    headers: dict[str, str],
    http: dict[str, Any],
    transport: httpx.BaseTransport | None = None,
) -> httpx.AsyncClient:
    """One client per adapter, with connect and read timeouts separated.

    A DNS or TLS stall and a slow generation are different faults with very
    different acceptable durations; one timeout number cannot express both.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(
            connect=http.get("connect_timeout_seconds"),
            read=http.get("read_timeout_seconds"),
            write=http.get("read_timeout_seconds"),
            pool=http.get("connect_timeout_seconds"),
        ),
        transport=transport,
    )


#: Vendor error codes that mean "this account is out of budget", read from the
#: body's own `type`/`code` field. Preferred over the prose below because a
#: field is a contract and a sentence is copy: OpenAI can reword
#: `insufficient_quota`'s message tomorrow and this still holds.
_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",       # OpenAI
        "billing_hard_limit_reached",  # OpenAI, spend cap rather than balance
        "quota_exceeded",
    }
)

#: **String-matching a vendor's prose, and it will need revisiting when either
#: of them rewords it.** Stated plainly because it is the weak half of this
#: rule and reads like the strong half.
#:
#: It is here because Anthropic gives us nothing better. Captured from a live
#: exhausted account on 2026-08-25, the body is `400` with
#: `error.type: "invalid_request_error"` — the *generic* 400 type, identical to
#: the one a malformed `max_tokens` produces. There is no quota-specific code
#: to match, so for that vendor the sentence is not a fallback, it is the only
#: discriminator there is.
#:
#: Kept narrow for that reason. Every phrase here is one no malformed request
#: would contain, and the tests that assert a real 400 still fails hard are the
#: thing that notices when that stops being true.
_QUOTA_PHRASES = (
    "usage limit",       # Anthropic: "You have reached your specified API usage limits."
    "credit balance",    # Anthropic: "Your credit balance is too low"
    "exceeded your current quota",
    "billing",
)


def _is_out_of_budget(response: httpx.Response) -> bool:
    """Whether this 4xx is a spend cap rather than a bad request.

    A spend cap is exactly what failover exists for and it arrives wearing the
    status code of our own bug — Anthropic reports it as 400, not 429. Before
    this, a configured and healthy failover was never reached: the account hit
    its limit mid-run on 2026-08-25 and every turn failed hard while OpenAI sat
    idle.
    """
    try:
        body = response.json()
    except ValueError:
        return False
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return False
    # The field first. It is a contract; the sentence is copy.
    for key in ("code", "type"):
        value = error.get(key)
        if isinstance(value, str) and value in _QUOTA_CODES:
            return True
    message = error.get("message")
    if not isinstance(message, str):
        return False
    lowered = message.lower()
    return any(phrase in lowered for phrase in _QUOTA_PHRASES)


def raise_for_status(response: httpx.Response, provider: str) -> None:
    if response.status_code < _CLIENT_ERROR:
        return
    detail = _detail(response)
    if response.status_code == _RATE_LIMITED or response.status_code >= _SERVER_ERROR:
        raise ProviderUnavailable(f"{provider} returned {response.status_code}: {detail}")
    if _is_out_of_budget(response):
        # Not our bug, and the one 4xx a second provider can actually answer.
        raise ProviderUnavailable(f"{provider} returned {response.status_code}: {detail}")
    raise ProviderRequestError(f"{provider} returned {response.status_code}: {detail}")


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or body)[:200]


async def request_with_retries(
    client: httpx.AsyncClient,
    *,
    provider: str,
    path: str,
    payload: dict[str, Any],
    http: dict[str, Any],
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """POST `payload`, retrying transient failures with jittered backoff.

    Jitter matters more than it looks: without it, every worker that failed on
    the same provider blip retries at the same instant and reproduces the blip.
    """
    attempts = http.get("max_retries", 0) + 1
    base = http.get("backoff_base_seconds", 0)
    jitter = http.get("backoff_jitter_seconds", 0)
    rng = rng or random.Random(  # noqa: S311 - backoff jitter, not security
        provider
    )

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.post(path, json=payload)
            raise_for_status(response, provider)
            return response.json()
        except ProviderRequestError:
            # Our bug. Retrying spends money to receive the same rejection.
            raise
        except (ProviderUnavailable, httpx.TransportError) as exc:
            last = exc
            if attempt == attempts - 1:
                break
            await sleep(base * (2**attempt) + rng.random() * jitter)

    if isinstance(last, httpx.TransportError):
        raise ProviderUnavailable(f"{provider} transport error: {last}") from last
    raise last  # type: ignore[misc]
