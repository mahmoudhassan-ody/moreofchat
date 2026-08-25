"""Direct API adapters, mocked at the HTTP layer — never at the SDK.

There is no SDK: the adapters speak HTTP with httpx, so these transports assert
the exact bytes we send and the exact shapes we parse. A provider changing a
response field breaks a test rather than being mocked away.

Every fixture below is a trimmed copy of a real response captured on
2026-08-17 while resolving the model ids, so the shapes are observed rather
than remembered.
"""

import httpx
import pytest

from moc.llm.anthropic_direct import AnthropicDirect
from moc.llm.base import Message, ProviderRequestError, ProviderUnavailable
from moc.llm.openai_direct import OpenAIDirect

MESSAGES = [Message(role="user", content="المصاريف كام؟")]

# ── Captured 2026-08-17 from api.anthropic.com/v1/messages ────────────────────
ANTHROPIC_OK = {
    "type": "message",
    "model": "claude-sonnet-5",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "الرسوم 1400 جنيه"}],
    "usage": {
        "input_tokens": 11,
        "output_tokens": 4,
        "cache_creation_input_tokens": 0,
        # Anthropic reports cache reads *outside* input_tokens.
        "cache_read_input_tokens": 4564,
    },
}

# ── Captured 2026-08-17 from api.openai.com/v1/chat/completions ───────────────
OPENAI_OK = {
    "model": "gpt-5.5-2026-04-23",
    "choices": [
        {"message": {"content": "الرسوم 1400 جنيه"}, "finish_reason": "stop"},
    ],
    "usage": {
        "prompt_tokens": 2535,
        "completion_tokens": 17,
        # OpenAI reports cached tokens *inside* prompt_tokens.
        "prompt_tokens_details": {"cached_tokens": 2000},
        "completion_tokens_details": {"reasoning_tokens": 7},
    },
}

OPENAI_EMBEDDING = {
    "model": "text-embedding-3-large",
    "data": [{"embedding": [0.1] * 1024, "index": 0}],
    "usage": {"prompt_tokens": 15, "total_tokens": 15},
}

NO_RETRY = {"max_retries": 0, "backoff_base_seconds": 0, "backoff_jitter_seconds": 0}


def transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def responder(payload, status=200, capture=None):
    def handle(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        return httpx.Response(status, json=payload)

    return handle


def anthropic(handler, **overrides) -> AnthropicDirect:
    return AnthropicDirect(
        api_key="test-key",
        http={**NO_RETRY, **overrides},
        transport=transport(handler),
    )


def openai(handler, **overrides) -> OpenAIDirect:
    return OpenAIDirect(
        api_key="test-key",
        http={**NO_RETRY, **overrides},
        transport=transport(handler),
    )


# ─────────────────────────── response mapping ───────────────────────────


async def test_maps_anthropic_response_to_completion():
    result = await anthropic(responder(ANTHROPIC_OK)).complete(
        model="claude-sonnet-5", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert result.text == "الرسوم 1400 جنيه"
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet-5"
    assert result.output_tokens == 4
    assert result.stop_reason == "end_turn"
    assert result.degraded is False


async def test_maps_openai_response_to_completion():
    result = await openai(responder(OPENAI_OK)).complete(
        model="gpt-5.5", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert result.text == "الرسوم 1400 جنيه"
    assert result.provider == "openai"
    assert result.output_tokens == 17
    assert result.stop_reason == "stop"


async def test_anthropic_joins_multiple_text_blocks():
    payload = {**ANTHROPIC_OK, "content": [
        {"type": "text", "text": "الرسوم "},
        {"type": "thinking", "thinking": "ignore me"},
        {"type": "text", "text": "1400 جنيه"},
    ]}
    result = await anthropic(responder(payload)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert result.text == "الرسوم 1400 جنيه"


async def test_a_thinking_only_response_maps_to_empty_text():
    """Observed live on 2026-08-17: claude-sonnet-5 thinks by default.

    With a small max_tokens the whole budget goes to thinking and no text block
    is produced — output_tokens is nonzero, stop_reason is max_tokens, and text
    is empty. The adapter reports that faithfully rather than raising, because
    only the orchestrator can decide what a customer sees. Task 14 must treat
    empty text as a failed turn: an empty WhatsApp message is worse than a
    scripted fallback.
    """
    payload = {
        **ANTHROPIC_OK,
        "stop_reason": "max_tokens",
        "content": [{"type": "thinking", "thinking": "", "signature": "..."}],
        "usage": {"input_tokens": 39, "output_tokens": 32, "cache_read_input_tokens": 0},
    }
    result = await anthropic(responder(payload)).complete(
        model="claude-sonnet-5", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=32
    )
    assert result.text == ""
    assert result.output_tokens == 32
    assert result.stop_reason == "max_tokens"


async def test_openai_null_content_becomes_empty_string():
    """A refusal or a length stop can return null content; None would crash callers."""
    payload = {**OPENAI_OK, "choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    result = await openai(responder(payload)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert result.text == ""


# ─────────────────────────── cached-token accounting ───────────────────────────


async def test_reports_cached_tokens_from_the_cache_read_field():
    result = await anthropic(responder(ANTHROPIC_OK)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert result.cached_tokens == 4564


async def test_input_tokens_exclude_cached_tokens_on_both_providers():
    """Normalized on purpose: input_tokens means 'billed at full rate'.

    The providers disagree. Anthropic's input_tokens already excludes cache
    reads; OpenAI's prompt_tokens includes them. Passing both through raw would
    make a cached OpenAI turn look far more expensive than the identical
    Anthropic turn in usage_ledger, and the whole point of that table is
    cross-provider cost comparison.
    """
    a = await anthropic(responder(ANTHROPIC_OK)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    o = await openai(responder(OPENAI_OK)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert (a.input_tokens, a.cached_tokens) == (11, 4564)
    assert (o.input_tokens, o.cached_tokens) == (2535 - 2000, 2000)


async def test_missing_cache_fields_default_to_zero():
    payload = {**OPENAI_OK, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    result = await openai(responder(payload)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert (result.input_tokens, result.cached_tokens) == (10, 0)


# ─────────────────────────── request shape ───────────────────────────


async def test_openai_sends_max_completion_tokens_not_max_tokens():
    """gpt-5.x rejects max_tokens outright — a 400 observed while resolving ids."""
    import json

    captured = []
    await openai(responder(OPENAI_OK, capture=captured)).complete(
        model="gpt-5.5", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    body = json.loads(captured[0].content)
    assert body["max_completion_tokens"] == 16
    assert "max_tokens" not in body


async def test_anthropic_marks_cache_blocks_for_caching():
    """Design §10's primary cost lever. Explicit on Anthropic, automatic on OpenAI."""
    import json

    captured = []
    await anthropic(responder(ANTHROPIC_OK, capture=captured)).complete(
        model="m",
        messages=MESSAGES,
        system="persona",
        cache_blocks=["tenant script", "policy preamble"],
        max_tokens=16,
    )
    system = json.loads(captured[0].content)["system"]
    cached = [b for b in system if "cache_control" in b]
    assert [b["text"] for b in cached] == ["tenant script", "policy preamble"]
    assert system[-1]["text"] == "persona"
    assert "cache_control" not in system[-1]


async def test_openai_folds_cache_blocks_into_the_system_message():
    """No cache_control to send — OpenAI caches automatically on prefix match."""
    import json

    captured = []
    await openai(responder(OPENAI_OK, capture=captured)).complete(
        model="m", messages=MESSAGES, system="persona", cache_blocks=["script"], max_tokens=16
    )
    body = json.loads(captured[0].content)
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"].startswith("script")
    assert "persona" in body["messages"][0]["content"]


async def test_anthropic_sends_the_version_header():
    captured = []
    await anthropic(responder(ANTHROPIC_OK, capture=captured)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert captured[0].headers["anthropic-version"]
    assert captured[0].headers["x-api-key"] == "test-key"


async def test_openai_sends_a_bearer_token():
    captured = []
    await openai(responder(OPENAI_OK, capture=captured)).complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert captured[0].headers["authorization"] == "Bearer test-key"


# ─────────────────────────── error translation ───────────────────────────


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
async def test_translates_rate_limit_and_5xx_to_provider_unavailable(status):
    with pytest.raises(ProviderUnavailable):
        await anthropic(responder({"error": "nope"}, status=status)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_does_not_translate_a_4xx_to_provider_unavailable(status):
    """A malformed request is our bug. Failing over would hide it and double
    the cost of every broken call."""
    with pytest.raises(ProviderRequestError) as exc:
        await openai(responder({"error": {"message": "bad"}}, status=status)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )
    assert not isinstance(exc.value, ProviderUnavailable)


async def test_a_connection_error_is_provider_unavailable():
    def boom(request):
        raise httpx.ConnectError("dns failed")

    with pytest.raises(ProviderUnavailable):
        await anthropic(boom).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )


async def test_a_timeout_is_provider_unavailable():
    def slow(request):
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(ProviderUnavailable):
        await openai(slow).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )


async def test_the_error_message_carries_the_status_for_triage():
    with pytest.raises(ProviderRequestError, match="400"):
        await openai(responder({"error": {"message": "bad"}}, status=400)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )


# ─────────────────────────── retries ───────────────────────────


async def test_retries_transient_failures_then_succeeds():
    attempts = []

    def flaky(request):
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json=ANTHROPIC_OK)

    provider = AnthropicDirect(
        api_key="k",
        http={"max_retries": 2, "backoff_base_seconds": 0, "backoff_jitter_seconds": 0},
        transport=transport(flaky),
        sleep=_no_sleep,
    )
    result = await provider.complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
    )
    assert len(attempts) == 3
    assert result.text


async def test_does_not_retry_a_4xx():
    """Retrying our own bad request just spends money on the same 400."""
    attempts = []

    def bad(request):
        attempts.append(request)
        return httpx.Response(400, json={"error": {"message": "bad"}})

    provider = OpenAIDirect(
        api_key="k",
        http={"max_retries": 3, "backoff_base_seconds": 0, "backoff_jitter_seconds": 0},
        transport=transport(bad),
        sleep=_no_sleep,
    )
    with pytest.raises(ProviderRequestError):
        await provider.complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )
    assert len(attempts) == 1


async def test_gives_up_after_the_configured_retry_count():
    attempts = []

    def always_down(request):
        attempts.append(request)
        return httpx.Response(503, json={"error": "down"})

    provider = AnthropicDirect(
        api_key="k",
        http={"max_retries": 2, "backoff_base_seconds": 0, "backoff_jitter_seconds": 0},
        transport=transport(always_down),
        sleep=_no_sleep,
    )
    with pytest.raises(ProviderUnavailable):
        await provider.complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )
    assert len(attempts) == 3, "one attempt plus two retries"


async def _no_sleep(seconds: float) -> None:
    return None


# ─────────────────────────── embeddings ───────────────────────────


async def test_embeddings_truncate_to_the_configured_dimension():
    """1024 via Matryoshka, from config. Design §7.3."""
    import json

    captured = []
    embedding = await openai(responder(OPENAI_EMBEDDING, capture=captured)).embed(
        model="text-embedding-3-large", texts=["رسوم الساعة"], dimensions=1024
    )
    assert [len(v) for v in embedding.vectors] == [1024]
    assert json.loads(captured[0].content)["dimensions"] == 1024


async def test_embeddings_preserve_input_order():
    """The API may return data out of order; index is authoritative."""
    payload = {
        "model": "text-embedding-3-large",
        "data": [
            {"embedding": [2.0], "index": 1},
            {"embedding": [1.0], "index": 0},
        ],
        "usage": {"prompt_tokens": 4},
    }
    embedding = await openai(responder(payload)).embed(
        model="m", texts=["first", "second"], dimensions=1
    )
    assert embedding.vectors == [[1.0], [2.0]]


async def test_anthropic_has_no_embedding_api():
    """§7.3 pins embeddings to one provider; this makes the reason explicit."""
    with pytest.raises(NotImplementedError, match="embedding"):
        await anthropic(responder(ANTHROPIC_OK)).embed(
            model="m", texts=["x"], dimensions=1024
        )


# The Task 11 endpoint guard covers this file's counterpart — see
# test_no_endpoints_outside_llm.py, which runs in the same session and now has
# real endpoints in src/moc/llm/ to be exempt about.


# ─────────────────────────── reasoning control ───────────────────────────
#
# All four behaviours below were verified against the live APIs on 2026-08-17
# before being encoded here — see the module docstring's note on captured
# fixtures. The mocks assert the request we send; the live smoke tests assert
# the providers still accept it.


async def test_anthropic_reasoning_off_sends_disabled_thinking():
    """claude-sonnet-5 thinks by default; answer composition opts out (§2.5)."""
    import json

    captured = []
    await anthropic(responder(ANTHROPIC_OK, capture=captured)).complete(
        model="claude-sonnet-5",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
    )
    assert json.loads(captured[0].content)["thinking"] == {"type": "disabled"}


async def test_anthropic_reasoning_auto_sends_adaptive_thinking():
    import json

    captured = []
    await anthropic(responder(ANTHROPIC_OK, capture=captured)).complete(
        model="claude-opus-5",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="auto",
    )
    assert json.loads(captured[0].content)["thinking"] == {"type": "adaptive"}


async def test_anthropic_sends_effort_only_when_configured():
    """claude-haiku-4-5 rejects the effort parameter with a 400.

    Its routing entries carry no effort, so the adapter must omit the key
    entirely rather than send a default.
    """
    import json

    captured = []
    provider = anthropic(responder(ANTHROPIC_OK, capture=captured))
    await provider.complete(
        model="claude-haiku-4-5-20251001",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
    )
    assert "output_config" not in json.loads(captured[0].content)

    await provider.complete(
        model="claude-sonnet-5",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
        effort="medium",
    )
    assert json.loads(captured[1].content)["output_config"] == {"effort": "medium"}


async def test_openai_reasoning_off_sends_effort_none():
    """The symmetric control: failover must not change how the turn is produced."""
    import json

    captured = []
    await openai(responder(OPENAI_OK, capture=captured)).complete(
        model="gpt-5.6-sol",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
        effort="medium",
    )
    body = json.loads(captured[0].content)
    assert body["reasoning_effort"] == "none", "effort must not override reasoning: none"


async def test_openai_reasoning_auto_passes_the_configured_effort():
    import json

    captured = []
    await openai(responder(OPENAI_OK, capture=captured)).complete(
        model="gpt-5.6-sol",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="auto",
        effort="high",
    )
    assert json.loads(captured[0].content)["reasoning_effort"] == "high"


async def test_openai_auto_without_effort_omits_the_key():
    import json

    captured = []
    await openai(responder(OPENAI_OK, capture=captured)).complete(
        model="gpt-5.6-luna",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="auto",
    )
    assert "reasoning_effort" not in json.loads(captured[0].content)


async def test_an_unknown_reasoning_mode_is_rejected():
    """A typo must not silently fall through to the provider's default."""
    with pytest.raises(ValueError, match="not a valid Reasoning"):
        await anthropic(responder(ANTHROPIC_OK)).complete(
            model="m",
            messages=MESSAGES,
            system=None,
            cache_blocks=[],
            max_tokens=16,
            reasoning="sometimes",
        )


# ──────────────────── temperature (harness spec §2, §2.4) ────────────────────


async def test_anthropic_sends_temperature_only_when_configured():
    """claude-sonnet-5 answers a request carrying `temperature` with a 400:

        `temperature` is deprecated for this model.

    Measured live 2026-08-20. claude-haiku-4-5 accepts it and returns 200, so
    the parameter is per-candidate — the adapter must omit the key entirely
    rather than send a default that breaks composition on every turn.
    """
    import json

    captured = []
    provider = anthropic(responder(ANTHROPIC_OK, capture=captured))
    await provider.complete(
        model="claude-sonnet-5",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
    )
    assert "temperature" not in json.loads(captured[0].content)

    await provider.complete(
        model="claude-haiku-4-5-20251001",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
        temperature=0.0,
    )
    assert json.loads(captured[1].content)["temperature"] == 0.0


async def test_openai_sends_temperature_only_when_configured():
    """gpt-5.6-luna and gpt-5.6-sol both return 200 with temperature 0
    (measured live 2026-08-20), but the key stays absent unless configured —
    the failover must not acquire a control the primary does not have."""
    import json

    captured = []
    provider = openai(responder(OPENAI_OK, capture=captured))
    await provider.complete(
        model="gpt-5.6-luna",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
    )
    assert "temperature" not in json.loads(captured[0].content)

    await provider.complete(
        model="gpt-5.6-luna",
        messages=MESSAGES,
        system=None,
        cache_blocks=[],
        max_tokens=16,
        reasoning="none",
        temperature=0.0,
    )
    assert json.loads(captured[1].content)["temperature"] == 0.0


async def test_the_embedding_carries_the_tokens_it_was_billed_for():
    """The usage block was read and discarded, which is why `embedding_call`
    could never be a row worth writing: a ledger entry with no token count
    prices as NULL, and a NULL cost is indistinguishable from an unpriced
    model. The provider bills the request whole, so the count is the batch's,
    not per text."""
    payload = {
        "model": "text-embedding-3-large",
        "data": [{"embedding": [1.0], "index": 0}, {"embedding": [2.0], "index": 1}],
        "usage": {"prompt_tokens": 37},
    }
    embedding = await openai(responder(payload)).embed(
        model="text-embedding-3-large", texts=["a", "b"], dimensions=1
    )
    assert embedding.input_tokens == 37
    assert embedding.model == "text-embedding-3-large"
    assert embedding.provider == "openai"


async def test_an_embedding_response_without_usage_reports_zero_not_a_guess():
    """A provider that omits the block has told us nothing, and estimating
    from character counts would put a fabricated number in a billing column."""
    payload = {"model": "m", "data": [{"embedding": [1.0], "index": 0}]}
    embedding = await openai(responder(payload)).embed(
        model="m", texts=["a"], dimensions=1
    )
    assert embedding.input_tokens == 0


# ────────── a spend cap is not a malformed request — 2026-08-25 ──────────
#
# The Anthropic account hit its configured usage limit mid-run and the whole
# system stopped answering. Failover was configured, OpenAI was healthy, and
# nothing reached it: the vendor reports quota exhaustion as **400**, and a 400
# is deliberately `ProviderRequestError` so that a malformed body is not sent
# twice and hidden behind a fallback.
#
# That reasoning is right for a malformed body and wrong for this. A spend cap
# is precisely the condition failover exists for, and it arrives wearing the
# status code of our own bug.
#
# **These bodies are copied from real responses**, which is the only reason
# they are trustworthy. The Anthropic one was captured from this account on
# 2026-08-25 while it was exhausted.

#: Verbatim, 2026-08-25, `request_id` removed. Note `type` — Anthropic reports
#: this with the *generic* 400 type, so for this vendor the prose is the only
#: discriminator there is.
ANTHROPIC_EXHAUSTED = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": (
            "You have reached your specified API usage limits. "
            "You will regain access on 2026-09-01 at 00:00 UTC."
        ),
    },
}

#: OpenAI's documented shape. **Not captured from a live exhausted account** —
#: that account is healthy and deliberately not being exhausted to produce a
#: fixture. It carries a specific `type` and `code`, which is what makes the
#: code path preferred over the prose one.
OPENAI_EXHAUSTED = {
    "error": {
        "message": (
            "You exceeded your current quota, please check your plan and "
            "billing details."
        ),
        "type": "insufficient_quota",
        "param": None,
        "code": "insufficient_quota",
    },
}


async def test_anthropics_usage_limit_is_provider_unavailable():
    """The real body, and the real status. 400, generic type, prose only."""
    with pytest.raises(ProviderUnavailable):
        await anthropic(responder(ANTHROPIC_EXHAUSTED, status=400)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )


async def test_openais_insufficient_quota_is_provider_unavailable():
    """Matched on `code`/`type`, not on the sentence — which is the point of
    preferring the field: OpenAI can reword this and the classification holds.
    """
    with pytest.raises(ProviderUnavailable):
        await openai(responder(OPENAI_EXHAUSTED, status=400)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )


async def test_the_openai_code_is_what_matches_not_the_message():
    """Same code, prose replaced with something that matches no phrase. If this
    fails, the field lookup is not doing the work and the prose is."""
    reworded = {
        "error": {
            "message": "Payment required for this organisation.",
            "type": "insufficient_quota",
            "code": "insufficient_quota",
        }
    }
    with pytest.raises(ProviderUnavailable):
        await openai(responder(reworded, status=400)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"type": "invalid_request_error", "message": "max_tokens: must be >= 1"}},
        {"error": {"type": "invalid_request_error", "message": "model: unknown model"}},
        {"error": {"message": "messages.0.content: field required"}},
    ],
)
async def test_an_actual_malformed_request_is_still_our_bug(body):
    """The narrowing must not swallow the case the original rule was written
    for. A 400 that is genuinely our fault still fails hard, still does not
    fail over, and still costs one call rather than two."""
    with pytest.raises(ProviderRequestError) as exc:
        await anthropic(responder(body, status=400)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )
    assert not isinstance(exc.value, ProviderUnavailable)


async def test_a_missing_key_is_still_our_bug():
    """401 is a credential this host does not have. Failing over to a second
    provider cannot fix it and would hide which key is wrong."""
    body = {"error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    with pytest.raises(ProviderRequestError) as exc:
        await anthropic(responder(body, status=401)).complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=16
        )
    assert not isinstance(exc.value, ProviderUnavailable)
