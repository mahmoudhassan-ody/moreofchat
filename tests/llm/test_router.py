"""Task-level routing with cross-provider failover — design §2.6.

No network anywhere in this file. Every provider is a FakeProvider, which is
the point: Tasks 12-15 test against it too, so flow logic never depends on a
vendor being up or on spending money.
"""

import pytest
from sqlalchemy import text

from moc import config_store
from moc.llm.base import (
    Completion,
    Message,
    NoFailoverConfigured,
    ProviderUnavailable,
    Task,
    UnknownTask,
)
from moc.llm.fake import FakeProvider
from moc.llm.router import AllProvidersUnavailable, Router
from moc.tenancy.context import tenant_session
from moc.tenancy.metering import UsageKind, record_usage

MESSAGES = [Message(role="user", content="المصاريف كام؟")]


class Clock:
    """Injectable time. A breaker test that sleeps is a test nobody reruns."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def routing() -> dict:
    return config_store.load("llm/routing")


@pytest.fixture
def anthropic() -> FakeProvider:
    return FakeProvider("anthropic", text="الرسوم 1400 جنيه")


@pytest.fixture
def openai() -> FakeProvider:
    return FakeProvider("openai", text="الرسوم 1400 جنيه", embedding_dimensions=1024)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def router(routing, anthropic, openai, clock) -> Router:
    return Router(
        config=routing, providers={"anthropic": anthropic, "openai": openai}, clock=clock
    )


# ─────────────────────────── routing ───────────────────────────


async def test_routes_answer_composition_to_the_configured_primary(router, anthropic):
    result = await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert result.provider == "anthropic"
    assert result.degraded is False
    assert anthropic.calls


async def test_model_name_comes_from_config_not_from_code(router, routing, anthropic):
    await router.complete(task=Task.answer_composition, messages=MESSAGES)
    expected = routing["tasks"]["answer_composition"]["primary"]["model"]
    assert anthropic.calls[0]["model"] == expected


async def test_max_tokens_comes_from_the_task_config(router, routing, anthropic):
    await router.complete(task=Task.slot_extraction, messages=MESSAGES)
    assert anthropic.calls[0]["max_tokens"] == routing["tasks"]["slot_extraction"]["max_tokens"]


async def test_unknown_task_is_a_config_error_not_a_silent_default(router):
    with pytest.raises(UnknownTask, match="not_a_task"):
        await router.complete(task="not_a_task", messages=MESSAGES)


def test_task_naming_an_unconfigured_provider_is_a_config_error(routing, anthropic, clock):
    """Caught at construction, not on first use.

    A lazily-discovered missing failover surfaces during the outage that first
    needs it, which is the worst possible moment to learn about a wiring error.
    """
    with pytest.raises(UnknownTask, match="openai"):
        Router(config=routing, providers={"anthropic": anthropic}, clock=clock)


# ─────────────────────────── failover ───────────────────────────


async def test_fails_over_to_the_secondary_on_provider_error(router, anthropic, openai):
    anthropic.fail_with = ProviderUnavailable("rate limited")
    result = await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert result.provider == "openai"
    assert result.degraded is True
    assert openai.calls


async def test_a_non_provider_error_does_not_trigger_failover(router, anthropic, openai):
    """A malformed request is our bug (plan Task 13). Failing over hides it."""
    anthropic.fail_with = ValueError("malformed request")
    with pytest.raises(ValueError):
        await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert openai.calls == []


async def test_primary_success_is_never_marked_degraded(router):
    result = await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert result.degraded is False


async def test_both_providers_down_raises(router, anthropic, openai):
    anthropic.fail_with = ProviderUnavailable("down")
    openai.fail_with = ProviderUnavailable("down")
    with pytest.raises(AllProvidersUnavailable):
        await router.complete(task=Task.answer_composition, messages=MESSAGES)


# ─────────────────────────── circuit breaker ───────────────────────────


async def test_breaker_opens_after_n_consecutive_failures(router, routing, anthropic, openai):
    threshold = routing["breaker"]["failure_threshold"]
    anthropic.fail_with = ProviderUnavailable("down")

    for _ in range(threshold):
        await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert len(anthropic.calls) == threshold

    await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert len(anthropic.calls) == threshold, "breaker should have stopped the call"


async def test_breaker_open_raises_rather_than_calling_the_provider(
    router, routing, anthropic, openai
):
    """Scripted fallback is the orchestrator's job (Task 14), not the router's."""
    anthropic.fail_with = ProviderUnavailable("down")
    openai.fail_with = ProviderUnavailable("down")
    for _ in range(routing["breaker"]["failure_threshold"]):
        with pytest.raises(AllProvidersUnavailable):
            await router.complete(task=Task.answer_composition, messages=MESSAGES)

    before = (len(anthropic.calls), len(openai.calls))
    with pytest.raises(AllProvidersUnavailable):
        await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert (len(anthropic.calls), len(openai.calls)) == before


async def test_a_success_resets_the_failure_count(router, routing, anthropic):
    anthropic.fail_times = routing["breaker"]["failure_threshold"] - 1
    anthropic.fail_with = ProviderUnavailable("blip")
    for _ in range(routing["breaker"]["failure_threshold"] - 1):
        await router.complete(task=Task.answer_composition, messages=MESSAGES)

    result = await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert result.provider == "anthropic"

    calls_before = len(anthropic.calls)
    await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert len(anthropic.calls) == calls_before + 1, "counter should have reset on success"


async def test_breaker_closes_after_the_configured_reset_window(
    router, routing, anthropic, clock
):
    anthropic.fail_with = ProviderUnavailable("down")
    for _ in range(routing["breaker"]["failure_threshold"]):
        await router.complete(task=Task.answer_composition, messages=MESSAGES)
    calls_while_open = len(anthropic.calls)

    clock.advance(routing["breaker"]["reset_seconds"] + 1)
    anthropic.fail_with = None
    result = await router.complete(task=Task.answer_composition, messages=MESSAGES)

    assert len(anthropic.calls) > calls_while_open, "breaker should have half-opened"
    assert result.provider == "anthropic"


async def test_breaker_is_per_provider_not_global(router, routing, anthropic, openai):
    anthropic.fail_with = ProviderUnavailable("down")
    for _ in range(routing["breaker"]["failure_threshold"] + 2):
        result = await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert result.provider == "openai", "openai's breaker must be unaffected"


# ─────────────────────────── embeddings (§7.3) ───────────────────────────


async def test_embeddings_route_to_the_configured_provider(router, openai, routing):
    vectors = await router.embed(texts=["رسوم الساعة"])
    assert len(vectors) == 1
    assert openai.calls[0]["model"] == routing["tasks"]["embedding"]["primary"]["model"]


async def test_embedding_dimensions_come_from_config(router, routing, openai):
    await router.embed(texts=["رسوم الساعة"])
    assert openai.calls[0]["dimensions"] == routing["tasks"]["embedding"]["primary"]["dimensions"]


async def test_no_failover_for_embeddings(router, openai):
    """A second embedding provider returns vectors from a different space —
    retrieval quality would collapse silently. Spec §7.3."""
    with pytest.raises(NoFailoverConfigured):
        await router.embed(texts=["رسوم الساعة"], force_failover=True)


async def test_embedding_failure_does_not_silently_fail_over(router, openai, anthropic):
    """Down means ingest queues and search degrades to Meilisearch — §7.3."""
    openai.fail_with = ProviderUnavailable("down")
    with pytest.raises(ProviderUnavailable):
        await router.embed(texts=["رسوم الساعة"])
    assert anthropic.calls == [], "embeddings must never reach the other provider"


def test_the_config_declares_no_embedding_failover(routing):
    """The rule is data, so a future edit that adds one is visible in review."""
    assert routing["tasks"]["embedding"]["failover"] is None


# ─────────────────────────── usage ledger (Task 6) ───────────────────────────


async def test_records_degraded_on_the_usage_ledger(
    app_engine, two_tenants, routing, anthropic, openai, clock
):
    """The degraded flag from Task 6 exists for exactly this."""
    tenant, _ = two_tenants
    recorded = []

    async def sink(event) -> None:
        recorded.append(event)
        async with tenant_session(app_engine, tenant.id) as s:
            await record_usage(
                s,
                kind=UsageKind.llm_call,
                model=event.model,
                provider=event.provider,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cached_tokens=event.cached_tokens,
                degraded=event.degraded,
            )
            await s.commit()

    router = Router(
        config=routing,
        providers={"anthropic": anthropic, "openai": openai},
        clock=clock,
        usage_sink=sink,
    )
    anthropic.fail_with = ProviderUnavailable("rate limited")
    await router.complete(task=Task.answer_composition, messages=MESSAGES)

    assert [e.degraded for e in recorded] == [True]
    async with tenant_session(app_engine, tenant.id) as s:
        rows = (
            await s.execute(text("SELECT provider, degraded FROM usage_ledger"))
        ).all()
    assert [(r.provider, r.degraded) for r in rows] == [("openai", True)]


async def test_usage_sink_is_optional(router):
    """A router without a sink still completes — metering is the caller's wiring."""
    assert (await router.complete(task=Task.answer_composition, messages=MESSAGES)).text


async def test_sink_receives_token_counts(routing, anthropic, openai, clock):
    events = []

    async def sink(event) -> None:
        events.append(event)

    anthropic.input_tokens, anthropic.output_tokens, anthropic.cached_tokens = 120, 30, 90
    router = Router(
        config=routing,
        providers={"anthropic": anthropic, "openai": openai},
        clock=clock,
        usage_sink=sink,
    )
    await router.complete(task=Task.answer_composition, messages=MESSAGES)
    assert (events[0].input_tokens, events[0].output_tokens, events[0].cached_tokens) == (
        120,
        30,
        90,
    )


# ─────────────────────────── the fake itself ───────────────────────────


async def test_fake_records_every_call(anthropic):
    await anthropic.complete(
        model="m", messages=MESSAGES, system="s", cache_blocks=["a"], max_tokens=10
    )
    call = anthropic.calls[0]
    assert call["system"] == "s"
    assert call["cache_blocks"] == ["a"]
    assert call["messages"] == MESSAGES


async def test_fake_returns_a_completion(anthropic):
    result = await anthropic.complete(
        model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=10
    )
    assert isinstance(result, Completion)
    assert result.provider == "anthropic"


async def test_fake_fails_only_the_configured_number_of_times(anthropic):
    anthropic.fail_with = ProviderUnavailable("blip")
    anthropic.fail_times = 1
    with pytest.raises(ProviderUnavailable):
        await anthropic.complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=10
        )
    assert (
        await anthropic.complete(
            model="m", messages=MESSAGES, system=None, cache_blocks=[], max_tokens=10
        )
    ).provider == "anthropic"


async def test_fake_embeddings_have_the_requested_dimension(openai):
    vectors = await openai.embed(model="m", texts=["a", "b"], dimensions=1024)
    assert [len(v) for v in vectors] == [1024, 1024]
