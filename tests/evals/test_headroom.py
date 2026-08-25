"""Refusing to start a graded run that cannot honestly finish.

Two failures on 2026-08-25, an hour apart, and the second was caused by the fix
for the first.

**The account was exhausted by a graded run** — 19 cases, 3 runs, and the
budget went partway through run 1. Runs 2 and 3 then executed 38 more cases
against a provider that had already said no, and reported `0.0% (0 scored, 19
errored)` twice. Nothing checked before starting, and nothing stopped after the
first refusal.

**Then quota became `ProviderUnavailable`,** so the router now fails over to
OpenAI — which is right for a customer's turn and wrong for a measurement. A
graded run whose primary is exhausted no longer errors: it quietly completes on
`gpt-5.6-luna` and reports a number for `claude-haiku-4-5`. §2.3 pins
`prompt_version` and `config_hash` precisely so a run is comparable, and a
silently substituted model defeats both.

So the guard is about the primary specifically, not about reachability in
general. A failover that works is not a reason to measure.
"""

import pytest

from moc.evals.headroom import NoHeadroom, check_primaries, projected_cost


class FakeCompletion:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.text = "ok"
        self.input_tokens = self.output_tokens = self.cached_tokens = 0
        self.degraded = provider != "anthropic"


class FakeRouter:
    """Answers as whichever provider is currently serving the task."""

    def __init__(self, serving: dict[str, tuple[str, str]]) -> None:
        self._serving = serving
        self.calls: list[str] = []
        self.asked: list[dict] = []

    async def complete(self, *, task, messages, system=None, **kwargs):
        self.calls.append(str(task))
        # Recorded, not swallowed. What the probe *asks for* is half of what
        # this check is worth: a judge probe that forgot to exclude the
        # composing provider reports the wrong model and still reads green.
        self.asked.append({"task": str(task), **kwargs})
        provider, model = self._serving[str(task)]
        return FakeCompletion(provider, model)


ROUTING = {
    "tasks": {
        "slot_extraction": {
            "max_tokens": 512,
            "primary": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            "failover": {"provider": "openai", "model": "gpt-5.6-luna"},
        },
        "answer_composition": {
            "max_tokens": 1024,
            "primary": {"provider": "anthropic", "model": "claude-sonnet-5"},
            "failover": {"provider": "openai", "model": "gpt-5.6-sol"},
        },
        # Grades the other four (§5.2), and never on the provider that
        # composed — which makes its `primary` the wrong thing to check.
        "eval_grading": {
            "max_tokens": 2048,
            "primary": {"provider": "anthropic", "model": "claude-opus-5"},
            "failover": {"provider": "openai", "model": "gpt-5.6-sol"},
        },
        # Routed, has a primary, and is not a completion. The router reads
        # `max_tokens` off the spec, so probing this one raises KeyError
        # instead of checking anything — found against the live router.
        "embedding": {
            "primary": {"provider": "openai", "model": "text-embedding-3-large"},
        },
    }
}

HEALTHY = {
    "slot_extraction": ("anthropic", "claude-haiku-4-5-20251001"),
    "answer_composition": ("anthropic", "claude-sonnet-5"),
    # Composition is on Anthropic above, so §5.2 puts the judge here. Not the
    # task's primary, and healthy all the same.
    "eval_grading": ("openai", "gpt-5.6-sol"),
}
EXHAUSTED = {
    "slot_extraction": ("openai", "gpt-5.6-luna"),
    "answer_composition": ("openai", "gpt-5.6-sol"),
    "eval_grading": ("anthropic", "claude-opus-5"),
}


async def test_a_healthy_primary_lets_the_run_start():
    router = FakeRouter(HEALTHY)
    await check_primaries(router=router, routing=ROUTING)
    assert len(router.calls) == 3, "every task with a primary is checked, not just one"
    assert "embedding" not in router.calls, (
        "a task with no max_tokens is not a completion, and probing it raises "
        "rather than checks"
    )


async def test_a_run_served_by_failover_is_refused():
    """The failure this exists for, and it is silent without the check: the run
    completes, the report is populated, and every number belongs to a model
    nobody chose."""
    with pytest.raises(NoHeadroom) as exc:
        await check_primaries(router=FakeRouter(EXHAUSTED), routing=ROUTING)
    assert "slot_extraction" in str(exc.value)
    assert "gpt-5.6-luna" in str(exc.value), "name what answered instead"


async def test_one_exhausted_task_is_enough_to_refuse():
    """A suite is one measurement. Extraction on the incumbent and composition
    on a substitute is not a partial baseline, it is not a baseline."""
    mixed = {**HEALTHY, "answer_composition": ("openai", "gpt-5.6-sol")}
    with pytest.raises(NoHeadroom, match="answer_composition"):
        await check_primaries(router=FakeRouter(mixed), routing=ROUTING)


def test_the_projection_multiplies_measured_turns_by_the_suite():
    """The ledger knows what a turn costs. Nothing compared it to a run."""
    estimate = projected_cost(turn_costs=[0.0055, 0.0093], turns=57)
    assert estimate.turns == 57
    assert estimate.low == pytest.approx(0.0055 * 57)
    assert estimate.high == pytest.approx(0.0093 * 57)
    assert estimate.samples == 2


def test_a_projection_with_no_measured_turns_says_so_rather_than_guessing():
    """`unmeasured is not zero`. A run with no prior data must report that it
    cannot be costed, not report zero and start."""
    estimate = projected_cost(turn_costs=[], turns=57)
    assert estimate.samples == 0
    assert estimate.low is None and estimate.high is None
    assert "not measured" in estimate.render().lower()


def test_the_projection_renders_its_own_sample_count():
    """Two samples is a range, not a distribution, and the sentence that says
    so travels with the number."""
    rendered = projected_cost(turn_costs=[0.0055, 0.0093], turns=57).render()
    assert "n=2" in rendered


# ─────────────── a ceiling per invocation, not per month ───────────────


def test_a_run_inside_the_ceiling_is_allowed():
    from moc.evals.headroom import within_budget

    estimate = projected_cost(turn_costs=[0.0055, 0.0093], turns=57)
    within_budget(estimate, ceiling_usd=5.00)  # does not raise


def test_a_run_over_the_ceiling_is_refused_on_its_worst_case():
    """The high end, not the mean. A ceiling that lets a run start on its
    average and blow past on its spread is a ceiling that fires after the
    money is gone."""
    from moc.evals.headroom import NoHeadroom, within_budget

    estimate = projected_cost(turn_costs=[0.0055, 0.0093], turns=5000)
    assert estimate.low < 40 < estimate.high, "the ceiling must sit inside the spread"
    with pytest.raises(NoHeadroom, match="ceiling"):
        within_budget(estimate, ceiling_usd=40.00)


def test_an_uncosted_run_is_refused_rather_than_waved_through():
    """`unmeasured is not zero`. A run nobody can price must not be treated as
    a free one — which is exactly how a $0.00 default would read."""
    from moc.evals.headroom import NoHeadroom, within_budget

    with pytest.raises(NoHeadroom, match="not measured"):
        within_budget(projected_cost(turn_costs=[], turns=57), ceiling_usd=5.00)


def test_the_ceiling_is_configuration():
    """§19: a number that changes with circumstances is config. This one
    changes with whose budget is paying."""
    from moc.evals.headroom import ceiling_usd

    assert ceiling_usd() > 0


# ───────────────── the judge is not graded by its primary ─────────────────
#
# `eval_grading` routes to Opus first, and with composition on Anthropic the
# judge never reaches it: §5.2 excludes the answering provider outright, so
# every judge call in a graded run lands on `gpt-5.6-sol`. A check that probed
# the task's primary reported `anthropic/claude-opus-5` serving — true, and
# about a model that would not grade a single turn of the run it was gating.


async def test_the_judge_is_probed_on_the_provider_5_2_will_actually_force():
    router = FakeRouter(HEALTHY)
    await check_primaries(router=router, routing=ROUTING)
    asked = [row for row in router.asked if row["task"] == "eval_grading"]
    assert asked, "the judge is probed like every other completion task"
    assert asked[0]["exclude_provider"] == "anthropic", (
        "the judge probe must route around the composing provider, exactly as "
        "Judge.grade does — otherwise it measures a model the run will not use"
    )


async def test_a_judge_answered_by_the_composing_provider_is_refused():
    """The violation §5.2 exists to make impossible, seen from the probe.

    Not a failover problem: a judge on the answering provider grades its own
    output, and self-preference reads exactly like quality.
    """
    serving = {**HEALTHY, "eval_grading": ("anthropic", "claude-opus-5")}
    with pytest.raises(NoHeadroom) as exc:
        await check_primaries(router=FakeRouter(serving), routing=ROUTING)
    assert "eval_grading" in str(exc.value)
    assert "claude-opus-5" in str(exc.value), "name what answered instead"


async def test_the_probe_caps_what_it_may_generate():
    """A check that costs real money is a check people switch off.

    `eval_grading` carries `max_tokens: 2048` and `effort: high`, which is
    right for grading a turn and absurd for establishing who picked up the
    phone: the word "ok" cost $0.0035 of a $0.0042 check.
    """
    router = FakeRouter(HEALTHY)
    await check_primaries(router=router, routing=ROUTING)
    caps = [row.get("max_tokens") for row in router.asked]
    assert caps and all(cap is not None and cap <= 32 for cap in caps), (
        f"every probe must cap its own generation, got {caps}"
    )
