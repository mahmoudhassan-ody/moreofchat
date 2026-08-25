"""A full education run — the first real number in the project.

Marked `live`: real Meilisearch, real ingestion, real model calls. Run with

    uv run pytest -m live tests/evals/test_runner_live.py -s

**Read the numbers. Do not tune to them in the same change.** `min_score` can
finally be measured rather than guessed, and measuring it is the point of this
file — but a threshold moved in the same commit that first measured it is a
threshold fitted to one run of 17 cases.

Education only. The real-estate cases need a real-estate script and the
structured-inventory connector, neither of which exists yet; running them now
would report failures that measure the missing connector rather than quality.
"""

import collections
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from moc.agent.extraction import LlmSlotExtractor
from moc.agent.orchestrator import Orchestrator
from moc.agent.script_engine import ScriptEngine
from moc.config_store import load
from moc.evals.judge import Judge
from moc.evals.load import load_cases
from moc.evals.repeatability import MetricSpread, default_runs, render_all, repeat, unmeasurable
from moc.evals.runner import (
    CaseRunner,
    composition_models,
    detail_run,
    metrics,
    phase_breakdown,
    summarize,
)
from moc.llm.anthropic_direct import AnthropicDirect
from moc.llm.openai_direct import OpenAIDirect
from moc.llm.router import Router
from moc.retrieval.chunker import embedding_text
from moc.retrieval.embedding_cache import EmbeddingCache
from moc.retrieval.fusion import FusionRetriever
from moc.retrieval.lexical import (
    LexicalDocument,
    MeilisearchAdmin,
    MeilisearchRepository,
)
from moc.retrieval.vectors import QdrantAdmin, QdrantRepository, VectorPoint
from moc.tenancy.context import tenant_session
from moc.tenancy.metering import UsageKind, record_usage

pytestmark = pytest.mark.live

ROUTING = load("llm/routing")
LEXICAL = load("retrieval/lexical")
CASES = Path(__file__).parents[2] / "evals" / "cases" / "education.yaml"
#: Gitignored. Survives between invocations on purpose — that is the saving.
CACHE_ROOT = Path(__file__).parents[2] / ".cache" / "embeddings"
EMBEDDING = ROUTING["tasks"]["embedding"]["primary"]
SCRIPT_ID = "scripts/education/fees"
SINAI = Path(__file__).parents[2] / "evals" / "fixtures" / "sinai_demo" / "chunks.jsonl"

RUN_INDEXES = {"education": "run_kb_education", "realestate": "run_kb_realestate"}
RUN_CONFIG = {**LEXICAL, "meilisearch": {**LEXICAL["meilisearch"], "indexes": RUN_INDEXES}}


def key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        pytest.skip(f"{name} not set")
    return value


class Embedder:
    """`Router.embed` behind the one-method shape fusion asks for."""

    def __init__(self, router: Router) -> None:
        self._router = router

    async def embed(self, *, texts):
        return await self._router.embed(texts=texts)


@pytest_asyncio.fixture(loop_scope="session")
async def corpus(app_engine, engine):
    """The frozen sinai fixture in BOTH arms, under one tenant.

    Both, deliberately. An earlier version of this fixture built the retriever
    with no dense arm, so the suite silently measured §7.3's degraded shape —
    Meilisearch alone — and the recall it reported was read as fusion recall.
    A missing arm has no behavioural signature: every case still runs, the
    number is merely lower and nobody can tell why.
    """
    import json

    from meilisearch_python_sdk import AsyncClient
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy.ext.asyncio import AsyncSession

    from moc.config import settings
    from moc.tenancy.models import Tenant

    async with AsyncSession(engine, expire_on_commit=False) as s:
        from sqlalchemy import text as sql

        for table in ("kb_outbox", "kb_chunks", "kb_documents", "usage_ledger",
                      "conversations", "tenants"):
            await s.execute(sql(f"DELETE FROM {table}"))  # noqa: S608
        tenant = Tenant(slug="run", name="Run", vertical="education")
        s.add(tenant)
        await s.commit()

    client = AsyncClient(
        f"http://{settings.meili_host}:{settings.meili_port}", settings.meili_key or None
    )
    for name in RUN_INDEXES.values():
        await client.delete_index_if_exists(name)
    await MeilisearchAdmin(client=client, config=RUN_CONFIG).ensure_indexes()

    lexical = MeilisearchRepository(client=client, config=RUN_CONFIG)
    records = [json.loads(line) for line in SINAI.read_text(encoding="utf-8").splitlines()]
    await lexical.add(
        tenant_id=tenant.id,
        vertical="education",
        documents=[
            LexicalDocument(
                point_id=f"{tenant.id}-{r['chunk_id']}",
                chunk_id=r["chunk_id"],
                content=r["content"],
                title=r["title"],
            )
            for r in records
        ],
    )
    qdrant = AsyncQdrantClient(
        url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
        api_key=settings.qdrant_key or None,
    )
    await QdrantAdmin(client=qdrant).ensure_collections()
    dense = QdrantRepository(client=qdrant)
    # Both providers, not just the one embedding uses: `Router` validates
    # every configured task at construction (§7.3's wiring check), so a router
    # built with a subset raises before it is ever asked to embed.
    embedder = Embedder(
        Router(
            config=ROUTING,
            providers={
                "anthropic": AnthropicDirect(
                    api_key=key("MOC_ANTHROPIC_API_KEY"), http=ROUTING["http"]
                ),
                "openai": OpenAIDirect(
                    api_key=key("MOC_OPENAI_API_KEY"), http=ROUTING["http"]
                ),
            },
        )
    )
    texts = [embedding_text(title=r["title"], content=r["content"]) for r in records]
    # The 102 chunks do not change between invocations and were paid for on
    # every one. Content-addressed and per text, so editing one chunk of the
    # fixture costs one embedding rather than a hundred and two.
    ingest = await EmbeddingCache(
        root=CACHE_ROOT,
        model=EMBEDDING["model"],
        dimensions=EMBEDDING["dimensions"],
    ).embed(embedder, texts)
    # What onboarding a tenant costs, on the ledger. A cache hit reports zero
    # tokens and writes nothing, so the row appears once per corpus rather than
    # once per run — which is the number to quote when someone asks.
    if ingest.input_tokens:
        async with tenant_session(app_engine, tenant.id) as s:
            await record_usage(
                s,
                kind=UsageKind.embedding_call,
                model=ingest.model,
                provider=ingest.provider,
                input_tokens=ingest.input_tokens,
                quantity=len(texts),
            )
            await s.commit()
    vectors = ingest.vectors
    await dense.upsert(
        tenant_id=tenant.id,
        vertical="education",
        points=[
            VectorPoint(
                chunk_id=r["chunk_id"],
                vector=v,
                # The title as well as the body. `titles` feeds the fallback
                # clarification, and an arm that indexes only content answers
                # first with a payload that has none.
                payload={"content": r["content"], "title": r["title"]},
            )
            for r, v in zip(records, vectors, strict=True)
        ],
    )

    yield lexical, dense, embedder, tenant, len(records)

    await dense.delete(
        tenant_id=tenant.id,
        vertical="education",
        chunk_ids=[r["chunk_id"] for r in records],
    )
    await qdrant.close()
    for name in RUN_INDEXES.values():
        await client.delete_index_if_exists(name)
    await client.aclose()


async def test_live_the_education_suite_produces_a_report(corpus, app_engine, capsys):
    """All 17 education cases, real retrieval, real models, both stages — N times.

    N, not once. One run of 17 cases graded partly by a model is a sample: four
    consecutive runs of the sibling real-estate suite over one unchanged commit
    read 52.2%, 42.9%, 39.1% and 45.5%. Every metric here therefore reports
    mean with min-max and its run count, and one whose spread exceeds the
    configured bar is flagged as not yet measurable at this suite size rather
    than compared against.
    """
    lexical, dense, embedder, tenant, chunk_count = corpus
    providers = {
        "anthropic": AnthropicDirect(
            api_key=key("MOC_ANTHROPIC_API_KEY"), http=ROUTING["http"]
        ),
        "openai": OpenAIDirect(api_key=key("MOC_OPENAI_API_KEY"), http=ROUTING["http"]),
    }
    router = Router(config=_composition_routing(), providers=providers)
    # Before a single case runs, and only when grading. Two completions of four
    # tokens each against a suite that costs dollars.
    substituted = await _budget_and_primaries(
        router, cases=len(load_cases(CASES)), runs=default_runs()
    ) if _grading() else []
    retriever = FusionRetriever(
        lexical=lexical,
        dense=dense,
        embedder=embedder,
        tenant_id=tenant.id,
        vertical="education",
        config=RUN_CONFIG,
    )
    runner = CaseRunner(
        orchestrator=Orchestrator(
            engine=ScriptEngine.from_config(SCRIPT_ID),
            router=router,
            retriever=retriever,
            extractor=LlmSlotExtractor(router=router, script=SCRIPT_ID),
        ),
        retriever=retriever,
        judge=Judge.from_config(router=router),
        script=SCRIPT_ID,
    )

    started_at = datetime.now(UTC)
    cases = load_cases(CASES)
    times = default_runs()
    grade = _grading()
    runs: list[list] = []

    async def once():
        async with tenant_session(app_engine, tenant.id) as session:
            outcomes = await runner.run(cases, session=session, grade=grade)
            # Commit, or the ledger is decorative. `tenant_session` closes
            # without committing, so every llm_call and embedding_call row a
            # run writes was rolled back — about 186 per invocation. "What did
            # that run cost" was then answerable only by reading code paths
            # and estimating tokens, which is how a billing table comes to
            # have a `provider_cost_usd` column that has never held a value.
            await session.commit()
        runs.append(outcomes)
        with capsys.disabled():
            summary = summarize(outcomes)
            print(
                f"  run {len(runs)}/{times}: {summary.accuracy:.1%} "
                f"({summary.passed}/{summary.scored} scored, {summary.errored} errored)"
            )
            # Errors, distinctly and immediately. An errored case is an
            # outage, not a quality signal, and a run that degrades mid-way
            # needs its cause visible in the same output as its number —
            # otherwise the next reader sees only that the suite got worse.
            reasons = collections.Counter(
                o.error.split("(")[0][:70] for o in outcomes if o.errored
            )
            for reason, count in reasons.most_common():
                print(f"      {count:2} errored: {reason}")
        return metrics(outcomes)

    with capsys.disabled():
        print(f"\n{'=' * 68}")
        print(
            f"  EDUCATION SUITE — {len(cases)} cases, {chunk_count} chunks, "
            f"{times} runs"
        )
        print(
            "  stage 2: the judge GRADED this run"
            if grade
            else f"  stage 2: NOT RUN — set {GRADE}=1 to grade. "
            "Accuracy below is stage 1 only."
        )
        print(f"{'=' * 68}")

    spreads = await repeat(once, times=times)

    with capsys.disabled():
        print(f"{'-' * 68}")
        for line in render_all(spreads):
            print(f"  {line}")
        print(f"{'-' * 68}")
        wide = unmeasurable(spreads)
        if wide:
            print(
                f"  Not measurable at {len(cases)} cases over {times} runs: "
                f"{', '.join(wide)}"
            )
            print("  A delta smaller than the spread is not a result.")
        else:
            # Measurability, never gate compliance. The previous wording —
            # "settled within the configured bar" — read as "every gate
            # passed" on a run where hallucinated_figure_rate sat at 8.1%
            # against a zero-tolerance gate. A spread is how noisy a number
            # is, not whether it is allowed.
            print(
                f"  Every metric's spread is under "
                f"{MetricSpread.measurable_spread_pp()} points at this suite size."
            )
            print("  That is measurability, not gate compliance — read the values above.")
        print(f"{'-' * 68}")
        # Everything below renders ONE run, and it is the last run that
        # produced turns rather than simply the last. A run killed by an
        # outage has nothing to show, and rendering it silently discards the
        # replies, the phases and the composed-by count an earlier run already
        # measured — which cost the haiku comparison its register number on
        # 2026-08-21 after run 1 had measured it.
        shown, detail = detail_run(runs)
        print(f"{'-' * 68}")
        # The two producers of `hallucinated_figure_rate`, split. The
        # deterministic one sees every turn that stated a figure; the judge
        # sees the subset that also cleared stage 1, and only it can catch a
        # figure lifted from a passage and relabelled. How often that fires is
        # what decides whether a runtime claim-citation pass is worth its
        # latency.
        for name in ("hallucinated_figure", "figure_labelling"):
            fed = [
                c
                for o in detail
                for t in o.turns
                for c in t.checks
                if c.name == name and not c.skipped
            ]
            failed = [c for c in fed if not c.passed]
            print(
                f"  {name:20} {len(failed)}/{len(fed)} failed, run {shown} of {times}"
            )
        print(f"{'-' * 68}")
        # Compositions the RUNTIME gate discarded (§19.3). These never reached
        # a customer, so they cannot count against `hallucinated_figure_rate`
        # — but how often the model tried, and on which figure, is the signal
        # the gate exists to produce.
        #
        # WHICH figure had no source, and what it was checked against. A gate
        # refusing compositions while recall is 100% is either the gate being
        # wrong or the passages being unusable, and those have opposite fixes;
        # a count cannot tell them apart.
        rejected = [
            (o.case_id, t)
            for o in detail
            for t in o.turns
            if t.grounding is not None and not t.grounding.passed
        ]
        attempted = [
            t for o in detail for t in o.turns if t.grounding is not None
        ]
        print(
            f"  the runtime gate discarded {len(rejected)} of {len(attempted)} "
            f"compositions, run {shown} of {times}:"
        )
        for case_id, t in rejected:
            print(f"    {case_id} t{t.turn_index}")
            print(f"      orphans:   {t.grounding.orphan_numbers}")
            print(f"      in reply:  {t.grounding.reply_numbers}")
            print(f"      in source: {t.grounding.source_numbers}")
            print(f"      composed:  {t.composed[:180]!r}")
            for passage in t.passages[:2]:
                print(f"      passage:   {passage[:180]!r}")
        print(f"{'-' * 68}")
        # WHICH model actually composed, counted over the turns rather than
        # read off the config the run was launched with. §2.6 fails composition
        # over silently: an arm of a model comparison that lost its primary
        # mid-run reports the failover's quality under the candidate's name,
        # and every other line of this report still looks clean.
        composed_by = composition_models([t for o in detail for t in o.turns])
        print(f"  Composed by, run {shown} of {times}:")
        for model, count in composed_by.most_common():
            print(f"    {model:34} {count:3} turns")
        if len(composed_by) > 1:
            print("    MIXED — this run measured no single model. Do not compare it.")
        print(f"{'-' * 68}")
        await _compare_judge_tier(detail, cases, capsys)
        # Where the p95 goes. §2.5's budget was breached on its first
        # measurement with no breakdown behind it, and a total that only lists
        # the phases somebody instrumented hides the part nobody looked at —
        # so `unattributed` is a row like any other.
        rows = phase_breakdown(
            [t.timings for o in detail for t in o.turns if t.timings]
        )
        print(f"  Phase breakdown, run {shown} of {times} — ms:")
        print(f"    {'phase':16} {'mean':>8} {'p95':>8} {'turns':>7}")
        # Counted phases first, then the dotted details of each — a detail is
        # inside its parent, not beside it, and interleaving them by size reads
        # as double counting.
        for phase, row in sorted(
            rows.items(), key=lambda kv: ("." in kv[0], -kv[1]["mean"])
        ):
            print(
                f"    {phase:16} {row['mean']:8.0f} {row['p95']:8.0f} "
                f"{row['turns']:7.0f}"
            )
        print(f"{'-' * 68}")
        # Per-case, from the LAST run only — labelled as such, because a case
        # that passes twice and fails once is not the same as one that fails
        # every time, and this table cannot tell them apart.
        print(f"  Per-case detail, run {shown} of {times}:")
        for outcome in detail:
            mark = "err " if outcome.errored else ("PASS" if outcome.passed else "fail")
            failed = [c.name for t in outcome.turns for c in t.checks if not c.passed]
            note = outcome.error[:60] if outcome.errored else (",".join(failed) or "-")
            print(f"  {mark}  {outcome.case_id:12} {outcome.category:22} {note}")
            # The reply, for any check that failed. A check name says which
            # gate moved; only the text says why, and reconstructing it meant
            # a second full run every time.
            for turn in outcome.turns:
                misses = [c for c in turn.checks if not c.passed and not c.skipped]
                judged = turn.verdict is not None and not turn.verdict.meets_rubric
                if not misses and not judged:
                    continue
                print(f"        t{turn.turn_index} {turn.action}: {turn.reply[:220]!r}")
                if turn.action == "clarify":
                    # What the fallback had to offer. A clarification that
                    # names nothing is either a node with no options or a
                    # retrieval that returned none, and the reply text alone
                    # cannot tell those apart.
                    print(f"          retrieved titles: {list(turn.titles)}")
                for check in misses:
                    print(f"          {check.name}: {check.detail[:110]}")
                # The judge's own words. A case that passes every
                # deterministic check and fails here failed on answer
                # quality, and "the judge said no" is not a finding.
                if judged:
                    v = turn.verdict
                    print(f"          judge scores: {v.scores()}")
                    if v.forbidden_violated:
                        print(f"          judge forbidden: {list(v.forbidden_violated)}")
                    if v.fact_coverage:
                        print(f"          judge facts: {dict(v.fact_coverage)}")
                    print(f"          judge says: {' '.join(v.reasoning.split())[:400]}")
        # Cases that changed verdict between runs are the suite's own
        # instability, and they are invisible in any single run's table.
        verdicts: dict[str, set[bool]] = {}
        for outcomes in runs:
            for outcome in outcomes:
                verdicts.setdefault(outcome.case_id, set()).add(outcome.passed)
        flaky = sorted(cid for cid, seen in verdicts.items() if len(seen) > 1)
        print(f"{'-' * 68}")
        print(f"  Cases that changed verdict across runs: {', '.join(flaky) or 'none'}")
        print(f"{'=' * 68}")

    assert len(runs) == times
    assert all(len(outcomes) == len(cases) for outcomes in runs)
    # Before this database goes away. `moc_test` is dropped and recreated
    # every session, so a run's ledger rows die with the run that wrote them —
    # which is why the spend report could account for live traffic and nothing
    # else while graded runs were exhausting the account.
    from moc.evals.spend import collect, record

    async with tenant_session(app_engine, tenant.id) as session:
        spend = await collect(session)
    run_id = await record(
        spend=spend, suite="education", graded=grade, runs=times,
        cases=len(cases), turns=sum(len(c.turns) for c in cases) * times,
        started_at=started_at, substituted=substituted or None,
    )
    with capsys.disabled():
        print(f"\n  this run cost {spend.render()}")
        print(f"  recorded as eval_runs {run_id}")

    assert spreads["overall_accuracy"].attempts == times


# ───────── is a cheaper judge the same judge? (§5.2, and the OpenAI bill) ─────────
#
# `eval_grading` routes to Opus first, but §5.2 excludes the answering provider
# and composition runs on Anthropic — so every judge call in the suite lands on
# the OpenAI failover. ~57 per invocation at ~1,450 input tokens each, which by
# volume is ten times all embedding traffic combined and is the largest OpenAI
# line item the project has.
#
# §5.2 constrains the provider, not the tier, and a judge grading against a
# rubric with the passages in front of it is not the hardest task in the
# system. So a cheaper tier may reach the same verdicts — but "may" is not a
# measurement, and switching on the assumption is how a suite quietly starts
# applying a different standard while every baseline still claims comparability.
#
# Off unless asked. Set MOC_JUDGE_TIER_COMPARE to a model name and the run
# re-grades every turn it already graded, printing agreement. Costs one extra
# judge call per graded turn and changes nothing about the run's own numbers —
# the incumbent's verdicts are what the report is built from either way.
TIER_COMPARE = "MOC_JUDGE_TIER_COMPARE"


# ─────────────────────── the composition model, swapped ───────────────────────
#
# Composition is the largest line item in both bills the project has: 2684 ms of
# a 3429 ms mean turn, and sonnet-5 at $2/$10 against haiku-4-5 at $1/$5. So the
# question of whether a cheaper, faster model composes as well is worth a
# measurement rather than an argument — and worth one that cannot be run by
# accident, because the answer only means anything against a baseline taken the
# same day on the same corpus.
#
# Set MOC_COMPOSITION_MODEL to a model name and answer_composition's PRIMARY
# becomes that model for the run. Everything else — the failover, the judge,
# the corpus, the cases — is untouched.
COMPOSITION_MODEL = "MOC_COMPOSITION_MODEL"


# ─────────────────────────── stage 2, opt-in ───────────────────────────
#
# Stage 1 always runs and costs nothing beyond the turn itself. Stage 2 is the
# judge, and the judge is the single largest provider cost a run has: 19 calls
# on gpt-5.6-sol at $0.203 against $0.074 for all 40 Anthropic calls the turns
# made. Repeating it N times over unchanged code buys N independent verdicts on
# the same replies, which is one measurement paid for three times.
#
# So it is asked for: MOC_GRADE=1 for a full graded run — a baseline, a nightly,
# a PR gate — and nothing for the cheap loop. What keeps the cheap loop honest
# is that a stage-1-only run reports `stage_one_accuracy` rather than
# `overall_accuracy`, and every judge-fed gate reads "not measured" rather than
# passing by default.
GRADE = "MOC_GRADE"


def _grading() -> bool:
    return os.environ.get(GRADE, "") not in ("", "0", "false", "no")


#: Set when a run is knowingly served by something other than its primaries —
#: the only honest way to measure anything while a provider is exhausted. The
#: run still happens; the `eval_runs` row records what answered, so the number
#: can never be quoted later as the incumbent's.
ALLOW_SUBSTITUTED = "MOC_ALLOW_SUBSTITUTED"


async def _turn_costs() -> list[float]:
    """What a turn has actually cost, from the durable ledger.

    Live traffic, not eval runs: eval spend lands in `eval_runs` and is a
    per-run total rather than a per-turn one. A turn is a turn either way, and
    this is the only per-turn price anything records.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from moc.config import settings

    engine = create_async_engine(settings.database_url)
    try:
        async with AsyncSession(engine) as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT created_at, sum(coalesce(provider_cost_usd, 0)) "
                        "FROM usage_ledger GROUP BY created_at"
                    )
                )
            ).all()
    finally:
        await engine.dispose()
    return [float(cost) for _, cost in rows if cost]


async def _budget_and_primaries(router, *, cases: int, runs: int) -> list[str]:
    """Cost the run, refuse if it is too big or would measure the wrong model,
    and print the estimate either way.

    Both checks live here because both answer the same question — should this
    command run — and splitting them would mean two places to bypass.

    Returns the substituted tasks, so the run's own record can carry them.
    """
    from moc.evals.headroom import (
        NoHeadroom,
        ceiling_usd,
        check_primaries,
        projected_cost,
        within_budget,
    )

    estimate = projected_cost(turn_costs=await _turn_costs(), turns=cases * runs)
    print(
        f"\n  projected: {estimate.render()}  ceiling ${ceiling_usd():,.2f}"
    )
    within_budget(estimate, ceiling_usd=ceiling_usd())

    try:
        await check_primaries(router=router, routing=ROUTING)
    except NoHeadroom as exc:
        if os.environ.get(ALLOW_SUBSTITUTED, "") in ("", "0", "false", "no"):
            raise
        print(f"  ⚠ {ALLOW_SUBSTITUTED} set — running anyway, and recording it")
        detail = str(exc).split("— ", 1)[-1]
        for part in detail.split("; "):
            print(f"      {part}")
        return detail.split("; ")
    return []


async def _refuse_unless_the_primaries_are_serving(router) -> None:
    """A graded run must measure the models it names, or not run.

    Before 2026-08-25 an exhausted provider made this loud: the run errored
    every case. Then quota exhaustion became `ProviderUnavailable` — correct,
    a spend cap is what failover is for — and the same run now completes
    quietly on OpenAI and reports the number under Anthropic's name.

    Only for graded runs. The cheap stage-1 loop is a developer iterating, and
    a substituted model there is visible in the report's `Composed by` line
    without costing two extra provider calls on every invocation.
    """
    from moc.evals.headroom import check_primaries

    if not _grading():
        return
    await check_primaries(router=router, routing=ROUTING)


def _composition_routing() -> dict:
    """ROUTING, or a copy with answer_composition repinned.

    **`effort` is dropped with the model, not carried onto it.** It is passed
    to the provider verbatim and is therefore provider-native and per-model:
    claude-haiku-4-5 answers a request carrying `output_config.effort` with a
    400, and a 400 is a ProviderRequestError, which does not fail over. Keeping
    the incumbent's `effort: medium` while swapping the model would end every
    composition turn in the run — which at least fails loudly. `reasoning: none`
    stays, because that is task policy for the latency path (§2.5) rather than
    a property of sonnet.
    """
    candidate = os.environ.get(COMPOSITION_MODEL)
    if not candidate:
        return ROUTING
    config = copy.deepcopy(ROUTING)
    task = config["tasks"]["answer_composition"]["primary"]
    task["model"] = candidate
    task.pop("effort", None)
    return config


def _judge_for(model: str) -> Judge:
    """A judge whose eval_grading FAILOVER is `model`.

    The failover, not the primary: `grade` passes `exclude_provider=anthropic`,
    so for a reply Anthropic composed the Anthropic primary is never reachable
    and the failover slot is the one that answers.
    """
    config = copy.deepcopy(ROUTING)
    config["tasks"]["eval_grading"]["failover"]["model"] = model
    return Judge.from_config(
        router=Router(
            config=config,
            providers={
                "anthropic": AnthropicDirect(
                    api_key=key("MOC_ANTHROPIC_API_KEY"), http=config["http"]
                ),
                "openai": OpenAIDirect(
                    api_key=key("MOC_OPENAI_API_KEY"), http=config["http"]
                ),
            },
        )
    )


def _agrees(a, b) -> tuple[bool, int]:
    """`reconcile`'s definition, reused rather than restated."""
    gap = max(abs(a.scores()[d] - b.scores()[d]) for d in a.scores())
    within = gap <= load("evals/judge")["disagreement"]["max_score_gap"]
    return bool(within and a.meets_rubric == b.meets_rubric), gap


async def _compare_judge_tier(outcomes, cases, capsys) -> None:
    candidate = os.environ.get(TIER_COMPARE)
    if not candidate:
        return
    by_id = {case.id: case for case in cases}
    challenger = _judge_for(candidate)
    rows = []
    for outcome in outcomes:
        case = by_id[outcome.case_id]
        for turn_outcome, turn in zip(outcome.turns, case.turns, strict=False):
            verdict = turn_outcome.verdict
            if verdict is None or verdict.malformed:
                continue
            other = await challenger.grade(
                question=turn.user,
                reply=turn_outcome.reply,
                retrieved_passages=list(turn_outcome.passages),
                expected_facts=list(turn.expected_facts),
                forbidden_claims=list(turn.forbidden_claims),
                expected_register=turn.expected_register,
                answer_provider="anthropic",
                # Exactly what the incumbent saw. A re-grade on different
                # evidence reports the difference between two inputs as
                # disagreement between two graders.
                script_statements=list(turn_outcome.script_statements),
            )
            if other.malformed:
                rows.append((f"{outcome.case_id} t{turn_outcome.turn_index}", None, None))
                continue
            ok, gap = _agrees(verdict, other)
            rows.append(
                (
                    f"{outcome.case_id} t{turn_outcome.turn_index}",
                    gap,
                    (ok, verdict.meets_rubric, other.meets_rubric),
                )
            )

    with capsys.disabled():
        graded = [r for r in rows if r[2] is not None]
        malformed = len(rows) - len(graded)
        agreed = sum(1 for _, _, d in graded if d[0])
        flips = [(n, d[1], d[2]) for n, _, d in graded if d[1] != d[2]]
        print(f"  JUDGE TIER — incumbent vs {candidate}, {len(graded)} graded turns")
        print(f"    agreement    : {agreed}/{len(graded)}")
        print(f"    widest gap   : {max((g for _, g, _ in graded), default=0)} rubric points")
        print(f"    verdict flips: {len(flips)}")
        for name, was, now in flips:
            print(f"      {name:16} incumbent={was}  {candidate}={now}")
        if malformed:
            print(f"    malformed    : {malformed} — a judge that returned prose graded nothing")
        print("    Only a flip costs anything: a gap that crosses no floor changes no case.")
        print(f"{'-' * 68}")
