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

import os
from pathlib import Path

import pytest
import pytest_asyncio

from moc.agent.orchestrator import Orchestrator
from moc.agent.script_engine import ScriptEngine
from moc.agent.state import ConversationState, TurnInput
from moc.config_store import load
from moc.evals.judge import Judge
from moc.evals.load import load_cases
from moc.evals.runner import CaseRunner, summarize
from moc.llm.anthropic_direct import AnthropicDirect
from moc.llm.openai_direct import OpenAIDirect
from moc.llm.router import Router
from moc.retrieval.chunker import embedding_text
from moc.retrieval.fusion import FusionRetriever
from moc.retrieval.lexical import (
    LexicalDocument,
    MeilisearchAdmin,
    MeilisearchRepository,
)
from moc.retrieval.vectors import QdrantAdmin, QdrantRepository, VectorPoint
from moc.tenancy.context import tenant_session

pytestmark = pytest.mark.live

ROUTING = load("llm/routing")
LEXICAL = load("retrieval/lexical")
CASES = Path(__file__).parents[2] / "evals" / "cases" / "education.yaml"
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


class SlotExtractor:
    """Intent and slots without a model call.

    The extraction prompt is not built yet, and a real Haiku call here would
    add a second variable to a measurement whose point is retrieval and
    composition. Keyword matching against the script's own intents, stated
    plainly so the number is read with it in mind.
    """

    INTENTS = {
        "مصاريف": "fees", "رسوم": "fees", "خصم": "fees", "منحة": "fees",
        "منح": "fees", "manh": "fees", "khasm": "fees", "قسط": "instalments",
        "تحويل": "transfer_rules", "سكن": "fees", "قبول": "fees",
    }

    async def extract(self, *, text: str, state: ConversationState) -> TurnInput:
        intent = next(
            (value for token, value in self.INTENTS.items() if token in text.lower()),
            None,
        )
        return TurnInput(intent=intent, slots={"faculty": "engineering"})


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
    vectors = await embedder.embed(
        texts=[embedding_text(title=r["title"], content=r["content"]) for r in records]
    )
    await dense.upsert(
        tenant_id=tenant.id,
        vertical="education",
        points=[
            VectorPoint(chunk_id=r["chunk_id"], vector=v, payload={"content": r["content"]})
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
    """All 17 education cases, real retrieval, real models, both stages."""
    lexical, dense, embedder, tenant, chunk_count = corpus
    providers = {
        "anthropic": AnthropicDirect(
            api_key=key("MOC_ANTHROPIC_API_KEY"), http=ROUTING["http"]
        ),
        "openai": OpenAIDirect(api_key=key("MOC_OPENAI_API_KEY"), http=ROUTING["http"]),
    }
    router = Router(config=ROUTING, providers=providers)
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
            engine=ScriptEngine.from_config("scripts/education/fees"),
            router=router,
            retriever=retriever,
            extractor=SlotExtractor(),
        ),
        retriever=retriever,
        judge=Judge.from_config(router=router),
        script="scripts/education/fees",
    )

    cases = load_cases(CASES)
    async with tenant_session(app_engine, tenant.id) as session:
        outcomes = await runner.run(cases, session=session)
    summary = summarize(outcomes)

    with capsys.disabled():
        print(f"\n{'=' * 62}")
        print(f"  EDUCATION SUITE — {len(cases)} cases, {chunk_count} chunks indexed")
        print(f"{'=' * 62}")
        print(f"  overall_accuracy      {summary.accuracy:.1%}"
              f"  ({summary.passed}/{summary.scored} scored)")
        print(f"  errored               {summary.errored}")
        recall = (
            f"{summary.recall_at_5:.1%} over {summary.recall_cases} cases"
            if summary.recall_at_5 is not None
            else "unmeasured"
        )
        print(f"  retrieval_recall_at_5 {recall}")
        # §2.1's two hard gates, both at zero in config/evals/gates.yaml.
        # Printed with their denominators: "0 of 0" is an unmeasured gate, not
        # a clean one, and the two read identically without the count.
        #
        # These measure the reply that was SENT. A composition the runtime
        # gate rejected never reached anyone, so it cannot count against the
        # gate — but how often the model tried is a real signal about
        # composition, so it is reported below rather than lost.
        for metric in ("hallucinated_figure_rate", "hedged_figure_rate"):
            fed = [
                check
                for outcome in outcomes
                for turn in outcome.turns
                for check in turn.checks
                if check.metric == metric and not check.skipped
            ]
            failed = [check for check in fed if not check.passed]
            rate = f"{len(failed) / len(fed):.1%}" if fed else "unmeasured"
            print(
                f"  {metric:21} {rate}"
                f"  ({len(failed)}/{len(fed)} turns that stated a figure)"
            )
        attempted = [
            turn.grounding
            for outcome in outcomes
            for turn in outcome.turns
            if turn.grounding is not None
        ]
        rejected = [g for g in attempted if not g.passed]
        print(f"{'-' * 62}")
        print(
            f"  tracked: the runtime gate rejected {len(rejected)} of {len(attempted)} "
            "compositions"
        )
        print("  (§19.3 discards a reply whose figure has no source and sends a")
        print("  scripted one instead, so these never reached a customer and do")
        print("  not count against the gate above — but they are what it is for.)")
        print(f"{'-' * 62}")
        for outcome in outcomes:
            mark = "err " if outcome.errored else ("PASS" if outcome.passed else "fail")
            failed = [c.name for t in outcome.turns for c in t.checks if not c.passed]
            detail = outcome.error[:60] if outcome.errored else (",".join(failed) or "-")
            print(f"  {mark}  {outcome.case_id:12} {outcome.category:22} {detail}")
        print(f"{'=' * 62}")

    assert len(outcomes) == len(cases)
    assert summary.scored + summary.errored == len(cases)
