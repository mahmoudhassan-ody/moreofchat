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
from moc.retrieval.fusion import FusionRetriever
from moc.retrieval.lexical import (
    LexicalDocument,
    MeilisearchAdmin,
    MeilisearchRepository,
)
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
    """The frozen sinai fixture in Meilisearch, under one tenant."""
    import json

    from meilisearch_python_sdk import AsyncClient
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
    yield lexical, tenant, len(records)
    for name in RUN_INDEXES.values():
        await client.delete_index_if_exists(name)
    await client.aclose()


async def test_live_the_education_suite_produces_a_report(corpus, app_engine, capsys):
    """All 17 education cases, real retrieval, real models, both stages."""
    lexical, tenant, chunk_count = corpus
    providers = {
        "anthropic": AnthropicDirect(
            api_key=key("MOC_ANTHROPIC_API_KEY"), http=ROUTING["http"]
        ),
        "openai": OpenAIDirect(api_key=key("MOC_OPENAI_API_KEY"), http=ROUTING["http"]),
    }
    router = Router(config=ROUTING, providers=providers)
    retriever = FusionRetriever(
        lexical=lexical, tenant_id=tenant.id, vertical="education", config=RUN_CONFIG
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
        print(f"{'-' * 62}")
        for outcome in outcomes:
            mark = "err " if outcome.errored else ("PASS" if outcome.passed else "fail")
            failed = [c.name for t in outcome.turns for c in t.checks if not c.passed]
            detail = outcome.error[:60] if outcome.errored else (",".join(failed) or "-")
            print(f"  {mark}  {outcome.case_id:12} {outcome.category:22} {detail}")
        print(f"{'=' * 62}")

    assert len(outcomes) == len(cases)
    assert summary.scored + summary.errored == len(cases)
