"""The worker processes — design §3, §14, demo plan Task 39.

`python -m moc.workers.run inbound` and `python -m moc.workers.run outbound`.

Everything slow and everything tenant-scoped happens here, behind the queue.
These processes hold `moc_app`, the model providers, Qdrant and Meilisearch;
the webhook process holds none of it (see `moc.api.main`).

**Three things are per tenant, and each of them produced a reply when it was
process-wide.** They are the reason this file exists rather than a
`for_each_message` loop:

1. **The retriever.** `FusionRetriever` is built with a tenant id and a
   vertical. One held for the life of the process answers every tenant from
   whichever corpus it was started with — a cross-tenant read that arrives as a
   fluent, correctly-cited reply about somebody else's fees. RLS cannot catch
   it: the retriever holds the id it filters on, and it holds the wrong one.

2. **The sender.** Every adapter is built from one tenant's credentials. One
   per channel for the life of the process sends every tenant's replies from
   one number, under one name. See `moc.channels.senders`.

3. **The vertical.** Real estate is a different agent with a different result
   type — a price came from a row, not a passage. Both are wired here, keyed by
   the tenant's vertical, and a vertical with no runner is refused loudly:
   answering a broker's customer with the education script produces a fluent
   reply about credit-hour fees, which is the worst outcome available because
   it is indistinguishable from working.

   The real-estate extractor is built **per turn**, not per process. The
   catalogue it resolves compounds against is the tenant's own inventory, so
   one extractor per process resolves every broker's areas against whichever
   broker started first — and the failure is a confident answer about the wrong
   compound.

**One process per channel is not what this is.** §14 lists `worker-inbound` and
`worker-outbound`, one each. Channels are a routing key inside the job, not a
process boundary; the rate limit is per tenant and a bucket per process would
each allow the full rate.
"""

import asyncio
import os
import sys
from typing import Any

from moc.config_store import load

_QUEUES = "workers/queues"
_ROUTING = "llm/routing"
_LEXICAL = "retrieval/lexical"
_SCRIPT = "scripts/education/fees"
_REALESTATE_SCRIPT = "scripts/realestate/search"
_EDUCATION = "education"
_REALESTATE = "realestate"


class RouterEmbedder:
    """`Router.embed` behind the one-method shape fusion asks for.

    Five lines, and until Task 39 they existed only inside a test file — which
    is the shape of the whole problem this module addresses: the production
    object graph had never been built.
    """

    def __init__(self, router: Any) -> None:
        self._router = router

    async def embed(self, *, texts: Any) -> Any:
        return await self._router.embed(texts=texts)


class TenantRetrievers:
    """(tenant, vertical) -> a retriever that reads *their* corpus.

    Holds the shared clients and builds a cheap per-tenant view over them.
    Cached, because a `FusionRetriever` is a few references and the clients
    underneath are what cost something to open.
    """

    def __init__(self, *, lexical: Any, dense: Any, embedder: Any) -> None:
        self._lexical = lexical
        self._dense = dense
        self._embedder = embedder
        self._cache: dict[tuple[str, str], Any] = {}

    async def for_tenant(self, *, tenant_id: Any, vertical: str) -> Any:
        from moc.retrieval.fusion import FusionRetriever

        key = (str(tenant_id), vertical)
        if key not in self._cache:
            self._cache[key] = FusionRetriever(
                lexical=self._lexical,
                dense=self._dense,
                embedder=self._embedder,
                tenant_id=tenant_id,
                vertical=vertical,
            )
        return self._cache[key]


def _providers() -> dict[str, Any]:
    """Both model providers, not only the one a task uses.

    `Router` validates every configured task at construction (§7.3's wiring
    check), so a router built with a subset raises before it is ever asked.
    """
    from moc.llm.anthropic_direct import AnthropicDirect
    from moc.llm.openai_direct import OpenAIDirect

    routing = load(_ROUTING)
    return {
        "anthropic": AnthropicDirect(
            api_key=_required("MOC_ANTHROPIC_API_KEY"), http=routing["http"]
        ),
        "openai": OpenAIDirect(
            api_key=_required("MOC_OPENAI_API_KEY"), http=routing["http"]
        ),
    }


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise KeyError(
            f"{name} is not set. A provider built with an empty key fails every "
            "call with the vendor's authentication error, which reads as an "
            "outage and gets escalated to the vendor."
        )
    return value


async def _valkey() -> Any:
    from moc.channels.valkey import valkey_client

    return valkey_client()


def _app_engine() -> Any:
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.config import settings

    return create_async_engine(settings.app_database_url())


async def inbound() -> None:
    """One turn per message, forever."""
    from moc.agent.extraction import LlmSlotExtractor
    from moc.agent.orchestrator import Orchestrator
    from moc.agent.script_engine import ScriptEngine
    from moc.agent.scripts import ScriptStore
    from moc.channels.accounts import EnvSecretResolver
    from moc.channels.senders import SqlSenderRegistry
    from moc.llm.router import Router
    from moc.retrieval.lexical import MeilisearchRepository, meilisearch_client
    from moc.retrieval.vectors import QdrantRepository, qdrant_client
    from moc.verticals.realestate.runner import InventoryRunner
    from moc.workers.inbound import InboundWorker, Served

    engine = _app_engine()
    client = await _valkey()
    router = Router(config=load(_ROUTING), providers=_providers())

    meili = meilisearch_client()
    qdrant = qdrant_client()
    # §2.5's perceived-latency mitigation, per tenant. The same registry the
    # sender worker uses: the indicator authenticates as the tenant's own
    # Twilio account, and it is not a message — it never touches the outbound
    # queue, because a courtesy that queues behind replies arrives after the
    # thing it was meant to precede.
    indicators = SqlSenderRegistry(engine=engine, secrets=EnvSecretResolver())
    retrievers = TenantRetrievers(
        lexical=MeilisearchRepository(client=meili, config=load(_LEXICAL)),
        dense=QdrantRepository(client=qdrant),
        embedder=RouterEmbedder(router),
    )

    worker = InboundWorker(
        client=client,
        engine=engine,
        orchestrator=Orchestrator(
            engine=ScriptEngine.from_config(_SCRIPT),
            router=router,
            # The construction-time retriever is never used: every turn is
            # given its tenant's. Passed because the constructor requires one,
            # and deliberately something that answers nothing, so a turn that
            # somehow reached it would hand off rather than answer from a
            # corpus chosen by startup order.
            retriever=_NoCorpus(),
            extractor=LlmSlotExtractor(router=router, script=_SCRIPT),
        ),
        script=_SCRIPT,
        config=load(_QUEUES),
        scripts=ScriptStore(engine=engine),
        retrievers=retrievers,
        indicators=indicators,
        vertical=_EDUCATION,
        runners={
            _REALESTATE: Served(
                runner=InventoryRunner(
                    script=_REALESTATE_SCRIPT,
                    # A factory, because the catalogue is the tenant's own
                    # inventory and there is no tenant-independent extractor.
                    extractor=lambda catalogue: LlmSlotExtractor(
                        router=router, script=_REALESTATE_SCRIPT, catalogue=catalogue
                    ),
                ),
                script=_REALESTATE_SCRIPT,
            )
        },
    )

    try:
        while True:
            await worker.run_once(block=True)
    finally:
        await indicators.aclose()
        await client.aclose()
        await engine.dispose()
        await qdrant.close()
        await meili.aclose()


class _NoCorpus:
    """The retriever a turn must never reach.

    `Orchestrator` requires one at construction and every turn here supplies
    its tenant's, so this is what is left holding the seam. It returns nothing
    at no confidence, which routes the turn to the script's fallback — the
    correct behaviour for "we do not know whose corpus this is", and the one
    that cannot answer from the wrong tenant's.
    """

    async def search(self, *, query: str) -> Any:
        from moc.agent.orchestrator import Retrieval

        return Retrieval(passages=(), confidence=None)


async def outbound() -> None:
    """One send per queued reply, forever."""
    from moc.channels.accounts import EnvSecretResolver
    from moc.channels.senders import SqlSenderRegistry
    from moc.workers.outbound import OutboundWorker

    engine = _app_engine()
    client = await _valkey()
    senders = SqlSenderRegistry(engine=engine, secrets=EnvSecretResolver())

    worker = OutboundWorker(client=client, providers=senders, config=load(_QUEUES))
    try:
        while True:
            await worker.run_once(block=True)
    finally:
        await senders.aclose()
        await client.aclose()
        await engine.dispose()


_ENTRYPOINTS = {"inbound": inbound, "outbound": outbound}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in _ENTRYPOINTS:
        print(f"usage: python -m moc.workers.run {{{'|'.join(_ENTRYPOINTS)}}}")
        return 2
    asyncio.run(_ENTRYPOINTS[argv[0]]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
