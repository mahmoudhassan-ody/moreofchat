"""The processes that actually run — design §14, demo plan Task 39.

Until this file existed there was no way to start this system. Every app in
this codebase was assembled inside a test, from fakes, over `ASGITransport`.
That is a real gap rather than a missing convenience: an object graph that has
only ever been built by tests is one whose production shape nobody has checked,
and three single-tenant assumptions were sitting in it — the retriever, the
sender and the vertical — each of which produces a *reply* when it is wrong.

**What the internet-facing process can reach, and what it cannot.**

This is the one endpoint an unauthenticated stranger can talk to, so it is
given the smallest possible reach, and the split is deliberate rather than an
accident of packaging:

- `moc_lookup`, whose only privilege in the database is SELECT on a five-column
  view (migration 0007). It resolves which tenant an address belongs to.
- Valkey, to publish onto the inbound stream and claim message ids.
- The signing secrets, from the environment.

It holds **no `moc_app` credentials at all**. The webhook process therefore
cannot read a conversation, a message, a fee or a unit price — not because it
declines to, but because it has no login that would let it. Everything slow and
everything tenant-scoped happens in the workers, behind the queue.

The console API is a separate process for the same reason: agent sessions,
tenant data and the whole `moc_app` surface do not belong in the process that
strangers can post to. Caddy routes `/webhooks/*` here and everything else
there.

**The console process publishes replies rather than sending them.** §6.2's one
send path: an agent's reply is a message to a customer on a messaging platform,
subject to the same rate limit and the same 24-hour window as a bot reply, so
it goes onto the outbound stream and `worker-outbound` delivers it. The
import-linter contract that forbids `moc.api.inbox` from reaching a channel
adapter is what makes that a property of the wiring rather than a habit — and
this file is the wiring, which is why the adapters are constructed here and
injected rather than imported there.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response

from moc.channels.accounts import EnvSecretResolver, SqlChannelAccountRegistry, lookup_engine
from moc.channels.valkey import ValkeyEventLog, ValkeyInboundQueue
from moc.config_store import load

_QUEUES = "workers/queues"
_ROUTING = "llm/routing"
_HEALTHZ = "/healthz"

#: FastAPI publishes these by default. On a public host they are a map of the
#: attack surface handed to whoever asks — every path, every parameter name,
#: every payload shape, in a form built for tooling. Removed rather than
#: disabled at construction because `build_app` is a security-reviewed file
#: and this is a property of the *process*, not of the routes.
_DOCUMENTATION = ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")


def webhook_app() -> FastAPI:
    """The public webhook process.

    Built here rather than in `webhooks.py` so that "what can the
    pre-authentication path reach?" is answerable by reading one file — the
    same rule `channels/accounts.py` follows about the bootstrap read.
    """
    from moc.api.webhooks import build_app
    from moc.channels.valkey import valkey_client

    engine = lookup_engine()
    client = valkey_client()
    queues = load(_QUEUES)

    app = build_app(
        registry=SqlChannelAccountRegistry(engine=engine),
        queue=ValkeyInboundQueue(client=client, config=queues),
        events=ValkeyEventLog(client=client, config=queues),
        secrets=EnvSecretResolver(),
    )

    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) not in _DOCUMENTATION
    ]

    @app.get(_HEALTHZ)
    async def healthz() -> Response:
        """For the container healthcheck, and for nothing else.

        Empty on purpose. A health endpoint that reports which dependencies are
        up is a reconnaissance endpoint on a public host — it tells a stranger
        what to attack and when it is degraded enough to be worth trying.
        """
        return Response(status_code=200)

    # Closed on the way down. `build_app` owns no lifespan of its own — it is
    # constructed from collaborators and does not know what they are — so the
    # composition root that opened these closes them.
    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(scope: Any):
        async with previous(scope):
            try:
                yield
            finally:
                await client.aclose()
                await engine.dispose()

    app.router.lifespan_context = lifespan
    return app


def console_app() -> FastAPI:
    """The authenticated console API.

    Everything the tenant console talks to: the inbox, knowledge, settings,
    scripts, analytics and branding, behind one cookie session. Separate from
    the webhook process because agent sessions and the whole `moc_app` surface
    do not belong in the process strangers can post to.

    Static assets are not served here. Caddy serves `console/dist` directly —
    a Python process in front of a stylesheet is latency with no upside, and
    the SPA uses hash routing so there is no history fallback to arrange.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.agent.scripts import ScriptStore
    from moc.api.analytics import build_analytics_router
    from moc.api.auth import build_auth_router, cookie_authenticator
    from moc.api.inbox import build_inbox
    from moc.api.knowledge import build_knowledge_router
    from moc.api.settings import build_settings_router
    from moc.api.tenant import build_tenant_router
    from moc.channels.valkey import (
        ValkeyInboxEvents,
        ValkeyOutboundPublisher,
        valkey_client,
    )
    from moc.config import settings as config
    from moc.llm.router import Router
    from moc.retrieval.knowledge import KnowledgeService
    from moc.retrieval.lexical import MeilisearchRepository, meilisearch_client
    from moc.retrieval.vectors import QdrantRepository, qdrant_client
    from moc.tenancy.agent_auth import AgentDirectory
    from moc.tenancy.analytics import AnalyticsStore
    from moc.tenancy.branding import BrandingStore
    from moc.tenancy.settings import SettingsStore

    engine = create_async_engine(config.app_database_url())
    # Also `moc_lookup`: resolving a session token to a tenant is the same
    # bootstrap problem the webhook has — the answer is what establishes the
    # tenant context, so it cannot run inside one.
    lookup = lookup_engine()
    client = valkey_client()
    queues = load(_QUEUES)

    directory = AgentDirectory(engine=engine, lookup=lookup)
    authenticate = cookie_authenticator(directory=directory)

    app = build_inbox(
        engine=engine,
        publisher=ValkeyOutboundPublisher(client=client, config=queues),
        events=ValkeyInboxEvents(client=client),
        authenticate=authenticate,
    )
    app.include_router(build_auth_router(directory=directory))
    app.include_router(
        build_settings_router(
            settings=SettingsStore(engine=engine),
            scripts=ScriptStore(engine=engine),
            authenticate=authenticate,
        )
    )
    app.include_router(
        build_analytics_router(store=AnalyticsStore(engine=engine), authenticate=authenticate)
    )
    app.include_router(
        build_tenant_router(store=BrandingStore(engine=engine), authenticate=authenticate)
    )

    meili = meilisearch_client()
    qdrant = qdrant_client()
    router = Router(config=load(_ROUTING), providers=_model_providers())
    app.include_router(
        build_knowledge_router(
            service=KnowledgeService(
                engine=engine,
                embedder=_RouterEmbedder(router),
                dense=QdrantRepository(client=qdrant),
                lexical=MeilisearchRepository(client=meili),
            ),
            authenticate=authenticate,
        )
    )

    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) not in _DOCUMENTATION
    ]

    @app.get(_HEALTHZ)
    async def healthz() -> Response:
        return Response(status_code=200)

    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(scope: Any):
        async with previous(scope):
            try:
                yield
            finally:
                await client.aclose()
                await meili.aclose()
                await qdrant.close()
                await engine.dispose()
                await lookup.dispose()

    app.router.lifespan_context = lifespan
    return app


class _RouterEmbedder:
    """`Router.embed` behind the one-method shape ingestion asks for.

    The same five lines the inbound worker needs. Duplicated rather than shared
    because the alternative is `moc.api` importing `moc.workers`, which the
    layered contract forbids in exactly that direction — and correctly: the
    console must not be able to reach a worker.
    """

    def __init__(self, router: Any) -> None:
        self._router = router

    async def embed(self, *, texts: Any) -> Any:
        return await self._router.embed(texts=texts)


def _model_providers() -> dict[str, Any]:
    """Both providers, because `Router` validates every configured task at
    construction — a router built with a subset raises before it is asked."""
    from moc.llm.anthropic_direct import AnthropicDirect
    from moc.llm.openai_direct import OpenAIDirect

    routing = load(_ROUTING)
    return {
        "anthropic": AnthropicDirect(
            api_key=os.environ.get("MOC_ANTHROPIC_API_KEY", ""), http=routing["http"]
        ),
        "openai": OpenAIDirect(
            api_key=os.environ.get("MOC_OPENAI_API_KEY", ""), http=routing["http"]
        ),
    }


def missing_environment() -> list[str]:
    """Variables this process needs and does not have.

    Returned rather than raised so the preflight can report all of them at
    once. A process that dies on the first missing name teaches an operator to
    fix one variable per restart.
    """
    required = ["MOC_LOOKUP_PASSWORD", "MOC_VALKEY_PASSWORD"]
    return [name for name in required if not os.environ.get(name)]


def console_missing_environment() -> list[str]:
    """What the console process needs on top of the webhook process's set."""
    required = [
        "MOC_APP_PASSWORD",
        "MOC_LOOKUP_PASSWORD",
        "MOC_VALKEY_PASSWORD",
        "MOC_ANTHROPIC_API_KEY",
        "MOC_OPENAI_API_KEY",
    ]
    return [name for name in required if not os.environ.get(name)]


__all__ = [
    "console_app",
    "console_missing_environment",
    "missing_environment",
    "webhook_app",
]
