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
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response

from moc.channels.accounts import EnvSecretResolver, SqlChannelAccountRegistry, lookup_engine
from moc.channels.valkey import ValkeyEventLog, ValkeyInboundQueue
from moc.config_store import load

_QUEUES = "workers/queues"
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


def missing_environment() -> list[str]:
    """Variables this process needs and does not have.

    Returned rather than raised so the preflight can report all of them at
    once. A process that dies on the first missing name teaches an operator to
    fix one variable per restart.
    """
    required = ["MOC_LOOKUP_PASSWORD", "MOC_VALKEY_PASSWORD"]
    return [name for name in required if not os.environ.get(name)]


__all__ = ["missing_environment", "webhook_app"]
