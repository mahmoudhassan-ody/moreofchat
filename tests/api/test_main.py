"""The processes that run — demo plan Task 39.

Everything in this codebase was assembled inside a test before this. These are
the assertions about the *process*: what it publishes, and what it can reach.

`test_the_public_process_holds_no_application_database_credentials` is the one
that matters. The webhook process is the only thing an unauthenticated stranger
can talk to, and its whole security value is what it cannot get to — the same
argument migration 0007 makes about the `moc_lookup` role. A convenience import
of the app engine here would erase that in one line, with nothing failing.
"""

import ast
import inspect
from pathlib import Path

from moc.api import main

MODULE = Path(main.__file__)
SOURCE = MODULE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def test_the_public_process_publishes_no_api_documentation():
    """A map of every path, parameter and payload shape, in a form built for
    tooling, on the one host strangers can reach."""
    app = main.webhook_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    assert not paths & {"/docs", "/redoc", "/openapi.json"}
    assert "/webhooks/twilio/whatsapp" in paths, "the filter removed the routes too"


async def test_the_public_process_answers_a_health_check_and_says_nothing():
    """Empty on purpose. A health endpoint that names its dependencies tells a
    stranger what to attack and when it is degraded enough to be worth
    trying."""
    import httpx

    app = main.webhook_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.content == b""


def test_the_public_process_holds_no_application_database_credentials():
    """Structural. The webhook process resolves a tenant through `moc_lookup`
    — SELECT on five columns — and publishes to a queue. It has no login that
    could read a conversation, a message, a fee or a unit price.

    Asserted on names rather than behaviour because the failure has none: an
    `app_database_url` here works perfectly, and the process merely becomes
    able to read everything.
    """
    builder = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "webhook_app"
    )
    reached = {
        node.attr for node in ast.walk(builder) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(builder) if isinstance(node, ast.Name)}

    forbidden = {"app_database_url", "database_url", "AgentDirectory", "tenant_session"}
    assert not reached & forbidden, (
        f"the public process reaches {reached & forbidden}; its security value is "
        "what it cannot get to"
    )
    assert "lookup_engine" in reached


def test_a_missing_credential_is_reported_all_at_once(monkeypatch):
    """A process that dies on the first missing variable teaches an operator to
    fix one per restart."""
    monkeypatch.delenv("MOC_LOOKUP_PASSWORD", raising=False)
    monkeypatch.delenv("MOC_VALKEY_PASSWORD", raising=False)
    assert set(main.missing_environment()) == {
        "MOC_LOOKUP_PASSWORD",
        "MOC_VALKEY_PASSWORD",
    }


def test_the_composition_root_is_the_only_place_these_are_built():
    """`webhook_app` is a function, not an import-time singleton.

    An app object built at import time opens a connection pool in every process
    that imports the module — including the test collector, and including the
    worker, which has no business holding the lookup role.
    """
    assert inspect.isfunction(main.webhook_app)
    assert "app = webhook_app()" not in SOURCE
