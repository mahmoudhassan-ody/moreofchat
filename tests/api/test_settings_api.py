"""Settings over HTTP — demo plan Task 33.

The property under test is what the screen is *allowed to draw*. The console
renders one control per setting the API declares, so the refusal and the
absence are the same mechanism seen from two ends.
"""

import uuid
from contextlib import asynccontextmanager

import httpx
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.agent.scripts import ScriptStore
from moc.api.auth import build_auth_router, cookie_authenticator
from moc.api.settings import build_settings_router
from moc.tenancy.settings import SettingsStore

PASSWORD = "correct horse battery staple"  # noqa: S105 - a test fixture, not a secret
DRAFT = {
    "version": 1,
    "script_id": "education_fees",
    "entry": "fees",
    "settings": {"max_consecutive_clarifications": 3},
    "nodes": {"fees": {"intents": ["fees"], "slots": ["faculty"]}},
}


@pytest_asyncio.fixture(loop_scope="session")
async def console(engine, app_engine, lookup_engine, tenant_tables):
    from moc.tenancy.agent_auth import AgentDirectory
    from moc.tenancy.models import Tenant

    ids = {"a": uuid.uuid4(), "b": uuid.uuid4()}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add_all(
            [
                Tenant(id=ids["a"], slug="cfg-a", name="A", vertical="education"),
                Tenant(id=ids["b"], slug="cfg-b", name="B", vertical="realestate"),
            ]
        )
        await s.commit()

    directory = AgentDirectory(engine=app_engine, lookup=lookup_engine)
    await directory.create_agent(tenant_id=ids["a"], email="ali@cfg-a.example",
                                 password=PASSWORD, display_name="Ali")
    await directory.create_agent(tenant_id=ids["b"], email="basma@cfg-b.example",
                                 password=PASSWORD, display_name="Basma")

    app = FastAPI()
    app.include_router(build_auth_router(directory=directory))
    app.include_router(
        build_settings_router(
            settings=SettingsStore(engine=app_engine),
            scripts=ScriptStore(engine=app_engine),
            authenticate=cookie_authenticator(directory=directory),
        )
    )
    yield app, ids

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


@asynccontextmanager
async def signed_in(app, email: str):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        await http.post("/auth/login", json={"email": email, "password": PASSWORD})
        yield http


async def test_the_response_carries_the_bounds_the_console_renders_from(console):
    """The console draws one control per declared setting. A setting the
    engine refuses is one the screen cannot draw — which is the honest
    rendering of "not settable", rather than a control that is there and
    greyed out."""
    app, _ = console
    async with signed_in(app, "ali@cfg-a.example") as http:
        body = (await http.get("/settings")).json()

    assert "min_score" in body["bounds"]
    assert "confidence_threshold" not in body["bounds"]
    assert body["values"]["min_score"] == body["bounds"]["min_score"]["min"]


async def test_lowering_the_gate_is_refused_with_the_reason(console):
    app, _ = console
    async with signed_in(app, "ali@cfg-a.example") as http:
        refused = await http.put("/settings", json={"changes": {"min_score": -0.2}})
        after = (await http.get("/settings")).json()

    assert refused.status_code == 422
    assert "floor" in refused.json()["detail"]
    assert after["values"]["min_score"] == after["bounds"]["min_score"]["min"]


async def test_raising_the_gate_is_allowed(console):
    app, _ = console
    async with signed_in(app, "ali@cfg-a.example") as http:
        raised = await http.put("/settings", json={"changes": {"min_score": 0.5}})

    assert raised.status_code == 200
    assert raised.json()["values"]["min_score"] == 0.5


async def test_a_setting_the_api_does_not_declare_is_refused(console):
    """A client that made one up. The console cannot produce this, which is
    exactly why the check has to be on the server."""
    app, _ = console
    async with signed_in(app, "ali@cfg-a.example") as http:
        refused = await http.put(
            "/settings", json={"changes": {"confidence_threshold": 0.01}}
        )
    assert refused.status_code == 422


async def test_synonyms_round_trip_and_are_tenant_scoped(console):
    """The demo moment: a broker adds their own name for an area."""
    app, _ = console
    async with signed_in(app, "basma@cfg-b.example") as http:
        await http.put(
            "/settings",
            json={"changes": {"synonyms": {"التجمع": ["التجمع الخامس"]}}},
        )
        theirs = (await http.get("/settings")).json()
    async with signed_in(app, "ali@cfg-a.example") as http:
        others = (await http.get("/settings")).json()

    assert theirs["values"]["synonyms"]["التجمع"] == ["التجمع الخامس"]
    assert others["values"]["synonyms"] == {}


async def test_a_script_is_drafted_previewed_then_published(console):
    app, _ = console
    async with signed_in(app, "ali@cfg-a.example") as http:
        assert (await http.get("/scripts/education_fees")).json()["publishedVersion"] is None
        await http.put("/scripts/education_fees", json={"body": DRAFT})
        preview = (await http.get("/scripts/education_fees/preview")).json()
        published = await http.post("/scripts/education_fees/publish")
        after = (await http.get("/scripts/education_fees")).json()

    assert preview["nodes"] == ["fees"]
    assert published.json()["version"] == 1
    assert after["publishedVersion"] == 1


async def test_publishing_nothing_is_a_404_not_a_success(console):
    """A publish button that reports "done" over an empty draft teaches a
    tenant to distrust the screen."""
    app, _ = console
    async with signed_in(app, "ali@cfg-a.example") as http:
        assert (await http.post("/scripts/never_drafted/publish")).status_code == 404


async def test_every_settings_route_requires_a_session(console):
    app, _ = console
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    ) as http:
        assert (await http.get("/settings")).status_code == 401
        assert (await http.put("/settings", json={"changes": {}})).status_code == 401
        assert (await http.get("/scripts/x")).status_code == 401
        assert (await http.post("/scripts/x/publish")).status_code == 401
