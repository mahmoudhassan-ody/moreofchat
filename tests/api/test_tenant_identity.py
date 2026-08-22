"""The tenant identity API — demo plan Task 30.

Through the real app with the real cookie authenticator, because the property
under test is not "the store filters correctly" — that is
`tests/tenancy/test_branding.py` — but that **no route here can be pointed at
another tenant**. There is no id in any path, so there is nothing to point.
"""

import uuid
from contextlib import asynccontextmanager

import httpx
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.api.auth import build_auth_router, cookie_authenticator
from moc.api.tenant import build_tenant_router
from moc.config_store import load
from moc.tenancy.branding import BrandingStore

COOKIE = load("security/agents")["session"]["cookie"]["name"]
PASSWORD = "correct horse battery staple"  # noqa: S105 - a test fixture, not a secret
PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 40
JPEG = bytes.fromhex("ffd8ffe0") + b"\x00" * 40


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
                Tenant(id=ids["a"], slug="sinai-u", name="Sinai University",
                       vertical="education"),
                Tenant(id=ids["b"], slug="cairo-h", name="Cairo Homes",
                       vertical="realestate"),
            ]
        )
        await s.commit()

    directory = AgentDirectory(engine=app_engine, lookup=lookup_engine)
    await directory.create_agent(tenant_id=ids["a"], email="ali@sinai.example",
                                 password=PASSWORD, display_name="Ali")
    await directory.create_agent(tenant_id=ids["b"], email="basma@cairo.example",
                                 password=PASSWORD, display_name="Basma")

    store = BrandingStore(engine=app_engine)
    app = FastAPI()
    app.include_router(build_auth_router(directory=directory))
    app.include_router(
        build_tenant_router(store=store, authenticate=cookie_authenticator(directory=directory))
    )
    yield app, ids, store

    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://moc.example"
    )


@asynccontextmanager
async def signed_in(app, email: str):
    """A client holding a live session cookie.

    A context manager rather than a coroutine returning a client: httpx
    refuses to be entered twice, and `async with signed_in(...)` opened
    the client on the first request and then tried again on the `with`.
    """
    async with client(app) as http:
        await http.post("/auth/login", json={"email": email, "password": PASSWORD})
        yield http


async def test_the_header_shows_the_tenants_name_and_logo(console):
    app, ids, store = console
    await store.set_logo(tenant_id=ids["a"], content=PNG, filename="crest.png")

    async with signed_in(app, "ali@sinai.example") as http:
        brand = (await http.get("/tenant")).json()
        logo = await http.get("/tenant/logo")

    assert brand["name"] == "Sinai University"
    assert brand["hasLogo"] is True
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"
    assert logo.content == PNG


async def test_a_tenant_without_a_logo_falls_back_to_its_initials(console):
    """404 on the image, initials in the payload. Not a placeholder: a tenant
    with no crest sees their own initials, never our mark or a grey square."""
    app, _, _ = console
    async with signed_in(app, "basma@cairo.example") as http:
        brand = (await http.get("/tenant")).json()
        logo = await http.get("/tenant/logo")

    assert brand["initials"] == "CH"
    assert brand["hasLogo"] is False
    assert logo.status_code == 404


async def test_the_logo_is_served_with_nosniff(console):
    """The response header is the only thing between a stored file and a
    rendered page: a browser allowed to guess the type of tenant-supplied
    bytes can be talked into guessing text/html."""
    app, ids, store = console
    await store.set_logo(tenant_id=ids["a"], content=PNG)

    async with signed_in(app, "ali@sinai.example") as http:
        logo = await http.get("/tenant/logo")

    assert logo.headers["x-content-type-options"] == "nosniff"


async def test_logo_upload_rejects_a_non_image_by_content_not_extension(console):
    app, ids, store = console
    async with signed_in(app, "ali@sinai.example") as http:
        refused = await http.put("/tenant/logo", content=b"#!/bin/sh\nrm -rf /\n")

    assert refused.status_code == 415
    assert (await store.brand(tenant_id=ids["a"])).has_logo is False


async def test_an_uploaded_logo_is_typed_from_its_bytes(console):
    app, ids, store = console
    async with signed_in(app, "ali@sinai.example") as http:
        accepted = await http.put("/tenant/logo", content=JPEG)

    assert accepted.json()["mediaType"] == "image/jpeg"
    assert (await store.brand(tenant_id=ids["a"])).media_type == "image/jpeg"


async def test_tenant_branding_is_tenant_scoped(console):
    """Two agents, two tenants, one endpoint. The route names no tenant, so
    there is nothing for a client to change."""
    app, ids, store = console
    await store.set_logo(tenant_id=ids["a"], content=PNG)

    async with signed_in(app, "ali@sinai.example") as http:
        mine = (await http.get("/tenant")).json()
    async with signed_in(app, "basma@cairo.example") as http:
        theirs = (await http.get("/tenant")).json()
        # The header a frontend "already sends", naming the other tenant.
        spoofed = await http.get("/tenant", headers={"X-Tenant-Id": str(ids["a"])})
        their_logo = await http.get("/tenant/logo")

    assert mine["name"] == "Sinai University"
    assert theirs["name"] == "Cairo Homes"
    assert spoofed.json()["name"] == "Cairo Homes"
    assert their_logo.status_code == 404, "tenant A's crest is not reachable from B"


async def test_no_route_here_takes_a_tenant_id(console):
    """Structural, and the reason the behavioural test above can be short.

    A `/tenants/{tenant_id}/logo` would make authorization a decision on every
    request. There is no id in any path, so there is no decision to get wrong.
    """
    app, _, _ = console
    # From the OpenAPI document rather than by walking `app.routes`: this
    # FastAPI wraps an included router in a private object that exposes
    # neither `.path` nor `.routes`, and a walk that silently found nothing
    # would pass this test by describing an empty app.
    paths = [path for path in app.openapi()["paths"] if path.startswith("/tenant")]

    assert paths, "no tenant routes registered"
    assert not any("{" in path for path in paths), paths


async def test_every_tenant_route_requires_a_session(console):
    app, ids, store = console
    await store.set_logo(tenant_id=ids["a"], content=PNG)

    async with client(app) as http:
        assert (await http.get("/tenant")).status_code == 401
        assert (await http.get("/tenant/logo")).status_code == 401
        assert (await http.put("/tenant/logo", content=PNG)).status_code == 401
        assert (await http.delete("/tenant/logo")).status_code == 401


async def test_clearing_a_logo_returns_the_tenant_to_initials(console):
    app, ids, store = console
    await store.set_logo(tenant_id=ids["a"], content=PNG)

    async with signed_in(app, "ali@sinai.example") as http:
        await http.delete("/tenant/logo")
        brand = (await http.get("/tenant")).json()

    assert brand["hasLogo"] is False
    assert brand["initials"] == "SU"
