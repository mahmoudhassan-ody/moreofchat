"""Tenant identity — demo plan Task 30.

Three tenants see the demo, and each has to see *themselves*: their name in
the header, their logo, their data. A console that shows our mark where their
logo belongs reads as a product that does not know who they are, which is the
opposite of what a pilot is meant to demonstrate.

**White-label is deliberately not this.** Their colours and their domain stay
out of scope: our mark in the corner of a working product is how the
university's IT director tells the broker about us.

`test_logo_upload_rejects_a_non_image_by_content_not_extension` is the one with
teeth. An extension is a claim by whoever is uploading, and the whole point of
checking is that the claim might be false.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from moc.tenancy.context import tenant_session

PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 40
JPEG = bytes.fromhex("ffd8ffe0") + b"\x00" * 40
GIF = b"GIF89a" + b"\x00" * 40
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 40
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'


@pytest_asyncio.fixture(loop_scope="session")
async def branded(engine, tenant_tables):
    from moc.tenancy.models import Tenant

    ids = {"a": uuid.uuid4(), "b": uuid.uuid4()}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        s.add_all(
            [
                Tenant(id=ids["a"], slug="sinai", name="Sinai University",
                       vertical="education"),
                Tenant(id=ids["b"], slug="cairo-homes", name="Cairo Homes",
                       vertical="realestate"),
            ]
        )
        await s.commit()
    yield ids
    async with AsyncSession(engine, expire_on_commit=False) as s:
        for table in tenant_tables:
            await s.execute(text(f"DELETE FROM {table}"))  # noqa: S608
        await s.commit()


# ─────────────────────────── the header ───────────────────────────


async def test_the_header_shows_the_tenants_name_and_logo(app_engine, branded):
    from moc.tenancy.branding import BrandingStore

    store = BrandingStore(engine=app_engine)
    await store.set_logo(tenant_id=branded["a"], content=PNG, filename="crest.png")
    brand = await store.brand(tenant_id=branded["a"])

    assert brand.name == "Sinai University"
    assert brand.has_logo is True
    assert brand.media_type == "image/png"


async def test_a_tenant_without_a_logo_falls_back_to_its_initials(app_engine, branded):
    """Not to the More Of Chat mark — a tenant seeing our logo where theirs
    should be reads as a product that does not know who they are."""
    from moc.tenancy.branding import BrandingStore

    brand = await BrandingStore(engine=app_engine).brand(tenant_id=branded["a"])

    assert brand.has_logo is False
    assert brand.initials == "SU"


async def test_initials_come_from_an_arabic_name_too(app_engine, engine, branded):
    """Every tenant in the pilot writes its own name in Arabic somewhere, and
    a fallback that only works in Latin is a fallback that fails in the demo's
    own language."""
    from moc.tenancy.branding import initials

    assert initials("جامعة سيناء") == "جس"
    assert initials("Sinai") == "SI"
    assert initials("  spaced   out  name ") == "SO"


async def test_a_tenant_with_no_usable_name_still_renders_something(app_engine):
    """Empty initials would render an empty box, which reads as a broken
    console rather than as a missing logo."""
    from moc.tenancy.branding import initials

    assert initials("") == "?"
    assert initials("   ") == "?"


# ─────────────────────────── the upload ───────────────────────────


@pytest.mark.parametrize(
    ("content", "media_type"),
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp")],
)
async def test_a_real_image_is_accepted_and_typed_from_its_bytes(
    app_engine, branded, content, media_type
):
    from moc.tenancy.branding import BrandingStore

    store = BrandingStore(engine=app_engine)
    # A deliberately wrong extension: the stored type must come from the
    # content, so a .txt holding a PNG is stored as a PNG.
    await store.set_logo(tenant_id=branded["a"], content=content, filename="logo.txt")

    assert (await store.brand(tenant_id=branded["a"])).media_type == media_type


async def test_logo_upload_rejects_a_non_image_by_content_not_extension(
    app_engine, branded
):
    """The extension is a claim by whoever is uploading, and the reason to
    check at all is that the claim might be false.

    The payload here is the shape that matters: a file named `.png` whose
    bytes are a script. Trusting the name stores it, and whatever serves it
    later decides what it is.
    """
    from moc.tenancy.branding import BrandingStore, NotAnImage

    store = BrandingStore(engine=app_engine)
    with pytest.raises(NotAnImage):
        await store.set_logo(
            tenant_id=branded["a"],
            content=b"#!/bin/sh\nrm -rf /\n",
            filename="logo.png",
        )
    assert (await store.brand(tenant_id=branded["a"])).has_logo is False


async def test_an_svg_is_refused_even_though_it_is_an_image(app_engine, branded):
    """SVG is a document format that executes.

    A tenant-supplied SVG can carry `<script>`, and a console that renders one
    inline has handed an uploader a cross-tenant XSS. Refused rather than
    sanitised: sanitising SVG is a losing arms race, and nobody's logo needs
    to be one.
    """
    from moc.tenancy.branding import BrandingStore, NotAnImage

    with pytest.raises(NotAnImage, match="SVG"):
        await BrandingStore(engine=app_engine).set_logo(
            tenant_id=branded["a"], content=SVG, filename="logo.svg"
        )


async def test_an_oversized_logo_is_refused(app_engine, branded):
    """A cap, because this column is read on every console page load and the
    row travels through RLS with the rest of the tenant."""
    from moc.tenancy.branding import MAX_LOGO_BYTES, BrandingStore, NotAnImage

    with pytest.raises(NotAnImage, match="too large"):
        await BrandingStore(engine=app_engine).set_logo(
            tenant_id=branded["a"],
            content=PNG + b"\x00" * MAX_LOGO_BYTES,
            filename="logo.png",
        )


# ─────────────────────────── isolation ───────────────────────────


async def test_tenant_branding_is_tenant_scoped(app_engine, branded):
    """The same guarantee as everything else, from the app role.

    Not "filtered to the right one" — the other tenant's row does not exist
    under this session, which is why a miss is None rather than a different
    tenant's crest.
    """
    from moc.tenancy.branding import BrandingStore

    store = BrandingStore(engine=app_engine)
    await store.set_logo(tenant_id=branded["a"], content=PNG, filename="a.png")
    await store.set_logo(tenant_id=branded["b"], content=JPEG, filename="b.png")

    assert (await store.brand(tenant_id=branded["a"])).media_type == "image/png"
    assert (await store.brand(tenant_id=branded["b"])).media_type == "image/jpeg"

    async with tenant_session(app_engine, branded["a"]) as session:
        visible = (
            await session.execute(text("SELECT count(*) FROM tenants"))
        ).scalar_one()
    assert visible == 1, "a console session sees exactly its own tenant"


async def test_reading_a_logo_that_was_never_uploaded_is_none(app_engine, branded):
    from moc.tenancy.branding import BrandingStore

    assert await BrandingStore(engine=app_engine).logo(tenant_id=branded["b"]) is None
