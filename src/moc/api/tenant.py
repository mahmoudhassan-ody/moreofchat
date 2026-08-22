"""The tenant's own identity, over HTTP — demo plan Task 30.

Four routes and one rule: **the tenant is the authenticated one, always.** No
route here takes a tenant id, in a path segment or anywhere else. `/tenant`
means "the tenant whose session cookie you presented", which is Task 28's
guarantee carried forward — a `/tenants/{id}/logo` would be an authorization
question on every request, and this way there is no question to get wrong.

**The logo is uploaded as a raw body, not as a multipart form.** Multipart
would add a dependency for parsing a filename this module refuses to trust
anyway: the media type is sniffed from the bytes, so the envelope carrying them
has nothing to contribute. It also keeps the upload path free of a parser.

**It is served back under the type it was sniffed as, with `nosniff`.** A
browser asked to guess the type of a tenant-supplied file is a browser that can
be talked into guessing `text/html`, and the response header is the only thing
standing between a stored file and a rendered page.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from moc.api.inbox import AgentPrincipal
from moc.tenancy.branding import Brand, BrandingStore, NotAnImage

_NOT_FOUND = 404
_UNSUPPORTED = 415

#: The console reloads the header on every navigation and the crest changes
#: about never. Private, because it is one tenant's image and a shared cache
#: has no business holding it.
_CACHE = "private, max-age=300"


def build_tenant_router(*, store: BrandingStore, authenticate: Any) -> APIRouter:
    router = APIRouter(prefix="/tenant")

    @router.get("")
    async def read(request: Request) -> dict[str, Any]:
        """Name, initials, timezone — everything the header renders except the
        image, which has its own route because it is an image."""
        brand = await _brand(store, await authenticate(request))
        return {
            "name": brand.name,
            # Sent whether or not there is a logo. The console should not have
            # to compute a fallback, and two implementations of "what stands
            # for this tenant" would disagree the day a name changes.
            "initials": brand.initials,
            "hasLogo": brand.has_logo,
            "timezone": brand.timezone,
            # The tenant's default REPLY language (§8.3), not the console's —
            # the console's is per agent. Named in full because the two have
            # been conflated before.
            "defaultReplyLanguage": brand.default_language,
        }

    @router.get("/logo")
    async def read_logo(request: Request) -> Response:
        principal = await authenticate(request)
        found = await store.logo(tenant_id=principal.tenant_id)
        if found is None:
            # 404, and the console renders initials. Not a placeholder image:
            # a tenant with no crest should see their own initials, never our
            # mark or a grey square.
            raise HTTPException(status_code=_NOT_FOUND, detail="no logo")
        content, media_type = found
        return Response(
            content=content,
            media_type=media_type,
            headers={
                # The one header that matters here. Without it a browser may
                # sniff a tenant-supplied file into something executable, and
                # the type we carefully derived from the bytes stops being the
                # type the browser acts on.
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": _CACHE,
            },
        )

    @router.put("/logo")
    async def write_logo(request: Request) -> dict[str, str]:
        principal = await authenticate(request)
        try:
            media_type = await store.set_logo(
                tenant_id=principal.tenant_id, content=await request.body()
            )
        except NotAnImage as refused:
            # 415, with the reason. A tenant refused for uploading an SVG has
            # done something reasonable and deserves to know it was refused
            # for what SVG can do rather than for what it is.
            raise HTTPException(status_code=_UNSUPPORTED, detail=str(refused)) from None
        return {"mediaType": media_type}

    @router.delete("/logo")
    async def clear_logo(request: Request) -> dict[str, bool]:
        principal = await authenticate(request)
        await store.clear_logo(tenant_id=principal.tenant_id)
        return {"ok": True}

    return router


async def _brand(store: BrandingStore, principal: AgentPrincipal) -> Brand:
    brand = await store.brand(tenant_id=principal.tenant_id)
    if brand is None:  # pragma: no cover - a live session whose tenant vanished
        raise HTTPException(status_code=_NOT_FOUND, detail="no tenant")
    return brand


__all__ = ["build_tenant_router"]
