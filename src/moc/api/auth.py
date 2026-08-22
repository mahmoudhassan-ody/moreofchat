"""The console's login surface — demo plan Task 28.

**Flagged for line-by-line human review.** This module is thin on purpose: it
turns a cookie into an `AgentPrincipal` and it does nothing else. Every
decision worth reviewing lives in `moc.tenancy.agent_auth`, and the reason to
keep this file short is that it is the one an HTTP framework's conveniences
would otherwise creep into.

**The only thing read off the request is the session cookie.** Not a header,
not a query parameter, not a path segment. A structural test parses this module
and asserts that `request.cookies` is the sole attribute of `request` it
touches, because the bypass this whole task exists to prevent arrives as a
convenience — the frontend already knows the tenant and already sends it, and
the value is correct in every test anyone writes until the day someone edits
it.

A cookie is client input too, and that is fine for a different reason: it
carries a 256-bit random token that is checked against a stored hash. It is a
bearer secret rather than a claim. `X-Tenant-Id: <uuid>` is a claim, and no
amount of it being usually true makes it checkable.

**401 for both "no cookie" and "bad cookie".** An expired session and a forged
one are the same answer, because the difference is what tells someone whether a
token they hold is live.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from moc.api.inbox import AgentPrincipal
from moc.tenancy.agent_auth import AgentDirectory

_UNAUTHENTICATED = 401
_SECONDS_PER_HOUR = 3600


class Credentials(BaseModel):
    """A body model rather than parameters pulled off the request.

    Pydantic validates and FastAPI keeps them out of the URL — a password in a
    query string is a password in the access log, the browser history and every
    proxy in between.
    """

    email: str
    password: str


def cookie_authenticator(*, directory: AgentDirectory) -> Any:
    """An `AgentAuthenticator` — a request in, a principal out, or 401.

    Returns the principal built from the *session row*. The tenant on it was
    read from the database under a token nobody can forge, which is the whole
    contract `moc.api.inbox` depends on: "whatever produces the principal is
    the only thing that decides which tenant's data a request sees."
    """
    name = _cookie()["name"]

    async def authenticate(request: Request) -> AgentPrincipal:
        session = await directory.resolve(token=request.cookies.get(name, ""))
        if session is None:
            # One answer for no cookie, an expired cookie and a forged one.
            raise HTTPException(status_code=_UNAUTHENTICATED, detail="not authenticated")
        return AgentPrincipal(
            tenant_id=session.tenant_id, agent_id=str(session.agent_id)
        )

    return authenticate


def build_auth_router(*, directory: AgentDirectory) -> APIRouter:
    """`/auth/login` and `/auth/logout`.

    Logout revokes the row *and* clears the cookie, in that order. Clearing
    only the cookie is the client agreeing to stop using a token that still
    works, which is no help at all against the case logout exists for.
    """
    router = APIRouter(prefix="/auth")
    settings = _cookie()

    @router.post("/login")
    async def login(credentials: Credentials, response: Response) -> dict[str, str]:
        issued = await directory.login(
            email=credentials.email, password=credentials.password
        )
        if issued is None:
            # Deliberately identical for an unknown address, a wrong password
            # and a suspended account. The distinction is what enumerates who
            # has an account here.
            raise HTTPException(
                status_code=_UNAUTHENTICATED, detail="check your email and password"
            )
        response.set_cookie(
            key=settings["name"],
            value=issued.token,
            httponly=bool(settings["httponly"]),
            samesite=settings["samesite"],
            secure=bool(settings["secure"]),
            path=settings["path"],
            # Matched to the server-side TTL rather than computed from the
            # session, so the browser stops presenting a token at the moment
            # the database stops accepting it. A cookie outliving its row is a
            # 401 the user reads as the console being broken.
            max_age=_max_age(),
        )
        # The token is not in the body. It is in an httpOnly cookie precisely
        # so that no script can read it, and echoing it here would hand it back
        # to the script that could not.
        return {"agent_id": str(issued.session.agent_id)}

    @router.post("/logout")
    async def logout(request: Request, response: Response) -> dict[str, bool]:
        await directory.logout(token=request.cookies.get(settings["name"], ""))
        response.delete_cookie(key=settings["name"], path=settings["path"])
        # True regardless. A logout that reported whether the token had been
        # live would answer the one question a stolen-token holder has.
        return {"ok": True}

    return router


def _max_age() -> int:
    from moc.config_store import load

    return int(load("security/agents")["session"]["ttl_hours"]) * _SECONDS_PER_HOUR


def _cookie() -> dict[str, Any]:
    from moc.config_store import load

    return load("security/agents")["session"]["cookie"]


__all__ = ["Credentials", "build_auth_router", "cookie_authenticator"]
