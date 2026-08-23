"""Settings and scripts over HTTP — demo plan Task 33.

**`GET /settings` returns the bounds as well as the values, and that is the
whole design.** The console renders one control per declared setting, so a
setting the engine refuses is one the screen cannot draw. A frontend holding
its own list would show a control for something the backend has withdrawn —
which is the disabled-slider failure arriving from the other side.

A greyed-out control implies the setting exists and you are not allowed it. The
truth is that a floor is a property of the system rather than a permission
level, and the honest rendering of "not settable" is "not there".

**A refusal is 422 with the reason.** A tenant who tried to make retrieval more
permissive has done something reasonable and gets told why it is not on offer,
rather than a silent clamp that leaves the screen saying one thing and the
system doing another.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from moc.agent.scripts import NoDraft, ScriptStore
from moc.api.inbox import AgentPrincipal
from moc.tenancy.settings import BelowFloor, SettingsStore, bounds

_REFUSED = 422
_NOT_FOUND = 404


class Changes(BaseModel):
    changes: dict[str, Any]


class Draft(BaseModel):
    body: dict[str, Any]


def build_settings_router(
    *, settings: SettingsStore, scripts: ScriptStore, authenticate: Any
) -> APIRouter:
    router = APIRouter()

    @router.get("/settings")
    async def read(request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        return {
            # What may be offered, and within what. The console draws from
            # this and never from a list of its own.
            "bounds": bounds(),
            "values": await settings.effective(tenant_id=principal.tenant_id),
        }

    @router.put("/settings")
    async def write(payload: Changes, request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        try:
            values = await settings.put(
                tenant_id=principal.tenant_id,
                changes=payload.changes,
                agent_id=principal.agent_id,
            )
        except BelowFloor as refused:
            raise HTTPException(status_code=_REFUSED, detail=str(refused)) from None
        return {"values": values}

    @router.get("/scripts/{script_id}")
    async def read_script(script_id: str, request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        published = await scripts.current(
            tenant_id=principal.tenant_id, script_id=script_id
        )
        return {
            "scriptId": script_id,
            "publishedVersion": published.version if published else None,
            "body": published.body if published else None,
        }

    @router.put("/scripts/{script_id}")
    async def save_draft(script_id: str, payload: Draft, request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        draft = await scripts.save_draft(
            tenant_id=principal.tenant_id, script_id=script_id, body=payload.body
        )
        return {"version": draft.version, "published": draft.published}

    @router.get("/scripts/{script_id}/preview")
    async def preview(script_id: str, request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        try:
            return await scripts.preview(
                tenant_id=principal.tenant_id, script_id=script_id
            )
        except NoDraft as missing:
            raise HTTPException(status_code=_NOT_FOUND, detail=str(missing)) from None

    @router.post("/scripts/{script_id}/publish")
    async def publish(script_id: str, request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        try:
            version = await scripts.publish(
                tenant_id=principal.tenant_id,
                script_id=script_id,
                agent_id=principal.agent_id,
            )
        except NoDraft as missing:
            # 404 rather than a success that published nothing. A publish
            # button that reports "done" over an empty draft is a button that
            # teaches a tenant to distrust the screen.
            raise HTTPException(status_code=_NOT_FOUND, detail=str(missing)) from None
        return {"version": version.version, "published": True}

    return router


__all__ = ["Changes", "Draft", "build_settings_router"]
