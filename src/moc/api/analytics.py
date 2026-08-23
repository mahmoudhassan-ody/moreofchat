"""Analytics over HTTP — demo plan Task 34.

One route. The shape it returns is the argument: `costIsComplete` travels with
the total, so a screen cannot render a number as final that the ledger could
not fully price.

**`containmentRate` is null rather than 1.0 for a tenant with no traffic.** A
rate over an empty denominator is a number nobody measured, and on this metric
it is the most flattering one available — a brand-new tenant would open the
console to a perfect score.
"""

from typing import Any

from fastapi import APIRouter, Request

from moc.api.inbox import AgentPrincipal
from moc.tenancy.analytics import AnalyticsStore


def build_analytics_router(*, store: AnalyticsStore, authenticate: Any) -> APIRouter:
    router = APIRouter(prefix="/analytics")

    @router.get("")
    async def read(request: Request) -> dict[str, Any]:
        principal: AgentPrincipal = await authenticate(request)
        report = await store.report(tenant_id=principal.tenant_id)
        return {
            "conversations": report.conversations,
            "handedOff": report.handed_off,
            "messages": report.messages,
            "containmentRate": report.containment_rate,
            # Strings, not floats. These are currency and the console renders
            # them; a float would round somewhere between here and the screen
            # and the rounding would be invisible.
            "costUsd": str(report.cost_usd),
            "costPerConversation": (
                str(report.cost_per_conversation)
                if report.cost_per_conversation is not None
                else None
            ),
            # The two fields that stop a partial total reading as a total.
            "unpricedCalls": report.unpriced_calls,
            "unpricedModels": list(report.unpriced_models),
            "costIsComplete": report.cost_is_complete,
            "byModel": [
                {
                    "model": row.model,
                    "calls": row.calls,
                    "inputTokens": row.input_tokens,
                    "outputTokens": row.output_tokens,
                    # Null, never "0.00": a model nothing could price has an
                    # unknown cost, and zero is a claim.
                    "costUsd": str(row.cost_usd) if row.cost_usd is not None else None,
                }
                for row in report.by_model
            ],
            "handoffReasons": [
                {"reason": reason, "times": times}
                for reason, times in report.handoff_reasons
            ],
        }

    return router


__all__ = ["build_analytics_router"]
