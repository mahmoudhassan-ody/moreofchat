"""Usage metering.

Every billable event lands in `usage_ledger` the moment it happens. The row is
attributed from the transaction's `moc.tenant_id`, never from a caller-supplied
argument — so a caller cannot bill the wrong tenant, and a caller that forgot to
open a tenant session writes nothing at all.
"""

from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Defined in models.py so the Postgres enum type has exactly one source, and
# re-exported here because metering is the import surface callers use.
from moc.tenancy.models import UsageKind

__all__ = ["UsageKind", "record_usage"]


_INSERT = text(
    """
INSERT INTO usage_ledger (
  id, tenant_id, kind, channel, quantity, model, provider,
  input_tokens, output_tokens, cached_tokens, provider_cost_usd, degraded
) VALUES (
  gen_random_uuid(),
  nullif(current_setting('moc.tenant_id', true), '')::uuid,
  :kind, :channel, :quantity, :model, :provider,
  :input_tokens, :output_tokens, :cached_tokens, :provider_cost_usd, :degraded
)
"""
)


async def record_usage(
    session: AsyncSession,
    *,
    kind: UsageKind,
    channel: str | None = None,
    quantity: int = 1,
    model: str | None = None,
    provider: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    provider_cost_usd: Decimal | None = None,
    degraded: bool = False,
) -> None:
    """Append one billable event to the ledger.

    `degraded` marks usage served by the failover provider (design doc §2.4);
    those rows are priced differently and are the signal for a provider incident.
    """
    await session.execute(
        _INSERT,
        {
            "kind": UsageKind(kind).value,
            "channel": channel,
            "quantity": quantity,
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "provider_cost_usd": provider_cost_usd,
            "degraded": degraded,
        },
    )
