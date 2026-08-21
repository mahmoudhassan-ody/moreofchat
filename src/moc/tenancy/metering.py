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
from moc.tenancy.pricing import cost_usd

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

    `provider_cost_usd` is computed from `config/llm/pricing.yaml` when the
    caller does not supply one. Priced here rather than at each call site
    because there are four of them and four is four chances to forget — and
    the one that forgot would be invisible, which is how the column came to be
    null on every row it has ever held.

    An explicit value still wins: a caller holding the provider's own reported
    figure — a batch discount, a negotiated rate — must not have it recomputed
    from a list price. And a model with no confirmed rate stores NULL rather
    than 0, so a run total says how much of itself it could not price.
    """
    if provider_cost_usd is None and model:
        provider_cost_usd = cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
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
