from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


@asynccontextmanager
async def tenant_session(engine: AsyncEngine, tenant_id: UUID) -> AsyncIterator[AsyncSession]:
    """Open a session scoped to one tenant.

    set_config's third argument is `true`, making the setting local to the
    transaction. With connection pooling a session-level setting would leak
    the tenant id into the next request that reuses the connection.

    Transaction-local means it is also *lost* on commit: the next statement
    starts a fresh transaction where `moc.tenant_id` is unset, RLS filters
    everything, and the session silently reads nothing. The `after_begin` hook
    re-applies the setting to every transaction the session opens, so the scope
    is the session for the caller and still the transaction on the wire.
    """
    async with AsyncSession(engine, expire_on_commit=False) as s:

        @event.listens_for(s.sync_session, "after_begin")
        def _set_tenant(session, transaction, connection) -> None:
            connection.execute(
                text("SELECT set_config('moc.tenant_id', :t, true)"), {"t": str(tenant_id)}
            )

        try:
            # Force a transaction now so the setting is live before the first
            # statement — `after_begin` has nothing to fire on until then.
            await s.execute(text("SELECT 1"))
            yield s
        finally:
            event.remove(s.sync_session, "after_begin", _set_tenant)
