from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


@asynccontextmanager
async def tenant_session(engine: AsyncEngine, tenant_id: UUID) -> AsyncIterator[AsyncSession]:
    """Open a session scoped to one tenant.

    set_config's third argument is `true`, making the setting local to the
    transaction. With connection pooling a session-level setting would leak
    the tenant id into the next request that reuses the connection.
    """
    async with AsyncSession(engine, expire_on_commit=False) as s:
        await s.execute(
            text("SELECT set_config('moc.tenant_id', :t, true)"),
            {"t": str(tenant_id)},
        )
        yield s
