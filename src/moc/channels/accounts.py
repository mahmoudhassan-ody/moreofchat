"""Channel-account resolution — design §5, §6.1, and P1 Task 21.

**The one read that happens before a tenant exists.**

Every tenant-scoped query in this codebase runs as `moc_app` with
`moc.tenant_id` set, under RLS. This one cannot: it runs to *find out* which
tenant an inbound message belongs to, and the standard predicate returns
nothing when no tenant is set. Migration 0007 resolves that with a second,
much smaller door — the `moc_lookup` role, whose only privilege in the
database is SELECT on `channel_account_lookup`, a view over five columns.

This module is the only thing that opens that door. Not because the grants
would otherwise be unsafe — they bound the role regardless — but because "what
can the pre-authentication path reach?" should be answerable by reading one
file. A test asserts no other module builds a lookup connection.

**The secret is not here.** The view exposes `secret_ref`, a name; the signing
secret itself stays on the base table and out of the bootstrap path entirely.
That matters because the signing secret is exactly what an attacker needs to
forge a tenant's inbound traffic, and the role that runs before any signature
has been checked is the last place it belongs. `SecretResolver` turns the
reference into the value, from somewhere the lookup role cannot read.
"""

import os
from typing import Any

from sqlalchemy import text

from moc.channels.base import Channel, ChannelAccount, SecretResolver

_RESOLVE = text(
    "SELECT id, tenant_id, channel, address, secret_ref "
    "FROM channel_account_lookup "
    "WHERE channel = :channel AND address = :address"
)


class EnvSecretResolver:
    """Secrets from the process environment, keyed by reference.

    The deployable default on a single VPS, and a seam for a real secret store
    later. A reference like `twilio/acme/wa` reads
    `MOC_SECRET_TWILIO__ACME__WA` — the mangling is mechanical so an operator
    can derive the variable name from the database row without a lookup table.
    """

    prefix = "MOC_SECRET_"

    def variable_for(self, secret_ref: str) -> str:
        return self.prefix + secret_ref.replace("/", "__").replace("-", "_").upper()

    def for_ref(self, secret_ref: str) -> str:
        try:
            return os.environ[self.variable_for(secret_ref)]
        except KeyError:
            # Raised rather than returning "": an empty secret makes the
            # constant-time comparison fail every message, which presents as a
            # vendor outage rather than as a missing configuration value.
            raise KeyError(
                f"no secret configured for {secret_ref!r} "
                f"(expected {self.variable_for(secret_ref)})"
            ) from None


class SqlChannelAccountRegistry:
    """Vendor address -> tenant's channel account, through `moc_lookup`.

    Takes an engine rather than a session: this read happens outside any
    tenant transaction by definition, and accepting a session would invite a
    caller to hand it one already bound to `moc_app` — which would return
    nothing under RLS and look like an unknown address.
    """

    def __init__(self, *, engine: Any) -> None:
        self._engine = engine

    async def resolve(
        self, *, channel: Channel, account_ref: str
    ) -> ChannelAccount | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _RESOLVE, {"channel": str(channel), "address": account_ref}
                )
            ).one_or_none()
        if row is None:
            return None
        return ChannelAccount(
            id=row.id,
            tenant_id=row.tenant_id,
            channel=Channel(row.channel),
            account_ref=row.address,
            secret_ref=row.secret_ref,
        )


def lookup_engine(*, database: str | None = None) -> Any:
    """The one place a connection as `moc_lookup` is built.

    Deliberately not a module-level singleton: the caller owns the engine's
    lifetime, and an import-time connection would make every test that touches
    this package need a database.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from moc.config import settings

    return create_async_engine(settings.lookup_database_url(database))


__all__ = [
    "EnvSecretResolver",
    "SecretResolver",
    "SqlChannelAccountRegistry",
    "lookup_engine",
]
