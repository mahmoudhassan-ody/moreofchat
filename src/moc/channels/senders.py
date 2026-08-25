"""One tenant's own outbound credentials, resolved per job — demo plan Task 39.

Every adapter in this package is built from one tenant's credentials. The
Twilio adapter says so in its own docstring: the sender comes from that
tenant's `channel_accounts` row and is never platform-wide. Until this file
existed there was no way to honour that — the worker took a mapping of channel
to adapter and held it for the life of the process, so every tenant's replies
would have gone out from whichever number the process was started with. The
customer gets an answer from a business they never wrote to: correctly worded,
on the right channel, under the wrong name, with nothing raising at either end.

`typing_for` returns the same account's typing indicator (§2.5, Task 40) —
a different host, a different API version and a different client, cached under
its own key so that closing one does not close the other's connection pool.

**Which secret is which.** The inbound-verification secret is what
`channel_accounts.secret_ref` names. Outbound credentials are different values
and hang off the same reference with a suffix, so an operator can derive every
variable name from one database row:

    twilio/sinai/wa          the Twilio auth token   (verifies inbound signatures)
    twilio/sinai/wa/sid      the Twilio account SID  (identifies the account)
    telegram/sinai/bot       the webhook secret token (verifies inbound)
    telegram/sinai/bot/token the bot token            (sends)

The Telegram pair is the one worth stating twice: the value that verifies an
inbound update and the value that sends a message are *different secrets*, and
an adapter handed the wrong one authenticates nothing and sends nothing, in
that order.

**A missing credential is an error, never an empty string.** `EnvSecretResolver`
raises rather than returning "", because an adapter built with a blank token
fails every send with the vendor's authentication error — which reads as an
outage and gets escalated to the vendor rather than to whoever forgot the
variable.
"""

from typing import Any

from sqlalchemy import text

from moc.channels.base import Channel, MessagingProvider
from moc.tenancy.context import tenant_session

_ACCOUNT = text(
    "SELECT id, address, secret_ref FROM channel_accounts WHERE channel = :channel"
)

#: The suffixes above, in one place.
_SID = "/sid"
_TOKEN = "/token"  # noqa: S105 - a reference suffix, not a secret
_API_KEY = "/apikey"
#: A cache-key discriminator, not a reference suffix. The typing indicator is a
#: different client on a different host from the sender for the same account.
_TYPING = "typing"


class NoAccount(LookupError):
    """This tenant has no connected account on this channel."""


class SqlSenderRegistry:
    """(tenant, channel) -> the adapter built from that tenant's row.

    Cached per pair, because each adapter owns an `httpx.AsyncClient` and
    building one per reply would open a connection pool per message. The cache
    is the reason `aclose` exists: the worker owns this object's lifetime and
    the clients' with it.

    No query here names a tenant. The account is read inside a `tenant_session`
    and RLS does the scoping, so a registry asked for tenant A's sender cannot
    return tenant B's row even if the channel matches — which is exactly the
    mistake this class exists to prevent, and it would be a filter somebody
    could forget rather than a policy.
    """

    def __init__(self, *, engine: Any, secrets: Any, transport: Any = None) -> None:
        self._engine = engine
        self._secrets = secrets
        #: Injected only by tests. Production passes nothing and httpx opens
        #: real sockets.
        self._transport = transport
        self._cache: dict[tuple[str, ...], Any] = {}

    async def for_job(self, *, tenant_id: str, channel: str) -> MessagingProvider | None:
        key = (str(tenant_id), channel, "send")
        if key not in self._cache:
            built = await self._build(tenant_id=tenant_id, channel=channel)
            if built is None:
                return None
            self._cache[key] = built
        return self._cache[key]

    async def _account(self, *, tenant_id: str, channel: str) -> Any:
        """The tenant's row for this channel. No query here names a tenant —
        RLS does the scoping, so a registry asked for tenant A cannot return
        tenant B's row even when the channel matches."""
        import uuid as _uuid

        async with tenant_session(self._engine, _uuid.UUID(str(tenant_id))) as session:
            return (await session.execute(_ACCOUNT, {"channel": channel})).one_or_none()

    async def _build(self, *, tenant_id: str, channel: str) -> MessagingProvider | None:
        row = await self._account(tenant_id=tenant_id, channel=channel)
        if row is None:
            # None rather than an exception: the worker turns it into a dead
            # letter naming the tenant and the channel, which is a better
            # message than a traceback and reaches the same place.
            return None

        ref = row.secret_ref
        if channel == Channel.whatsapp:
            from moc.channels.twilio_wa import TwilioWhatsApp

            return TwilioWhatsApp(
                account_sid=self._secrets.for_ref(ref + _SID),
                auth_token=self._secrets.for_ref(ref),
                sender=row.address,
                transport=self._transport,
            )
        if channel == Channel.telegram:
            from moc.channels.telegram import TelegramBot

            # Not `for_ref(ref)`: that is the webhook secret, which verifies
            # inbound updates and cannot send anything.
            return TelegramBot(
                token=self._secrets.for_ref(ref + _TOKEN), transport=self._transport
            )
        if channel in (Channel.messenger, Channel.instagram):
            from moc.channels.meta import MetaMessenger

            return MetaMessenger(
                page_id=row.address,
                access_token=self._secrets.for_ref(ref + _TOKEN),
                transport=self._transport,
            )
        if channel == Channel.email:
            from moc.channels.sendgrid_email import SendGridEmail

            return SendGridEmail(
                api_key=self._secrets.for_ref(ref + _API_KEY),
                sender=row.address,
                transport=self._transport,
            )
        return None

    async def typing_for(self, *, tenant_id: str, channel: str) -> Any:
        """That tenant's typing indicator, or None where the channel has none.

        WhatsApp and Telegram have one; the rest return None. Returning None
        rather than raising is the point: the worker treats an absent
        indicator and a failed one identically, because both mean "no hint"
        and neither is worth a turn.

        Cached separately from the sender: it is a different host, a different
        API version and a different client, and sharing one would mean the
        first `aclose` closed the other's connection pool.
        """
        key = (str(tenant_id), channel, _TYPING)
        if key in self._cache:
            return self._cache[key]
        if channel not in (Channel.whatsapp, Channel.telegram):
            return None

        row = await self._account(tenant_id=tenant_id, channel=channel)
        if row is None:
            return None

        if channel == Channel.telegram:
            from moc.channels.telegram import TelegramTypingIndicator

            # The bot token, the same one the sender uses. No second
            # credential, which is why this channel got an indicator for the
            # cost of a config block.
            indicator: Any = TelegramTypingIndicator(
                token=self._secrets.for_ref(row.secret_ref + _TOKEN),
                transport=self._transport,
            )
            self._cache[key] = indicator
            return indicator

        from moc.channels.twilio_wa import TwilioTypingIndicator

        indicator = TwilioTypingIndicator(
            account_sid=self._secrets.for_ref(row.secret_ref + _SID),
            auth_token=self._secrets.for_ref(row.secret_ref),
            transport=self._transport,
        )
        self._cache[key] = indicator
        return indicator

    async def aclose(self) -> None:
        for provider in self._cache.values():
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
        self._cache.clear()


__all__ = ["NoAccount", "SqlSenderRegistry"]
