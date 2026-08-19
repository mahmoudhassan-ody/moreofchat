from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOC_", env_file=".env", extra="ignore")

    pg_password: str
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_database: str = "moc_dev"
    app_password: str
    app_user: str = "moc_app"

    # The pre-tenant bootstrap role (migration 0007). Separate credentials
    # from `moc_app` on purpose: this one authenticates nothing and runs
    # before a tenant is known, so its reach is a security property and
    # sharing a login with the application role would erase the distinction.
    lookup_password: str = ""
    lookup_user: str = "moc_lookup"

    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_key: str = ""

    meili_host: str = "127.0.0.1"
    meili_port: int = 7700
    meili_key: str = ""

    valkey_password: str = ""
    valkey_host: str = "127.0.0.1"
    valkey_port: int = 6379
    valkey_db: int = 0

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    def valkey_url(self, db: int | None = None) -> str:
        """Streams and the token bucket both live here.

        Password is inline rather than a separate client argument because
        `redis.from_url` is the one construction path every caller uses, and a
        second way to build a client is a second way to forget the password.
        """
        secret = f":{self.valkey_password}@" if self.valkey_password else ""
        index = self.valkey_db if db is None else db
        return f"redis://{secret}{self.valkey_host}:{self.valkey_port}/{index}"

    def app_database_url(self, database: str | None = None) -> str:
        """URL for the restricted moc_app role. RLS applies to this role."""
        return (
            f"postgresql+asyncpg://{self.app_user}:{self.app_password}"
            f"@{self.pg_host}:{self.pg_port}/{database or self.pg_database}"
        )

    def lookup_database_url(self, database: str | None = None) -> str:
        """URL for `moc_lookup`, whose only privilege is SELECT on the
        channel-account view. Its whole security value is what it cannot
        reach, so it never shares a connection pool with `moc_app`."""
        return (
            f"postgresql+asyncpg://{self.lookup_user}:{self.lookup_password}"
            f"@{self.pg_host}:{self.pg_port}/{database or self.pg_database}"
        )


settings = Settings()
