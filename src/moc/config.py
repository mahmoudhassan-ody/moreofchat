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

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    def app_database_url(self, database: str | None = None) -> str:
        """URL for the restricted moc_app role. RLS applies to this role."""
        return (
            f"postgresql+asyncpg://{self.app_user}:{self.app_password}"
            f"@{self.pg_host}:{self.pg_port}/{database or self.pg_database}"
        )


settings = Settings()
