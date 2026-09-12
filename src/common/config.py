"""Shared settings, loaded from environment variables / .env (never hardcoded)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me-min-8-chars"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "fraud_detection"
    postgres_user: str = "fraud_app"
    postgres_password: str = "change-me"

    # Pinned, never "latest": the scoring service loads one specific version so
    # a bad retrain cannot silently reach the demo (locked decision).
    model_version: str = "gnn_v2"
    embedding_version: str = "gnn_emb_v2"

    # Above this score a transaction is flagged for analyst review. Not tuned to
    # a metric - it is an operational choice about analyst queue volume.
    flag_threshold: float = 0.9

    service_api_key: str = "change-me"
    jwt_secret_key: str = "change-me-use-a-long-random-value"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    @field_validator("model_version", "embedding_version", mode="before")
    @classmethod
    def _blank_falls_back_to_default(cls, value: str | None, info: ValidationInfo) -> str:
        """`.env.example` ships these keys blank, and a blank env var otherwise
        beats the code default - which silently produced an unversioned
        artifact path and a confusing FileNotFoundError at startup.
        """
        if value:
            return value
        return str(cls.model_fields[info.field_name].default)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
