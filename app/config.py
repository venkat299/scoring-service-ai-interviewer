"""Application configuration utilities."""
from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    gemini_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    # Support both DEFAULT_TIMEOUT_SECONDS and legacy REQUEST_TIMEOUT env vars
    default_timeout_seconds: int = Field(
        default=60,
        validation_alias=AliasChoices("DEFAULT_TIMEOUT_SECONDS", "REQUEST_TIMEOUT"),
    )
    
    # Handle empty strings coming from env files (e.g. VAR=)
    @field_validator("default_timeout_seconds", mode="before")
    @classmethod
    def _coerce_timeout(cls, v: Optional[str]) -> int | str | None:  # type: ignore[override]
        if isinstance(v, str) and v.strip() == "":
            return 60
        return v
    # Ignore unrelated env vars present in .env or the environment
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of :class:`Settings`."""

    return Settings()
