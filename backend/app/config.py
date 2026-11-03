"""Configuration settings for the LandAssist FastAPI backend."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    app_name: str = "LandAssist OSM Rapid API"
    app_version: str = "0.1.0"
    database_url: str | None = None
    default_model_version: str = "v1.0.0"
    cors_origins: list[str] = ["*"]
    max_features_per_request: int = 5000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
