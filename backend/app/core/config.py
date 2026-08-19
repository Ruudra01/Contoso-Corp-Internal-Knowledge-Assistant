"""Configuration. Every value comes from the environment or a .env file — no
credential is ever hard-coded or committed."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseSettings):
    """Structure-aware chunker tuning. Defaults match solution design §4.1."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", env_file=".env", extra="ignore")

    target_tokens: int = Field(default=800, gt=0, description="Preferred chunk size.")
    overlap_tokens: int = Field(default=120, ge=0, description="Overlap between splits of one section.")
    max_tokens: int = Field(default=1200, gt=0, description="Hard ceiling before a section is split.")
    min_tokens: int = Field(default=40, ge=0, description="Chunks below this are merged into a neighbour.")

    @model_validator(mode="after")
    def _check_bounds(self) -> "ChunkingSettings":
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS")
        if self.max_tokens < self.target_tokens:
            raise ValueError("CHUNK_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS")
        return self


class AzureOpenAISettings(BaseSettings):
    """Embedding model access. api_key is optional: when absent the client uses
    Managed Identity / DefaultAzureCredential, which is the deployed path."""

    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_", env_file=".env", extra="ignore")

    endpoint: str | None = None
    api_key: str | None = None
    api_version: str = "2024-10-21"
    embedding_deployment: str = "text-embedding-3-large"
    embedding_dimensions: int = Field(default=3072, gt=0)
    embedding_batch_size: int = Field(default=16, gt=0)


class AzureSearchSettings(BaseSettings):
    """Azure AI Search target index."""

    model_config = SettingsConfigDict(env_prefix="AZURE_SEARCH_", env_file=".env", extra="ignore")

    endpoint: str | None = None
    api_key: str | None = None
    index_name: str = "contoso-policies-v1"
    # Azure caps an indexing batch at 1000 documents / 16 MB.
    upload_batch_size: int = Field(default=100, gt=0, le=1000)
    # Candidate pool per query; solution design 4.2 retrieves 30 and reranks.
    default_top: int = Field(default=30, gt=0, le=1000)
    # The semantic ranker is where most of the precision gain comes from on a
    # corpus this size. Requires a Basic tier or above.
    use_semantic_ranker: bool = True


class IngestionSettings(BaseSettings):
    """Top-level ingestion settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    corpus_root: Path = Field(default=Path("data"), validation_alias="CORPUS_ROOT")
    source_uri_base: str = Field(
        default="",
        validation_alias="SOURCE_URI_BASE",
        description="Blob container base, e.g. https://acct.blob.core.windows.net/corpus/raw. "
        "Empty falls back to a file:// URI.",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    search: AzureSearchSettings = Field(default_factory=AzureSearchSettings)


@lru_cache
def get_settings() -> IngestionSettings:
    return IngestionSettings()
