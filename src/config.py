"""Configuration and environment validation.

This module centralizes runtime settings so every part of the project reads
configuration in one consistent way. Keeping settings in one place prevents
"mystery defaults" and makes debugging easier when something behaves oddly.

How this module is designed:
1. `Settings` loads values from environment variables and optional `.env` file.
2. `validate_env()` explicitly checks required variables for a given workflow.
3. We keep required checks out of import-time so tools like `pytest` can run
   even when remote-service keys are not present.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Why this class exists:
    - To provide typed defaults for local development.
    - To map all environment knobs to explicit fields.
    - To give a single object (`settings`) every module can depend on.

    Note:
    We intentionally allow some keys to be optional at load time. Required-key
    checks are deferred to `validate_env()` so we can fail fast only when a
    specific feature is invoked.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    debug: bool = Field(default=False, alias="DEBUG")
    small_mode_limit: int = Field(default=100, alias="SMALL_MODE_LIMIT")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    fallback_embedding_dim: int = Field(default=1536, alias="FALLBACK_EMBEDDING_DIM")

    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="earningscall-rag", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    pinecone_namespace: str = Field(default="earnings-call-rag", alias="PINECONE_NAMESPACE")

    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="earningscall-rag", alias="LANGSMITH_PROJECT")
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")


settings = Settings()


def validate_env(required_vars: Iterable[str]) -> None:
    """Fail fast if required environment variables are missing.

    Parameters
    ----------
    required_vars:
        Iterable of variable names (for example, `OPENAI_API_KEY`) that must be
        present and non-empty before a workflow can proceed.

    Raises
    ------
    RuntimeError
        If one or more required variables are missing.

    Teaching note:
    We keep this function simple and explicit rather than "magical" because
    reliability is more important than cleverness in MLOps pipelines.
    """

    missing: list[str] = []
    for var_name in required_vars:
        # Convert ENV-like names to Settings attribute names.
        attr_name = var_name.lower()
        if hasattr(settings, attr_name):
            value = getattr(settings, attr_name)
        else:
            # Fall back to direct environment lookup for any variables not
            # modeled as attributes.
            import os

            value = os.getenv(var_name)

        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(var_name)

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(sorted(missing))
            + ". Please copy .env.example to .env and fill these values."
        )
