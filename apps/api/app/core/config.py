"""
TRUSTRAG API — core settings.

Reads from environment (via .env) and from config/models.yaml.
Business code must import from this module — never read env vars directly.

Separation of concerns:
  .env          → secrets, deployment-specific values (GEMINI_API_KEY, URIs, etc.)
  models.yaml   → model IDs, thresholds, tuning parameters, retrieval config
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Paths ────────────────────────────────────────────────────────────────────

# apps/api/ root (one level above app/)
_API_ROOT = Path(__file__).resolve().parents[2]
_MODELS_YAML_PATH = _API_ROOT / "config" / "models.yaml"
# Repo root config/ports.yaml — canonical port registry (see scripts/apply_ports.py)
_PORTS_YAML_PATH = _API_ROOT.parent.parent / "config" / "ports.yaml"

# P0-CFG FIX (2026-09-06 audit): ModelConfig reads os.environ directly while
# Settings loads .env via pydantic-settings (which does NOT export to
# os.environ). Without this, .env values like EMBEDDING_PROVIDER/AI_PROVIDER
# were silently ignored and models.yaml defaults won (e.g. Settings said
# huggingface while ModelConfig reported google_genai). Loading .env into
# os.environ here keeps both paths consistent.
load_dotenv(_API_ROOT / ".env", override=False)
load_dotenv(_API_ROOT.parent.parent / ".env", override=False)


def _load_models_yaml() -> dict[str, Any]:
    """Load and parse config/models.yaml. Fails loudly on missing/malformed file."""
    if not _MODELS_YAML_PATH.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {_MODELS_YAML_PATH}. "
            "This file must exist — it is the centralized config registry."
        )
    with _MODELS_YAML_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"models.yaml must be a YAML mapping. Got: {type(data)}")
    return data


def _load_ports_yaml() -> dict[str, int]:
    """Load repo-root config/ports.yaml. Returns {} if absent (dev fallback)."""
    try:
        if not _PORTS_YAML_PATH.exists():
            return {}
        with _PORTS_YAML_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        ports = (data or {}).get("ports", {}) if isinstance(data, dict) else {}
        return {k: int(v) for k, v in ports.items() if isinstance(v, int)}
    except Exception:
        return {}


# Local LLM base URLs derived once from the canonical port registry so a fresh
# checkout works with zero provider config — explicit env vars still win
# (pydantic env > Field default), and models.yaml stays the ID source.
_PORTS_FALLBACK = _load_ports_yaml()
_DEFAULT_OLLAMA_BASE_URL = f"http://localhost:{_PORTS_FALLBACK.get('ollama', 11434)}"
_DEFAULT_LLAMACPP_BASE_URL = f"http://127.0.0.1:{_PORTS_FALLBACK.get('llamacpp', 8080)}/v1"


# ─── Settings ─────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Application settings.

    Values come from environment variables (or .env file).
    Model/AI configuration is read from models.yaml via model_config property.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    app_name: str = "TRUSTRAG"
    app_version: str = "0.1.0"

    # ── Security ──────────────────────────────────────────────────────────────
    jwt_secret: str
    jwt_expiry_minutes: int = 60
    cors_origins: str = "http://localhost:5173"
    trusted_proxy_ips: str = ""  # Comma-separated proxy IPs/CIDRs allowed to supply X-Forwarded-For

    # ── Hugging Face ──────────────────────────────────────────────────────────
    hf_token: str = ""  # Optional read-only token to prevent download rate-limits

    # ── Google Gemini (Optional if using local LLMs) ───────────────────────────
    gemini_api_key: str = ""

    # ── Local LLM Providers (Ollama & llama.cpp) ──────────────────────────────
    # Defaults derive from config/ports.yaml (see _DEFAULT_*_BASE_URL above);
    # set OLLAMA_BASE_URL / LLAMACPP_BASE_URL env vars to override per deploy
    # (e.g. host.docker.internal inside containers).
    ollama_base_url: str = Field(
        default=_DEFAULT_OLLAMA_BASE_URL,
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_HOST"),
        description="Ollama local API server endpoint",
    )
    ollama_model: str = Field(
        default="",
        validation_alias=AliasChoices("OLLAMA_MODEL"),
        description="Override Ollama model name from models.yaml via env",
    )
    llamacpp_base_url: str = Field(
        default=_DEFAULT_LLAMACPP_BASE_URL,
        validation_alias=AliasChoices("LLAMACPP_BASE_URL", "LLAMA_CPP_BASE_URL"),
        description="llama.cpp server OpenAI-compatible base URL",
    )
    llamacpp_model: str = Field(
        default="",
        validation_alias=AliasChoices("LLAMACPP_MODEL", "LLAMA_CPP_MODEL"),
        description="Override llama.cpp model identifier from models.yaml via env",
    )

    # ── NVIDIA NIM & Tavily Search ─────────────────────────────────────────────
    nvidia_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("NVIDIA_API_KEY", "NIM_API_KEY"),
        description="NVIDIA NIM API key for Llama, Mistral, and Nemotron models",
    )
    tavily_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("TAVILY_API_KEY"),
        description="Tavily AI Search API key",
    )

    # ── Multi-Provider Engine Selectors ────────────────────────────────────────
    ai_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("AI_PROVIDER", "LLM_PROVIDER"),
        description=(
            "Active AI generation & verification provider: 'ollama', "
            "'llama_cpp', 'gemini', or 'nvidia'"
        ),
    )
    embedding_provider: str = Field(
        default="huggingface",
        validation_alias=AliasChoices("EMBEDDING_PROVIDER", "EMBEDDING_BACKEND"),
        description="Active embedding engine: 'huggingface' (local-only)",
    )
    search_provider: str = Field(
        default="auto",
        validation_alias=AliasChoices("SEARCH_PROVIDER"),
        description="Web search engine: 'auto', 'tavily', 'duckduckgo', or 'both'",
    )

    # ── MongoDB Atlas ──────────────────────────────────────────────────────────
    mongodb_uri: str
    mongodb_database: str = "trustrag_db"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""  # Empty string = no auth (local dev)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_analyses_per_minute: int = 10
    rate_limit_auth_per_minute: int = 20
    rate_limit_upload_per_minute: int = 10
    rate_limit_url_ingest_per_minute: int = 10

    # ── Model Configuration Overrides (env takes precedence over models.yaml) ──
    gemini_model: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_MODEL", "LLM_MODEL"),
        description="Override primary LLM model ID in .env",
    )
    gemini_verification_model: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_VERIFICATION_MODEL", "VERIFICATION_MODEL"),
        description="Override verification LLM model ID in .env",
    )
    gemini_embedding_model: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "LOCAL_EMBEDDING_MODEL"),
        description="Override embedding model ID in .env",
    )
    embedding_dim: int | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_DIM", "EMBEDDING_DIMENSIONALITY"),
        description="Override embedding dimensionality in .env",
    )

    # ── Derived: parsed CORS list ─────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def trusted_proxy_list(self) -> list[str]:
        return [p.strip() for p in self.trusted_proxy_ips.split(",") if p.strip()]

    # ── Validation ────────────────────────────────────────────────────────────
    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_must_be_strong(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError(
                "JWT_SECRET must be at least 32 characters. "
                'Generate with: python -c "import secrets; print(secrets.token_hex(64))"'
            )
        return v

    @field_validator("app_env")
    @classmethod
    def valid_app_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}, got '{v}'")
        return v

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v_upper

    @model_validator(mode="after")
    def production_must_have_qdrant_key(self) -> Settings:
        if self.app_env == "production" and not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY must be set in production")
        return self

    def is_production(self) -> bool:
        return self.app_env == "production"

    def is_development(self) -> bool:
        return self.app_env == "development"


# ─── Model config (from models.yaml) ─────────────────────────────────────────


class ModelConfig:
    """
    Typed access to models.yaml sections.

    This is the ONLY place application code reads model IDs, thresholds,
    retrieval parameters, and reliability policy. Never read models.yaml
    directly in routes, services, or LangGraph nodes.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def _get(self, *keys: str, required: bool = True) -> Any:
        node: Any = self._data
        path = ".".join(keys)
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                if required:
                    raise KeyError(f"Required key '{path}' missing from models.yaml")
                return None
            node = node[key]
        return node

    # ── Config version ────────────────────────────────────────────────────────
    @property
    def config_version(self) -> str:
        return str(self._get("runtime", "config_version"))

    # ── LLM ──────────────────────────────────────────────────────────────────
    @property
    def llm_provider(self) -> str:
        val = self._get("llm", "provider", required=False)
        env_val = os.environ.get("AI_PROVIDER") or os.environ.get("LLM_PROVIDER")
        if env_val:
            return env_val.lower()
        return str(val or "ollama").lower()

    @property
    def llm_model(self) -> str:
        self._get("llm")
        return self.llm_model_for(self.llm_provider)

    def llm_model_for(self, provider: str) -> str:
        """Resolve the model id for an explicit provider (not the configured one)."""
        self._get("llm")
        p = provider.lower()
        if p == "ollama":
            env_model = os.environ.get("OLLAMA_MODEL")
            return (
                env_model
                or str(self._get("llm", "model_ollama", required=False) or "")
                or str(self._get("llm", "model") or "granite4.2:3b-q4_K_M")
            )
        if p in ("llama_cpp", "llamacpp"):
            env_model = os.environ.get("LLAMACPP_MODEL") or os.environ.get("LLAMA_CPP_MODEL")
            return (
                env_model
                or str(self._get("llm", "model_llamacpp", required=False) or "")
                or str(self._get("llm", "model") or "ibm-granite/granite-4.2-3b-GGUF:Q4_K_M")
            )
        env_model = os.environ.get("LLM_MODEL") or os.environ.get("GEMINI_MODEL")
        if env_model:
            return env_model
        if self.llm_provider in ("nvidia", "nim"):
            return "meta/llama-3.3-70b-instruct"
        return str(self._get("llm", "model") or "gemini-3.5-flash-lite")

    @property
    def ollama_base_url(self) -> str:
        env_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
        if env_url:
            return env_url
        port = get_ports().get("ollama")
        if port:
            return f"http://localhost:{port}"
        return str(self._get("llm", "ollama_base_url", required=False) or "http://localhost:11434")

    @property
    def llamacpp_base_url(self) -> str:
        env_url = os.environ.get("LLAMACPP_BASE_URL") or os.environ.get("LLAMA_CPP_BASE_URL")
        if env_url:
            return env_url
        port = get_ports().get("llamacpp")
        if port:
            return f"http://127.0.0.1:{port}/v1"
        return str(
            self._get("llm", "llamacpp_base_url", required=False) or "http://127.0.0.1:8080/v1"
        )

    @property
    def llm_temperature(self) -> float:
        return float(self._get("llm", "temperature"))

    @property
    def llm_top_p(self) -> float:
        return float(self._get("llm", "top_p"))

    @property
    def llm_max_output_tokens(self) -> int:
        return int(self._get("llm", "max_output_tokens"))

    @property
    def llm_timeout_seconds(self) -> int:
        return int(self._get("llm", "timeout_seconds"))

    @property
    def llm_max_retries(self) -> int:
        return int(self._get("llm", "max_retries"))

    # ── Embedding ─────────────────────────────────────────────────────────────
    @property
    def embedding_provider(self) -> str:
        val = self._get("embedding", "provider", required=False)
        env_val = os.environ.get("EMBEDDING_PROVIDER") or os.environ.get("EMBEDDING_BACKEND")
        if env_val:
            return env_val.lower()
        return str(val or "huggingface").lower()

    @property
    def embedding_model(self) -> str:
        val = self._get("embedding", "model")
        env_model = os.environ.get("EMBEDDING_MODEL") or os.environ.get("LOCAL_EMBEDDING_MODEL")
        if env_model:
            return env_model
        return str(val or "BAAI/bge-small-en-v1.5")

    @property
    def embedding_dimensionality(self) -> int:
        val = int(self._get("embedding", "output_dimensionality"))
        env_dim = os.environ.get("EMBEDDING_DIM") or os.environ.get("EMBEDDING_DIMENSIONALITY")
        return int(env_dim) if env_dim is not None else val

    @property
    def embedding_version(self) -> str:
        return str(self._get("embedding", "version"))

    @property
    def embedding_cache_dir(self) -> str:
        return str(self._get("embedding", "cache_dir", required=False) or ".model_cache")

    @property
    def embedding_max_seq_length(self) -> int:
        val = self._get("embedding", "max_seq_length", required=False)
        if val is not None:
            return int(val)
        return 512  # Default for BGE-small

    # ── Verification ──────────────────────────────────────────────────────────
    @property
    def verification_provider(self) -> str:
        val = self._get("verification", "provider", required=False)
        env_val = os.environ.get("AI_PROVIDER") or os.environ.get("LLM_PROVIDER")
        if env_val:
            return env_val.lower()
        return str(val or "ollama").lower()

    @property
    def verification_model(self) -> str:
        return self.verification_model_for(self.verification_provider)

    def verification_model_for(self, provider: str) -> str:
        """Resolve the verifier model id for an explicit provider."""
        p = provider.lower()
        if p == "ollama":
            env_model = os.environ.get("OLLAMA_MODEL")
            return (
                env_model
                or str(self._get("verification", "model_ollama", required=False) or "")
                or str(self._get("verification", "model") or "granite4.2:3b-q4_K_M")
            )
        if p in ("llama_cpp", "llamacpp"):
            env_model = os.environ.get("LLAMACPP_MODEL") or os.environ.get("LLAMA_CPP_MODEL")
            return (
                env_model
                or str(self._get("verification", "model_llamacpp", required=False) or "")
                or str(
                    self._get("verification", "model") or "ibm-granite/granite-4.2-3b-GGUF:Q4_K_M"
                )
            )
        val = self._get("verification", "model")
        env_model = os.environ.get("GEMINI_VERIFICATION_MODEL") or os.environ.get(
            "VERIFICATION_MODEL"
        )
        if env_model:
            return env_model
        if self.verification_provider in ("nvidia", "nim"):
            return "meta/llama-3.3-70b-instruct"
        return str(val or "gemini-3.5-flash-lite")

    @property
    def verification_temperature(self) -> float:
        return float(self._get("verification", "temperature"))

    @property
    def verification_max_output_tokens(self) -> int:
        return int(self._get("verification", "max_output_tokens"))

    @property
    def verification_timeout_seconds(self) -> int:
        return int(self._get("verification", "timeout_seconds"))

    @property
    def fused_decompose_verify(self) -> bool:
        val = self._get("verification", "fused_decompose_verify", required=False)
        env_val = os.environ.get("FUSED_DECOMPOSE_VERIFY")
        if env_val is not None:
            return env_val.strip().lower() in ("1", "true", "yes", "on")
        if val is None:
            return True
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    @property
    def max_verification_time_seconds(self) -> int:
        return int(self._get("verification", "max_verification_time_seconds"))

    # ── Reranker ─────────────────────────────────────────────────────────────
    @property
    def reranker_enabled(self) -> bool:
        return bool(self._get("reranker", "enabled"))

    @property
    def reranker_model(self) -> str:
        return self._get("reranker", "model")

    @property
    def reranker_top_k(self) -> int:
        return int(self._get("reranker", "top_k"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    @property
    def dense_top_k(self) -> int:
        return int(self._get("retrieval", "dense_top_k"))

    @property
    def sparse_top_k(self) -> int:
        return int(self._get("retrieval", "sparse_top_k"))

    @property
    def fusion_method(self) -> str:
        return self._get("retrieval", "fusion_method")

    @property
    def rrf_k(self) -> int:
        return int(self._get("retrieval", "rrf_k"))

    @property
    def fusion_top_k(self) -> int:
        return int(self._get("retrieval", "fusion_top_k"))

    @property
    def sparse_k1(self) -> float:
        return float(self._get("retrieval", "sparse_k1", required=False) or 1.2)

    @property
    def sparse_b(self) -> float:
        return float(self._get("retrieval", "sparse_b", required=False) or 0.75)

    @property
    def sparse_avg_len_tokens(self) -> int:
        return int(self._get("retrieval", "sparse_avg_len_tokens", required=False) or 128)

    @property
    def max_context_chunks(self) -> int:
        return int(self._get("retrieval", "max_context_chunks"))

    @property
    def router_enabled(self) -> bool:
        return bool(self._get("retrieval", "query_router", "enabled", required=False) is not False)

    @property
    def max_fanout_sub_queries(self) -> int:
        value = self._get("retrieval", "query_router", "max_sub_queries", required=False)
        return int(value) if value is not None else 3

    # ── Ingestion ─────────────────────────────────────────────────────────────
    @property
    def chunk_size(self) -> int:
        return int(self._get("ingestion", "chunk_size"))

    @property
    def chunk_overlap(self) -> int:
        return int(self._get("ingestion", "chunk_overlap"))

    @property
    def supported_formats(self) -> list[str]:
        return list(self._get("ingestion", "supported_formats"))

    @property
    def max_file_size_mb(self) -> int:
        return int(self._get("ingestion", "max_file_size_mb"))

    # ── OCR fallback ─────────────────────────────────────────────────────
    @property
    def ocr_enabled(self) -> bool:
        return bool(self._get("ingestion", "ocr", "enabled", required=False) is not False)

    @property
    def ocr_min_native_chars(self) -> int:
        return int(self._get("ingestion", "ocr", "min_native_chars", required=False) or 50)

    @property
    def ocr_dpi(self) -> int:
        return int(self._get("ingestion", "ocr", "dpi", required=False) or 300)

    @property
    def ocr_min_confidence(self) -> float:
        return float(self._get("ingestion", "ocr", "min_confidence", required=False) or 0.5)

    # ── Reliability ──────────────────────────────────────────────────────────
    @property
    def minimum_evidence_coverage(self) -> float:
        return float(self._get("reliability", "minimum_evidence_coverage"))

    @property
    def maximum_contradiction_rate(self) -> float:
        return float(self._get("reliability", "maximum_contradiction_rate"))

    @property
    def abstain_below(self) -> float:
        return float(self._get("reliability", "abstain_below"))

    # ── Recovery ─────────────────────────────────────────────────────────────
    @property
    def max_recovery_attempts(self) -> int:
        return int(self._get("recovery", "max_recovery_attempts"))

    @property
    def recovery_strategy_priority(self) -> list[str]:
        """Return ordered recovery strategies, e.g. ['query_rewrite', 're_retrieve']."""
        val = self._get("recovery", "strategy_priority", required=False)
        if isinstance(val, list) and val:
            return [str(s) for s in val]
        return ["query_rewrite", "re_retrieve"]

    # ── Cost controls ─────────────────────────────────────────────────────────
    @property
    def max_input_tokens(self) -> int:
        return int(self._get("cost_controls", "max_input_tokens"))

    @property
    def max_verification_claims(self) -> int:
        return int(self._get("cost_controls", "max_verification_claims"))

    @property
    def max_individual_nli_fallback(self) -> int:
        value = self._get("cost_controls", "max_individual_nli_fallback", required=False)
        return int(value) if value is not None else 5

    @property
    def max_claim_retrievals(self) -> int:
        value = self._get("cost_controls", "max_claim_retrievals", required=False)
        return int(value) if value is not None else 3

    @property
    def claim_retrieval_top_k(self) -> int:
        value = self._get("cost_controls", "claim_retrieval_top_k", required=False)
        return int(value) if value is not None else 5

    def as_snapshot(self) -> dict[str, Any]:
        """Return a flat dict for recording with each analysis run."""
        return {
            "config_version": self.config_version,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "embedding_dimensionality": self.embedding_dimensionality,
            "verification_provider": self.verification_provider,
            "verification_model": self.verification_model,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": self.reranker_model if self.reranker_enabled else None,
            "fusion_method": self.fusion_method,
            "fusion_top_k": self.fusion_top_k,
            "max_fanout_sub_queries": self.max_fanout_sub_queries,
            "ocr_enabled": self.ocr_enabled,
            "sparse_k1": self.sparse_k1,
            "sparse_b": self.sparse_b,
            "sparse_avg_len_tokens": self.sparse_avg_len_tokens,
            "max_context_chunks": self.max_context_chunks,
            "abstain_below": self.abstain_below,
            "max_recovery_attempts": self.max_recovery_attempts,
            "max_claim_retrievals": self.max_claim_retrievals,
        }


# ─── Singletons ───────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application Settings singleton."""
    return Settings()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    """Return the cached ModelConfig singleton loaded from models.yaml."""
    raw = _load_models_yaml()
    return ModelConfig(raw)


@lru_cache(maxsize=1)
def get_ports() -> dict[str, int]:
    """Return the canonical port registry from repo-root config/ports.yaml."""
    return _load_ports_yaml()


def reload_ports() -> dict[str, int]:
    """Clear cached ports and re-read config/ports.yaml."""
    get_ports.cache_clear()
    return get_ports()


def reload_settings() -> Settings:
    """Clear cached settings singleton and re-read environment variables."""
    get_settings.cache_clear()
    return get_settings()
