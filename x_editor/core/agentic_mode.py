"""Agentic provider/mode selection for X-Editor."""

from __future__ import annotations

import os
from pathlib import Path


GEMINI_MODE = "gemini"
OPEN_GLM_QWEN_OMNI_MODE = "open_glm_qwen_omni"
DEFAULT_AGENTIC_MODE_ENV = "X_EDITOR_AGENTIC_MODE"

GLM_PROVIDER = "glm_4_7_flash"
QWEN3_OMNI_PROVIDER = "qwen3_omni"

DEFAULT_GLM_MODEL = "glm-4.7-flash"
DEFAULT_QWEN3_OMNI_MODEL = "Qwen3Omni-Instruct"


def normalize_agentic_mode(value: str | None) -> str:
    """Return a canonical agentic mode name."""

    raw = (value or GEMINI_MODE).strip().lower().replace("-", "_")
    aliases = {
        "": GEMINI_MODE,
        "default": GEMINI_MODE,
        "llm": GEMINI_MODE,
        "gemini_planner": GEMINI_MODE,
        "gemini_executor": GEMINI_MODE,
        "open": OPEN_GLM_QWEN_OMNI_MODE,
        "open_source": OPEN_GLM_QWEN_OMNI_MODE,
        "open_glm": OPEN_GLM_QWEN_OMNI_MODE,
        "glm": OPEN_GLM_QWEN_OMNI_MODE,
        "glm_qwen": OPEN_GLM_QWEN_OMNI_MODE,
        "glm_qwen_omni": OPEN_GLM_QWEN_OMNI_MODE,
    }
    mode = aliases.get(raw, raw)
    if mode not in {GEMINI_MODE, OPEN_GLM_QWEN_OMNI_MODE}:
        raise ValueError(
            f"Unsupported X-Editor agentic mode: {value!r}; "
            f"expected {GEMINI_MODE!r} or {OPEN_GLM_QWEN_OMNI_MODE!r}"
        )
    return mode


def resolve_agentic_mode(value: str | None = None) -> str:
    """Resolve mode from an explicit value or ``X_EDITOR_AGENTIC_MODE``."""

    return normalize_agentic_mode(value or os.getenv(DEFAULT_AGENTIC_MODE_ENV))


def is_open_agentic_mode(value: str | None = None) -> bool:
    return resolve_agentic_mode(value) == OPEN_GLM_QWEN_OMNI_MODE


def planner_provider_name(mode: str | None = None) -> str:
    return GLM_PROVIDER if is_open_agentic_mode(mode) else GEMINI_MODE


def perception_provider_name(mode: str | None = None) -> str:
    return QWEN3_OMNI_PROVIDER if is_open_agentic_mode(mode) else GEMINI_MODE


def resolve_openai_compatible_api_key(provider: str, explicit: str | None = None) -> str | None:
    """Resolve a non-Gemini OpenAI-compatible provider key without committing secrets."""

    if explicit:
        return explicit.strip()
    provider_key = _provider_key(provider)
    for env_name in _provider_env_names(provider_key, suffix="API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    for env_name in _provider_env_names(provider_key, suffix="API_KEY_FILE"):
        path = os.getenv(env_name)
        if path:
            value = _read_secret_file(Path(path))
            if value:
                return value
    return None


def resolve_openai_compatible_base_url(provider: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit.strip()
    provider_key = _provider_key(provider)
    for env_name in _provider_env_names(provider_key, suffix="OPENAI_BASE_URL"):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    for env_name in _provider_env_names(provider_key, suffix="BASE_URL"):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    return None


def resolve_openai_compatible_model(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    provider_key = _provider_key(provider)
    for env_name in _provider_env_names(provider_key, suffix="MODEL"):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    if provider_key == "GLM":
        return DEFAULT_GLM_MODEL
    if provider_key == "QWEN3_OMNI":
        return DEFAULT_QWEN3_OMNI_MODEL
    return os.getenv("X_EDITOR_OPENAI_COMPAT_MODEL", "").strip() or "openai-compatible-model"


def provider_service_hint(provider: str) -> str:
    """Human-readable setup hint for service-unavailable errors."""

    provider_key = _provider_key(provider)
    if provider_key == "GLM":
        return (
            "configure X_EDITOR_GLM_OPENAI_BASE_URL and X_EDITOR_GLM_API_KEY "
            "(or X_EDITOR_OPENAI_COMPAT_BASE_URL / X_EDITOR_OPENAI_COMPAT_API_KEY)"
        )
    if provider_key == "QWEN3_OMNI":
        return (
            "configure X_EDITOR_QWEN3_OMNI_OPENAI_BASE_URL and X_EDITOR_QWEN3_OMNI_API_KEY "
            "(or X_EDITOR_OPENAI_COMPAT_BASE_URL / X_EDITOR_OPENAI_COMPAT_API_KEY)"
        )
    return "configure the provider OpenAI-compatible base URL and API key"


def _provider_key(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"glm", "glm_4_7_flash", "glm_47_flash", "glm_flash"}:
        return "GLM"
    if normalized in {"qwen", "qwen3", "qwen3_omni", "open_omni"}:
        return "QWEN3_OMNI"
    return normalized.upper()


def _provider_env_names(provider_key: str, *, suffix: str) -> list[str]:
    return [
        f"X_EDITOR_{provider_key}_{suffix}",
        f"X_EDITOR_OPENAI_COMPAT_{suffix}",
    ]


def _read_secret_file(path: Path) -> str | None:
    try:
        value = path.expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None
