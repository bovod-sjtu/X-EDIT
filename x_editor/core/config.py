"""Configuration for X-Editor."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# Load .env file if present
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)
    except ImportError:
        pass


DEFAULT_RUNTIME_ARTIFACT_ROOT = (
    Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "x-editor"
    / "artifacts"
)


def runtime_artifact_root() -> Path:
    """Return the default root for ignored runtime outputs and demos."""

    return Path(os.getenv("X_EDITOR_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ARTIFACT_ROOT))).expanduser()


def runtime_artifact_path(*parts: str) -> Path:
    """Return a path below the runtime artifact root."""

    return runtime_artifact_root().joinpath(*parts)


def resolve_gemini_api_key(explicit: str | None = None) -> str | None:
    """Resolve the shared Gemini API key used by Planner and Executor.

    The lookup intentionally keeps secrets out of tracked source files:
    explicit argument -> GEMINI_API_KEY -> X_EDITOR_GEMINI_API_KEY_FILE.
    Release builds never search sibling projects for credentials.
    """

    if explicit:
        return explicit.strip()
    env_value = os.getenv("GEMINI_API_KEY")
    if env_value:
        return env_value.strip()

    key_file = os.getenv("X_EDITOR_GEMINI_API_KEY_FILE")
    if key_file:
        value = _read_secret_file(Path(key_file))
        if value:
            return value


    return None


def gemini_api_key_source() -> str | None:
    """Return a non-secret description of the selected Gemini key source."""

    if os.getenv("GEMINI_API_KEY"):
        return "env:GEMINI_API_KEY"
    if os.getenv("X_EDITOR_GEMINI_API_KEY_FILE"):
        path = Path(os.environ["X_EDITOR_GEMINI_API_KEY_FILE"])
        if _read_secret_file(path):
            return f"file:{path}"


    return None


def _read_secret_file(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    exported = _exported_gemini_key_from_text(value)
    if exported:
        return exported
    return value or None


def _read_exported_gemini_key(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _exported_gemini_key_from_text(text)


def _exported_gemini_key_from_text(text: str) -> str | None:
    match = re.search(r"^\s*export\s+GEMINI_API_KEY=(['\"]?)([^'\"\s]+)\1\s*$", text, re.MULTILINE)
    return match.group(2).strip() if match else None


class AudioEditConfig(BaseModel):
    """Configuration for the audio editing agent."""

    # Model settings
    omni_model: str = Field(default="gemini-2.5-pro", description="Omni agent model name")
    omni_api_key: Optional[str] = Field(default=None, description="API key for omni agent")
    omni_base_url: str = Field(
        default_factory=lambda: os.getenv("X_EDITOR_GEMINI_BASE_URL", ""),
        description="Base URL for omni agent API",
    )
    omni_temperature: float = Field(default=0.01, ge=0.0, le=2.0)
    omni_max_tokens: int = Field(default=40960, ge=1)
    omni_timeout: float = Field(default=60.0, ge=1.0)

    text_model: str = Field(default="gemini-3.1-flash", description="Text agent model name")
    text_api_key: Optional[str] = Field(default=None, description="API key for text agent")
    text_base_url: str = Field(
        default_factory=lambda: os.getenv("X_EDITOR_GEMINI_BASE_URL", ""),
        description="Base URL for text agent API",
    )
    text_temperature: float = Field(default=0.01, ge=0.0, le=2.0)
    text_max_tokens: int = Field(default=8192, ge=1)
    text_timeout: float = Field(default=30.0, ge=1.0)

    # Execution settings
    max_steps: int = Field(default=20, ge=1, description="Max tool execution steps")
    max_revisions_per_step: int = Field(default=3, ge=1, description="Max revisions for same step")
    max_clarification_rounds: int = Field(default=2, ge=0, description="Max clarification rounds")

    # Paths
    prompt_dir: Path = Field(
        default=Path(__file__).parent.parent / "prompts",
        description="Directory containing prompt markdown files",
    )
    temp_dir: Path = Field(
        default_factory=runtime_artifact_root,
        description="Temp directory for audio processing",
    )

    def model_post_init(self, __context: object) -> None:
        # Auto-load API key from env if not set
        if self.omni_api_key is None:
            self.omni_api_key = resolve_gemini_api_key()
        if self.text_api_key is None:
            self.text_api_key = resolve_gemini_api_key()
