"""Audio perception frontends."""

from x_editor.frontend.gemini_audio_frontend import GeminiAudioFrontend
from x_editor.frontend.qwen3_omni_audio_frontend import (
    OpenAICompatibleStrongOmniAudioFrontend,
    Qwen3OmniAudioFrontend,
)

__all__ = [
    "GeminiAudioFrontend",
    "OpenAICompatibleStrongOmniAudioFrontend",
    "Qwen3OmniAudioFrontend",
]
