"""Speech perception helpers used by benchmark adapters and planners."""

from .speech import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_SEDICOW_URL,
    SpeechPerceptionClient,
    SpeechPerceptionError,
    SpeechPerceptionServiceUnavailableError,
)
from .gemini_enrichment import (
    enrich_perception_with_gemini,
    merge_gemini_perception,
    needs_gemini_audio_perception,
)
from .fallback import (
    FallbackPerceptionResult,
    GeminiAudioPerceptionFallback,
    PerceptionFallbackError,
    PerceptionRequirements,
    Qwen3OmniAudioPerceptionFallback,
    build_audio_perception_fallback,
    missing_requirements,
)
from .qwen3_omni_enrichment import (
    enrich_perception_with_qwen3_omni,
    merge_qwen3_omni_perception,
    needs_qwen3_omni_audio_perception,
)
from .speaker_attributed_review import (
    GeminiSpeakerReviewFrontend,
    GLMSpeakerEvidenceTextFrontend,
    OpenAICompatibleSpeakerReviewFrontend,
    Qwen3OmniSpeakerReviewFrontend,
    ReviewVerificationResult,
    run_speaker_attributed_review,
    verify_review_output,
)
from .timestamped_caption import (
    CaptionVerificationResult,
    GeminiTimestampedCaptionFrontend,
    OpenAICompatibleTimestampedCaptionFrontend,
    build_event_candidates,
    run_timestamped_caption_review,
    verify_timestamped_caption_output,
)

__all__ = [
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_SEDICOW_URL",
    "FallbackPerceptionResult",
    "GeminiAudioPerceptionFallback",
    "GeminiSpeakerReviewFrontend",
    "GeminiTimestampedCaptionFrontend",
    "GLMSpeakerEvidenceTextFrontend",
    "OpenAICompatibleSpeakerReviewFrontend",
    "OpenAICompatibleTimestampedCaptionFrontend",
    "PerceptionFallbackError",
    "PerceptionRequirements",
    "Qwen3OmniAudioPerceptionFallback",
    "Qwen3OmniSpeakerReviewFrontend",
    "ReviewVerificationResult",
    "CaptionVerificationResult",
    "SpeechPerceptionClient",
    "SpeechPerceptionError",
    "SpeechPerceptionServiceUnavailableError",
    "build_event_candidates",
    "build_audio_perception_fallback",
    "enrich_perception_with_gemini",
    "enrich_perception_with_qwen3_omni",
    "merge_gemini_perception",
    "merge_qwen3_omni_perception",
    "missing_requirements",
    "needs_gemini_audio_perception",
    "needs_qwen3_omni_audio_perception",
    "run_speaker_attributed_review",
    "run_timestamped_caption_review",
    "verify_timestamped_caption_output",
    "verify_review_output",
]
