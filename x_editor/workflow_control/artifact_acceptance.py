"""Operation policies for framework-level artifact acceptance."""

from __future__ import annotations

from collections.abc import Iterable

from x_editor.workflow_control.schemas import ArtifactAcceptancePolicy

OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS: dict[str, tuple[str, ...]] = {
    "export_wav": ("export_wav.source_equivalent",),
    "artifact_ops.apply_gain": (
        "artifact_ops.apply_gain.contract",
        "artifact_ops.apply_gain.timeline",
        "artifact_ops.apply_gain.signal",
        "artifact_ops.apply_gain.preservation",
    ),
    "artifact_ops.silence": (
        "artifact_ops.silence.duration",
        "artifact_ops.silence.format",
        "artifact_ops.silence.signal",
    ),
    "artifact_ops.concat": (
        "artifact_ops.concat.sources",
        "artifact_ops.concat.duration",
    ),
    "artifact_ops.extract_spans": (
        "artifact_ops.extract_spans.spans",
        "artifact_ops.extract_spans.duration",
    ),
    "artifact_ops.remove_spans": (
        "artifact_ops.remove_spans.spans",
        "artifact_ops.remove_spans.duration",
    ),
    "artifact_ops.mute_spans": (
        "artifact_ops.mute_spans.spans",
        "artifact_ops.mute_spans.duration",
        "artifact_ops.mute_spans.signal",
    ),
    "artifact_ops.beep_spans": (
        "artifact_ops.beep_spans.spans",
        "artifact_ops.beep_spans.duration",
        "artifact_ops.beep_spans.signal",
    ),
    "artifact_ops.reconstruct": (
        "artifact_ops.reconstruct.contract",
        "artifact_ops.reconstruct.inputs",
        "artifact_ops.reconstruct.duration",
    ),
    "artifact_ops.reorder_spans": (
        "artifact_ops.reorder_spans.spans",
        "artifact_ops.reorder_spans.order",
        "artifact_ops.reorder_spans.duration",
        "artifact_ops.reorder_spans.content",
    ),
    "edit_data_pipeline.volume_normalize": (
        "edit_data_pipeline.volume_normalize.contract",
        "edit_data_pipeline.volume_normalize.duration",
        "edit_data_pipeline.volume_normalize.level",
    ),
    "audio_effects": (
        "audio_effects.contract",
        "audio_effects.timeline",
        "audio_effects.scope_signal",
    ),
    "spatial_reverb": (
        "spatial_reverb.contract",
        "spatial_reverb.duration",
        "spatial_reverb.signal",
    ),
    "speech_separation": (
        "speech_separation.contract",
        "speech_separation.stems",
        "speech_separation.duration",
    ),
    "speech_enhancement": (
        "speech_enhancement.contract",
        "speech_enhancement.duration",
        "speech_enhancement.signal",
    ),
}

AUDIO_ARTIFACT_OPERATIONS: frozenset[str] = frozenset(
    {
        "accent_conversion",
        "ace_step_music.timeline_edit",
        "artifact_ops.apply_gain",
        "artifact_ops.beep_spans",
        "artifact_ops.concat",
        "artifact_ops.extract_spans",
        "artifact_ops.mute_spans",
        "artifact_ops.reconstruct",
        "artifact_ops.remove_spans",
        "artifact_ops.reorder_spans",
        "artifact_ops.silence",
        "artifact_ops.speaker_scoped_edit",
        "audio_effects",
        "edit_data_pipeline.audio_assembly",
        "edit_data_pipeline.boundary_smooth",
        "edit_data_pipeline.emotion_style_edit",
        "edit_data_pipeline.pause_edit",
        "edit_data_pipeline.pitch_rate_edit",
        "edit_data_pipeline.pitch_shift",
        "edit_data_pipeline.speech_rate",
        "edit_data_pipeline.text_edit",
        "edit_data_pipeline.volume_normalize",
        "emphasis_stress_edit",
        "export_wav",
        "foreground_background_remix",
        "intra_audio_voice_transfer",
        "no_reference_voice_conversion",
        "paralinguistic_event_add",
        "paralinguistic_event_remove",
        "qwen3_tts",
        "seamless_expressive_s2st",
        "auk",
        "seed_vc",
        "spatial_reverb",
        "speech_enhancement",
        "speech_separation",
        "tse",
    }
)


DELIVERY_CONTRACT_CHECK_IDS: tuple[str, ...] = (
    "framework.delivery_contract.execution",
    "framework.delivery_contract.inputs",
    "framework.delivery_contract.artifact",
)

DELIVERY_CONTRACT_OPERATIONS: frozenset[str] = frozenset(
    AUDIO_ARTIFACT_OPERATIONS - OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS.keys()
)

# These operations can legitimately synthesize audio without consuming an audio
# input. Their executed mode and optional references remain visible in the
# delivery contract, while semantic generation quality belongs to Evaluator.
DELIVERY_AUDIO_INPUT_OPTIONAL_OPERATIONS: frozenset[str] = frozenset(
    {
        "qwen3_tts",
    }
)

DEFAULT_ARTIFACT_ACCEPTANCE_CHECKS: dict[str, tuple[str, ...]] = {
    **OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS,
    **{
        operation: DELIVERY_CONTRACT_CHECK_IDS
        for operation in sorted(DELIVERY_CONTRACT_OPERATIONS)
    },
}

EVALUATOR_DEFERRED_ARTIFACT_OPERATIONS: frozenset[str] = frozenset(
    AUDIO_ARTIFACT_OPERATIONS - DEFAULT_ARTIFACT_ACCEPTANCE_CHECKS.keys()
)


class ArtifactAcceptancePolicyRegistry:
    """Map an operation to the principle checks required for its artifact."""

    def __init__(
        self,
        policies: Iterable[ArtifactAcceptancePolicy] | None = None,
    ) -> None:
        self._policies: dict[str, ArtifactAcceptancePolicy] = {}
        for policy in policies or []:
            self.register(policy)

    def register(self, policy: ArtifactAcceptancePolicy) -> None:
        if policy.operation in self._policies:
            raise ValueError(
                f"artifact acceptance policy already registered: {policy.operation}"
            )
        self._policies[policy.operation] = policy

    def policy_for(self, operation: str | None) -> ArtifactAcceptancePolicy | None:
        if not operation:
            return None
        return self._policies.get(operation)

    def describe(self) -> list[dict[str, object]]:
        return [
            self._policies[operation].model_dump(mode="json")
            for operation in sorted(self._policies)
        ]


def build_default_artifact_acceptance_policy_registry(
) -> ArtifactAcceptancePolicyRegistry:
    """Return the reviewed framework acceptance policies for audio artifacts."""

    return ArtifactAcceptancePolicyRegistry(
        ArtifactAcceptancePolicy(
            policy_id=f"{operation}/framework-principles/v1",
            operation=operation,
            required_check_ids=list(check_ids),
            plain_summary=(
                "The artifact satisfies the operation's deterministic framework "
                "principles; semantic objective evaluation remains pending."
            ),
        )
        for operation, check_ids in DEFAULT_ARTIFACT_ACCEPTANCE_CHECKS.items()
    )
