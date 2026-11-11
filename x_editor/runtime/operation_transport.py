"""Exact field mappings at a service boundary, without semantic inference.

The reference aliases below are the voice-cloning service's existing input
contract. Equivalence requires one unchanged reference and preserves every
other argument, including text and synthesis mode.
"""

from collections.abc import Mapping
from typing import Any

VOICE_CLONE_REFERENCE_KEYS = (
    "ref_audio_path",
    "voice_clone_reference_path",
    "voice_clone_reference_audio_path",
    "voice_clone_reference",
    "voice_clone_ref_path",
    "reference_audio_path",
)


def voice_clone_transport_equivalent(
    *,
    before_inputs: Mapping[str, Any],
    before_params: Mapping[str, Any],
    after_inputs: Mapping[str, Any],
    after_params: Mapping[str, Any],
) -> bool:
    """Prove reference-field transport and the declared embedding-only default."""
    if before_params.get("mode") != "voice_clone" or after_params.get("mode") != "voice_clone":
        return False
    before_reference = _unique_reference(before_inputs, before_params)
    after_reference = _unique_reference(after_inputs, after_params)
    if before_reference is None or before_reference != after_reference:
        return False
    if after_params.get("ref_audio_path") != before_reference:
        return False

    before = _without_reference_fields(before_inputs, before_params)
    after = _without_reference_fields(after_inputs, after_params)
    if (
        "x_vector_only_mode" not in before_params
        and before_params.get("ref_text") in (None, "")
        and after_params.get("x_vector_only_mode") is True
    ):
        after["params"].pop("x_vector_only_mode")
    return before == after


def _unique_reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> str | None:
    values = [
        value
        for mapping in (inputs, params)
        for key in VOICE_CLONE_REFERENCE_KEYS
        if (value := mapping.get(key)) not in (None, "")
    ]
    if not values or any(not isinstance(value, str) for value in values):
        return None
    references = set(values)
    return next(iter(references)) if len(references) == 1 else None


def _without_reference_fields(
    inputs: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        name: {key: value for key, value in mapping.items() if key not in VOICE_CLONE_REFERENCE_KEYS}
        for name, mapping in (("inputs", inputs), ("params", params))
    }
