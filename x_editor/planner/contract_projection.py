"""Lossless presentation of argument ownership and available audio evidence.

These projections describe existing contracts before generation. They neither
select an operation nor edit the returned plan or its provenance.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from x_editor.runtime.audio_usage import canonical_speaker_evidence_references


def group_step_arguments(contract: dict[str, Any]) -> dict[str, Any]:
    """Group declared paths by their actual PlanIR container, without defaults."""
    projected = deepcopy(contract)
    arguments = projected.pop("planner_selectable_arguments", None)
    if arguments is None:
        return projected
    containers: dict[str, dict[str, Any]] = {"inputs": {}, "params": {}}
    for argument in arguments:
        path = argument["argument_path"]
        namespace, separator, relative_path = path.partition(".")
        if not separator or namespace not in containers or not relative_path:
            raise ValueError(f"invalid declared operation argument path: {path!r}")
        if relative_path in containers[namespace]:
            raise ValueError(f"duplicate declared operation argument path: {path!r}")
        # Relative paths remain exact (including any declared nested path).
        # The schema and descriptions are not rewritten or semantically inferred.
        containers[namespace][relative_path] = {
            key: value for key, value in argument.items() if key != "argument_path"
        }
    projected["step_argument_contract"] = containers
    return projected


def audio_usage_evidence_inventory(
    *,
    audio_refs: list[str],
    evidence_path_sources: dict[str, list[str]],
    evidence_path_audio_refs: dict[str, str],
) -> list[dict[str, Any]]:
    """Expose the same per-asset speaker prerequisite used by provenance binding."""
    return [
        {
            "audio_ref": audio_ref,
            "speaker_grounded_reference_evidence_available": bool(
                canonical_speaker_evidence_references(
                    audio_ref=audio_ref,
                    evidence_path_sources=evidence_path_sources,
                    evidence_path_audio_refs=evidence_path_audio_refs,
                )
            ),
        }
        for audio_ref in dict.fromkeys(audio_refs)
    ]
