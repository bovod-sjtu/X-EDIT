"""Executor-authored delivery evidence for framework-valid audio artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from x_editor.core.schemas import AudioEditStep
from x_editor.workflow_control.artifact_acceptance import (
    DELIVERY_AUDIO_INPUT_OPTIONAL_OPERATIONS,
    DELIVERY_CONTRACT_OPERATIONS,
)
from x_editor.workflow_control.schemas import (
    ArtifactDeliveryContract,
    ArtifactDeliveryInvocation,
)

ARTIFACT_DELIVERY_CONTRACT_KEY = "artifact_delivery_contract"

_OUTPUT_PATH_KEYS = (
    "output_audio_path",
    "target_audio_path",
    "output_path",
    "generated_audio_path",
)
_INPUT_PATH_KEYS = frozenset(
    {
        "audio_path",
        "audio_paths",
        "input_audio",
        "input_audio_path",
        "input_audio_paths",
        "mixture_audio",
        "mixture_audio_path",
        "ref_audio_path",
        "reference_audio",
        "reference_audio_path",
        "reference_audio_paths",
        "source_audio",
        "source_audio_path",
        "speaker_audio_path",
        "target_audio_path",
        "target_speaker_reference_path",
    }
)


def attach_artifact_delivery_contract(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
    handler_name: str,
    invocation_start: int,
) -> dict[str, Any]:
    """Attach trusted delivery evidence after a registered handler succeeds."""

    if step.operation not in DELIVERY_CONTRACT_OPERATIONS:
        return output
    output_path = _extract_output_path(output)
    if output_path is None:
        return output
    artifact_path = Path(output_path).expanduser()
    if not artifact_path.is_file():
        return output

    invocation_rows = state.get("tool_invocations")
    observed_invocations = (
        invocation_rows[invocation_start:]
        if isinstance(invocation_rows, list)
        else []
    )
    invocations = [_summarize_invocation(item) for item in observed_invocations]
    invocations = [item for item in invocations if item is not None]
    delivered_operation = _nonempty_string(output.get("operation"))
    operation_evidence = "handler_result"
    if delivered_operation is None:
        delivered_operation = step.operation
        operation_evidence = "executor_dispatch"

    input_audio_paths = _collect_input_audio_paths(
        step=step,
        state=state,
        output=output,
        invocations=observed_invocations,
        output_path=output_path,
    )
    contract = ArtifactDeliveryContract(
        step_id=step.id,
        requested_operation=step.operation,
        delivered_operation=delivered_operation,
        operation_evidence=operation_evidence,
        handler=handler_name,
        handler_completed=True,
        provenance=_delivery_provenance(invocations, output_path=str(artifact_path)),
        input_audio_paths=input_audio_paths,
        output_audio_path=str(artifact_path),
        output_size_bytes=artifact_path.stat().st_size,
        output_sha256=_sha256(artifact_path),
        tool_invocations=invocations,
    )
    delivered = dict(output)
    # The executor owns this key. A handler-provided value cannot impersonate
    # post-success delivery evidence.
    delivered[ARTIFACT_DELIVERY_CONTRACT_KEY] = contract.model_dump(mode="json")
    return delivered


def collect_consumed_audio_paths(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
) -> list[str]:
    """Return audio paths consumed by this step, excluding ambient state."""

    raw_contract = output.get(ARTIFACT_DELIVERY_CONTRACT_KEY)
    if isinstance(raw_contract, dict) and isinstance(
        raw_contract.get("input_audio_paths"), list
    ):
        return _unique_input_paths(
            raw_contract["input_audio_paths"],
            output_path=_extract_output_path(output),
        )
    return _collect_input_audio_paths(
        step=step,
        state=state,
        output=output,
        invocations=[],
        output_path=_extract_output_path(output) or "",
    )


def _summarize_invocation(value: Any) -> ArtifactDeliveryInvocation | None:
    if not isinstance(value, dict):
        return None
    tool_id = _nonempty_string(value.get("tool_id"))
    if tool_id is None or not isinstance(value.get("success"), bool):
        return None
    output = value.get("output")
    output_path = _extract_output_path(output) if isinstance(output, dict) else None
    return ArtifactDeliveryInvocation(
        tool_id=tool_id,
        success=value["success"],
        output_audio_path=output_path,
        error_message=_nonempty_string(value.get("error_message")),
    )


def _delivery_provenance(
    invocations: list[ArtifactDeliveryInvocation],
    *,
    output_path: str,
) -> str:
    if not invocations:
        return "local"
    if (
        len(invocations) == 1
        and invocations[0].success
        and invocations[0].output_audio_path is not None
        and _same_path(invocations[0].output_audio_path, output_path)
    ):
        return "tool"
    return "composite"


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except (OSError, RuntimeError):
        return str(left) == str(right)


def _collect_input_audio_paths(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
    invocations: list[Any],
    output_path: str,
) -> list[str]:
    candidates: list[str] = []
    for invocation in invocations:
        if isinstance(invocation, dict):
            _collect_named_audio_paths(invocation.get("arguments"), candidates)
    _collect_named_audio_paths(output, candidates)
    _collect_named_audio_paths(step.inputs, candidates)
    _collect_named_audio_paths(step.params, candidates)
    if not candidates and step.operation not in DELIVERY_AUDIO_INPUT_OPTIONAL_OPERATIONS:
        for state_key in ("current_audio_path", "input_audio"):
            value = _nonempty_string(state.get(state_key))
            if value is not None:
                candidates.append(value)

    return _unique_input_paths(candidates, output_path=output_path)


def _unique_input_paths(
    candidates: list[Any],
    *,
    output_path: str | None,
) -> list[str]:
    normalized = [
        item.strip()
        for item in candidates
        if isinstance(item, str) and item.strip()
    ]

    output_key = _path_key(output_path) if output_path else None
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in normalized:
        key = _path_key(candidate)
        if (output_key is not None and key == output_key) or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _collect_named_audio_paths(value: Any, output: list[str]) -> None:
    if not isinstance(value, dict):
        return
    for key, candidate in value.items():
        if key not in _INPUT_PATH_KEYS:
            continue
        if isinstance(candidate, str) and candidate.strip():
            output.append(candidate.strip())
        elif isinstance(candidate, list):
            output.extend(
                item.strip()
                for item in candidate
                if isinstance(item, str) and item.strip()
            )


def _extract_output_path(value: dict[str, Any]) -> str | None:
    for key in _OUTPUT_PATH_KEYS:
        path = _nonempty_string(value.get(key))
        if path is not None:
            return path
    return None


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _path_key(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError):
        return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
