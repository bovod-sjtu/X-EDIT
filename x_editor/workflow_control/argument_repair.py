"""Field-level argument repair proposals and deterministic admission."""

from __future__ import annotations

import json
from typing import Any

from x_editor.core.schemas import (
    AudioEditStep,
    ExecutorArgumentBinding,
    ExecutorToolCall,
)
from x_editor.runtime.operation_normalizers import canonicalize_volume_normalize_method
from x_editor.workflow_control.schemas import (
    ArgumentRepairAdmissionFinding,
    ArgumentRepairAdmissionResult,
    ArgumentRepairFieldChange,
    ArgumentRepairProposal,
    RepairSignal,
)


class ArgumentRepairAdmissionGate:
    """Reject ambiguous, ungrounded, or over-broad step patches."""

    def audit(
        self,
        proposal: ArgumentRepairProposal,
        *,
        evidence_sources: dict[str, Any] | None = None,
    ) -> ArgumentRepairAdmissionResult:
        sources = {
            str(key): _evidence_text(value)
            for key, value in dict(evidence_sources or {}).items()
        }
        findings: list[ArgumentRepairAdmissionFinding] = []

        if proposal.resolution != "resolved":
            findings.append(
                _blocking(
                    "repair_not_resolved",
                    "The proposer did not produce one unambiguous argument binding.",
                    details={"resolution": proposal.resolution},
                )
            )
        if proposal.operation_before != proposal.operation_after:
            findings.append(
                _blocking(
                    "operation_changed",
                    "Argument repair cannot change the selected operation.",
                    details={
                        "before": proposal.operation_before,
                        "after": proposal.operation_after,
                    },
                )
            )
        if not proposal.changes:
            findings.append(
                _blocking(
                    "repair_has_no_changes",
                    "The proposal does not change any executor argument.",
                )
            )

        duplicate_bindings = proposal.metadata.get("duplicate_binding_fields") or []
        for field_path in duplicate_bindings:
            findings.append(
                _blocking(
                    "duplicate_field_binding",
                    "A changed field has more than one binding record.",
                    field_path=str(field_path),
                )
            )

        allowed = set(proposal.allowed_field_paths)
        semantic_required = set(
            proposal.metadata.get("semantic_binding_required_fields") or []
        )
        for change in proposal.changes:
            if change.field_path not in allowed:
                findings.append(
                    _blocking(
                        "field_not_allowlisted",
                        "The proposal modifies a field not named by the tool-contract violation.",
                        field_path=change.field_path,
                    )
                )
            if change.binding_kind == "unknown":
                findings.append(
                    _blocking(
                        "field_binding_missing",
                        "Every changed field needs an explicit binding record.",
                        field_path=change.field_path,
                    )
                )
            if (
                change.field_path in semantic_required
                and change.binding_kind != "semantic"
            ):
                findings.append(
                    _blocking(
                        "semantic_binding_required",
                        "The tool contract requires semantic binding for this field.",
                        field_path=change.field_path,
                    )
                )
            if change.alternatives:
                findings.append(
                    _blocking(
                        "field_binding_ambiguous",
                        "The field still has viable alternatives and cannot be auto-applied.",
                        field_path=change.field_path,
                        details={"alternatives": change.alternatives},
                    )
                )
            if change.binding_kind == "semantic" and not _has_grounded_semantic_evidence(
                change.evidence,
                sources,
            ):
                findings.append(
                    _blocking(
                        "semantic_evidence_missing",
                        "A semantic field change needs an exact excerpt from user, step, or perception evidence.",
                        field_path=change.field_path,
                    )
                )

        return ArgumentRepairAdmissionResult(
            proposal_id=proposal.proposal_id,
            admitted=not any(finding.severity == "blocking" for finding in findings),
            findings=findings,
            checked_field_paths=[change.field_path for change in proposal.changes],
        )


def proposal_from_executor_call(
    *,
    signal: RepairSignal,
    before: AudioEditStep,
    candidate: AudioEditStep,
    call: ExecutorToolCall,
    violations: list[dict[str, Any]],
    producer: str,
    provider: str | None,
    model: str | None,
    prompt_version: str | None,
    producer_kind: str = "executor_llm",
) -> ArgumentRepairProposal:
    """Project a normalized executor call into an auditable field patch."""

    binding_counts: dict[str, int] = {}
    bindings = {}
    for binding in call.argument_bindings:
        binding_counts[binding.field_path] = binding_counts.get(binding.field_path, 0) + 1
        bindings[binding.field_path] = binding
    changes = []
    for field_path, before_value, after_value in _step_changes(before, candidate):
        binding = bindings.get(field_path)
        matches_value = binding is not None and binding.proposed_value == after_value
        changes.append(
            ArgumentRepairFieldChange(
                field_path=field_path,
                before=before_value,
                after=after_value,
                binding_kind=binding.binding_kind if matches_value else "unknown",
                semantic_value=binding.semantic_value if matches_value else None,
                evidence=[item.model_dump(mode="json") for item in binding.evidence]
                if matches_value
                else [],
                alternatives=list(binding.alternatives) if matches_value else [],
                rationale=binding.rationale if matches_value else "",
            )
        )
    allowed_fields = sorted(
        {
            str(field)
            for violation in violations
            for field in violation.get("repairable_fields") or []
            if field
        }
    )
    semantic_required_fields = sorted(
        {
            str(field)
            for violation in violations
            if violation.get("requires_semantic_binding")
            for field in violation.get("repairable_fields") or []
            if field
        }
    )
    return ArgumentRepairProposal(
        signal_id=signal.signal_id,
        step_id=before.id,
        operation_before=before.operation,
        operation_after=candidate.operation,
        producer_kind=producer_kind,
        producer=producer,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        resolution=call.repair_resolution or "unresolved",
        allowed_field_paths=allowed_fields,
        changes=changes,
        metadata={
            "allow_execute": call.allow_execute,
            "missing_inputs": list(call.missing_inputs),
            "blocked_reason": call.blocked_reason,
            "semantic_binding_required_fields": semantic_required_fields,
            "duplicate_binding_fields": sorted(
                field_path
                for field_path, count in binding_counts.items()
                if count > 1
            ),
        },
    )


def deterministic_argument_repair_candidate(
    *,
    step: AudioEditStep,
    violations: list[dict[str, Any]],
    operation_spec: dict[str, Any],
) -> tuple[ExecutorToolCall, AudioEditStep] | None:
    """Repair only mechanical aliases/defaults with no semantic choice."""

    repairable_fields = sorted(
        {
            str(field)
            for violation in violations
            if not violation.get("requires_semantic_binding")
            for field in violation.get("repairable_fields") or []
            if field
        }
    )
    if len(repairable_fields) != 1:
        return None
    field_path = repairable_fields[0]
    if field_path not in {
        "params.edit_op_params.method",
        "params.edit_params.method",
        "params.sub_operation_params.method",
        "params.method",
    }:
        return None
    current = _step_field_value(step, field_path)
    canonical = canonicalize_volume_normalize_method(current, strict=False)
    if current == canonical:
        return None

    candidate = step.model_copy(deep=True)
    _set_step_field_value(candidate, field_path, canonical)
    call = ExecutorToolCall(
        step_id=step.id,
        operation=step.operation,
        tool_id=operation_spec.get("tool_id"),
        normalized_params=dict(candidate.params),
        repair_resolution="resolved",
        argument_bindings=[
            ExecutorArgumentBinding(
                field_path=field_path,
                binding_kind="mechanical",
                semantic_value=current,
                proposed_value=canonical,
                rationale=(
                    "The selected nested volume_normalize contract accepts only "
                    "rms or peak; its deterministic fallback is rms."
                ),
            )
        ],
    )
    return call, candidate


def argument_repair_evidence_sources(
    step: AudioEditStep, state: dict[str, Any]
) -> dict[str, Any]:
    """Return only prompt-visible evidence classes accepted by admission."""

    return {
        "user_request": state.get("user_prompt"),
        "step_description": step.description,
        "perception": state.get("perception_report"),
        "source_text": state.get("source_text"),
    }


def _step_changes(
    before: AudioEditStep, after: AudioEditStep
) -> list[tuple[str, Any, Any]]:
    changes = []
    changes.extend(_mapping_changes(before.inputs, after.inputs, prefix="inputs"))
    changes.extend(_mapping_changes(before.params, after.params, prefix="params"))
    return changes


def _step_field_value(step: AudioEditStep, field_path: str) -> Any:
    parts = field_path.split(".")
    if not parts or parts[0] not in {"inputs", "params"}:
        return None
    value: Any = getattr(step, parts[0])
    for part in parts[1:]:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _set_step_field_value(
    step: AudioEditStep, field_path: str, value: Any
) -> None:
    parts = field_path.split(".")
    if len(parts) < 2 or parts[0] not in {"inputs", "params"}:
        raise ValueError(f"unsupported argument-repair field path: {field_path}")
    mapping = getattr(step, parts[0])
    for part in parts[1:-1]:
        child = mapping.get(part)
        if child is None:
            child = {}
            mapping[part] = child
        if not isinstance(child, dict):
            raise ValueError(  # noqa: TRY004 - the path value is invalid for this proposal.
                f"argument-repair path crosses non-mapping field: {field_path}"
            )
        mapping = child
    mapping[parts[-1]] = value


def _mapping_changes(
    before: dict[str, Any], after: dict[str, Any], *, prefix: str
) -> list[tuple[str, Any, Any]]:
    changes: list[tuple[str, Any, Any]] = []
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key)
        after_value = after.get(key)
        field_path = f"{prefix}.{key}"
        if isinstance(before_value, dict) and isinstance(after_value, dict):
            changes.extend(
                _mapping_changes(before_value, after_value, prefix=field_path)
            )
        elif before_value != after_value:
            changes.append((field_path, before_value, after_value))
    return changes


def _has_grounded_semantic_evidence(
    evidence: list[dict[str, str]], sources: dict[str, str]
) -> bool:
    semantic_sources = {"user_request", "step_description", "perception", "source_text"}
    for item in evidence:
        source_ref = str(item.get("source_ref") or "")
        excerpt = str(item.get("excerpt") or "").strip()
        source_text = sources.get(source_ref, "")
        if source_ref in semantic_sources and excerpt and excerpt in source_text:
            return True
    return False


def _evidence_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _blocking(
    code: str,
    summary: str,
    *,
    field_path: str | None = None,
    details: dict[str, Any] | None = None,
) -> ArgumentRepairAdmissionFinding:
    return ArgumentRepairAdmissionFinding(
        code=code,
        severity="blocking",
        plain_summary=summary,
        field_path=field_path,
        details=dict(details or {}),
    )
