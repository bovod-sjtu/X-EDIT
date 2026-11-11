"""Shadow-only final delivery manifest projection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from x_editor.core.schemas import AudioEditPlan
from x_editor.workflow_control.schemas import (
    ArtifactAuditRecord,
    ArtifactFinalizationAudit,
    ArtifactStoreCommit,
    FinalAssemblerFinding,
    FinalDeliveryArtifact,
    FinalDeliveryManifest,
)


class FinalAssemblerShadow:
    """Describe delivery structure without changing ArtifactStore authority."""

    def project(
        self,
        plan: AudioEditPlan,
        audit: ArtifactFinalizationAudit,
        commit: ArtifactStoreCommit,
    ) -> FinalDeliveryManifest:
        if commit.status != "committed":
            return FinalDeliveryManifest(
                commit_id=commit.commit_id,
                audit_id=commit.audit_id,
                attempt_id=commit.attempt_id,
                commit_status=commit.status,
                status="not_applicable",
                findings=[
                    FinalAssemblerFinding(
                        code="artifact_store_not_committed",
                        severity="info",
                        plain_summary=(
                            "ArtifactStore did not commit a final output, so no delivery "
                            "authority was projected."
                        ),
                        details={"failure_codes": list(commit.failure_codes)},
                    )
                ],
            )

        findings: list[FinalAssemblerFinding] = []
        if commit.audit_id != audit.audit_id or commit.attempt_id != audit.attempt_id:
            findings.append(
                FinalAssemblerFinding(
                    code="commit_audit_binding_mismatch",
                    severity="blocking",
                    plain_summary=(
                        "The ArtifactStore commit does not identify the supplied artifact audit."
                    ),
                )
            )

        final_matches = [
            artifact
            for artifact in audit.artifacts
            if artifact.artifact_ref_id == commit.final_artifact_ref_id
        ]
        final_artifact = final_matches[0] if len(final_matches) == 1 else None
        if len(final_matches) != 1:
            findings.append(
                FinalAssemblerFinding(
                    code="committed_artifact_not_unique",
                    severity="blocking",
                    plain_summary=(
                        "The committed final artifact is missing from the audit or appears more than once."
                    ),
                    artifact_ref_id=commit.final_artifact_ref_id,
                    details={"match_count": len(final_matches)},
                )
            )
        elif not _same_path(final_artifact.path, commit.final_output_path):
            findings.append(
                FinalAssemblerFinding(
                    code="committed_output_path_mismatch",
                    severity="blocking",
                    plain_summary=(
                        "The committed final path does not match the audited artifact path."
                    ),
                    artifact_ref_id=final_artifact.artifact_ref_id,
                    path=final_artifact.path,
                )
            )
        elif final_artifact.disposition != "accepted":
            findings.append(
                FinalAssemblerFinding(
                    code="committed_artifact_not_accepted",
                    severity="blocking",
                    plain_summary="The committed final artifact is not accepted in the audit.",
                    artifact_ref_id=final_artifact.artifact_ref_id,
                    path=final_artifact.path,
                )
            )

        ordered_components: list[FinalDeliveryArtifact] = []
        component_order_source = "none"
        component_artifact_refs: set[str] = set()
        component_path_keys: set[str] = set()
        if final_artifact is not None:
            final_step = next(
                (
                    step
                    for step in plan.steps
                    if step.id == final_artifact.producer_step_id
                ),
                None,
            )
            if final_step is not None and final_step.operation == "artifact_ops.concat":
                component_order_source = "concat_step_result.sources"
                sources = (
                    final_step.result.get("sources")
                    if isinstance(final_step.result, dict)
                    else None
                )
                if not isinstance(sources, list) or not sources:
                    findings.append(
                        FinalAssemblerFinding(
                            code="concat_component_order_missing",
                            severity="blocking",
                            plain_summary=(
                                "The concat final does not expose the ordered source list used to build it."
                            ),
                            artifact_ref_id=final_artifact.artifact_ref_id,
                            path=final_artifact.path,
                        )
                    )
                else:
                    for index, raw_source in enumerate(sources):
                        source = raw_source if isinstance(raw_source, dict) else {}
                        source_path = _string_or_none(source.get("path"))
                        if source_path is None:
                            findings.append(
                                FinalAssemblerFinding(
                                    code="concat_component_path_missing",
                                    severity="blocking",
                                    plain_summary=(
                                        "One concat component has an order position but no source path."
                                    ),
                                    details={"order_index": index},
                                )
                            )
                            continue
                        source_artifact = _artifact_for_component(
                            source_path,
                            final_artifact,
                            audit.artifacts,
                        )
                        item = _delivery_artifact(
                            source_artifact,
                            path=source_path,
                            delivery_role="ordered_component",
                            order_index=index,
                            metadata={
                                key: value
                                for key, value in source.items()
                                if key != "path"
                            },
                        )
                        ordered_components.append(item)
                        component_path_keys.add(_path_key(source_path))
                        if item.artifact_ref_id:
                            component_artifact_refs.add(item.artifact_ref_id)
                    findings.append(
                        FinalAssemblerFinding(
                            code="concat_component_order_projected",
                            severity="info",
                            plain_summary=(
                                "The concat source order was projected from the executed step result."
                            ),
                            details={"component_count": len(ordered_components)},
                        )
                    )

        non_authoritative_outputs = [
            _delivery_artifact(
                artifact,
                path=artifact.path,
                delivery_role="non_authoritative_output",
            )
            for artifact in audit.artifacts
            if artifact.artifact_ref_id != commit.final_artifact_ref_id
            and artifact.artifact_ref_id not in component_artifact_refs
            and _path_key(artifact.path) not in component_path_keys
        ]
        if non_authoritative_outputs:
            findings.append(
                FinalAssemblerFinding(
                    code="non_authoritative_outputs_separated",
                    severity="info",
                    plain_summary=(
                        "Other observed outputs were retained as evidence without receiving final authority."
                    ),
                    details={"output_count": len(non_authoritative_outputs)},
                )
            )

        classified_refs = {
            item.artifact_ref_id
            for item in ordered_components + non_authoritative_outputs
            if item.artifact_ref_id
        }
        if final_artifact is not None:
            classified_refs.add(final_artifact.artifact_ref_id)
        missing_required_refs = [
            ref for ref in audit.required_artifact_refs if ref not in classified_refs
        ]
        complete_required_artifacts = not missing_required_refs
        if missing_required_refs:
            findings.append(
                FinalAssemblerFinding(
                    code="required_delivery_artifacts_missing",
                    severity="blocking",
                    plain_summary=(
                        "One or more required accepted artifacts are absent from the delivery structure."
                    ),
                    details={"artifact_ref_ids": missing_required_refs},
                )
            )

        authoritative_output = (
            _delivery_artifact(
                final_artifact,
                path=final_artifact.path,
                delivery_role="authoritative_output",
            )
            if final_artifact is not None
            and _same_path(final_artifact.path, commit.final_output_path)
            and final_artifact.disposition == "accepted"
            else None
        )
        format_contract_complete = bool(
            authoritative_output
            and authoritative_output.format
            and authoritative_output.media_type
        )
        requested_output_format = _requested_output_format(plan)
        requested_output_path = _string_or_none(plan.output_audio)
        path_matches_request = (
            _same_path(commit.final_output_path, requested_output_path)
            if requested_output_path is not None
            else None
        )
        actual_output_format = (
            authoritative_output.format if authoritative_output is not None else None
        )
        format_matches_request = (
            actual_output_format == requested_output_format
            if requested_output_format is not None and actual_output_format is not None
            else None
        )
        if path_matches_request is False:
            findings.append(
                FinalAssemblerFinding(
                    code="final_output_path_mismatch",
                    severity="blocking",
                    plain_summary=(
                        "The committed final path differs from the caller's explicit output path."
                    ),
                    artifact_ref_id=(
                        authoritative_output.artifact_ref_id
                        if authoritative_output is not None
                        else None
                    ),
                    path=commit.final_output_path,
                    details={
                        "requested_path": requested_output_path,
                        "committed_path": commit.final_output_path,
                    },
                )
            )
        if format_matches_request is False:
            format_contract_complete = False
            findings.append(
                FinalAssemblerFinding(
                    code="final_output_format_mismatch",
                    severity="blocking",
                    plain_summary=(
                        "The committed final output format differs from the caller's explicit output path."
                    ),
                    artifact_ref_id=(
                        authoritative_output.artifact_ref_id
                        if authoritative_output is not None
                        else None
                    ),
                    path=(
                        authoritative_output.path
                        if authoritative_output is not None
                        else commit.final_output_path
                    ),
                    details={
                        "requested_format": requested_output_format,
                        "actual_format": actual_output_format,
                    },
                )
            )
        if authoritative_output is not None and not format_contract_complete:
            findings.append(
                FinalAssemblerFinding(
                    code="final_output_format_unknown",
                    severity="blocking",
                    plain_summary=(
                        "The committed final output does not expose a usable delivery format."
                    ),
                    artifact_ref_id=authoritative_output.artifact_ref_id,
                    path=authoritative_output.path,
                )
            )
        if authoritative_output is not None and authoritative_output.content_sha256 is None:
            findings.append(
                FinalAssemblerFinding(
                    code="final_output_identity_unavailable",
                    severity="blocking",
                    plain_summary=(
                        "The committed final output could not be bound to its current file bytes."
                    ),
                    artifact_ref_id=authoritative_output.artifact_ref_id,
                    path=authoritative_output.path,
                )
            )

        lineage_items = [*ordered_components, *non_authoritative_outputs]
        lineage_complete = all(
            item.artifact_ref_id is not None or item.content_sha256 is not None
            for item in lineage_items
        )
        blocking = any(item.severity == "blocking" for item in findings)
        ready = (
            authoritative_output is not None
            and authoritative_output.content_sha256 is not None
            and complete_required_artifacts
            and format_contract_complete
            and lineage_complete
            and not blocking
        )
        return FinalDeliveryManifest(
            commit_id=commit.commit_id,
            audit_id=commit.audit_id,
            attempt_id=commit.attempt_id,
            commit_status=commit.status,
            status="ready" if ready else "incomplete",
            committed_final_artifact_ref_id=commit.final_artifact_ref_id,
            committed_final_output_path=commit.final_output_path,
            authoritative_output=authoritative_output if ready else None,
            ordered_components=ordered_components,
            non_authoritative_outputs=non_authoritative_outputs,
            component_order_source=component_order_source,
            requested_output_path=requested_output_path,
            path_matches_request=path_matches_request,
            requested_output_format=requested_output_format,
            actual_output_format=actual_output_format,
            format_matches_request=format_matches_request,
            required_artifact_refs=list(audit.required_artifact_refs),
            missing_required_artifact_refs=missing_required_refs,
            complete_required_artifacts=complete_required_artifacts,
            format_contract_complete=format_contract_complete,
            lineage_complete=lineage_complete,
            findings=findings,
        )


def _artifact_for_component(
    path: str,
    final_artifact: ArtifactAuditRecord,
    artifacts: list[ArtifactAuditRecord],
) -> ArtifactAuditRecord | None:
    matches = [
        artifact
        for artifact in artifacts
        if artifact.artifact_ref_id != final_artifact.artifact_ref_id
        and _same_path(artifact.path, path)
    ]
    parent_matches = [
        artifact
        for artifact in matches
        if artifact.artifact_ref_id in final_artifact.parent_artifact_ref_ids
    ]
    considered = parent_matches or matches
    accepted = [artifact for artifact in considered if artifact.disposition == "accepted"]
    preferred = accepted or considered
    return preferred[0] if len(preferred) == 1 else None


def _delivery_artifact(
    artifact: ArtifactAuditRecord | None,
    *,
    path: str,
    delivery_role: str,
    order_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> FinalDeliveryArtifact:
    path_obj = Path(path).expanduser()
    size_bytes: int | None = None
    content_sha256: str | None = None
    try:
        if path_obj.is_file():
            size_bytes = path_obj.stat().st_size
            content_sha256 = _sha256(path_obj)
    except OSError:
        pass
    output_format = path_obj.suffix.lower().lstrip(".") or None
    media_type = {"wav": "audio/wav"}.get(output_format)
    return FinalDeliveryArtifact(
        artifact_ref_id=artifact.artifact_ref_id if artifact is not None else None,
        path=str(path),
        delivery_role=delivery_role,  # type: ignore[arg-type]
        disposition=(
            artifact.disposition if artifact is not None else "external_source"
        ),
        producer_step_id=artifact.producer_step_id if artifact is not None else None,
        producer_operation=(
            artifact.producer_operation if artifact is not None else None
        ),
        order_index=order_index,
        format=output_format,
        media_type=media_type,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
        audio_properties=(
            _audio_properties(artifact) if artifact is not None else {}
        ),
        metadata=dict(metadata or {}),
    )


def _audio_properties(artifact: ArtifactAuditRecord) -> dict[str, Any]:
    for check in artifact.integrity_checks:
        if check.check_id == "audio_container" and check.status == "passed":
            return dict(check.details)
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(left: Any, right: Any) -> bool:
    return bool(left and right and _path_key(left) == _path_key(right))


def _path_key(value: Any) -> str:
    try:
        return str(Path(str(value)).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(value)


def _string_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _requested_output_format(plan: AudioEditPlan) -> str | None:
    if not plan.output_audio:
        return None
    return Path(plan.output_audio).suffix.lower().lstrip(".") or None
