"""Minimal authoritative artifact commit and final-output assembly."""

from __future__ import annotations

from x_editor.workflow_control.schemas import (
    ArtifactAuditRecord,
    ArtifactFinalizationAudit,
    ArtifactStoreCommit,
)


class ArtifactStore:
    """Commit one accepted legacy final or fail closed without choosing a prefix."""

    def commit(self, audit: ArtifactFinalizationAudit) -> ArtifactStoreCommit:
        final_artifact = _legacy_final_artifact(audit)
        failure_codes = _failure_codes(audit, final_artifact)
        if failure_codes:
            return ArtifactStoreCommit(
                audit_id=audit.audit_id,
                attempt_id=audit.attempt_id,
                status="blocked",
                legacy_final_path=audit.legacy_final_path,
                accepted_artifact_refs=list(audit.accepted_artifact_refs),
                candidate_artifact_refs=list(audit.candidate_artifact_refs),
                rejected_artifact_refs=list(audit.rejected_artifact_refs),
                failure_codes=failure_codes,
                plain_summary=(
                    "No accepted complete final artifact was committed; the authoritative "
                    "output is empty."
                ),
                behavior_changed=audit.legacy_final_path is not None,
            )

        assert final_artifact is not None
        return ArtifactStoreCommit(
            audit_id=audit.audit_id,
            attempt_id=audit.attempt_id,
            status="committed",
            legacy_final_path=audit.legacy_final_path,
            final_artifact_ref_id=final_artifact.artifact_ref_id,
            final_output_path=final_artifact.path,
            accepted_artifact_refs=list(audit.accepted_artifact_refs),
            candidate_artifact_refs=list(audit.candidate_artifact_refs),
            rejected_artifact_refs=list(audit.rejected_artifact_refs),
            failure_codes=[],
            plain_summary="The accepted complete final artifact was committed atomically.",
            behavior_changed=False,
        )


def assemble_final_output(commit: ArtifactStoreCommit) -> str | None:
    """Return only the accepted artifact named by a successful store commit."""

    if commit.status != "committed":
        return None
    if commit.final_artifact_ref_id not in commit.accepted_artifact_refs:
        return None
    return commit.final_output_path


def _legacy_final_artifact(
    audit: ArtifactFinalizationAudit,
) -> ArtifactAuditRecord | None:
    if not audit.legacy_final_artifact_ref_id:
        return None
    return next(
        (
            artifact
            for artifact in audit.artifacts
            if artifact.artifact_ref_id == audit.legacy_final_artifact_ref_id
        ),
        None,
    )


def _failure_codes(
    audit: ArtifactFinalizationAudit,
    final_artifact: ArtifactAuditRecord | None,
) -> list[str]:
    codes: list[str] = []
    if audit.attempt_outcome != "selected":
        codes.append("attempt_not_selected")
    if audit.status != "would_finalize":
        codes.append(f"artifact_audit_{audit.status}")
    if audit.unresolved_required_steps:
        codes.append("required_artifacts_unaccepted")
    if final_artifact is None:
        codes.append("legacy_final_artifact_missing")
    elif final_artifact.disposition != "accepted":
        codes.append(f"legacy_final_{final_artifact.disposition}")
    codes.extend(finding.code for finding in audit.findings)
    return _dedupe(codes)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
