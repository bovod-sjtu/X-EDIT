"""Offline-only state projection for a future strict delivery contract."""

from __future__ import annotations

from typing import Literal

from x_editor.workflow_control.schemas import (
    ArtifactStoreCommit,
    FinalDeliveryManifest,
    StrictDeliveryContractProjection,
)


class StrictDeliveryContractProjector:
    """Propose strict delivery behavior without mutating Runner or Session state."""

    def project(
        self,
        *,
        source_run_status: Literal["completed", "failed"],
        commit: ArtifactStoreCommit,
        manifest: FinalDeliveryManifest | None,
    ) -> StrictDeliveryContractProjection:
        if manifest is None:
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=None,
                decision="reject_contract",
                failure_stage="configuration",
                reason_codes=["strict_delivery_manifest_missing"],
                plain_summary=(
                    "Strict delivery was requested without a FinalAssembler manifest."
                ),
            )

        binding_failures = _binding_failures(commit, manifest)
        if binding_failures:
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=manifest,
                decision="reject_contract",
                failure_stage="configuration",
                reason_codes=binding_failures,
                plain_summary=(
                    "The FinalAssembler manifest is not bound to the supplied "
                    "ArtifactStore decision."
                ),
            )

        if commit.status == "blocked" and manifest.status != "not_applicable":
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=manifest,
                decision="reject_contract",
                failure_stage="configuration",
                reason_codes=["artifact_store_manifest_status_inconsistent"],
                plain_summary=(
                    "A blocked ArtifactStore decision must have a not-applicable "
                    "delivery manifest."
                ),
            )
        if commit.status == "committed" and manifest.status == "not_applicable":
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=manifest,
                decision="reject_contract",
                failure_stage="configuration",
                reason_codes=["artifact_store_manifest_status_inconsistent"],
                plain_summary=(
                    "A committed ArtifactStore decision requires a ready or incomplete "
                    "delivery manifest."
                ),
            )

        if source_run_status == "failed":
            stage = "artifact_store" if commit.status == "blocked" else "runner"
            reasons = (
                ["artifact_store_not_committed", *commit.failure_codes]
                if commit.status == "blocked"
                else ["upstream_run_failed"]
            )
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=manifest,
                decision="preserve_upstream_failure",
                failure_stage=stage,
                reason_codes=_ordered_unique(reasons),
                plain_summary=(
                    "The existing upstream failure remains authoritative; strict "
                    "delivery does not convert it into success."
                ),
            )

        if commit.status == "blocked":
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=manifest,
                decision="preserve_upstream_failure",
                failure_stage="artifact_store",
                reason_codes=_ordered_unique(
                    ["artifact_store_not_committed", *commit.failure_codes]
                ),
                plain_summary=(
                    "ArtifactStore did not commit a final artifact, so strict delivery "
                    "preserves a failed run."
                ),
            )

        if manifest.status == "incomplete":
            finding_codes = [
                item.code for item in manifest.findings if item.severity == "blocking"
            ]
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=manifest,
                decision="block_delivery",
                failure_stage="final_assembler",
                reason_codes=_ordered_unique(
                    ["delivery_manifest_incomplete", *finding_codes]
                ),
                plain_summary=(
                    "ArtifactStore committed a file, but the delivery contract is "
                    "incomplete. The file remains evidence and is not proposed as the "
                    "official result."
                ),
            )

        if manifest.status != "ready" or manifest.authoritative_output is None:
            return _projection(
                source_run_status=source_run_status,
                commit=commit,
                manifest=manifest,
                decision="reject_contract",
                failure_stage="configuration",
                reason_codes=["delivery_manifest_not_ready_or_incomplete"],
                plain_summary="The delivery manifest has no valid strict-delivery state.",
            )

        return _projection(
            source_run_status=source_run_status,
            commit=commit,
            manifest=manifest,
            decision="deliver",
            failure_stage="none",
            reason_codes=["delivery_manifest_ready"],
            plain_summary=(
                "The completed run, ArtifactStore commit, and ready delivery manifest "
                "agree on one official output."
            ),
        )


def _binding_failures(
    commit: ArtifactStoreCommit,
    manifest: FinalDeliveryManifest,
) -> list[str]:
    checks = (
        (manifest.commit_id == commit.commit_id, "delivery_commit_id_mismatch"),
        (manifest.audit_id == commit.audit_id, "delivery_audit_id_mismatch"),
        (manifest.attempt_id == commit.attempt_id, "delivery_attempt_id_mismatch"),
        (
            manifest.commit_status == commit.status,
            "delivery_commit_status_mismatch",
        ),
        (
            manifest.committed_final_artifact_ref_id == commit.final_artifact_ref_id,
            "delivery_artifact_ref_mismatch",
        ),
        (
            manifest.committed_final_output_path == commit.final_output_path,
            "delivery_output_path_mismatch",
        ),
    )
    return [code for passed, code in checks if not passed]


def _projection(
    *,
    source_run_status: Literal["completed", "failed"],
    commit: ArtifactStoreCommit,
    manifest: FinalDeliveryManifest | None,
    decision: Literal[
        "deliver",
        "block_delivery",
        "preserve_upstream_failure",
        "reject_contract",
    ],
    failure_stage: Literal[
        "none", "runner", "artifact_store", "final_assembler", "configuration"
    ],
    reason_codes: list[str],
    plain_summary: str,
) -> StrictDeliveryContractProjection:
    deliver = decision == "deliver"
    committed_ref = commit.final_artifact_ref_id if commit.status == "committed" else None
    committed_path = commit.final_output_path if commit.status == "committed" else None
    return StrictDeliveryContractProjection(
        source_run_status=source_run_status,
        proposed_run_status="completed" if deliver else "failed",
        decision=decision,
        failure_stage=failure_stage,
        artifact_store_commit_id=commit.commit_id,
        artifact_store_status=commit.status,
        manifest_id=manifest.manifest_id if manifest is not None else None,
        manifest_status=manifest.status if manifest is not None else None,
        official_final_output_path=committed_path if deliver else None,
        evidence_committed_final_artifact_ref_id=committed_ref,
        evidence_committed_final_output_path=committed_path,
        session_current_action=("advance_current" if deliver else "preserve_current"),
        output_visibility=(
            "official_and_evidence"
            if deliver
            else "evidence_only"
            if committed_path is not None
            else "none"
        ),
        reason_codes=reason_codes,
        plain_summary=plain_summary,
    )


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
