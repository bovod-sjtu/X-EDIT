"""Cross-record validation for immutable flow-capacity releases."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from .projection import FORBIDDEN_STANDARD_KEYS, audit_runtime_projection, find_forbidden_key_paths
from .schemas import (
    AtomicObjective,
    BudgetProfile,
    EvidenceRef,
    MetricBinding,
    MigrationRecord,
    ModuleAssessmentRecord,
    ReleaseManifest,
    ReleaseValidationReport,
    RuntimeCaseRecord,
    SplitMembership,
    StandardInputRecord,
    ValidationIssue,
)


def validate_release(
    *,
    manifest: ReleaseManifest,
    runtime_cases: Sequence[RuntimeCaseRecord],
    assessments: Sequence[ModuleAssessmentRecord],
    standard_inputs: Sequence[StandardInputRecord],
    objectives: Sequence[AtomicObjective],
    metric_bindings: Sequence[MetricBinding],
    evidence_refs: Sequence[EvidenceRef],
    budget_profiles: Sequence[BudgetProfile],
    split_memberships: Sequence[SplitMembership] = (),
    migration_records: Sequence[MigrationRecord] = (),
) -> ReleaseValidationReport:
    issues: list[ValidationIssue] = []
    _check_manifest_counts(
        manifest,
        runtime_cases=runtime_cases,
        assessments=assessments,
        standard_inputs=standard_inputs,
        objectives=objectives,
        evidence_refs=evidence_refs,
        issues=issues,
    )

    runtime_by_id = _unique_by("runtime_record_uid", runtime_cases, lambda item: item.record_uid, issues)
    assessment_by_id = _unique_by(
        "assessment_record_id", assessments, lambda item: item.assessment_record_id, issues
    )
    standard_by_id = _unique_by(
        "standard_input_id", standard_inputs, lambda item: item.standard_input_id, issues
    )
    objective_by_id = _unique_by(
        "objective_instance_id", objectives, lambda item: item.objective_instance_id, issues
    )
    binding_by_id = _unique_by("binding_id", metric_bindings, lambda item: item.binding_id, issues)
    evidence_by_id = _unique_by(
        "evidence_ref_id", evidence_refs, lambda item: item.evidence_ref_id, issues
    )
    budget_by_id = _unique_by(
        "budget_profile_id", budget_profiles, lambda item: item.budget_profile_id, issues
    )

    for record in runtime_cases:
        for path in audit_runtime_projection(record):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="runtime_projection_forbidden_key",
                    record_ref=record.record_uid,
                    message=f"forbidden runtime key at {path}",
                )
            )
        for asset in record.audio_assets:
            if asset.evidence_ref_id not in evidence_by_id:
                _missing_ref(issues, "runtime_asset_evidence_missing", record.record_uid, asset.evidence_ref_id)

    exact_assessment_keys: set[tuple[str, str, str, str]] = set()
    for item in assessments:
        exact_key = (
            item.identity.case_fingerprint,
            item.identity.sidecar_record_key,
            item.boundary.module_id,
            item.boundary.boundary_id,
        )
        if exact_key in exact_assessment_keys:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_exact_module_assessment",
                    record_ref=item.assessment_record_id,
                    message="exact case, module, and boundary tuple is duplicated",
                )
            )
        exact_assessment_keys.add(exact_key)
        if item.runtime_record_uid not in runtime_by_id:
            _missing_ref(issues, "assessment_runtime_case_missing", item.assessment_record_id, item.runtime_record_uid)
        else:
            runtime_case = runtime_by_id[item.runtime_record_uid]
            if runtime_case.case_fingerprint != item.identity.case_fingerprint:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="assessment_runtime_fingerprint_mismatch",
                        record_ref=item.assessment_record_id,
                        message="assessment fingerprint does not match referenced runtime case",
                    )
                )
        if item.budget_profile_ref not in budget_by_id:
            _missing_ref(issues, "assessment_budget_missing", item.assessment_record_id, item.budget_profile_ref)
        for objective_ref in item.objective_refs:
            if objective_ref not in objective_by_id:
                _missing_ref(issues, "assessment_objective_missing", item.assessment_record_id, objective_ref)
        for evidence_ref in item.evidence_refs:
            if evidence_ref not in evidence_by_id:
                _missing_ref(issues, "assessment_evidence_missing", item.assessment_record_id, evidence_ref)
        for arm in item.input_arms:
            if arm.input_regime == "standard" and arm.payload_ref and arm.payload_ref not in standard_by_id:
                _missing_ref(issues, "assessment_standard_input_missing", item.assessment_record_id, arm.payload_ref)

    candidate_set_counts: Counter[tuple[str, str]] = Counter()
    canonical_by_candidate_set: Counter[tuple[str, str]] = Counter()
    for item in standard_inputs:
        if item.assessment_record_id not in assessment_by_id:
            _missing_ref(issues, "standard_assessment_missing", item.standard_input_id, item.assessment_record_id)
        candidate_set_counts[(item.assessment_record_id, item.candidate_set_id)] += 1
        if item.selection_role == "canonical":
            canonical_by_candidate_set[(item.assessment_record_id, item.candidate_set_id)] += 1
        for path in find_forbidden_key_paths(item.payload, forbidden_keys=FORBIDDEN_STANDARD_KEYS):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="standard_input_forbidden_key",
                    record_ref=item.standard_input_id,
                    message=f"forbidden standard-input key at {path}",
                )
            )
        assessment = assessment_by_id.get(item.assessment_record_id)
        if assessment is not None:
            if assessment.boundary.module_id != item.target_module:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="standard_input_target_module_mismatch",
                        record_ref=item.standard_input_id,
                        message="standard input target module does not match assessment",
                    )
                )
            if assessment.boundary.boundary_id != item.boundary_id:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="standard_input_boundary_mismatch",
                        record_ref=item.standard_input_id,
                        message="standard input boundary does not match assessment",
                    )
                )
    for key in candidate_set_counts:
        count = canonical_by_candidate_set[key]
        if count != 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="standard_input_canonical_cardinality",
                    record_ref=key[0],
                    message=f"candidate set {key[1]} has {count} canonical inputs",
                )
            )

    for item in objectives:
        for dependency in item.dependencies:
            if dependency not in objective_by_id:
                _missing_ref(issues, "objective_dependency_missing", item.objective_instance_id, dependency)
        for binding_ref in item.metric_binding_refs:
            if binding_ref not in binding_by_id:
                _missing_ref(issues, "objective_metric_binding_missing", item.objective_instance_id, binding_ref)
        if item.counting_role == "leaf_flow" and not item.metric_binding_refs and not item.boundary_expectations:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="leaf_objective_has_no_acceptance_path",
                    record_ref=item.objective_instance_id,
                    message="leaf flow objective needs a metric binding or boundary expectation",
                )
            )

    _check_objective_cycles(objectives, issues)
    for item in metric_bindings:
        if item.objective_instance_id not in objective_by_id:
            _missing_ref(issues, "metric_objective_missing", item.binding_id, item.objective_instance_id)
        elif item.binding_id not in objective_by_id[item.objective_instance_id].metric_binding_refs:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="metric_binding_not_declared_by_objective",
                    record_ref=item.binding_id,
                    message="metric binding must be declared by its objective",
                )
            )

    for item in evidence_refs:
        if item.visibility == "evaluator_private" and item.available_to:
            runtime_consumers = {
                "planner_reasoner",
                "planner",
                "reasoner",
                "perception",
                "executor",
                "tool",
                "verifier",
                "feedback_controller",
            }
            exposed = sorted(runtime_consumers.intersection(item.available_to))
            if exposed:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="evaluator_evidence_exposed_to_runtime",
                        record_ref=item.evidence_ref_id,
                        message=f"evaluator-private evidence exposed to {','.join(exposed)}",
                    )
                )

    _check_split_groups(split_memberships, issues)
    _check_migrations(manifest, migration_records, issues)
    return ReleaseValidationReport(
        passed=not any(item.severity == "error" for item in issues),
        issues=issues,
        checked_counts={
            "runtime_cases": len(runtime_cases),
            "assessments": len(assessments),
            "standard_inputs": len(standard_inputs),
            "objectives": len(objectives),
            "metric_bindings": len(metric_bindings),
            "evidence_refs": len(evidence_refs),
            "budget_profiles": len(budget_profiles),
            "split_memberships": len(split_memberships),
            "migration_records": len(migration_records),
        },
    )


def _check_manifest_counts(
    manifest: ReleaseManifest,
    *,
    runtime_cases: Sequence[RuntimeCaseRecord],
    assessments: Sequence[ModuleAssessmentRecord],
    standard_inputs: Sequence[StandardInputRecord],
    objectives: Sequence[AtomicObjective],
    evidence_refs: Sequence[EvidenceRef],
    issues: list[ValidationIssue],
) -> None:
    actual = {
        "runtime_case_count": len(runtime_cases),
        "assessment_count": len(assessments),
        "standard_input_count": len(standard_inputs),
        "objective_count": len(objectives),
        "evidence_ref_count": len(evidence_refs),
    }
    for field, count in actual.items():
        if getattr(manifest, field) != count:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="manifest_count_mismatch",
                    record_ref=manifest.release_id,
                    message=f"{field}: manifest={getattr(manifest, field)} actual={count}",
                )
            )


def _unique_by(label: str, items: Iterable, key, issues: list[ValidationIssue]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in items:
        value = str(key(item))
        if value in result:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_record_id",
                    record_ref=value,
                    message=f"duplicate {label}",
                )
            )
        result[value] = item
    return result


def _missing_ref(issues: list[ValidationIssue], code: str, record_ref: str, missing: str) -> None:
    issues.append(
        ValidationIssue(
            severity="error",
            code=code,
            record_ref=record_ref,
            message=f"missing reference: {missing}",
        )
    )


def _check_objective_cycles(
    objectives: Sequence[AtomicObjective], issues: list[ValidationIssue]
) -> None:
    graph = {item.objective_instance_id: list(item.dependencies) for item in objectives}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(dep in graph and visit(dep) for dep in graph.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    for node in graph:
        if visit(node):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="objective_dependency_cycle",
                    record_ref=node,
                    message="objective dependency graph contains a cycle",
                )
            )
            return


def _check_split_groups(
    memberships: Sequence[SplitMembership], issues: list[ValidationIssue]
) -> None:
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in memberships:
        for kind, group in (
            ("base_case", item.base_case_id),
            ("session", item.session_id),
            ("canonical", item.canonical_group_id),
            ("pair", item.pair_group_id),
        ):
            if group:
                group_splits[(kind, group)].add(item.split)
    for (kind, group), splits in group_splits.items():
        if len(splits) > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="group_split_leakage",
                    record_ref=group,
                    message=f"{kind} group crosses splits: {sorted(splits)}",
                )
            )


def _check_migrations(
    manifest: ReleaseManifest,
    migrations: Sequence[MigrationRecord],
    issues: list[ValidationIssue],
) -> None:
    by_id = {item.migration_id: item for item in migrations}
    if manifest.migration_ref is None:
        if migrations:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unreferenced_migration_records",
                    record_ref=manifest.release_id,
                    message="release contains migration records but manifest has no migration_ref",
                )
            )
        return
    migration = by_id.get(manifest.migration_ref)
    if migration is None:
        _missing_ref(issues, "manifest_migration_missing", manifest.release_id, manifest.migration_ref)
        return
    if migration.destination_release_id != manifest.release_id:
        issues.append(
            ValidationIssue(
                severity="error",
                code="migration_destination_mismatch",
                record_ref=migration.migration_id,
                message="migration destination does not match release id",
            )
        )
