"""Atomic writer for immutable three-plane flow-capacity releases."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schemas import (
    AtomicObjective,
    BudgetProfile,
    EvidenceRef,
    MetricBinding,
    MigrationRecord,
    ModuleAssessmentRecord,
    ReleaseManifest,
    RuntimeCaseRecord,
    SplitMembership,
    StandardInputRecord,
)
from .validation import validate_release


@dataclass(frozen=True)
class FlowCapacityRelease:
    manifest: ReleaseManifest
    runtime_cases: Sequence[RuntimeCaseRecord]
    assessments: Sequence[ModuleAssessmentRecord]
    standard_inputs: Sequence[StandardInputRecord]
    objectives: Sequence[AtomicObjective]
    metric_bindings: Sequence[MetricBinding]
    evidence_refs: Sequence[EvidenceRef]
    budget_profiles: Sequence[BudgetProfile]
    split_memberships: Sequence[SplitMembership] = ()
    evaluator_labels: Sequence[Mapping[str, Any]] = ()
    evaluator_rubrics: Sequence[Mapping[str, Any]] = ()
    evaluator_scorers: Sequence[Mapping[str, Any]] = ()
    provenance_records: Sequence[Mapping[str, Any]] = ()
    migration_records: Sequence[MigrationRecord] = ()


def write_release_bundle(
    output_dir: str | Path,
    release: FlowCapacityRelease,
) -> Path:
    """Validate and atomically create a release; never overwrite an old one."""

    validation = validate_release(
        manifest=release.manifest,
        runtime_cases=release.runtime_cases,
        assessments=release.assessments,
        standard_inputs=release.standard_inputs,
        objectives=release.objectives,
        metric_bindings=release.metric_bindings,
        evidence_refs=release.evidence_refs,
        budget_profiles=release.budget_profiles,
        split_memberships=release.split_memberships,
        migration_records=release.migration_records,
    )
    if not validation.passed:
        codes = sorted({item.code for item in validation.issues if item.severity == "error"})
        raise ValueError(f"flow-capacity release validation failed: {codes}")

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"immutable flow-capacity release already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        _write_json(temporary / "manifest.json", release.manifest)
        _write_jsonl(temporary / "runtime" / "cases.jsonl", release.runtime_cases)

        assessments_by_module: dict[str, list[ModuleAssessmentRecord]] = defaultdict(list)
        for item in release.assessments:
            assessments_by_module[item.boundary.module_id].append(item)
        for module, items in sorted(assessments_by_module.items()):
            _write_jsonl(temporary / "modules" / module / "assessments.jsonl", items)

        standards_by_module: dict[str, list[StandardInputRecord]] = defaultdict(list)
        for item in release.standard_inputs:
            standards_by_module[item.target_module].append(item)
        for module, items in sorted(standards_by_module.items()):
            _write_jsonl(temporary / "standards" / module / "inputs.jsonl", items)

        _write_jsonl(temporary / "evaluator" / "objectives.jsonl", release.objectives)
        _write_jsonl(temporary / "evaluator" / "metric_bindings.jsonl", release.metric_bindings)
        _write_jsonl(temporary / "evaluator" / "labels.jsonl", release.evaluator_labels)
        _write_jsonl(temporary / "evaluator" / "rubrics.jsonl", release.evaluator_rubrics)
        _write_jsonl(temporary / "evaluator" / "scorers.jsonl", release.evaluator_scorers)
        _write_jsonl(temporary / "evidence" / "index.jsonl", release.evidence_refs)
        _write_jsonl(temporary / "budgets" / "profiles.jsonl", release.budget_profiles)
        _write_jsonl(temporary / "splits" / "memberships.jsonl", release.split_memberships)
        _write_jsonl(temporary / "provenance" / "records.jsonl", release.provenance_records)
        _write_jsonl(temporary / "migrations" / "records.jsonl", release.migration_records)
        _write_json(temporary / "audits" / "validation.json", validation)
        _write_schemas(temporary / "schemas")
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_schemas(directory: Path) -> None:
    models: tuple[type[BaseModel], ...] = (
        ReleaseManifest,
        RuntimeCaseRecord,
        ModuleAssessmentRecord,
        StandardInputRecord,
        AtomicObjective,
        MetricBinding,
        EvidenceRef,
        BudgetProfile,
        SplitMembership,
        MigrationRecord,
    )
    for model in models:
        _write_json_value(
            directory / f"{model.__name__}.schema.json",
            model.model_json_schema(),
        )


def _write_json(path: Path, value: BaseModel | Mapping[str, Any]) -> None:
    _write_json_value(path, _json_value(value))


def _write_json_value(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[BaseModel | Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _json_value(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)
