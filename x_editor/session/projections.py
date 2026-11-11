"""Projection and validation helpers for edit-session sidecar records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from x_editor.session.schemas import (
    ArtifactRef,
    NormalizedSessionRecord,
    ObjectiveRef,
    SessionEvent,
    SourceTextRef,
    SpanRef,
    ValidationIssue,
    ValidationReport,
)


def build_normalized_session_record(
    graph: Any,
    *,
    plan: Any | None = None,
    evidence: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> NormalizedSessionRecord:
    """Project graph/plan/evidence into stable refs plus a replayable event log."""

    graph_data = _object_dump(graph)
    plan_data = _object_dump(plan)
    evidence_data = _mapping(evidence)
    session_id = str(graph_data.get("session_id") or evidence_data.get("sample_id") or "session")
    aliases = _graph_aliases(graph, graph_data)
    operation_graph_indexes = _operation_graph_indexes(plan_data)
    artifacts = _artifact_refs(
        graph_data,
        aliases,
        operation_graph_nodes_by_path=operation_graph_indexes["by_path"],
    )
    path_to_ref_id = _path_to_ref_id(artifacts)
    current_audio_ref_id = _current_audio_ref_id(graph_data, path_to_ref_id)
    span_refs = _span_refs(
        graph_data,
        evidence_data,
        artifacts,
        current_audio_ref_id,
        artifact_operation_node_by_ref=_artifact_operation_node_by_ref(artifacts),
    )
    objective_refs = _objective_refs(
        plan_data or _plan_from_evidence(evidence_data),
        artifacts,
        span_refs,
        operation_graph_nodes_by_step=operation_graph_indexes["by_step"],
    )
    source_text_refs = _source_text_refs(plan_data, evidence_data, path_to_ref_id, current_audio_ref_id)
    events = _session_events(
        graph_data=graph_data,
        evidence_data=evidence_data,
        artifacts=artifacts,
        span_refs=span_refs,
        objective_refs=objective_refs,
    )
    return NormalizedSessionRecord(
        session_id=session_id,
        current_node_id=graph_data.get("current_node_id"),
        current_audio_ref_id=current_audio_ref_id,
        artifact_refs=artifacts,
        span_refs=span_refs,
        objective_refs=objective_refs,
        source_text_refs=source_text_refs,
        events=events,
        aliases=dict(sorted(aliases.items())),
        metadata=_without_empty(
            {
                "graph_schema_version": graph_data.get("schema_version"),
                "round_index": graph_data.get("round_index"),
                "operation_graph_node_count": operation_graph_indexes["node_count"],
                **(metadata or {}),
            }
        ),
    )


def normalized_refs_for_planner(
    graph: Any,
    *,
    plan: Any | None = None,
    evidence: dict[str, Any] | None = None,
    max_artifacts: int = 24,
    max_spans: int = 32,
    max_objectives: int = 16,
    max_events: int = 16,
) -> dict[str, Any]:
    """Return a compact normalized-ref block suitable for planner context JSON."""

    record = build_normalized_session_record(graph, plan=plan, evidence=evidence)
    report = validate_session_record(record, graph=graph)
    artifact_rows = [_model_row(item) for item in record.artifact_refs[-max_artifacts:]]
    span_rows = [_model_row(item) for item in record.span_refs[-max_spans:]]
    objective_rows = [_model_row(item) for item in record.objective_refs[-max_objectives:]]
    event_rows = [_model_row(item) for item in record.events[-max_events:]]
    return _without_empty(
        {
            "schema_version": record.schema_version,
            "session_id": record.session_id,
            "current_node_id": record.current_node_id,
            "current_audio_ref_id": record.current_audio_ref_id,
            "alias_count": len(record.aliases),
            "aliases": _alias_rows(record.aliases, record.artifact_refs),
            "artifact_refs": artifact_rows,
            "span_refs": span_rows,
            "objective_refs": objective_rows,
            "events": event_rows,
            "validation": {
                "valid": report.valid,
                "issue_counts": report.issue_counts,
                "issues": [_model_row(item) for item in report.issues[:8]],
            },
            "planner_policy": {
                "default_editable_ref": record.current_audio_ref_id,
                "use_alias": "session_current",
                "span_rule": "Use SpanRef.audio_ref_id plus start_sec/end_sec; do not copy spans across audio refs.",
                "artifact_rule": "Use ArtifactRef.ref_id or aliases instead of guessing paths from filenames.",
                "objective_rule": "Every compound edit objective should map to objective_refs and executable step ids.",
            },
        }
    )


def validate_session_record(
    record: NormalizedSessionRecord | dict[str, Any],
    *,
    graph: Any | None = None,
    evidence: dict[str, Any] | None = None,
) -> ValidationReport:
    """Validate normalized refs against basic graph/evidence consistency rules."""

    if isinstance(record, dict):
        record = NormalizedSessionRecord(**record)
    issues: list[ValidationIssue] = []
    artifact_ids = {item.ref_id for item in record.artifact_refs}
    artifact_paths = {_path_key(item.path) for item in record.artifact_refs if item.path}

    if record.current_audio_ref_id and record.current_audio_ref_id not in artifact_ids:
        issues.append(
            _issue(
                "error",
                "current_audio_ref_missing",
                f"current_audio_ref_id {record.current_audio_ref_id!r} is not present in artifact_refs.",
                location="current_audio_ref_id",
                ref_id=record.current_audio_ref_id,
            )
        )
    for alias, path in record.aliases.items():
        if path and _path_key(path) not in artifact_paths:
            issues.append(
                _issue(
                    "warning",
                    "alias_target_not_projected",
                    f"alias {alias!r} points to a path that is not present in artifact_refs.",
                    location=f"aliases.{alias}",
                    metadata={"path": path},
                )
            )

    for index, artifact in enumerate(record.artifact_refs):
        if not artifact.path:
            issues.append(
                _issue(
                    "error",
                    "artifact_path_missing",
                    "ArtifactRef.path is required.",
                    location=f"artifact_refs[{index}]",
                    ref_id=artifact.ref_id,
                )
            )
        if not artifact.provenance:
            issues.append(
                _issue(
                    "info",
                    "artifact_provenance_missing",
                    "ArtifactRef has no provenance string.",
                    location=f"artifact_refs[{index}].provenance",
                    ref_id=artifact.ref_id,
                )
            )

    for index, span in enumerate(record.span_refs):
        if span.audio_ref_id and span.audio_ref_id not in artifact_ids:
            issues.append(
                _issue(
                    "error",
                    "span_audio_ref_missing",
                    f"SpanRef.audio_ref_id {span.audio_ref_id!r} is not present in artifact_refs.",
                    location=f"span_refs[{index}].audio_ref_id",
                    ref_id=span.span_id,
                )
            )
        if not span.audio_ref_id:
            issues.append(
                _issue(
                    "warning",
                    "span_audio_ref_unbound",
                    "SpanRef is not bound to an ArtifactRef.",
                    location=f"span_refs[{index}].audio_ref_id",
                    ref_id=span.span_id,
                )
            )
        if span.start_sec is not None and span.end_sec is not None and span.end_sec <= span.start_sec:
            issues.append(
                _issue(
                    "error",
                    "span_time_range_invalid",
                    "SpanRef.end_sec must be greater than start_sec.",
                    location=f"span_refs[{index}]",
                    ref_id=span.span_id,
                )
            )

    step_ids = _step_ids_from_graph_or_events(graph, record)
    for index, objective in enumerate(record.objective_refs):
        unknown_steps = [step_id for step_id in objective.covered_by_steps if step_ids and step_id not in step_ids]
        if unknown_steps:
            issues.append(
                _issue(
                    "warning",
                    "objective_coverage_unknown_step",
                    "ObjectiveRef.covered_by_steps contains step ids not present in graph/event projection.",
                    location=f"objective_refs[{index}].covered_by_steps",
                    ref_id=objective.objective_id,
                    metadata={"unknown_step_ids": unknown_steps},
                )
            )
        if objective.status in {"planned", "covered_by_constraint"} and not (
            objective.covered_by_steps
            or objective.artifact_ref_ids
            or objective.span_ref_ids
            or objective.covered_by_operation_node_ids
        ):
            issues.append(
                _issue(
                    "warning",
                    "objective_coverage_unlinked",
                    "ObjectiveRef is marked covered but has no step/artifact/span link.",
                    location=f"objective_refs[{index}]",
                    ref_id=objective.objective_id,
                )
            )

    for issue in _source_text_drift_issues(record):
        issues.append(issue)
    if graph is not None:
        issues.extend(_graph_consistency_issues(_object_dump(graph)))
    if evidence:
        issues.extend(_evidence_consistency_issues(_mapping(evidence), record))

    counts = Counter(issue.severity for issue in issues)
    return ValidationReport(
        valid=counts.get("error", 0) == 0,
        issue_counts=dict(sorted(counts.items())),
        issues=issues,
    )


def _artifact_refs(
    graph_data: dict[str, Any],
    aliases: dict[str, str],
    *,
    operation_graph_nodes_by_path: dict[str, str] | None = None,
) -> list[ArtifactRef]:
    operation_graph_nodes_by_path = operation_graph_nodes_by_path or {}
    aliases_by_path: dict[str, list[str]] = {}
    for alias, path in aliases.items():
        aliases_by_path.setdefault(_path_key(path), []).append(alias)
    refs: list[ArtifactRef] = []
    seen: set[str] = set()
    nodes = _nodes(graph_data)
    for node_id, node in sorted(nodes.items(), key=lambda item: (_int(item[1].get("round_index")), item[0])):
        node_path = _string_or_none(node.get("primary_audio_path"))
        node_metadata = _mapping(node.get("metadata"))
        audio_paths = _dict_list(node.get("audio_paths"))
        for index, asset in enumerate(audio_paths, start=1):
            path = _string_or_none(asset.get("path"))
            if not path:
                continue
            operation_graph_node_id = operation_graph_nodes_by_path.get(_path_key(path))
            ref = ArtifactRef(
                ref_id=_artifact_ref_id(node_id, asset.get("asset_id") or f"asset_{index}"),
                path=path,
                role=str(asset.get("role") or node_metadata.get("asset_role") or node.get("kind") or "audio"),
                kind=str(node.get("kind") or asset.get("role") or "audio"),
                node_id=node_id,
                asset_id=str(asset.get("asset_id") or f"asset_{index}"),
                audio_id=_string_or_none(asset.get("audio_id")),
                round_index=_int_or_none(asset.get("round_index"), fallback=_int_or_none(node.get("round_index"))),
                step_id=_string_or_none(asset.get("step_id") or node_metadata.get("step_id")),
                operation=_string_or_none(asset.get("operation") or node_metadata.get("operation")),
                operation_graph_node_id=operation_graph_node_id,
                aliases=sorted(aliases_by_path.get(_path_key(path), [])),
                provenance=_string_or_none(asset.get("source") or node_metadata.get("source_round_edge_id") or f"node:{node_id}"),
                metadata=_without_empty(
                    {
                        "label": asset.get("label"),
                        "node_kind": node.get("kind"),
                        "span": asset.get("span"),
                        "speaker_ids": _mapping(asset.get("metadata")).get("speaker_ids") or node_metadata.get("speaker_ids"),
                        "operation_graph_node_id": operation_graph_node_id,
                        "session_eligible": node_metadata.get("session_eligible", True),
                    }
                ),
            )
            if ref.ref_id not in seen:
                refs.append(ref)
                seen.add(ref.ref_id)
        if node_path and not any(_same_path(node_path, item.get("path")) for item in audio_paths):
            operation_graph_node_id = operation_graph_nodes_by_path.get(_path_key(node_path))
            ref = ArtifactRef(
                ref_id=_artifact_ref_id(node_id, "primary"),
                path=node_path,
                role=str(node_metadata.get("asset_role") or node.get("kind") or "audio"),
                kind=str(node.get("kind") or "audio"),
                node_id=node_id,
                round_index=_int_or_none(node.get("round_index")),
                step_id=_string_or_none(node_metadata.get("step_id")),
                operation=_string_or_none(node_metadata.get("operation")),
                operation_graph_node_id=operation_graph_node_id,
                aliases=sorted(aliases_by_path.get(_path_key(node_path), [])),
                provenance=f"node:{node_id}.primary_audio_path",
                metadata=_without_empty(
                    {
                        "node_kind": node.get("kind"),
                        "operation_graph_node_id": operation_graph_node_id,
                        "session_eligible": node_metadata.get("session_eligible", True),
                    }
                ),
            )
            if ref.ref_id not in seen:
                refs.append(ref)
                seen.add(ref.ref_id)
    return refs


def _span_refs(
    graph_data: dict[str, Any],
    evidence_data: dict[str, Any],
    artifacts: list[ArtifactRef],
    current_audio_ref_id: str | None,
    artifact_operation_node_by_ref: dict[str, str] | None = None,
) -> list[SpanRef]:
    refs: list[SpanRef] = []
    seen: set[str] = set()
    artifact_operation_node_by_ref = artifact_operation_node_by_ref or {}
    artifact_by_node = {item.node_id: item.ref_id for item in artifacts if item.node_id}
    artifact_by_step = _artifacts_by_step(artifacts)
    artifacts_by_path = _path_to_ref_id(artifacts)

    def add(
        rows: Any,
        *,
        source: str,
        default_audio_ref_id: str | None,
        round_index: int | None = None,
        step_id: str | None = None,
        operation: str | None = None,
    ) -> None:
        default_operation_graph_node_id = (
            artifact_operation_node_by_ref.get(default_audio_ref_id or "")
            if default_audio_ref_id
            else None
        )
        for row_index, row in enumerate(_dict_list(rows), start=1):
            span = _span_ref(
                row,
                source=source,
                index=row_index,
                default_audio_ref_id=default_audio_ref_id,
                round_index=round_index,
                step_id=step_id,
                operation=operation,
                operation_graph_node_id=default_operation_graph_node_id,
            )
            if span and span.span_id not in seen:
                refs.append(span)
                seen.add(span.span_id)

    for node_id, node in _nodes(graph_data).items():
        node_metadata = _mapping(node.get("metadata"))
        node_audio_ref = artifact_by_node.get(node_id)
        for asset in _dict_list(node.get("audio_paths")):
            asset_metadata = _mapping(asset.get("metadata"))
            audio_ref = artifacts_by_path.get(_path_key(asset.get("path"))) or node_audio_ref
            add(
                [asset.get("span")] if isinstance(asset.get("span"), dict) else asset_metadata.get("span_set"),
                source=f"graph.nodes.{node_id}.audio_paths.{asset.get('asset_id') or 'asset'}",
                default_audio_ref_id=audio_ref,
                round_index=_int_or_none(asset.get("round_index"), fallback=_int_or_none(node.get("round_index"))),
                step_id=_string_or_none(asset.get("step_id") or node_metadata.get("step_id")),
                operation=_string_or_none(asset.get("operation") or node_metadata.get("operation")),
            )
        add(
            node_metadata.get("span_set"),
            source=f"graph.nodes.{node_id}.metadata.span_set",
            default_audio_ref_id=node_audio_ref,
            round_index=_int_or_none(node.get("round_index")),
            step_id=_string_or_none(node_metadata.get("step_id")),
            operation=_string_or_none(node_metadata.get("operation")),
        )

    for edge_id, edge in _edges(graph_data).items():
        evidence = _mapping(edge.get("evidence"))
        step_id = _first_string(edge.get("plan_step_ids"))
        output_ref = artifact_by_node.get(str(edge.get("target_node_id") or ""))
        if not output_ref:
            output_ref = artifacts_by_path.get(_path_key(_first_string(edge.get("output_audio_paths"))))
        add(
            evidence.get("selected_spans"),
            source=f"graph.edges.{edge_id}.evidence.selected_spans",
            default_audio_ref_id=output_ref,
            round_index=_int_or_none(edge.get("round_index")),
            step_id=step_id,
            operation=_first_string(edge.get("operations")) or _string_or_none(edge.get("operation_type")),
        )

    localization = _mapping(evidence_data.get("localization"))
    for source_index, source in enumerate(_dict_list(localization.get("sources")), start=1):
        step_id = _string_or_none(source.get("step_id"))
        step_artifacts = artifact_by_step.get(step_id or "") or []
        default_audio_ref_id = step_artifacts[-1].ref_id if step_artifacts else current_audio_ref_id
        add(
            source.get("selected_spans"),
            source=f"evidence.localization.sources[{source_index}]",
            default_audio_ref_id=default_audio_ref_id,
            step_id=step_id,
            operation=_string_or_none(source.get("operation")),
        )
    add(
        localization.get("selected_spans"),
        source="evidence.localization.selected_spans",
        default_audio_ref_id=current_audio_ref_id,
    )
    return refs


def _objective_refs(
    plan_data: dict[str, Any],
    artifacts: list[ArtifactRef],
    spans: list[SpanRef],
    operation_graph_nodes_by_step: dict[str, list[str]] | None = None,
) -> list[ObjectiveRef]:
    metadata = _mapping(plan_data.get("metadata"))
    operation_graph_nodes_by_step = operation_graph_nodes_by_step or {}
    objectives = _dict_list(metadata.get("high_level_objectives"))
    coverage_rows = _dict_list(metadata.get("objective_coverage"))
    coverage_by_id = {
        str(item.get("objective_id") or item.get("id") or ""): item
        for item in coverage_rows
        if item.get("objective_id") or item.get("id")
    }
    artifacts_by_step = _artifacts_by_step(artifacts)
    spans_by_step: dict[str, list[SpanRef]] = {}
    for span in spans:
        if span.step_id:
            spans_by_step.setdefault(span.step_id, []).append(span)
    refs: list[ObjectiveRef] = []
    for index, item in enumerate(objectives, start=1):
        objective_id = str(item.get("id") or item.get("objective_id") or f"obj_{index}")
        coverage = coverage_by_id.get(objective_id, {})
        covered_steps = _string_list(coverage.get("covered_by_steps") or coverage.get("step_ids") or [])
        artifact_ref_ids = _dedupe(
            artifact.ref_id
            for step_id in covered_steps
            for artifact in artifacts_by_step.get(step_id, [])
        )
        span_ref_ids = _dedupe(
            span.span_id
            for step_id in covered_steps
            for span in spans_by_step.get(step_id, [])
        )
        covered_by_operation_node_ids = _dedupe(
            node_id
            for step_id in covered_steps
            for node_id in operation_graph_nodes_by_step.get(step_id, [])
        )
        refs.append(
            ObjectiveRef(
                objective_id=objective_id,
                objective=str(item.get("objective") or item.get("goal") or item.get("description") or ""),
                type=_string_or_none(item.get("type")),
                required=bool(item.get("required", True)),
                source_instruction_fragment=_string_or_none(item.get("source_instruction_fragment")),
                covered_by_steps=covered_steps,
                status=_string_or_none(coverage.get("status")),
                coverage_reason=_string_or_none(coverage.get("coverage_reason") or coverage.get("reason")),
                artifact_ref_ids=artifact_ref_ids,
                span_ref_ids=span_ref_ids,
                covered_by_operation_node_ids=covered_by_operation_node_ids,
                metadata=_without_empty({"raw_objective": item, "raw_coverage": coverage}),
            )
        )
    return refs


def _source_text_refs(
    plan_data: dict[str, Any],
    evidence_data: dict[str, Any],
    path_to_ref_id: dict[str, str],
    current_audio_ref_id: str | None,
) -> list[SourceTextRef]:
    rows: list[SourceTextRef] = []
    seen: set[tuple[str, str]] = set()

    def add(text: Any, *, source: str, audio_path: Any = None, provenance: str | None = None) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        normalized = " ".join(text.split())
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        audio_ref_id = path_to_ref_id.get(_path_key(audio_path)) if audio_path else current_audio_ref_id
        key = (digest, audio_ref_id or "")
        if key in seen:
            return
        seen.add(key)
        rows.append(
            SourceTextRef(
                ref_id=f"source_text:{len(rows) + 1}",
                text_hash=digest,
                text_preview=normalized[:240],
                audio_ref_id=audio_ref_id,
                source_audio_path=_string_or_none(audio_path),
                source=source,
                provenance=provenance,
            )
        )

    perception = _mapping(evidence_data.get("perception"))
    files = _mapping(evidence_data.get("files"))
    input_audio = _first_string(files.get("input_audios"))
    add(evidence_data.get("source_text"), source="evidence.source_text", audio_path=input_audio)
    add(perception.get("source_text"), source="evidence.perception.source_text", audio_path=perception.get("source_text_audio_path") or input_audio)
    global_params = _mapping(plan_data.get("global_params"))
    add(global_params.get("source_text"), source="plan.global_params.source_text", audio_path=plan_data.get("input_audio"))
    for step_index, step in enumerate(_dict_list(plan_data.get("steps")), start=1):
        params = _mapping(step.get("params"))
        add(
            params.get("source_text"),
            source=f"plan.steps[{step_index}].params.source_text",
            audio_path=step.get("input_audio") or _mapping(step.get("inputs")).get("input_audio_path") or plan_data.get("input_audio"),
            provenance=_string_or_none(step.get("id")),
        )
    return rows


def _session_events(
    *,
    graph_data: dict[str, Any],
    evidence_data: dict[str, Any],
    artifacts: list[ArtifactRef],
    span_refs: list[SpanRef],
    objective_refs: list[ObjectiveRef],
) -> list[SessionEvent]:
    events: list[SessionEvent] = []
    artifact_by_node = {item.node_id: item.ref_id for item in artifacts if item.node_id}
    artifacts_by_step = _artifacts_by_step(artifacts)
    spans_by_step: dict[str, list[str]] = {}
    for span in span_refs:
        if span.step_id:
            spans_by_step.setdefault(span.step_id, []).append(span.span_id)

    def append(event_type: str, **kwargs: Any) -> None:
        events.append(
            SessionEvent(
                event_id=f"event:{len(events) + 1:04d}:{event_type}",
                event_type=event_type,
                **kwargs,
            )
        )

    append(
        "session_initialized",
        timestamp=_string_or_none(graph_data.get("created_at")),
        node_id=_first_string(graph_data.get("root_node_ids")),
        actor="runtime",
        summary="Session graph initialized.",
        refs={"artifacts": [item.ref_id for item in artifacts if item.round_index == 0]},
    )
    for record in _dict_list(graph_data.get("round_history")):
        round_index = _int_or_none(record.get("round_index"))
        input_node_id = _string_or_none(record.get("input_node_id"))
        result_node_id = _string_or_none(record.get("result_node_id"))
        append(
            "user_intent_recorded",
            timestamp=_string_or_none(record.get("started_at")),
            round_index=round_index,
            node_id=input_node_id,
            actor="user",
            summary=_string_or_none(record.get("request")),
            refs={"artifacts": _dedupe([artifact_by_node.get(input_node_id or "")])},
            payload=_without_empty({"status": record.get("status")}),
        )
        if result_node_id:
            append(
                "round_result_recorded",
                timestamp=_string_or_none(record.get("ended_at")),
                round_index=round_index,
                node_id=result_node_id,
                edge_id=_string_or_none(record.get("edge_id")),
                actor="runtime",
                summary=f"Round {round_index} completed with status {record.get('status')}.",
                refs={"artifacts": _dedupe([artifact_by_node.get(result_node_id)])},
                payload=_without_empty({"evidence_path": record.get("evidence_path")}),
            )
    for edge_id, edge in sorted(_edges(graph_data).items(), key=lambda item: (_int(item[1].get("round_index")), item[0])):
        step_ids = _string_list(edge.get("plan_step_ids"))
        output_node = _string_or_none(edge.get("target_node_id"))
        artifact_ids = _dedupe(
            [
                artifact_by_node.get(output_node or ""),
                *[
                    artifact.ref_id
                    for step_id in step_ids
                    for artifact in artifacts_by_step.get(step_id, [])
                ],
            ]
        )
        span_ids = _dedupe([span_id for step_id in step_ids for span_id in spans_by_step.get(step_id, [])])
        append(
            "operation_edge_recorded",
            timestamp=_string_or_none(edge.get("created_at")),
            round_index=_int_or_none(edge.get("round_index")),
            node_id=output_node,
            edge_id=edge_id,
            step_id=step_ids[0] if len(step_ids) == 1 else None,
            actor="executor",
            summary=_string_or_none(edge.get("instruction") or edge.get("operation_type")),
            refs=_without_empty({"artifacts": artifact_ids, "spans": span_ids}),
            payload=_without_empty(
                {
                    "operation_type": edge.get("operation_type"),
                    "status": edge.get("status"),
                    "operations": edge.get("operations"),
                    "source_node_ids": edge.get("source_node_ids"),
                }
            ),
        )
        for invocation in _dict_list(_mapping(edge.get("evidence")).get("tool_invocations")):
            invocation_step_id = _string_or_none(invocation.get("step_id")) or (step_ids[0] if step_ids else None)
            append(
                "tool_call_recorded",
                timestamp=_string_or_none(edge.get("created_at")),
                round_index=_int_or_none(edge.get("round_index")),
                edge_id=edge_id,
                step_id=invocation_step_id,
                actor="executor",
                summary=_string_or_none(invocation.get("tool_id") or invocation.get("operation")),
                payload=invocation,
            )
    if objective_refs:
        append(
            "objective_coverage_recorded",
            timestamp=_string_or_none(graph_data.get("updated_at")),
            actor="planner",
            summary="Planner objective coverage projected.",
            refs={"objectives": [item.objective_id for item in objective_refs]},
        )
    evaluator = _mapping(evidence_data.get("evaluator"))
    evaluation = _mapping(evaluator.get("evaluation")) or _mapping(evidence_data.get("evaluation"))
    if evaluation:
        append(
            "scoring_recorded",
            timestamp=_string_or_none(graph_data.get("updated_at")),
            actor="evaluator",
            summary="Evaluation/scoring result recorded.",
            payload=_without_empty(
                {
                    "passed": evaluation.get("passed"),
                    "score": evaluation.get("score"),
                    "need_replan": evaluation.get("need_replan"),
                }
            ),
        )
    return events


def _graph_aliases(graph: Any, graph_data: dict[str, Any]) -> dict[str, str]:
    if hasattr(graph, "audio_aliases") and callable(graph.audio_aliases):
        aliases = graph.audio_aliases()
        if isinstance(aliases, dict):
            return {str(key): str(value) for key, value in aliases.items() if value}
    aliases: dict[str, str] = {}
    current_id = graph_data.get("current_node_id")
    for node_id, node in _nodes(graph_data).items():
        path = _string_or_none(node.get("primary_audio_path"))
        if not path:
            continue
        aliases[f"session_node:{node_id}"] = path
        aliases[f"node:{node_id}"] = path
        if node_id == current_id:
            aliases["session_current"] = path
            aliases["current_session_audio"] = path
        if node.get("is_round_result"):
            aliases[f"session_round:{node.get('round_index')}:result"] = path
            aliases[f"round_{node.get('round_index')}_result"] = path
        if node.get("kind") == "initial":
            aliases["session_initial"] = path
            aliases["session_original"] = path
        metadata = _mapping(node.get("metadata"))
        audio_id = metadata.get("audio_id")
        if metadata.get("is_reference_landmark") and audio_id:
            aliases[f"session_reference:{audio_id}"] = path
            aliases[f"reference:{audio_id}"] = path
            aliases.setdefault(str(audio_id), path)
        for asset in _dict_list(node.get("audio_paths")):
            if asset.get("audio_id") and asset.get("path"):
                aliases[f"session_audio:{asset['audio_id']}"] = str(asset["path"])
                aliases.setdefault(str(asset["audio_id"]), str(asset["path"]))
    return aliases


def _span_ref(
    raw: dict[str, Any],
    *,
    source: str,
    index: int,
    default_audio_ref_id: str | None,
    round_index: int | None = None,
    step_id: str | None = None,
    operation: str | None = None,
    operation_graph_node_id: str | None = None,
) -> SpanRef | None:
    if not isinstance(raw, dict):
        return None
    start = _float_or_none(_first_present(raw.get("start_sec"), raw.get("start")))
    end = _float_or_none(_first_present(raw.get("end_sec"), raw.get("end")))
    speaker = _first_present(raw.get("speaker_id"), raw.get("speaker"), raw.get("speaker_label"))
    text = _first_present(raw.get("text"), raw.get("transcript"))
    if start is None and end is None and not speaker and not text and not raw.get("segment_id") and not raw.get("id"):
        return None
    source_step = _string_or_none(raw.get("step_id")) or step_id
    source_operation = _string_or_none(raw.get("operation")) or operation
    identity = json.dumps(
        _json_safe(
            {
                "source": source,
                "index": index,
                "segment_id": raw.get("segment_id") or raw.get("id"),
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "step_id": source_step,
            }
        ),
        sort_keys=True,
        ensure_ascii=False,
    )
    return SpanRef(
        span_id=str(raw.get("span_id") or f"span:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}"),
        audio_ref_id=_string_or_none(raw.get("audio_ref_id")) or default_audio_ref_id,
        coordinate_space=str(raw.get("coordinate_space") or "audio_timeline_seconds"),
        start_sec=start,
        end_sec=end,
        speaker_id=_string_or_none(speaker),
        text=_string_or_none(text),
        role=_string_or_none(raw.get("role") or raw.get("span_role") or raw.get("purpose") or raw.get("label")),
        source=source,
        round_index=_int_or_none(raw.get("round_index"), fallback=round_index),
        step_id=source_step,
        operation=source_operation,
        operation_graph_node_id=_string_or_none(raw.get("operation_graph_node_id")) or operation_graph_node_id,
        provenance=_string_or_none(raw.get("source") or raw.get("provenance") or source),
        metadata=_without_empty(
            {
                "segment_id": raw.get("segment_id") or raw.get("id"),
                "confidence": raw.get("confidence"),
                "reason": raw.get("reason"),
                "raw": raw,
            }
        ),
    )


def _current_audio_ref_id(graph_data: dict[str, Any], path_to_ref_id: dict[str, str]) -> str | None:
    current_id = graph_data.get("current_node_id")
    current_node = _nodes(graph_data).get(str(current_id or ""))
    if not current_node:
        return None
    path = current_node.get("primary_audio_path")
    if path:
        return path_to_ref_id.get(_path_key(path))
    return None


def _path_to_ref_id(artifacts: list[ArtifactRef]) -> dict[str, str]:
    by_path: dict[str, str] = {}
    for item in artifacts:
        by_path.setdefault(_path_key(item.path), item.ref_id)
    return by_path


def _artifact_operation_node_by_ref(artifacts: list[ArtifactRef]) -> dict[str, str]:
    return {
        item.ref_id: item.operation_graph_node_id
        for item in artifacts
        if item.operation_graph_node_id
    }


def _operation_graph_indexes(plan_data: dict[str, Any]) -> dict[str, Any]:
    metadata = _mapping(plan_data.get("metadata"))
    graph = metadata.get("operation_graph_actual")
    if not isinstance(graph, dict):
        graph = metadata.get("operation_graph_plan")
    if not isinstance(graph, dict):
        return {"by_path": {}, "by_step": {}, "node_count": 0}
    by_path: dict[str, str] = {}
    by_step: dict[str, list[str]] = {}
    node_count = 0
    for node in _dict_list(graph.get("nodes")):
        node_id = _string_or_none(node.get("node_id") or node.get("id"))
        if not node_id:
            continue
        node_count += 1
        path = _string_or_none(node.get("path"))
        if path:
            by_path.setdefault(_path_key(path), node_id)
        step_id = _string_or_none(node.get("step_id") or node.get("producer_step_id"))
        if step_id:
            by_step.setdefault(step_id, []).append(node_id)
    return {"by_path": by_path, "by_step": by_step, "node_count": node_count}


def _artifacts_by_step(artifacts: list[ArtifactRef]) -> dict[str, list[ArtifactRef]]:
    by_step: dict[str, list[ArtifactRef]] = {}
    for item in artifacts:
        if item.step_id:
            by_step.setdefault(item.step_id, []).append(item)
    return by_step


def _alias_rows(aliases: dict[str, str], artifacts: list[ArtifactRef], *, max_rows: int = 48) -> list[dict[str, Any]]:
    path_to_ref_id = _path_to_ref_id(artifacts)
    rows = [
        _without_empty({"alias": alias, "artifact_ref_id": path_to_ref_id.get(_path_key(path)), "path": path})
        for alias, path in sorted(aliases.items())
    ]
    priority = {"session_current": 0, "current_session_audio": 1, "session_initial": 2, "session_original": 3}
    rows.sort(key=lambda item: (priority.get(str(item.get("alias")), 10), str(item.get("alias"))))
    return rows[:max_rows]


def _graph_consistency_issues(graph_data: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    nodes = _nodes(graph_data)
    edges = _edges(graph_data)
    current_id = graph_data.get("current_node_id")
    if current_id and str(current_id) not in nodes:
        issues.append(_issue("error", "graph_current_node_missing", "Graph current_node_id does not exist.", location="graph.current_node_id"))
    for edge_id, edge in edges.items():
        target = str(edge.get("target_node_id") or "")
        if target and target not in nodes:
            issues.append(_issue("error", "graph_edge_target_missing", "Edge target node does not exist.", location=f"graph.edges.{edge_id}.target_node_id"))
        for source_id in _string_list(edge.get("source_node_ids")):
            if source_id not in nodes:
                issues.append(_issue("error", "graph_edge_source_missing", "Edge source node does not exist.", location=f"graph.edges.{edge_id}.source_node_ids", metadata={"source_node_id": source_id}))
    for index, record in enumerate(_dict_list(graph_data.get("round_history"))):
        edge_id = _string_or_none(record.get("edge_id"))
        if edge_id and edge_id not in edges:
            issues.append(_issue("warning", "round_edge_missing", "Round record references a missing edge.", location=f"graph.round_history[{index}].edge_id", metadata={"edge_id": edge_id}))
    return issues


def _evidence_consistency_issues(evidence: dict[str, Any], record: NormalizedSessionRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    files = _mapping(evidence.get("files"))
    output = files.get("output_audio_path")
    if output:
        projected = {_path_key(item.path) for item in record.artifact_refs}
        if _path_key(output) not in projected:
            issues.append(
                _issue(
                    "warning",
                    "evidence_output_not_projected",
                    "Evidence output_audio_path is not present in normalized artifact refs.",
                    location="evidence.files.output_audio_path",
                    metadata={"path": output},
                )
            )
    return issues


def _source_text_drift_issues(record: NormalizedSessionRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_audio: dict[str, set[str]] = {}
    for item in record.source_text_refs:
        key = item.audio_ref_id or item.source_audio_path or "unknown"
        by_audio.setdefault(key, set()).add(item.text_hash)
    for audio_ref_id, hashes in by_audio.items():
        if len(hashes) > 1:
            issues.append(
                _issue(
                    "warning",
                    "source_text_drift",
                    "Multiple source_text hashes are attached to the same audio reference.",
                    location="source_text_refs",
                    metadata={"audio_ref_id": audio_ref_id, "text_hashes": sorted(hashes)},
                )
            )
    return issues


def _step_ids_from_graph_or_events(graph: Any | None, record: NormalizedSessionRecord) -> set[str]:
    step_ids = {item.step_id for item in record.artifact_refs if item.step_id}
    step_ids.update(item.step_id for item in record.span_refs if item.step_id)
    step_ids.update(item.step_id for item in record.events if item.step_id)
    if graph is not None:
        for edge in _edges(_object_dump(graph)).values():
            step_ids.update(_string_list(edge.get("plan_step_ids")))
    return {str(item) for item in step_ids if item}


def _plan_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    planner = _mapping(evidence.get("planner"))
    plan = planner.get("plan")
    return plan if isinstance(plan, dict) else {}


def _object_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        dumped = value.to_dict()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _model_row(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _without_empty(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return _without_empty(value)
    return {}


def _artifact_ref_id(node_id: str, asset_id: Any) -> str:
    return f"artifact:{node_id}:{_safe_id(asset_id)}"


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    location: str | None = None,
    ref_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        location=location,
        ref_id=ref_id,
        metadata=metadata or {},
    )


def _nodes(graph_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = graph_data.get("nodes")
    if not isinstance(nodes, dict):
        return {}
    return {str(key): value for key, value in nodes.items() if isinstance(value, dict)}


def _edges(graph_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    edges = graph_data.get("edges")
    if not isinstance(edges, dict):
        return {}
    return {str(key): value for key, value in edges.items() if isinstance(value, dict)}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _first_string(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            if item not in (None, ""):
                return str(item)
        return None
    if value not in (None, ""):
        return str(value)
    return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any, *, fallback: int | None = None) -> int | None:
    try:
        if value is None:
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _without_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _path_key(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(Path(str(value)).expanduser().resolve())
    except Exception:
        return str(value)


def _same_path(left: Any, right: Any) -> bool:
    return bool(left and right and _path_key(left) == _path_key(right))


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    safe = "".join(char if char.isalnum() or char in {"_", "-", ":"} else "_" for char in text)
    return safe.strip("_") or "item"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
