"""Mechanical compilation from ``AudioEditPlan`` into an execution graph.

The Planner owns the operation choices, parameters, step count, and dependency
shape. This module only resolves explicit artifact references and validates the
resulting DAG. It never routes by operation name, user wording, or benchmark.

The native step-output reference is the current PlanIR object::

    {"from_step": "step_id", "from_output": "output_audio_path"}

The legacy ``<step-id>.output`` string remains a compatibility alias for that
object. ``plan.input_audio``, ``current.audio``, and a literal string exactly
equal to the plan's input audio all resolve to the plan-input artifact.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from x_editor.core.schemas import AudioEditPlan, AudioEditStep, StrictModel

ExecutionNodeStatus = Literal["pending", "running", "succeeded", "failed", "skipped", "blocked"]
ExecutionGraphStatus = Literal["pending", "running", "succeeded", "failed"]
ArtifactStatus = Literal[
    "available",
    "pending",
    "candidate",
    "produced",
    "rejected",
    "failed",
]
ArtifactSource = Literal["plan_input", "step_output"]
ArtifactBindingKind = Literal["plan_input", "step_output"]

_PLAN_INPUT_ALIASES = frozenset({"plan.input_audio", "current.audio"})
_DEFAULT_STEP_OUTPUT = "output_audio_path"


class StepOutputReference(StrictModel):
    """Typed form of the existing PlanIR step-output reference object."""

    from_step: str = Field(min_length=1)
    from_output: str = Field(min_length=1)


class ExecutionArtifact(StrictModel):
    """An input or future step output known at compile time."""

    artifact_id: str
    reference: str | StepOutputReference
    source: ArtifactSource
    producer_node_id: str | None = None
    value: Any | None = None
    status: ArtifactStatus
    duration_sec: float | None = Field(default=None, gt=0.0)
    timeline_revision: int | None = Field(default=None, ge=0)
    parent_artifact_ids: list[str] = Field(default_factory=list)
    verification_status: Literal["not_required", "pending", "accepted", "rejected"] = (
        "not_required"
    )


class ArtifactInputPort(StrictModel):
    """One explicit artifact binding in a step payload."""

    port_id: str
    node_id: str
    field_path: str
    binding_kind: ArtifactBindingKind
    source_reference: str | StepOutputReference
    artifact_id: str


class ArtifactOutputPort(StrictModel):
    """One named result available from a step."""

    port_id: str
    node_id: str
    name: str
    reference: StepOutputReference
    artifact_id: str
    status: ArtifactStatus = "pending"


class ExecutionTerminalOutput(StrictModel):
    """A delivered terminal artifact declared by the existing plan contract."""

    artifact_id: str
    source_port_id: str
    source_artifact_id: str
    from_step: str
    from_output: str
    role: str
    primary: bool


class ExecutionGraphNode(StrictModel):
    """Runtime representation of one Planner-authored step."""

    node_id: str
    plan_step_id: str
    operation: str
    description: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    dependency_node_ids: list[str] = Field(default_factory=list)
    input_ports: list[ArtifactInputPort] = Field(default_factory=list)
    output_ports: list[ArtifactOutputPort] = Field(default_factory=list)
    expected_output: str | None = None
    required: bool = True
    continue_on_error: bool | None = None
    status: ExecutionNodeStatus = "pending"


class ExecutionGraph(StrictModel):
    """Typed, not-yet-executed graph compiled from a dynamic plan."""

    graph_id: str
    compiler_version: Literal["planir_execution_graph/v2"] = "planir_execution_graph/v2"
    request: str
    status: ExecutionGraphStatus = "pending"
    artifacts: list[ExecutionArtifact] = Field(default_factory=list)
    nodes: list[ExecutionGraphNode]
    topological_order: list[str]
    terminal_outputs: list[ExecutionTerminalOutput]
    primary_output: ExecutionTerminalOutput


class ExecutionGraphCompileError(ValueError):
    """A typed mechanical validation failure while compiling PlanIR."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        step_id: str | None = None,
        field_path: str | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.step_id = step_id
        self.field_path = field_path


@dataclass(frozen=True)
class _CollectedReference:
    field_path: str
    binding_kind: ArtifactBindingKind
    source_reference: str | StepOutputReference
    from_step: str | None = None
    from_output: str | None = None


@dataclass(frozen=True)
class _OutputDeclaration:
    artifact_id: str
    from_step: str
    from_output: str
    role: str
    primary: bool


class PlanIRExecutionGraphCompiler:
    """Compile Planner IR without selecting or rewriting operations."""

    def compile(self, plan: AudioEditPlan, *, graph_id: str | None = None) -> ExecutionGraph:
        runtime_graph_id = graph_id or f"exec_{uuid4().hex}"
        if not runtime_graph_id.strip():
            raise ExecutionGraphCompileError("invalid_graph_id", "graph_id must not be empty")

        steps_by_id, positions = _index_steps(plan.steps)
        dependencies, successors = _validate_dependencies(steps_by_id, positions)
        topological_step_ids = _topological_sort(dependencies, successors, positions)
        if not topological_step_ids:
            raise ExecutionGraphCompileError(
                "no_terminal_output",
                "an execution graph requires at least one step and terminal output",
            )

        references_by_step = {
            step_id: _collect_artifact_references(step, plan.input_audio)
            for step_id, step in steps_by_id.items()
        }
        _validate_references(
            references_by_step,
            all_step_ids=frozenset(steps_by_id),
            dependencies=dependencies,
            plan_input_available=plan.input_audio is not None,
        )
        terminal_step_ids = [
            step_id for step_id in topological_step_ids if not successors[step_id]
        ]
        declarations = _read_output_declarations(plan)
        _validate_output_declarations(declarations, terminal_step_ids, frozenset(steps_by_id))

        node_ids = {
            step_id: f"{runtime_graph_id}:node:{positions[step_id]:04d}" for step_id in steps_by_id
        }
        output_names = _collect_output_names(
            steps_by_id,
            references_by_step,
            declarations,
            terminal_step_ids,
        )
        output_ports, output_port_by_key = _build_output_ports(
            runtime_graph_id=runtime_graph_id,
            steps_by_id=steps_by_id,
            positions=positions,
            node_ids=node_ids,
            output_names=output_names,
        )
        plan_input_artifact_id = f"{runtime_graph_id}:artifact:plan_input"

        artifacts: list[ExecutionArtifact] = []
        if plan.input_audio is not None:
            artifacts.append(
                ExecutionArtifact(
                    artifact_id=plan_input_artifact_id,
                    reference="plan.input_audio",
                    source="plan_input",
                    value=plan.input_audio,
                    status="available",
                    timeline_revision=0,
                )
            )
        for step_id in steps_by_id:
            for port in output_ports[step_id]:
                artifacts.append(
                    ExecutionArtifact(
                        artifact_id=port.artifact_id,
                        reference=port.reference,
                        source="step_output",
                        producer_node_id=port.node_id,
                        status="pending",
                        verification_status="pending",
                    )
                )

        nodes: list[ExecutionGraphNode] = []
        for step_id, step in steps_by_id.items():
            input_ports = _build_input_ports(
                node_id=node_ids[step_id],
                references=references_by_step[step_id],
                output_port_by_key=output_port_by_key,
                plan_input_artifact_id=plan_input_artifact_id,
            )
            nodes.append(
                ExecutionGraphNode(
                    node_id=node_ids[step_id],
                    plan_step_id=step_id,
                    operation=step.operation,
                    description=step.description,
                    inputs=deepcopy(step.inputs),
                    params=deepcopy(step.params),
                    depends_on=list(dependencies[step_id]),
                    dependency_node_ids=[node_ids[item] for item in dependencies[step_id]],
                    input_ports=input_ports,
                    output_ports=output_ports[step_id],
                    expected_output=step.expected_output,
                    required=step.required,
                    continue_on_error=step.continue_on_error,
                )
            )

        terminal_outputs = _build_terminal_outputs(
            declarations=declarations,
            terminal_step_ids=terminal_step_ids,
            output_port_by_key=output_port_by_key,
        )
        primary_output = next(item for item in terminal_outputs if item.primary)

        return ExecutionGraph(
            graph_id=runtime_graph_id,
            request=plan.request,
            artifacts=artifacts,
            nodes=nodes,
            topological_order=[node_ids[step_id] for step_id in topological_step_ids],
            terminal_outputs=terminal_outputs,
            primary_output=primary_output,
        )


def compile_execution_graph(plan: AudioEditPlan, *, graph_id: str | None = None) -> ExecutionGraph:
    """Compile an ``AudioEditPlan`` into a pending, mechanically valid DAG."""

    return PlanIRExecutionGraphCompiler().compile(plan, graph_id=graph_id)


def _index_steps(
    steps: Sequence[AudioEditStep],
) -> tuple[dict[str, AudioEditStep], dict[str, int]]:
    steps_by_id: dict[str, AudioEditStep] = {}
    positions: dict[str, int] = {}
    for position, step in enumerate(steps):
        if not step.id.strip():
            raise ExecutionGraphCompileError("invalid_step_id", "step IDs must not be empty")
        if step.id in steps_by_id:
            raise ExecutionGraphCompileError(
                "duplicate_step_id",
                f"step ID {step.id!r} appears more than once",
                step_id=step.id,
            )
        steps_by_id[step.id] = step
        positions[step.id] = position
    return steps_by_id, positions


def _validate_dependencies(
    steps_by_id: Mapping[str, AudioEditStep],
    positions: Mapping[str, int],
) -> tuple[dict[str, tuple[str, ...]], dict[str, list[str]]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    successors: dict[str, list[str]] = {step_id: [] for step_id in steps_by_id}

    for step_id, step in steps_by_id.items():
        if len(step.depends_on) != len(set(step.depends_on)):
            raise ExecutionGraphCompileError(
                "duplicate_dependency",
                f"step {step_id!r} declares the same dependency more than once",
                step_id=step_id,
            )
        missing = [item for item in step.depends_on if item not in steps_by_id]
        if missing:
            raise ExecutionGraphCompileError(
                "unknown_dependency",
                f"step {step_id!r} depends on unknown step(s): {missing}",
                step_id=step_id,
            )
        dependencies[step_id] = tuple(step.depends_on)
        for dependency in step.depends_on:
            successors[dependency].append(step_id)

    for successor_ids in successors.values():
        successor_ids.sort(key=positions.__getitem__)
    return dependencies, successors


def _topological_sort(
    dependencies: Mapping[str, tuple[str, ...]],
    successors: Mapping[str, list[str]],
    positions: Mapping[str, int],
) -> list[str]:
    in_degree = {step_id: len(items) for step_id, items in dependencies.items()}
    ready = [(positions[step_id], step_id) for step_id, degree in in_degree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []

    while ready:
        _, step_id = heapq.heappop(ready)
        ordered.append(step_id)
        for successor in successors[step_id]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                heapq.heappush(ready, (positions[successor], successor))

    if len(ordered) != len(dependencies):
        cycle_members = [
            step_id
            for step_id in sorted(dependencies, key=positions.__getitem__)
            if in_degree[step_id] > 0
        ]
        raise ExecutionGraphCompileError(
            "dependency_cycle",
            f"step dependency graph contains a cycle involving: {cycle_members}",
        )
    return ordered


def _collect_artifact_references(
    step: AudioEditStep,
    plan_input_audio: str | None,
) -> list[_CollectedReference]:
    return [
        *_iter_artifact_references(
            step.inputs,
            ("inputs",),
            step_id=step.id,
            plan_input_audio=plan_input_audio,
        ),
        *_iter_artifact_references(
            step.params,
            ("params",),
            step_id=step.id,
            plan_input_audio=plan_input_audio,
        ),
    ]


def _iter_artifact_references(
    value: Any,
    path: tuple[str, ...],
    *,
    step_id: str,
    plan_input_audio: str | None,
) -> Iterator[_CollectedReference]:
    field_path = _json_pointer(path)
    if isinstance(value, str):
        if value in _PLAN_INPUT_ALIASES or (
            plan_input_audio is not None and value == plan_input_audio
        ):
            yield _CollectedReference(
                field_path=field_path,
                binding_kind="plan_input",
                source_reference=value,
            )
        elif _is_string_step_output_alias(value):
            from_step = value[: -len(".output")]
            reference = StepOutputReference(
                from_step=from_step,
                from_output=_DEFAULT_STEP_OUTPUT,
            )
            yield _CollectedReference(
                field_path=field_path,
                binding_kind="step_output",
                source_reference=reference,
                from_step=reference.from_step,
                from_output=reference.from_output,
            )
        return

    if isinstance(value, Mapping):
        if "from_step" in value:
            if set(value) != {"from_step", "from_output"}:
                raise ExecutionGraphCompileError(
                    "invalid_step_output_reference",
                    "step-output reference objects require exactly from_step and from_output",
                    step_id=step_id,
                    field_path=field_path,
                )
            from_step = value.get("from_step")
            from_output = value.get("from_output")
            if not isinstance(from_step, str) or not from_step.strip():
                raise ExecutionGraphCompileError(
                    "invalid_step_output_reference",
                    "from_step must be a non-empty string",
                    step_id=step_id,
                    field_path=field_path,
                )
            if not isinstance(from_output, str) or not from_output.strip():
                raise ExecutionGraphCompileError(
                    "invalid_step_output_reference",
                    "from_output must be a non-empty string",
                    step_id=step_id,
                    field_path=field_path,
                )
            reference = StepOutputReference(from_step=from_step, from_output=from_output)
            yield _CollectedReference(
                field_path=field_path,
                binding_kind="step_output",
                source_reference=reference,
                from_step=reference.from_step,
                from_output=reference.from_output,
            )
            return
        for key, nested in value.items():
            yield from _iter_artifact_references(
                nested,
                (*path, str(key)),
                step_id=step_id,
                plan_input_audio=plan_input_audio,
            )
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            yield from _iter_artifact_references(
                nested,
                (*path, str(index)),
                step_id=step_id,
                plan_input_audio=plan_input_audio,
            )


def _validate_references(
    references_by_step: Mapping[str, list[_CollectedReference]],
    *,
    all_step_ids: frozenset[str],
    dependencies: Mapping[str, tuple[str, ...]],
    plan_input_available: bool,
) -> set[str]:
    consumed_step_ids: set[str] = set()
    for step_id, references in references_by_step.items():
        for reference in references:
            if reference.binding_kind == "plan_input":
                if not plan_input_available:
                    raise ExecutionGraphCompileError(
                        "unresolved_plan_input",
                        "plan input is referenced but plan.input_audio is not set",
                        step_id=step_id,
                        field_path=reference.field_path,
                    )
                continue

            producer_step_id = reference.from_step
            if producer_step_id not in all_step_ids:
                raise ExecutionGraphCompileError(
                    "unknown_step_output",
                    f"artifact reference names unknown step {producer_step_id!r}",
                    step_id=step_id,
                    field_path=reference.field_path,
                )
            if producer_step_id not in dependencies[step_id]:
                raise ExecutionGraphCompileError(
                    "undeclared_output_dependency",
                    (
                        f"artifact reference to {producer_step_id!r} requires that step "
                        "in depends_on"
                    ),
                    step_id=step_id,
                    field_path=reference.field_path,
                )
            consumed_step_ids.add(producer_step_id)
    return consumed_step_ids


def _read_output_declarations(plan: AudioEditPlan) -> list[_OutputDeclaration]:
    raw_declarations = getattr(plan, "output_artifacts", None) or []
    declarations: list[_OutputDeclaration] = []
    for index, item in enumerate(raw_declarations):
        values = item if isinstance(item, Mapping) else vars(item)
        try:
            artifact_id = values["artifact_id"]
            from_step = values["from_step"]
            from_output = values["from_output"]
            role = values["role"]
            primary = values["primary"]
        except (KeyError, TypeError) as exc:
            raise ExecutionGraphCompileError(
                "invalid_output_artifact",
                f"output_artifacts[{index}] does not satisfy PlanOutputArtifact",
            ) from exc
        string_fields = {
            "artifact_id": artifact_id,
            "from_step": from_step,
            "from_output": from_output,
            "role": role,
        }
        invalid = [
            name
            for name, value in string_fields.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid or not isinstance(primary, bool):
            raise ExecutionGraphCompileError(
                "invalid_output_artifact",
                f"output_artifacts[{index}] has invalid field(s): {invalid or ['primary']}",
            )
        declarations.append(
            _OutputDeclaration(
                artifact_id=artifact_id,
                from_step=from_step,
                from_output=from_output,
                role=role,
                primary=primary,
            )
        )
    return declarations


def _validate_output_declarations(
    declarations: list[_OutputDeclaration],
    terminal_step_ids: list[str],
    all_step_ids: frozenset[str],
) -> None:
    if not terminal_step_ids:
        raise ExecutionGraphCompileError(
            "no_terminal_output",
            "execution graph has no terminal output",
        )
    if not declarations:
        if len(terminal_step_ids) != 1:
            raise ExecutionGraphCompileError(
                "ambiguous_primary_output",
                (
                    "multiple terminal steps require AudioEditPlan.output_artifacts "
                    "with exactly one PlanOutputArtifact.primary"
                ),
            )
        return

    artifact_ids = [item.artifact_id for item in declarations]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ExecutionGraphCompileError(
            "duplicate_output_artifact_id",
            "AudioEditPlan.output_artifacts contains duplicate artifact IDs",
        )
    unknown_steps = sorted({item.from_step for item in declarations} - all_step_ids)
    if unknown_steps:
        raise ExecutionGraphCompileError(
            "unknown_output_artifact_step",
            f"output artifact declarations name unknown step(s): {unknown_steps}",
        )
    declared_terminal_steps = {item.from_step for item in declarations}
    if declared_terminal_steps != set(terminal_step_ids):
        raise ExecutionGraphCompileError(
            "output_artifact_terminal_mismatch",
            (
                "output_artifacts must declare every terminal step and only terminal steps; "
                f"declared={sorted(declared_terminal_steps)}, terminal={terminal_step_ids}"
            ),
        )
    primary = [item for item in declarations if item.primary]
    if len(primary) != 1:
        raise ExecutionGraphCompileError(
            "invalid_primary_output",
            "AudioEditPlan.output_artifacts must declare exactly one primary artifact",
        )


def _collect_output_names(
    steps_by_id: Mapping[str, AudioEditStep],
    references_by_step: Mapping[str, list[_CollectedReference]],
    declarations: list[_OutputDeclaration],
    terminal_step_ids: list[str],
) -> dict[str, list[str]]:
    names = {step_id: [] for step_id in steps_by_id}
    for references in references_by_step.values():
        for reference in references:
            if reference.binding_kind == "step_output":
                _append_unique(names[reference.from_step], reference.from_output)
    for declaration in declarations:
        _append_unique(names[declaration.from_step], declaration.from_output)
    if not declarations:
        for step_id in terminal_step_ids:
            _append_unique(names[step_id], _DEFAULT_STEP_OUTPUT)
    return names


def _build_output_ports(
    *,
    runtime_graph_id: str,
    steps_by_id: Mapping[str, AudioEditStep],
    positions: Mapping[str, int],
    node_ids: Mapping[str, str],
    output_names: Mapping[str, list[str]],
) -> tuple[
    dict[str, list[ArtifactOutputPort]],
    dict[tuple[str, str], ArtifactOutputPort],
]:
    ports_by_step: dict[str, list[ArtifactOutputPort]] = {}
    ports_by_key: dict[tuple[str, str], ArtifactOutputPort] = {}
    for step_id in steps_by_id:
        ports: list[ArtifactOutputPort] = []
        for output_index, output_name in enumerate(output_names[step_id]):
            artifact_id = (
                f"{runtime_graph_id}:artifact:{positions[step_id]:04d}:output:{output_index:04d}"
            )
            port = ArtifactOutputPort(
                port_id=f"{node_ids[step_id]}:port:output:{output_index:04d}",
                node_id=node_ids[step_id],
                name=output_name,
                reference=StepOutputReference(from_step=step_id, from_output=output_name),
                artifact_id=artifact_id,
            )
            ports.append(port)
            ports_by_key[(step_id, output_name)] = port
        ports_by_step[step_id] = ports
    return ports_by_step, ports_by_key


def _build_input_ports(
    *,
    node_id: str,
    references: list[_CollectedReference],
    output_port_by_key: Mapping[tuple[str, str], ArtifactOutputPort],
    plan_input_artifact_id: str,
) -> list[ArtifactInputPort]:
    ports: list[ArtifactInputPort] = []
    for index, reference in enumerate(references):
        if reference.binding_kind == "plan_input":
            artifact_id = plan_input_artifact_id
        else:
            artifact_id = output_port_by_key[
                (reference.from_step, reference.from_output)
            ].artifact_id
        ports.append(
            ArtifactInputPort(
                port_id=f"{node_id}:port:input:{index:04d}",
                node_id=node_id,
                field_path=reference.field_path,
                binding_kind=reference.binding_kind,
                source_reference=reference.source_reference,
                artifact_id=artifact_id,
            )
        )
    return ports


def _build_terminal_outputs(
    *,
    declarations: list[_OutputDeclaration],
    terminal_step_ids: list[str],
    output_port_by_key: Mapping[tuple[str, str], ArtifactOutputPort],
) -> list[ExecutionTerminalOutput]:
    if declarations:
        return [
            _terminal_output_from_declaration(item, output_port_by_key) for item in declarations
        ]

    step_id = terminal_step_ids[0]
    port = output_port_by_key[(step_id, _DEFAULT_STEP_OUTPUT)]
    return [
        ExecutionTerminalOutput(
            artifact_id=port.artifact_id,
            source_port_id=port.port_id,
            source_artifact_id=port.artifact_id,
            from_step=step_id,
            from_output=_DEFAULT_STEP_OUTPUT,
            role="primary",
            primary=True,
        )
    ]


def _terminal_output_from_declaration(
    declaration: _OutputDeclaration,
    output_port_by_key: Mapping[tuple[str, str], ArtifactOutputPort],
) -> ExecutionTerminalOutput:
    port = output_port_by_key[(declaration.from_step, declaration.from_output)]
    return ExecutionTerminalOutput(
        artifact_id=declaration.artifact_id,
        source_port_id=port.port_id,
        source_artifact_id=port.artifact_id,
        from_step=declaration.from_step,
        from_output=declaration.from_output,
        role=declaration.role,
        primary=declaration.primary,
    )


def _append_unique(items: list[str], value: str | None) -> None:
    if value is not None and value not in items:
        items.append(value)


def _is_string_step_output_alias(value: str) -> bool:
    return value.endswith(".output") and bool(value[: -len(".output")])


def _json_pointer(path: tuple[str, ...]) -> str:
    escaped = [item.replace("~", "~0").replace("/", "~1") for item in path]
    return "/" + "/".join(escaped)


__all__ = [
    "ArtifactInputPort",
    "ArtifactOutputPort",
    "ExecutionArtifact",
    "ExecutionGraph",
    "ExecutionGraphCompileError",
    "ExecutionGraphNode",
    "ExecutionTerminalOutput",
    "PlanIRExecutionGraphCompiler",
    "StepOutputReference",
    "compile_execution_graph",
]
