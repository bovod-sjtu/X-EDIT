"""Declarative editing dialect, above the concrete tool program.

This module does not interpret instructions. It lowers explicit composition:
pipelines bind intermediate artifacts, local edits extract/edit/reinsert, and
concatenation builds ordered inputs. Effects and argument roles come from the
same capability cards available to every planner. Ambiguous implementations
must be selected by the author; the compiler never ranks tools.

AudioEditPlan is the legacy *concrete tool-program carrier*, not this dialect.
Graph and Actions do not pass through EditProgram or its lowering rules.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from x_editor.capabilities.schemas import OperationCapabilityCard
from x_editor.core.schemas import AudioEditPlan, AudioEditStep, PlanOutputArtifact, StrictModel
from x_editor.runtime.execution_graph import compile_execution_graph


class EditValue(StrictModel):
    value_of: str = Field(min_length=1)
    output: str | None = None


class Effect(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    effect: str = Field(min_length=1)
    implementation: str | None = None
    # Keys are semantic_role names in the public capability card, not tool
    # input/parameter paths. References use EditValue, never from_step.
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = "output_audio_path"


class Pipeline(StrictModel):
    kind: Literal["pipeline"]
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    source: str | EditValue | None
    effects: list[Effect] = Field(min_length=1)


class Region(StrictModel):
    start_sec: float = Field(ge=0, allow_inf_nan=False)
    end_sec: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("region must have positive duration")
        return self


class LocalEdit(StrictModel):
    kind: Literal["local_edit"]
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    source: str | EditValue
    region: Region
    effects: list[Effect] = Field(min_length=1)
    preserve_outside: Literal[True]
    duration_policy: Literal["retain_generated_duration", "fit_region"]

class Concatenate(StrictModel):
    kind: Literal["concatenate"]
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    sources: list[str | EditValue] = Field(min_length=1)
    crossfade_ms: int = Field(ge=0)
    pause_duration_sec: float = Field(ge=0, allow_inf_nan=False)


Edit = Annotated[Pipeline | LocalEdit | Concatenate, Field(discriminator="kind")]


class EditDelivery(StrictModel):
    artifact_id: str
    value: EditValue
    role: str
    primary: bool


class EditProgram(StrictModel):
    request: str
    input_audio: str = Field(min_length=1)
    edits: list[Edit] = Field(min_length=1)
    deliver: list[EditDelivery] = Field(min_length=1)


class LoweringError(ValueError):
    """Explicit unsupported/ambiguous lowering, never a fallback tool choice."""


def effect_catalog(cards: dict[str, Any]) -> list[dict[str, Any]]:
    """Mechanical public vocabulary; it contains no dataset or quality labels."""
    result = []
    for name, raw in sorted(cards.items()):
        card = raw if isinstance(raw, OperationCapabilityCard) else OperationCapabilityCard.model_validate(raw)
        roles = {}
        for argument in card.parameter_contract.planner_selectable:
            # semantic_role is historically also used for natural-language
            # descriptions. Such prose cannot be a stable program identifier.
            # Keep symbolic roles; otherwise use the explicit contract leaf.
            role = argument.semantic_role
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', role or ''):
                role = argument.path.rsplit('.', 1)[-1]
            if role in roles:
                raise LoweringError(f"ambiguous semantic argument role in {name}: {role}")
            roles[role] = argument.path
        for effect in card.semantic_effects:
            # Concrete argument leaf names are explicit, lossless aliases when
            # there is one destination. Do not require planners to invent a
            # second vocabulary merely to choose a higher-level composition.
            aliases = {}
            for path in roles.values():
                leaf = path.rsplit('.', 1)[-1]
                if sum(p.rsplit('.', 1)[-1] == leaf for p in roles.values()) == 1:
                    if leaf not in roles:
                        aliases[leaf] = path
            result.append(dict(effect=effect, implementation=name, arguments=roles,
                argument_aliases=aliases,
                threaded_source_role=next((role for role, path in roles.items()
                                           if path == 'inputs.input_audio'), None)))
    return result


class EditProgramCompiler:
    def __init__(self, cards: dict[str, Any]):
        self.catalog = effect_catalog(cards)
        self.operations = set(cards)
        self.steps: list[AudioEditStep] = []
        self.values: dict[str, dict] = {}
        self.mapping: list[dict] = []

    def resolve(self, value):
        if isinstance(value, EditValue):
            value = value.model_dump()
        if isinstance(value, dict):
            if 'value_of' in value:
                ref = EditValue.model_validate(value)
                if ref.value_of not in self.values:
                    raise LoweringError(f"unknown or forward edit value: {ref.value_of}")
                producer = self.values[ref.value_of]
                return dict(from_step=producer['from_step'], from_output=ref.output or producer['from_output'])
            if set(value) & {'from_step', 'from_output', 'result_of'}:
                raise LoweringError("concrete program references are not EditProgram values")
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        return value

    def emit(self, identifier, operation, inputs, params, origin, paths):
        if operation not in self.operations:
            raise LoweringError(f"lowering requires unavailable capability: {operation}")
        if any(s.id == identifier for s in self.steps):
            raise LoweringError(f"duplicate compiled step: {identifier}")
        deps = set()
        def visit(v):
            if isinstance(v, dict):
                if 'from_step' in v:
                    deps.add(v['from_step'])
                else:
                    for child in v.values(): visit(child)
            elif isinstance(v, list):
                for child in v: visit(child)
        visit([inputs, params])
        self.steps.append(AudioEditStep(id=identifier, operation=operation, inputs=inputs,
            params=params, depends_on=sorted(deps), description=f"Lowered from {origin}"))
        self.mapping.append(dict(step_id=identifier, origin=origin, operation=operation,
                                 argument_paths=paths))
        return dict(from_step=identifier, from_output='output_audio_path')

    def effects(self, effects, current, owner, *, local_coordinates=False):
        for effect in effects:
            matches = [row for row in self.catalog if row['effect'] == effect.effect and
                       (effect.implementation is None or row['implementation'] == effect.implementation)]
            if len(matches) != 1:
                raise LoweringError(f"effect {effect.effect}: specify exactly one available implementation")
            recipe = matches[0]
            accepted_arguments = {**recipe['arguments'], **recipe['argument_aliases']}
            inputs, params, paths = {}, {}, {}
            for role, value in effect.arguments.items():
                if role not in accepted_arguments:
                    raise LoweringError(f"unknown argument role {role} for {effect.effect}; "
                                        f"allowed roles: {', '.join(sorted(accepted_arguments))}")
                path = accepted_arguments[role]
                if path in paths:
                    raise LoweringError(f"argument supplied twice through aliases: {path}")
                namespace, key = path.split('.', 1)
                if '.' in key:
                    raise LoweringError(f"nested capability declaration requires explicit lowering: {path}")
                (inputs if namespace == 'inputs' else params)[key] = self.resolve(value)
                paths[path] = dict(origin=effect.id, path=f'arguments.{role}')
            if current is not None:
                if recipe['threaded_source_role'] is None:
                    raise LoweringError(f"{effect.effect} does not accept a threaded audio source; use source=null")
                if 'input_audio' in inputs and inputs['input_audio'] != current:
                    raise LoweringError("cannot override the pipeline source in effect arguments; "
                                        "omit the threaded source argument, especially inside local_edit "
                                        "where the compiler supplies the cropped region")
                # Repeating an identical binding has no semantic effect. Reject
                # different sources (including full source vs local crop) above.
                inputs['input_audio'] = current
                paths['inputs.input_audio'] = (None if isinstance(current, dict) and 'from_step' in current
                                               else dict(origin=owner, path='source'))
            current = self.emit(effect.id, recipe['implementation'], inputs, params, effect.id, paths)
            if local_coordinates and any(
                isinstance(container.get(key), list) and container[key]
                for container in (inputs, params)
                for key in ('selected_spans', 'spans', 'time_spans', 'segments_to_extract')
            ):
                # LocalEdit's effect arguments live on the threaded cropped input,
                # not on the source from which its region was selected.
                self.steps[-1].span_timebase = 'step_input'
            current = dict(current, from_output=effect.result)
            self.values[effect.id] = current
        return current

    def extract(self, identifier, source, start, end, owner, source_paths):
        return self.emit(identifier, 'artifact_ops.extract_spans', {'input_audio': source},
            {'selected_spans': [dict(start_sec=start, end_sec=end)], 'crossfade_ms': 0}, owner,
            {'inputs.input_audio': dict(origin=owner, path='source'),
             'params.selected_spans': dict(origin=owner, path=source_paths),
             'params.crossfade_ms': None})

    def compile(self, raw: dict) -> AudioEditPlan:
        self.steps, self.values, self.mapping = [], {}, []
        program = EditProgram.model_validate(raw)
        names = [e.id for e in program.edits] + [f.id for e in program.edits for f in getattr(e, 'effects', [])]
        if len(names) != len(set(names)):
            raise LoweringError("edit and effect IDs must be globally unique")
        self.authored_ids = set(names)
        for edit in program.edits:
            if isinstance(edit, Pipeline):
                current = self.effects(edit.effects, self.resolve(edit.source), edit.id)
            elif isinstance(edit, Concatenate):
                current = self.emit(edit.id, 'artifact_ops.concat',
                    {'audio_paths': self.resolve([v.model_dump() if isinstance(v, EditValue) else v for v in edit.sources])},
                    {'crossfade_ms': edit.crossfade_ms, 'pause_duration_sec': edit.pause_duration_sec}, edit.id,
                    {'inputs.audio_paths': dict(origin=edit.id, path='sources'),
                     'params.crossfade_ms': dict(origin=edit.id, path='crossfade_ms'),
                     'params.pause_duration_sec': dict(origin=edit.id, path='pause_duration_sec')})
            else:
                source = self.resolve(edit.source)
                start, end = edit.region.start_sec, edit.region.end_sec
                selected = self.extract(edit.id+'__crop', source, start, end, edit.id, 'region')
                changed = self.effects(edit.effects, selected, edit.id, local_coordinates=True)
                current = self.emit(edit.id, 'artifact_ops.reconstruct', {'input_audio': source},
                    {'edits': [dict(start_sec=start, end_sec=end, audio_path=changed,
                                    mode='replace' if edit.duration_policy == 'fit_region' else 'splice')],
                     'reconstruction_mode': 'timeline_edits', 'edited_artifact_scope': 'local_regions'}, edit.id,
                    {'inputs.input_audio': dict(origin=edit.id, path='source'),
                     'params.edits.0.start_sec': dict(origin=edit.id, path='region'),
                     'params.edits.0.end_sec': dict(origin=edit.id, path='region'),
                     'params.edits.0.audio_path': None, 'params.edits.0.mode': None,
                     'params.reconstruction_mode': None, 'params.edited_artifact_scope': None})
            self.values[edit.id] = current
        deliveries = []
        for output in program.deliver:
            ref = self.resolve(output.value)
            deliveries.append(PlanOutputArtifact(artifact_id=output.artifact_id, role=output.role,
                primary=output.primary, **ref))
        plan = AudioEditPlan(request=program.request, input_audio=program.input_audio, steps=self.steps,
                             output_artifacts=deliveries)
        compile_execution_graph(plan, graph_id='edit_program_validation')
        return plan

    def project_semantics(self, semantics):
        """Map authored evidence; compiler constants are not acoustic evidence."""
        if semantics is None:
            return None
        output = deepcopy(semantics)
        bindings = {b['plan_step_id']: b for b in semantics.get('bindings', [])}
        if len(bindings) != len(semantics.get('bindings', [])) or set(bindings) - self.authored_ids:
            raise LoweringError('duplicate or unknown abstract semantic binding')
        allowed_paths = {'source', 'region', 'sources', 'crossfade_ms', 'pause_duration_sec'}
        allowed_paths.update(f'arguments.{role}' for row in self.catalog for role in row['arguments'])
        allowed_paths.update(f'arguments.{role}' for row in self.catalog for role in row['argument_aliases'])
        if any(s['argument_path'] not in allowed_paths for b in bindings.values() for s in b.get('argument_sources', [])):
            raise LoweringError('abstract evidence annotations must use edit/effect paths, not concrete tool paths')
        projected = []
        for row in self.mapping:
            original = bindings.get(row['origin'])
            if original is None:
                continue  # Common admission reports missing evidence; never invent it.
            binding = deepcopy(original)
            binding.update(plan_step_id=row['step_id'], operation=row['operation'])
            sources = []
            for dest, source in row['argument_paths'].items():
                if source is None:
                    sources.append(dict(argument_path=dest, source_kind='deterministic_context'))
                    continue
                source_binding = bindings.get(source['origin'], {})
                for item in source_binding.get('argument_sources', []):
                    if item['argument_path'] == source['path']:
                        sources.append(dict(item, argument_path=dest))
            binding['argument_sources'] = sources
            projected.append(binding)
        output['bindings'] = projected
        usages = []
        for row in self.mapping:
            for dest, source in row['argument_paths'].items():
                if source:
                    for usage in semantics.get('audio_usage_bindings', []):
                        if usage['plan_step_id'] == source['origin'] and usage['argument_path'] == source['path']:
                            usages.append(dict(usage, plan_step_id=row['step_id'], argument_path=dest))
        output['audio_usage_bindings'] = usages
        return output
