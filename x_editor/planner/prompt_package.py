"""Auditable prompt packages for the native Planner/Reasoner workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Literal

from pydantic import Field

from x_editor.core.schemas import StrictModel
from x_editor.planner.composition_contract import (
    COMPOSITION_RULES,
    session_completion_choices,
)
from x_editor.planner.contract_projection import (
    audio_usage_evidence_inventory,
    group_step_arguments,
)
from x_editor.planner.evidence_index import EvidenceIndexFormat, factor_evidence_index
from x_editor.planner.argument_table import ArgumentContractFormat, factor_argument_contracts
from x_editor.planner.retry_context import ACTIONABLE_RETRY_POLICY, RetryGuidanceMode
from x_editor.planner.observation_contracts import observation_input_contracts
from x_editor.planner.reasoned_workflow import (
    MemoQualifiedPlannerInput,
)
from x_editor.reasoner.contracts import session_completion_proof_inventory

PlannerPromptPhase = Literal["pre_perception", "post_perception"]
PlannerPromptProfileName = Literal[
    "catalog_strict_v1",
    "objective_coverage_v2",
    "evidence_complete_v3",
    "robust_contract_v4",
    "robust_contract_v5",
    "robust_contract_v6",
]
DEFAULT_PLANNER_PROMPT_PROFILE: PlannerPromptProfileName = "objective_coverage_v2"
CandidateProjectionSource = Literal[
    "not_applicable",
    "frozen_edit_intent",
    "conservative_all_edit_operations",
    "complete_edit_catalog",
    "typed_effect_shortlist",
]

_OMITTED_REDUNDANT_SECTIONS = (
    "available_operations_full_specs",
    "duplicate_operation_capability_catalog",
    "output_schema_in_user_payload",
    "reverse_evidence_path_sources",
)


def _type_outline(*parts: str) -> str:
    return "".join(parts)


_COMMON_TURN_TYPE_OUTLINE = (
    _type_outline(
        "PlannerReasonerTurn={schema_version?:string,decision:string,",
        "edit_intent:EditIntent|null,perception_intent:PerceptionIntent|null,",
        "plan:AudioEditPlan|null,execution_semantics:ExecutionSemantics|null,",
        "tool_selection:ToolSelection|null,missing_information:list[string],",
        "decision_summary:string}",
    ),
    _type_outline(
        "EditIntent={schema_version?:string(exactly ",
        "x-editor.planner-edit-intent/v1),effect_kinds:list[string],",
        "target_scopes:list[string],edit_objectives:list[string],",
        "preservation_requirements:list[IntentRequirement],",
        "quality_requirements:list[IntentRequirement],required_evidence:list[string],",
        "candidate_operations:list[string],uncertainties:list[string]}",
    ),
    _type_outline(
        "IntentRequirement={dimension:string,priority:string(must|should|prefer),",
        "description:string}",
    ),
)

_PRE_PERCEPTION_TYPE_OUTLINE = (
    _type_outline(
        "PerceptionIntent={schema_version?:string(exactly ",
        "x-editor.planner-perception-intent/v1),request_id:string,",
        "scope:string(general|targeted),perception_operation?:string(auto or compatibility hint),",
        "edit_objective:string,perception_targets:list[PerceptionTarget],",
        "target_hints:list[string],evidence_questions:list[string],",
        "expected_evidence_fields:list[string],",
        "temporal_precision:string(coarse|segment|word),",
        "candidate_operations:list[string],ambiguities:list[string],",
        "audio_refs:list[string],primary_audio_ref:string,requirements:list[string],",
        "observation_jobs:list[ObservationJob]}",
    ),
    _type_outline(
        "ObservationJob={job_id:string,audio_ref:string,",
        "supporting_audio_refs:list[string],requirements:list[string],",
        "expected_evidence_fields:list[string],required:boolean}",
    ),
    _type_outline(
        "PerceptionTarget={kind:string(event_label|speaker_hint|quoted_text|",
        "semantic_property|audio_property),value:string}",
    ),
)

_POST_PERCEPTION_TYPE_OUTLINE = (
    _type_outline(
        "AudioEditPlan={request:string(required; copy the current user request),",
        "input_audio:string|null,output_audio:string|null,",
        "steps:list[AudioEditStep],global_params:object,metadata:object,",
        "missing_inputs:list[string]}",
    ),
    _type_outline(
        "AudioEditStep={id:string,operation:string,description:string,inputs:object,",
        "params:object,expected_output:string|null,depends_on:list[string],",
        "status:string,result:object|null,error:string|null,required:boolean,",
        "continue_on_error:boolean|null}",
    ),
    _type_outline(
        "ExecutionSemantics={schema_version?:string(exactly ",
        "x-editor.planner-execution-semantics/v1),bindings:list[SemanticBinding],",
        "audio_usage_bindings:list[AudioUsageBinding]}",
    ),
    _type_outline(
        "AudioUsageBinding={schema_version?:string,plan_step_id:string,",
        "argument_path:string,audio_ref:string,usage:string(edit_input|",
        "reference_input|speaker_grounded_reference|assembly_member|",
        "comparison_input)}",
    ),
    _type_outline(
        "SemanticBinding={plan_step_id:string,target:string(required),",
        "evidence_ids:list[string],evidence_paths:list[EvidencePath],operation:string,",
        "argument_sources:list[ArgumentSource],unresolved_requirements:list[string]}",
    ),
    "EvidencePath={evidence_id:string,field_path:string}; never emit a bare path string",
    _type_outline(
        "ArgumentSource={argument_path:string,source_kind:string(user_request|",
        "perception_evidence|derived_from_evidence|deterministic_context),",
        "evidence_id:string|null,evidence_path:string|null}",
    ),
    _type_outline(
        "ToolSelection={schema_version?:string(exactly ",
        "x-editor.planner-tool-selection/v1),steps:list[StepToolSelection]}",
    ),
    _type_outline(
        "StepToolSelection={plan_step_id:string,selected_operation:string,",
        "candidates:list[CandidateAssessment],",
        "parameter_selections:list[ParameterSelection],selection_summary:string}",
    ),
    _type_outline(
        "CandidateAssessment={operation:string,semantic_fit:string(full|partial|none|",
        "unknown),preservation_fit:string(full|partial|none|unknown),",
        "quality_fit:string(full|partial|none|unknown),legal_inputs_available:boolean,",
        "rejection_codes:list[string]}",
    ),
    _type_outline(
        "ParameterSelection={argument_path:string,selected_value:any,",
        "contract_operation:string}",
    ),
)

_COMMON_CURRENT_CALL_RULES = (
    (
        "Confirm target-selection criterion as well as location."
    ),
    (
        "Objects are closed: emit only outlined fields; no aliases, helpers, or "
        "alternate shapes."
    ),
    (
        "Every preservation/quality requirement needs non-empty dimension and "
        "description; priority must be exactly one of must, should, or prefer."
    ),
    (
        "Use workflow_prior_turn_semantic_context only when its result is an ancestor "
        "of the current source. Resolve coreference/undo; never reuse tool parameters."
    ),
)

_PRE_PERCEPTION_CURRENT_CALL_RULES = (
    (
        "Every edit_intent.candidate_operations item must be an operation from this "
        "call's edit_operation_capability_index."
    ),
    (
        "Do not select a concrete Perception provider or specialist route. Describe "
        "the observation need; set perception_operation to auto or omit it. A concrete "
        "perception.* value is accepted only as a backward-compatible hint and is not "
        "authoritative."
    ),
    (
        "perception_intent.primary_audio_ref must appear exactly in audio_refs, and "
        "every audio ref must come from this call's workflow source/reference refs."
    ),
    (
        "For multi-audio evidence, emit one observation_jobs item per audio that "
        "must actually be observed. Give each job only its own requirements and "
        "expected fields; do not claim that listing an audio_ref means it was observed."
    ),
)

_PRE_PERCEPTION_CURRENT_CALL_RULES_V4 = (
    (
        "Every edit_intent.effect_kinds item must be an effect_kind from this "
        "call's semantic_effect_capability_index."
    ),
    (
        "edit_intent.candidate_operations may be empty. When present, every item "
        "must be listed as a supporting_operation for at least one declared effect."
    ),
    *_PRE_PERCEPTION_CURRENT_CALL_RULES[1:],
)

_POST_PERCEPTION_CURRENT_CALL_RULES_V1 = (
    (
        "With frozen_edit_intent, copy it field-for-field into edit_intent; do not "
        "alter values."
    ),
    (
        "Plan fields are request, input_audio, output_audio, steps, global_params, "
        "metadata, and missing_inputs; plan.steps_summary and other fields are invalid."
    ),
    (
        "Step operations, candidate operations, and selected_operation must come from "
        "candidate_operation_capability_cards; selection must match its step."
    ),
    (
        "Fill every required planner_selectable path with a value legal under its "
        "type, enum, range, and conditions; never emit undeclared/framework_owned "
        "paths. If source_text is required, it cannot be omitted or guessed."
    ),
    (
        "tool_selection.steps needs exactly one item for every plan step and must "
        "include plan_step_id equal to that plan step's id. Mirror each authored value "
        "once with the same path/value and matching contract_operation."
    ),
    (
        "Use at most one binding per plan step; merge all evidence paths and argument "
        "sources there, with matching plan_step_id and operation."
    ),
    (
        "ArgumentSource.source_kind must be exactly user_request, "
        "perception_evidence, derived_from_evidence, or deterministic_context. The two "
        "evidence-backed kinds require evidence_id and evidence_path; the other two "
        "must omit both fields."
    ),
    (
        "For every plan argument that directly consumes a supplied audio ref, emit "
        "one execution_semantics.audio_usage_bindings item with the exact scalar "
        "argument path and this turn's semantic use. Asset source/reference labels are "
        "availability hints, not permanent roles. The framework attaches evidence "
        "requirements and provenance."
    ),
    (
        "If any required Planner-selectable value, evidence binding, or legal "
        "candidate is unavailable, return cannot_plan instead of inventing it."
    ),
)

_POST_PERCEPTION_CURRENT_CALL_RULES_V2 = (
    (
        "When frozen_edit_intent is present, preserve effect_kinds, target_scopes, "
        "edit_objectives, preservation_requirements, quality_requirements, and "
        "required_evidence. candidate_operations are provisional and may be revised "
        "after reviewing evidence and complete capability cards; uncertainties may "
        "be removed only when reviewed evidence resolves them."
    ),
    (
        "plan allows only request, input_audio, output_audio, steps, global_params, "
        "metadata, and missing_inputs. plan.steps_summary and every other extra field "
        "are invalid."
    ),
    (
        "Every plan step operation, CandidateAssessment.operation, and "
        "selected_operation must come from this call's "
        "candidate_operation_capability_cards; selected_operation must equal the "
        "matching plan step operation."
    ),
    (
        "A planner_selectable path is an address, not a literal nested key. For "
        "example, params.source_text means plan.steps[i].params.source_text, never "
        "plan.steps[i].params.params.source_text. inputs.input_audio follows the same "
        "rule."
    ),
    (
        "For each selected operation, fill every required planner_selectable address "
        "using only values legal under its type, enum, range, cardinality, value "
        "guidance, and conditional rules. Do not emit undeclared or framework_owned "
        "paths."
    ),
    (
        "One step has one operation contract. If no single capability card covers all "
        "effects assigned to a step, decompose the objective into dependent steps. An "
        "operation must never perform a semantic effect absent from its card."
    ),
    (
        "After the executable plan is complete, mirror each authored "
        "planner_selectable value exactly once into parameter_selections. Use the "
        "same address, value, and selected operation; do not create duplicate paths."
    ),
    (
        "execution_semantics.bindings may contain at most one binding per plan step. "
        "Bind every evidence-derived scalar leaf with an exact reviewed evidence path, "
        "including each start_sec and end_sec inside selected span arrays."
    ),
    (
        "ArgumentSource.source_kind must be exactly user_request, "
        "perception_evidence, derived_from_evidence, or deterministic_context. The two "
        "evidence-backed kinds require evidence_id and evidence_path; the other two "
        "must omit both fields."
    ),
    (
        "For every plan argument that directly consumes a supplied audio ref, emit "
        "one execution_semantics.audio_usage_bindings item with the exact scalar "
        "argument path and this turn's semantic use. The framework attaches evidence "
        "requirements and provenance."
    ),
    (
        "Before returning execute, verify objective coverage, required paths, array "
        "cardinality, unique parameter paths, evidence leaf bindings, and from_step "
        "dependencies. If any required value or safe route is unavailable, return "
        "cannot_plan instead of inventing it."
    ),
)

_POST_PERCEPTION_CURRENT_CALL_RULES_V3 = (
    (
        "When frozen_edit_intent is present, preserve target_scopes, "
        "edit_objectives, preservation_requirements, quality_requirements, and "
        "required_evidence. effect_kinds are ordered pre-evidence hypotheses: post-"
        "Perception may remove disproved entries but must not add or reorder them. "
        "candidate_operations are provisional and may be revised; uncertainties may "
        "be removed only when reviewed evidence resolves them."
    ),
    *_POST_PERCEPTION_CURRENT_CALL_RULES_V2[1:],
    (
        "depends_on declares order but does not redirect audio. When a step transforms "
        "a predecessor's result, inputs.input_audio must use that predecessor's "
        "from_step object rather than the original source ref. Do not fill optional "
        "parameters merely because a default exists; satisfy every conditional input "
        "for the chosen value or select another fully fitting legal value."
    ),
    (
        "For evidence_complete_v3 cards, planner_selectable.enum_values is the legal "
        "domain for this call after current-context constraints. catalog_enum_values "
        "is audit-only and its removed values must not be selected. An empty "
        "available_values list means that reference parameter cannot be authored."
    ),
    (
        "Frozen tool_prior_guidance is development-only route ranking. Apply one "
        "item only when its task_scope matches the current objective and every "
        "compared route already passes semantic, legal-input, preservation, and "
        "evidence checks. no_preference forbids an implicit tie-break. The prior "
        "never supplies parameters, admits a plan, or blocks execution."
    ),
)

_POST_PERCEPTION_CURRENT_CALL_RULES_V4 = (
    (
        "When frozen_edit_intent is present, preserve target_scopes, "
        "edit_objectives, preservation_requirements, quality_requirements, and "
        "required_evidence. effect_kinds and candidate_operations are pre-evidence "
        "representation hypotheses and may be revised only from the complete current "
        "capability catalog after reviewed evidence; do not add or weaken an objective."
    ),
    *_POST_PERCEPTION_CURRENT_CALL_RULES_V3[1:],
)

_POST_PERCEPTION_CURRENT_CALL_RULES_V5 = (
    (
        "Preserve frozen target_scopes, edit_objectives, preservation_requirements, "
        "quality_requirements, and required_evidence. Reviewed evidence may revise "
        "effect_kinds and candidate_operations, but may not delete or weaken an "
        "objective."
    ),
    (
        "Choose one capability card per plan step. Use only that card's "
        "parameter_contract.planner_selectable addresses and current legal values; "
        "never author framework_owned addresses."
    ),
    (
        "A parameter path is an address, not a nested literal key: params.source_text "
        "means plan.steps[i].params.source_text and inputs.input_audio means "
        "plan.steps[i].inputs.input_audio."
    ),
    (
        "For cumulative multi-step editing, connect every consuming step to the "
        "producer with depends_on and "
        '{"from_step":"step_id","from_output":"output_audio_path"}. '
        "Use the original source only for a genuinely independent branch."
    ),
    (
        "Build the plan first, then mirror every Planner-selected value exactly once "
        "in tool_selection.parameter_selections. Every tool-selection row requires a "
        "complete candidates array containing the selected operation."
    ),
    (
        "Bind every evidence-derived scalar leaf to an exact reviewed evidence id and "
        "path. If a required legal value, dependency, or evidence source is missing, "
        "return cannot_plan instead of guessing."
    ),
)

_POST_PERCEPTION_CURRENT_CALL_RULES_V6 = (
    (
        "Preserve frozen effect_kinds, target_scopes, edit_objectives, "
        "preservation_requirements, quality_requirements, and required_evidence. "
        "Reviewed evidence resolves targets and parameter values; it does not "
        "authorize an unrelated effect class. Return cannot_plan with an explicit "
        "intent/evidence mismatch if the frozen representation is contradicted."
    ),
    (
        "Choose one shortlisted capability card per plan step. The shortlist is the "
        "union of frozen candidate_operations and exact frozen effect matches. Use "
        "only that card's parameter_contract.planner_selectable addresses and "
        "current legal values; never author framework_owned addresses."
    ),
    (
        "Treat each planner_selectable value_guidance and value_examples entry as "
        "part of the legal value contract. A field described as tagged text, DSL, "
        "XML, or a structured label must contain the complete executable form from "
        "that card, never natural-language prose. Preserve the complete source text "
        "outside declared tags and use only the card's embedded enum values."
    ),
    *_POST_PERCEPTION_CURRENT_CALL_RULES_V5[2:],
)

_PRE_PERCEPTION_MINIMAL_EXAMPLES: dict[str, Any] = {
    "intent_requirement": {
        "dimension": "<current requirement dimension>",
        "priority": "must",
        "description": "<non-empty current requirement description>",
    },
}

_POST_PERCEPTION_MINIMAL_EXAMPLES: dict[str, Any] = {
    "single_step_cross_object_link": {
        "plan_step": {
            "id": "step_1",
            "operation": "<operation from current candidate cards>",
            "inputs": {"<required planner_selectable input>": "<legal value>"},
            "params": {"<required planner_selectable parameter>": "<legal value>"},
        },
        "one_merged_semantic_binding": {
            "plan_step_id": "step_1",
            "operation": "<same operation as plan_step.operation>",
            "argument_sources": [
                {
                    "argument_path": "<exact inputs.* or params.* path>",
                    "source_kind": "perception_evidence",
                    "evidence_id": "<current reviewed evidence id>",
                    "evidence_path": "<exact current reviewed evidence path>",
                }
            ],
        },
        "tool_selection_step": {
            "plan_step_id": "step_1",
            "selected_operation": "<same operation as plan_step.operation>",
            "parameter_selections": [
                {
                    "argument_path": "<exact planner_selectable path>",
                    "selected_value": "<same value as the plan>",
                    "contract_operation": "<same selected operation>",
                }
            ],
        },
    },
}

_POST_PERCEPTION_COMPLETE_EXAMPLE_V4: dict[str, Any] = {
    "execute_shape_for_each_plan_step": {
        "plan_step": {
            "id": "step_1",
            "operation": "<operation from current candidate cards>",
            "description": "<short objective-aligned action>",
            "inputs": {"<required input leaf>": "<legal current value>"},
            "params": {"<required parameter leaf>": "<legal current value>"},
            "expected_output": None,
            "depends_on": [],
            "status": "pending",
            "result": None,
            "error": None,
            "required": True,
            "continue_on_error": None,
        },
        "tool_selection_step": {
            "plan_step_id": "step_1",
            "selected_operation": "<same as plan_step.operation>",
            "candidates": [
                {
                    "operation": "<same selected operation>",
                    "semantic_fit": "full",
                    "preservation_fit": "full",
                    "quality_fit": "full",
                    "legal_inputs_available": True,
                    "rejection_codes": [],
                }
            ],
            "parameter_selections": [
                {
                    "argument_path": "<exact inputs.* or params.* address>",
                    "selected_value": "<same value as the plan>",
                    "contract_operation": "<same selected operation>",
                }
            ],
            "selection_summary": "<short comparison result>",
        },
        "evidence_binding_when_needed": {
            "plan_step_id": "step_1",
            "objective": "<objective covered by this step>",
            "target": "<grounded edit target>",
            "evidence_ids": ["<current reviewed evidence id>"],
            "evidence_paths": [
                {
                    "evidence_id": "<same current reviewed evidence id>",
                    "field_path": "<exact current reviewed evidence path>",
                }
            ],
            "operation": "<same as plan_step.operation>",
            "argument_sources": [
                {
                    "argument_path": "<exact inputs.* or params.* address>",
                    "source_kind": "perception_evidence",
                    "evidence_id": "<same current reviewed evidence id>",
                    "evidence_path": "<same exact reviewed evidence path>",
                }
            ],
            "unresolved_requirements": [],
        },
    },
}


class PromptBudgetReport(StrictModel):
    """Exact prompt sizes and catalog projection facts for one provider call."""

    schema_version: Literal["x-editor.planner-prompt-budget/v1"] = (
        "x-editor.planner-prompt-budget/v1"
    )
    phase: PlannerPromptPhase
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE
    system_prompt_chars: int = Field(ge=0)
    current_prompt_chars: int = Field(ge=0)
    total_prompt_chars: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    section_chars: dict[str, int] = Field(default_factory=dict)
    operation_index_count: int = Field(ge=0)
    full_capability_card_count: int = Field(ge=0)
    candidate_projection_source: CandidateProjectionSource
    candidate_operations: list[str] = Field(default_factory=list)
    unavailable_candidate_operations: list[str] = Field(default_factory=list)
    candidate_operations_without_full_contract: list[str] = Field(default_factory=list)
    omitted_redundant_sections: list[str] = Field(default_factory=list)
    critical_contracts_complete: bool = True


class PlannerPromptPackage(StrictModel):
    """One stable system prompt plus the exact current-call package."""

    schema_version: Literal["x-editor.planner-prompt-package/v1"] = (
        "x-editor.planner-prompt-package/v1"
    )
    phase: PlannerPromptPhase
    system_prompt: str
    current_payload: dict[str, Any]
    current_prompt: str
    package_fingerprint: str
    budget_report: PromptBudgetReport


def planner_reasoner_turn_contract_outline(
    phase: PlannerPromptPhase,
    *,
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
) -> dict[str, Any]:
    """Return a compact phase contract for providers using JSON-object mode."""

    if phase == "pre_perception":
        branch_rules = {
            "request_perception": (
                "decision=request_perception; emit edit_intent and perception_intent; "
                "omit plan, execution_semantics, and tool_selection"
            ),
            "cannot_plan": (
                "decision=cannot_plan; require non-empty missing_information:list[string]; "
                "omit work products"
            ),
        }
        phase_types = _PRE_PERCEPTION_TYPE_OUTLINE
        phase_rules = (
            _PRE_PERCEPTION_CURRENT_CALL_RULES_V4
            if prompt_profile == "robust_contract_v4"
            else _PRE_PERCEPTION_CURRENT_CALL_RULES
        )
        minimal_examples = _PRE_PERCEPTION_MINIMAL_EXAMPLES
    else:
        branch_rules = {
            "execute": (
                "decision=execute; emit edit_intent, plan, execution_semantics, and "
                "tool_selection; missing_information=[]"
            ),
            "cannot_plan": (
                "decision=cannot_plan; require non-empty missing_information:list[string]; "
                "omit work products"
            ),
        }
        phase_types = _POST_PERCEPTION_TYPE_OUTLINE
        if prompt_profile == "catalog_strict_v1":
            phase_rules = _POST_PERCEPTION_CURRENT_CALL_RULES_V1
        elif prompt_profile == "objective_coverage_v2":
            phase_rules = _POST_PERCEPTION_CURRENT_CALL_RULES_V2
        elif prompt_profile == "evidence_complete_v3":
            phase_rules = _POST_PERCEPTION_CURRENT_CALL_RULES_V3
        elif prompt_profile == "robust_contract_v4":
            phase_rules = _POST_PERCEPTION_CURRENT_CALL_RULES_V4
        else:
            phase_rules = _POST_PERCEPTION_CURRENT_CALL_RULES_V5
        minimal_examples = (
            _POST_PERCEPTION_COMPLETE_EXAMPLE_V4
            if prompt_profile == "robust_contract_v4"
            else _POST_PERCEPTION_MINIMAL_EXAMPLES
        )
    return {
        "contract_kind": "compact_type_outline_not_full_json_schema",
        "notation": "? optional; list[T] is a JSON array of T",
        "example_rule": "Examples show structure; replace <...> with current values.",
        "exactness_rule": "Use exact names/types; do not wrap the turn in decision.",
        "branch_rules": branch_rules,
        "exact_current_call_rules": [*_COMMON_CURRENT_CALL_RULES, *phase_rules],
        "type_outline": [*_COMMON_TURN_TYPE_OUTLINE, *phase_types],
        "minimal_structure_examples": copy.deepcopy(minimal_examples),
    }


def compact_operation_capability_index(
    operations: list[dict[str, Any]],
    *,
    perception_operation_ids: set[str],
) -> list[dict[str, Any]]:
    """Project semantic route facts without exposing full parameter contracts."""

    index: list[dict[str, Any]] = []
    for operation_spec in operations:
        operation = str(operation_spec.get("operation") or "").strip()
        if not operation:
            continue
        card = operation_spec.get("operation_capability")
        item: dict[str, Any] = {
            "operation": operation,
            "operation_kind": ("perception" if operation in perception_operation_ids else "edit"),
        }
        if isinstance(card, dict):
            item.update(
                {
                    "semantic_effects": _string_list(card.get("semantic_effects")),
                    "supported_scopes": _string_list(card.get("supported_scopes")),
                    "requirements": _requirements_index(card.get("requirements")),
                    "preservation_levels": _dimension_level_index(
                        card.get("preservation"),
                        name_field="dimension",
                    ),
                    "quality_levels": _dimension_level_index(
                        card.get("quality_characteristics"),
                        name_field="dimension",
                    ),
                    "artifact_risk_levels": _dimension_level_index(
                        card.get("artifact_risks"),
                        name_field="risk",
                    ),
                    "composition": copy.deepcopy(card.get("composition") or {}),
                    "limitations": _string_list(card.get("limitations")),
                    "parameter_contract_available": isinstance(
                        card.get("parameter_contract"), dict
                    ),
                }
            )
        else:
            item.update(
                {
                    "description": str(operation_spec.get("description") or ""),
                    "required_inputs": _string_list(operation_spec.get("required_inputs")),
                    "parameter_contract_available": False,
                }
            )
        index.append(item)
    return index


def compact_semantic_effect_capability_index(
    operations: list[dict[str, Any]],
    *,
    perception_operation_ids: set[str],
) -> list[dict[str, Any]]:
    """Group pre-Perception choices by typed effect instead of tool order."""

    effects: dict[str, dict[str, set[str]]] = {}
    for operation_spec in operations:
        operation = str(operation_spec.get("operation") or "").strip()
        if not operation or operation in perception_operation_ids:
            continue
        card = operation_spec.get("operation_capability")
        if not isinstance(card, dict):
            continue
        scopes = set(_string_list(card.get("supported_scopes")))
        evidence = set(_requirements_index(card.get("requirements"))["evidence"])
        for effect_kind in _string_list(card.get("semantic_effects")):
            item = effects.setdefault(
                effect_kind,
                {
                    "supporting_operations": set(),
                    "supported_scopes": set(),
                    "potential_evidence": set(),
                },
            )
            item["supporting_operations"].add(operation)
            item["supported_scopes"].update(scopes)
            item["potential_evidence"].update(evidence)
    return [
        {
            "effect_kind": effect_kind,
            "supporting_operations": sorted(item["supporting_operations"]),
            "supported_scopes": sorted(item["supported_scopes"]),
            "potential_evidence": sorted(item["potential_evidence"]),
        }
        for effect_kind, item in sorted(effects.items())
    ]


def frozen_edit_intent_from_context(
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """Read the explicit cross-turn EditIntent handoff when the runtime supplies it."""

    value = context.get("workflow_frozen_edit_intent")
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return None
    return copy.deepcopy(value)


def candidate_capability_projection(
    operations: list[dict[str, Any]],
    *,
    perception_operation_ids: set[str],
    frozen_edit_intent: dict[str, Any] | None,
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
) -> tuple[
    CandidateProjectionSource,
    list[str],
    list[str],
    list[dict[str, Any]],
]:
    """Select complete post-Perception cards without semantic keyword routing."""

    by_operation = {
        str(item.get("operation")): item
        for item in operations
        if isinstance(item, dict) and item.get("operation")
    }
    edit_operations = [
        operation for operation in by_operation if operation not in perception_operation_ids
    ]
    if prompt_profile in {"robust_contract_v4", "robust_contract_v5"}:
        source = "complete_edit_catalog"
        requested = edit_operations
    elif (
        prompt_profile in {"evidence_complete_v3", "robust_contract_v6"}
        and frozen_edit_intent is not None
    ):
        source = "typed_effect_shortlist"
        typed_effects = set(_string_list(frozen_edit_intent.get("effect_kinds")))
        provisional = _string_list(frozen_edit_intent.get("candidate_operations"))
        effect_matches = [
            operation
            for operation in edit_operations
            if typed_effects & set(_operation_semantic_effects(by_operation[operation]))
        ]
        requested = list(dict.fromkeys([*provisional, *effect_matches]))
    elif prompt_profile in {
        "objective_coverage_v2",
        "evidence_complete_v3",
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    }:
        source = "complete_edit_catalog"
        requested = edit_operations
    elif frozen_edit_intent is not None:
        source: CandidateProjectionSource = "frozen_edit_intent"
        requested = _string_list(frozen_edit_intent.get("candidate_operations"))
    else:
        source = "conservative_all_edit_operations"
        requested = edit_operations

    candidates: list[str] = []
    unavailable: list[str] = []
    cards: list[dict[str, Any]] = []
    for operation in requested:
        operation_spec = by_operation.get(operation)
        if operation_spec is None or operation in perception_operation_ids:
            unavailable.append(operation)
            continue
        candidates.append(operation)
        card = operation_spec.get("operation_capability")
        if isinstance(card, dict):
            if prompt_profile in {
                "robust_contract_v4",
                "robust_contract_v5",
                "robust_contract_v6",
            }:
                cards.append(_compact_v4_executable_capability_card(card))
            elif prompt_profile in {"objective_coverage_v2", "evidence_complete_v3"}:
                cards.append(_compact_executable_capability_card(card))
            else:
                cards.append(copy.deepcopy(card))
            continue
        cards.append(
            {
                "operation": operation,
                "capability_card_status": "missing",
                "planner_selectable": [],
                "planning_rule": (
                    "Do not invent parameters. Return cannot_plan when this "
                    "operation needs an undeclared Planner-selected value."
                ),
            }
        )
    return source, candidates, unavailable, cards


def build_planner_prompt_package(
    *,
    phase: PlannerPromptPhase,
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
    system_prompt: str,
    current_payload: dict[str, Any],
    operation_index_count: int,
    full_capability_card_count: int,
    candidate_projection_source: CandidateProjectionSource,
    candidate_operations: list[str],
    unavailable_candidate_operations: list[str] | None = None,
    candidate_operations_without_full_contract: list[str] | None = None,
) -> PlannerPromptPackage:
    """Render and measure the exact prompt without truncating any section."""

    frozen_payload = copy.deepcopy(current_payload)
    current_prompt = json.dumps(
        frozen_payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    section_chars = {
        key: len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for key, value in frozen_payload.items()
    }
    total_chars = len(system_prompt) + len(current_prompt)
    fingerprint_material = json.dumps(
        {
            "phase": phase,
            "system_prompt": system_prompt,
            "current_payload": frozen_payload,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    report = PromptBudgetReport(
        phase=phase,
        prompt_profile=prompt_profile,
        system_prompt_chars=len(system_prompt),
        current_prompt_chars=len(current_prompt),
        total_prompt_chars=total_chars,
        estimated_input_tokens=math.ceil(total_chars / 4),
        section_chars=section_chars,
        operation_index_count=operation_index_count,
        full_capability_card_count=full_capability_card_count,
        candidate_projection_source=candidate_projection_source,
        candidate_operations=list(candidate_operations),
        unavailable_candidate_operations=list(unavailable_candidate_operations or []),
        candidate_operations_without_full_contract=list(
            candidate_operations_without_full_contract or []
        ),
        omitted_redundant_sections=list(_OMITTED_REDUNDANT_SECTIONS),
        critical_contracts_complete=not (
            unavailable_candidate_operations or candidate_operations_without_full_contract
        ),
    )
    return PlannerPromptPackage(
        phase=phase,
        system_prompt=system_prompt,
        current_payload=frozen_payload,
        current_prompt=current_prompt,
        package_fingerprint=hashlib.sha256(fingerprint_material.encode("utf-8")).hexdigest(),
        budget_report=report,
    )


def _compact_executable_capability_card(card: dict[str, Any]) -> dict[str, Any]:
    """Keep semantic trade-offs and exact authoring contracts without audit prose."""

    return {
        "schema_version": card.get("schema_version"),
        "operation": card.get("operation"),
        "semantic_effects": copy.deepcopy(card.get("semantic_effects") or []),
        "supported_scopes": copy.deepcopy(card.get("supported_scopes") or []),
        "requirements": copy.deepcopy(card.get("requirements") or {}),
        "preservation_levels": _dimension_level_index(
            card.get("preservation"),
            name_field="dimension",
        ),
        "quality_levels": _dimension_level_index(
            card.get("quality_characteristics"),
            name_field="dimension",
        ),
        "artifact_risk_levels": _dimension_level_index(
            card.get("artifact_risks"),
            name_field="risk",
        ),
        "composition": copy.deepcopy(card.get("composition") or {}),
        "parameter_contract": copy.deepcopy(card.get("parameter_contract") or {}),
        "limitations": copy.deepcopy(card.get("limitations") or []),
    }


def _compact_v4_executable_capability_card(card: dict[str, Any]) -> dict[str, Any]:
    """Keep route boundaries, trade-offs, and the exact legal authoring surface."""

    return {
        "operation": card.get("operation"),
        "semantic_effects": copy.deepcopy(card.get("semantic_effects") or []),
        "supported_scopes": copy.deepcopy(card.get("supported_scopes") or []),
        "requirements": copy.deepcopy(card.get("requirements") or {}),
        "preservation_levels": _dimension_level_index(
            card.get("preservation"),
            name_field="dimension",
        ),
        "quality_levels": _dimension_level_index(
            card.get("quality_characteristics"),
            name_field="dimension",
        ),
        "artifact_risk_levels": _dimension_level_index(
            card.get("artifact_risks"),
            name_field="risk",
        ),
        "parameter_contract": copy.deepcopy(card.get("parameter_contract") or {}),
        "limitations": copy.deepcopy(card.get("limitations") or []),
    }


def _requirements_index(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {"inputs": [], "evidence": []}
    return {
        "inputs": _string_list(value.get("inputs")),
        "evidence": _string_list(value.get("evidence")),
    }


def _dimension_level_index(
    value: Any,
    *,
    name_field: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    projected: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or not item.get(name_field):
            continue
        projected.append(
            {
                name_field: str(item[name_field]),
                "level": str(item.get("level") or "unknown"),
            }
        )
    return projected


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item)))


def _operation_semantic_effects(operation_spec: dict[str, Any]) -> list[str]:
    card = operation_spec.get("operation_capability")
    if not isinstance(card, dict):
        return []
    return _string_list(card.get("semantic_effects"))


def list_planner_prompt_profile_names() -> tuple[PlannerPromptProfileName, ...]:
    """Return reproducible prompt profiles accepted by the shared runtime."""

    return (
        "catalog_strict_v1",
        "objective_coverage_v2",
        "evidence_complete_v3",
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    )


def memo_qualified_planner_system_prompt() -> str:
    """Stable Planner role after a natural-language semantic Reasoner memo."""

    return _MEMO_HANDOFF_ROLE + _COMMON_EXECUTION_RULES


def integrated_planner_system_prompt() -> str:
    """Same executable contracts, without an external Reasoner authority."""
    return (
        "You own interpretation and executable planning in one response. "
        "The original user request is authoritative. Any reference below to a "
        "Reasoner proposal or memo means your own current interpretation, not "
        "an external answer. Select an eligible Session Graph base yourself; "
        "use complete_from_session_state only when its exact proof authorizes "
        "the whole request. Do not mistake source material for a completed edit. "
        "Preserve every explicit transformation and preservation requirement.\n\n"
    ) + _COMMON_EXECUTION_RULES


_MEMO_HANDOFF_ROLE = """You are the execution Planner for X-Editor.

The Reasoner has described the user's intended audible result. When `requested_change_ledger_status` is `complete`, its typed requested changes and selected Session Graph base are authoritative and must be copied exactly. When the status is `provider_omitted`, the provider deliberately returned a prose memo: use that memo together with the original request to author the structured PlanIR, including selection of one eligible Session Graph base when a graph is supplied. This is the Planner's normal responsibility in prose mode, not a degraded handoff. Return `complete_from_session_state` only when the supplied `session_completion_proofs` authorize the exact selected node/audio pair. Merely finding requested source material in an unedited input does not complete an extraction, selection, or transformation request.

For a complete typed ledger, Runtime mechanically derives `execution_request` from `requested_changes` not covered by admitted Graph completion proofs. In prose mode it preserves the original request verbatim as `execution_request` so Planner structure cannot silently narrow the user's objective.

Use the original request to check handoff integrity and preserve exact user-supplied details. If a complete typed ledger or the prose memo contradicts an explicit transformation or preservation clause, return `cannot_plan` with a semantic conflict; do not reinterpret or silently repair the requested result.

"""

_COMMON_EXECUTION_RULES = """An explicit request to transform meaning, for example by translating or paraphrasing, authorizes target wording to be composed from adequately observed source content. Compile a supported Reasoner proposal, or compose the remaining target wording within that authorized transformation. Missing user-supplied target text or a separate translation service is not itself missing evidence. Keep source observations immutable: generated target wording is a user-requested result (`user_request` provenance), never a heard source fact or a deterministic evidence derivation. Preserve exact user-specified wording, the intended meaning, local scope and unaffected content. Request Perception for uncertain source content, and retain genuine ambiguity about the intended meaning; ordinary equivalent wording choices do not create a new user-intent ambiguity.

Your responsibilities are operational:
- copy the Reasoner's exact base node/audio pair when present; otherwise select one eligible Graph node/audio pair from the supplied Session Graph;
- distinguish selecting the current state from restoring an eligible historical state;
- request only evidence needed to execute the remaining work, using explicit independent observation jobs over the bound source/reference audio namespace; the Perception Gateway owns concrete route selection;
- author a typed `perception_targets` entry for each target, with exactly two keys: `{"kind": "...", "value": "..."}`. `quoted_text`, `semantic_property`, `speaker_hint`, and `audio_property` are values of `kind`, never alternate field names. For every `perception.semantic_span_review` job, supplement it with an explicit `evidence_questions` item stating exactly what the model should decide. Use `quoted_text` only to assert exact source wording; use the property kinds for meaning- or property-based source locators. A desired output value is not an observed source target. Never let Perception replace the Planner-authored target with wording it reports hearing;
- compare only supplied operations and choose by semantic completeness, preservation, audible quality and artifact risk, reliability, then cost and latency;
- Expert support for an event does not establish absence of other sources. If ASR reports speech but a caption denies it, resolve the conflict with same-version speech timing before destructive interval processing, regardless of event confidence. When only a non-overlapping portion can be removed safely, retain the overlapping remainder as an unmet requirement and consider offered component-selective tools; do not silently shrink the target and claim complete removal;
- An event span identifies when a sound occurs, not which simultaneous component a tool changes. Whole-interval mute/remove/gain also changes protected speech or background inside that span. For source-selective requests, check same-version coactivity evidence before choosing such operations. Reuse sufficient supplied speech/word timing; when timing or component identity is missing, request only that missing observation. If components overlap, compare available source-selective tools/compositions and their leakage risk. Do not automatically choose a generic speech stem for a particular vocal behavior, or call a separation residual clean background without evidence. A separator's named target/residual roles determine which output is retained; do not choose by list position. If no offered capability meets the constraints, report the limitation without silently deleting protected material;
- A model-only caption marked needs_review cannot by itself establish that a target interval contains no protected speech. Before destructive interval processing, request independent speech timing (speech_context with word_timestamps and word precision) or targeted listening for coactivity, unless sufficient evidence is already supplied. A report's 'before speech' wording is not a substitute for that missing evidence;
- A limitation of one operation is not a limitation of the whole catalog. For a reference-derived change, first separate what must be transferred (identity, wording, emotion over time, rate, or acoustic context). Obtain only the observations needed for that property. Consider supported local controls and explicit composition when a single operation cannot express the variation. Retain the requested variation and preservation constraints; a single global label is not an equivalent substitute for a changing trajectory. If available controls can only approximate that trajectory, state that limitation rather than claim exact transfer. A tool accepting only one source and text cannot directly condition on a second audio merely because the request mentions it;
- use only exact legal argument paths and runtime value domains from the selected operation's step_argument_contract to fill the matching PlanIR containers: each entry under inputs belongs under step.inputs, and each entry under params belongs under step.params, even when its value comes from an input observation. Copy required fields with their legal value types; an observation's field name never changes the operation argument's declared owner. Relative paths within a container are addresses, not literal dotted JSON keys. Do not emit a flattened arguments map;
- write only the literal operation argument value at every PlanIR `inputs` or `params` leaf. Never wrap a value in an object containing `value`, `source`, `evidence_id`, or `evidence_path`; put provenance only in the matching execution_semantics binding's `argument_sources` list;
- a complete typed from_step/from_output reference, or a non-empty argument made entirely of such references, has its provenance owned by the validated PlanIR graph. Do not repeat that argument in argument_sources or label it derived_from_evidence; the framework records its deterministic dependency source. The producer must be earlier, present in depends_on, and expose the named output. Mixed arguments and actual observation fields retain their ordinary source and evidence requirements; no missing or invalid dependency is repaired;
- map qualitative user requirements to the closest legal value when the supplied contract makes that choice operationally safe; missing an exact number is not blocking unless it changes meaning, safety, or legality;
- bind observed values to admitted evidence paths;
- emit at most one execution_semantics binding for each plan_step_id. That one binding describes the step target and groups every argument source for the step; do not create one binding per argument path;
- do not author a separate semantic-binding objective. The framework projects that
  non-authoritative audit label from the matching PlanIR step description, so it
  cannot duplicate or override the Reasoner memo or executable arguments;
- when a plan uses directly supplied audio assets in roles that are not mechanically obvious, emit exact per-argument audio usage semantics; omit that optional ledger for ordinary single-audio flow. Use `edit_input` for the asset being edited, `assembly_member` for an external clip whose samples enter the output, and `reference_input` for an asset used only to condition processing. Use `speaker_grounded_reference` only when a particular speaker, profile, or speaker segment was selected from reviewed speaker evidence. The framework attaches memo fingerprints, evidence requirements, and admitted evidence provenance. A from_step output is already authorized by its exact dependency object and must not receive a second external-asset binding; an asset's historical source/reference label does not permanently decide its use in this turn;
- audio_usage_evidence_inventory states which bound assets currently have admitted speaker facts. A whole recording used as a voice reference is reference_input; the fact that it supplies a voice does not itself mean a particular speaker was selected from evidence. If the request selects a particular speaker within a recording and that asset lacks the required facts, request the missing observation when allowed or report missing_evidence. Never downgrade a speaker selection to whole-recording conditioning, invent facts, or declare speaker_grounded_reference without its prerequisite;
- produce one cumulative executable audio path with explicit dependencies for ordinary edits. When the user requests multiple independent final audio files, declare `plan.output_artifacts` for every terminal step and mark exactly one as primary for Session Graph continuation; do not add a fake assembly step merely to hide multiple legitimate deliverables.

Reviewed observations are immutable, but an explicit contextual re-recognition can publish a new transcript_revision linked to the prior observation. Copy each complete admitted scalar value exactly and declare its source. Copy the current admitted source_text and its matching word_timestamps; never mix a revised transcript with superseded word IDs. When the bound audio has exactly one source_text_resolutions row with status=corroborated, its resolved_source_text is a complete admitted multi-source transcript and should be copied for transcript-conditioned tool arguments with that exact evidence path. Do not perform an additional text repair yourself; multiple, absent, or conflicted resolutions remain unresolved evidence. Give every Perception job its own primary audio, optional supporting audios, targets, precision, requirements, and expected fields; use `auto` for the route unless you are preserving an old compatibility hint. Never use one shared route to stand in for heterogeneous observations. Request each audio observation independently and never treat an audio merely listed in a request as already observed. For cumulative steps, use the exact {"from_step":"step_id","from_output":"output_audio_path"} input plus depends_on; do not add that derived output to audio_usage_bindings because it is not an external audio asset. Never invent a node, operation, parameter, enum value, evidence path, audio reference, or extra objective. If required evidence conflicts, request an observation capable of resolving it when available; an unresolved observation is not itself proof that the requested edit is unsupported.

An observation job is an invocation over its declared assets, not a container for
every asset mentioned by a comparison. Follow observation_input_contracts:
single_audio means one primary_audio_ref and empty supporting_audio_refs. To
compare independent measurements across assets or versions, author separate jobs
with the same needed measurement for each asset, then reason over their separately
bound results. Use supporting_audio_refs only for a capability whose paired_audio
or multi_audio contract actually observes those inputs together. Keep the editing
base unchanged. Never discard a reference just to make the request shape valid.
Request only facts the catalog can measure; model listening descriptions must not
be presented as numerical physical loudness, peak, or gain measurements.
First inspect the supplied audio_asset_facts for each exact audio ref/content
revision. sample_peak and rms_amplitude are measured linear amplitudes; RMS includes
all frames/channels and silence, and is not LUFS. Use an appropriate available
metric consistently when the request leaves that operational choice open, explain
the choice, and derive gain from the measured values rather than ask a listening
model to invent them. A whole-file metric cannot stand in for a selected region,
isolated speaker, or an explicitly requested perceptual loudness measurement.

An unspecified operational degree is not an acoustic fact to recognize. When the
user authorizes a qualitative change and the selected operation offers a suitable
legal range, choose a justified value within that range and state the choice in
the step rationale. Do not require Perception to choose that value or invent an
observation supporting it. User-specified numbers, direction, scope and preservation
constraints remain binding. A genuine uncertainty about which target or outcome
the user means still requires clarification; choosing an arbitrary occurrence is
not parameter autonomy.

For tagged edits, start from the complete admitted transcript of the actual input
artifact as a template, not an earlier whole recording. Emotion/prosody pipelines
also consume exact text: a request to preserve words does not waive this input
contract. Advice that a semantic target is resolved does not establish execution
readiness. Resolve conflicting transcripts or missing word coverage for the chosen
region with targeted recognition/alignment; never combine disagreeing reports.
Wrap only the selected local source words in the operation's tags, leaving all
other words and punctuation outside the tags unchanged. Full-sentence context does
not authorize a full-sentence replacement. Render the full target from those tags
and unchanged context; do not write a second independently paraphrased target.
Source-copy punctuation tolerance is not permission to alter words or edit scope.

After extraction, inherit complete words only through an accepted, same-version
single-span unit-time mapping. Overlaps outside the retained interval do not
invalidate its words. Multi-span joins, crossfades, cut words, or ambiguous local
alignment need an explicit observation of the actual crop before text-conditioned
editing. Reuse a valid crop-local observation and author matching local labels;
do not discard it merely because its parent mapping is unavailable. If the current
plan cannot obtain that evidence through a supported observation/continuation,
report the missing prerequisite rather than inventing future observations or
silently bypassing the binding check. Existing budgets and tool availability apply.
If the full source transcript and selected words are already grounded, local tags
on that complete source can avoid an unnecessary crop and its new evidence needs.
This does not authorize whole-file tagging or a change of target scope. When a crop
is necessary, plan its observation explicitly if inheritance cannot be proven;
future observations are not current facts and cannot justify guessed labels.

For a needed recheck of lexical content, recognition_context on a Perception job
provides unverified vocabulary or domain context to fresh ASR. State the disputed
source wording/question, including alternatives if useful; do not supply the edit's
replacement text as the transcript to recognize. Request source_text and request
word_timestamps when the chosen edit needs local timing. This observation can
resolve multiple words and repeated occurrences in one asset. Request separate
jobs for separate assets. Direct listening can establish a sound, while contextual
ASR and alignment supply executable wording and its corresponding boundaries.
Do not declare a word absent from an ASR omission or a low-confidence candidate
nonmatch; choose a new observation that can answer the remaining question.

Reviewed target candidates and final semantic selection are separate. When the Reasoner supplied a complete typed target ledger, use only framework-materialized `perception_report.selected_spans`, audio-scoped `selected_speaker_refs`, compatibility-only `selected_speaker_ids`, and `target_bindings` produced from its resolved candidate_refs. In the normal prose-memo mode, the Planner owns the final semantic selection among already admitted, audio-scoped candidates: resolve explicit cardinality and relation language such as all matches, an ordinal occurrence, a stated subset, a range, or exclusions against those candidates. Multiple candidates alone are not ambiguity when the request itself states how many or which occurrences to select. Copy selected candidate values and boundaries exactly into legal PlanIR arguments and cite their exact evidence paths; never invent a candidate, alter a measured boundary, or infer that an incomplete candidate search is exhaustive. A bare speaker ID never identifies an audio in a multi-audio task. Return `cannot_plan` only when the user's selector remains genuinely ambiguous or the admitted evidence cannot support the requested selection.

For `cannot_plan`, preserve the exact missing information and also emit concise typed `cannot_plan_reasons`. Choose the reason code from the supplied schema; do not infer success, hide a failed service, or replace a legal plan with a refusal merely because one candidate tool is inconvenient.

Return only the requested JSON contract.""" + "\n\n" + COMPOSITION_RULES


def memo_qualified_planner_current_call_prompt(
    planner_input: MemoQualifiedPlannerInput,
    *,
    output_schema: dict[str, Any] | None = None,
    evidence_index_format: EvidenceIndexFormat = "expanded",
    argument_contract_format: ArgumentContractFormat = "expanded",
    retry_guidance_mode: RetryGuidanceMode = "diagnostic",
) -> str:
    """Current Planner assignment with memo, evidence and exact legal candidates."""

    if evidence_index_format not in {"expanded", "prefix-plus-suffix/v1"}:
        raise ValueError("Unsupported evidence index format")
    if argument_contract_format not in {"expanded", "argument-table/v1"}:
        raise ValueError("Unsupported argument contract format")
    if retry_guidance_mode not in {"diagnostic", "actionable/v1"}:
        raise ValueError("Unsupported retry guidance mode")
    prose_handoff = (
        planner_input.semantic_memo.requested_change_ledger_status
        == "provider_omitted"
    )
    if planner_input.execution_request is None:
        stage_task = (
            "The Reasoner reports that the selected Session Graph base already satisfies "
            "all requested changes. Return complete_from_session_state, copy the exact "
            "node/audio pair and changes_satisfied_by_base, and use select_existing only "
            "for the current node or restore_history for a historical node. Do not emit a "
            "plan, tool selection, missing information, candidate operations, or Perception job."
        )
    elif planner_input.phase == "evidence_planning":
        graph_instruction = (
            "First inspect session_completion_proofs. Only when one proof authorizes a state "
            "that already satisfies the whole request may you return complete_from_session_state. "
            "Otherwise select the edit base and request "
            "the smallest genuinely useful Perception gateway batch. "
            if prose_handoff and planner_input.session_state is not None
            else (
                "Select one eligible Session Graph edit base, then request the smallest "
                "genuinely useful Perception gateway batch. "
                if prose_handoff
                else "Copy the Reasoner's base node/audio pair and request the smallest genuinely useful Perception gateway batch. "
            )
        )
        stage_task = (
            graph_instruction
            + "First decide whether any additional observation is genuinely required. If one "
            "of direct_execution_operation_contracts fully implements the residual request, "
            "its required inputs and evidence are already supplied, return execute "
            "directly with a legal PlanIR; do not manufacture a diagnostic Perception request. "
            "Otherwise request only the missing evidence. Use the Reasoner's object, scope, "
            "residual request, and the semantic memo "
            "with the original request as the requested-result authority. Do not reinterpret "
            "the requested observable domain. "
            "An unresolved source fact is missing evidence, not proof that the edit is unsupported. "
            "Optionally retain a small shortlist of plausible edit operations when the current "
            "evidence supports one; inspect their requirements.evidence prerequisites and do not "
            "choose by lexical name similarity; request the union of prerequisites needed by "
            "surviving candidates. Describe observation "
            "needs rather than providers. A selected span is not a substitute for a distinct "
            "typed prerequisite. Describe observation needs, not routes: emit independent "
            "perception_jobs, set perception_operation to auto or omit it, and let the Perception "
            "Gateway choose specialist operations from "
            "typed targets, expected evidence fields, audio refs, and temporal precision. When "
            "different audios need different evidence, emit separate jobs. Use quoted_text only for "
            "exact source wording and a property target for meaning-based localization. Keep desired "
            "output values out of source-observation targets. Exact argument schemas exposed under "
            "direct_execution_operation_contracts are available only when their declared evidence "
            "requirements are already present in this call; all other exact edit argument schemas "
            "remain deferred until evidence review."
        )
    else:
        stage_task = (
            "Use the frozen Session Graph base selected before Perception, choose legal candidates "
            "and exact arguments. If a required source fact remains unknown or contradictory, "
            "request_perception is still available: request only a useful additional observation "
            "on the same base and residual request, without assuming the missing fact or changing "
            "the user's goal. The workflow enforces its existing finite observation budget. "
            "When evidence is sufficient, emit a complete PlanIR for execution_request only. "
            "execution_semantics is needed only for actual evidence or multi-audio usage bindings; "
            "tool_selection is an optional audit explanation. For a complete typed Reasoner ledger, "
            "treat framework-materialized selected_spans, audio-scoped selected_speaker_refs, and "
            "target_bindings as exact semantic bindings. In prose-memo mode, select only among "
            "admitted audio-scoped candidates according to the user's explicit cardinality or "
            "relation, copy their values exactly, and cite their evidence paths. "
            "plan.request is your descriptive summary of the proposed edit; it may make "
            "an indirect instruction concrete using the supplied evidence. The compiler binds "
            "the execution plan to the caller-owned execution_request and retains your description "
            "for audit. Your description cannot authorize a different goal, relax preservation "
            "requirements, or replace the required argument and evidence bindings."
        )
    payload = {
        "current_call_role": "memo_qualified_execution_planning",
        "stage_task": stage_task,
        "observation_input_contracts": observation_input_contracts(
            planner_input.perception_capability_catalog
        ),
        "planner_output_contract": _memo_qualified_planner_output_contract(
            phase=planner_input.phase,
            state_only=planner_input.execution_request is None,
            evidence_index_format=evidence_index_format,
        ),
        "semantic_handling": {
            "authority": (
                "semantic_memo.content plus original request"
                if prose_handoff
                else (
                    "semantic_memo.requested_changes selected by "
                    "changes_satisfied_by_base, with semantic_memo.content for resolved details"
                )
            ),
            "execution_request_projection": (
                "verbatim_original_request"
                if prose_handoff
                else "requested_changes_minus_graph_completions"
            ),
            "original_request_is_visible": True,
            "original_request_use": "recover_consistent_exact_details_only",
            "silent_reinterpretation_allowed": False,
            "handoff_integrity": (
                "execution_request must retain the resolved observable domain, edit scope, "
                "requested transformations, and explicit preservation clauses"
            ),
            "evidence_conflict_action": "cannot_plan_with_semantic_conflict",
            "memo_normalization_applied": False,
            "source_text_resolution_policy": (
                "Use one corroborated source_text_resolutions.resolved_source_text for "
                "transcript-conditioned arguments; otherwise do not alter canonical source_text."
            ),
        },
        "planner_input": _memo_qualified_planner_prompt_projection(
            planner_input, evidence_index_format=evidence_index_format,
            argument_contract_format=argument_contract_format,
            retry_guidance_mode=retry_guidance_mode,
        ),
        "final_instruction": (
            "Return only the requested Planner decision JSON. Do not author framework "
            "provenance fields such as semantic_memo_fingerprint, evidence_requirement, "
            "or audio-usage evidence citations."
        ),
    }
    if planner_input.semantic_memo.semantic_drafts:
        payload["semantic_draft_policy"] = (
            "Reasoner drafts are tool-neutral proposals, not observations or service syntax. "
            "Compile supported content into the selected tool's arguments. Preserve declared "
            "local scope and unedited content; full source text can be synthesis context "
            "without being the replacement region. Correct unsupported drafts against current "
            "evidence and the user request; explain material scope changes in the plan rationale."
        )
    if planner_input.perception_capability_feedback is not None:
        payload["capability_feedback"] = (
            planner_input.perception_capability_feedback.model_dump(mode="json")
        )
    if planner_input.semantic_memo.observation_ledger:
        from x_editor.reasoner.evidence_obligations import pending_observations
        pending = pending_observations(planner_input.semantic_memo)
        payload["observation_obligations"] = {
            "pending": [item.model_dump(mode="json") for item in pending],
            "policy": (
                "For each pending need create a targeted Perception job, citing its need_id "
                "in observation_need_ids. Cover its audio scope and the actual missing "
                "property in evidence_questions; do not merely repeat ASR for an acoustic "
                "property. For sounds potentially omitted by ASR, ask for open direct "
                "audio observations with location and overlap, rather than restricting "
                "the search to existing transcript candidates. Select evidence fields "
                "that the offered capability actually returns. For disputed wording, "
                "ask direct audio review with candidate readings as uncertain hints; "
                "speech_context with recognition_context requests fresh contextual ASR "
                "and fresh alignment when the catalog declares that support. Hints "
                "remain unverified until recognition returns. Legitimate simultaneous speakers "
                "need not be made non-overlapping; only unresolved attribution needed "
                "by the selected edit requires further review. Distinguish a broad search "
                "window containing an event from the observed event boundaries. Preserve "
                "the target interval when building the operation graph: processing a "
                "whole asset does not select a local target. A generic foreground or "
                "speech stem does not distinguish particular speakers or vocal behaviors. "
                "Sequential, non-intersecting voices do not require separating them "
                "from each other. Do not knowingly select an operation that violates "
                "the requested preservation constraints merely because it is executable; "
                "obtain necessary evidence or state the actual capability limitation. "
                "Observed intervals belong to the observed asset version. Equal output "
                "duration alone does not authorize reusing those coordinates after an "
                "edit: the operation must provide a valid time mapping. If it does not, "
                "select the needed region on the observed asset before processing when "
                "the tool's context requirements permit, or observe the produced asset "
                "before using new intervals. Mapped coordinates do not automatically "
                "preserve old claims about words or speakers. "
                "An unresolved observation "
                "calls for Perception, not a negative "
                "conclusion or an expanded edit scope. If no offered capability can supply "
                "the observation, keep the capability failure explicit. Findings may be "
                "used only within the property and scope actually observed."
            ),
        }
        if pending and planner_input.perception_capability_catalog:
            payload["planner_output_contract"]["allowed_decisions"] = ["request_perception", "cannot_plan"]
            payload["observation_obligations"]["unavailable_observation"] = (
                "If no offered capability can observe the required property, return "
                "cannot_plan with unsupported_capability or missing_perception_route "
                "and identify the unresolved need. This is unknown, not not_detected. "
                "Ordinary missing_evidence is a reason to request observation, not stop."
            )
    # The provider receives a separate stable wire envelope. Keep this argument
    # for API compatibility without duplicating the large dynamic Runtime schema
    # or its evidence-path enum in the textual prompt.
    _ = output_schema
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def execute_decision_outline(plan_shape: str) -> str:
    """Keep evidence and asset-use declarations identical across plan formats."""
    return (
        "{decision, selected_base_state_node_id, selected_base_audio_ref, "
        "state_transition_kind, plan:" + plan_shape + ", "
        "execution_semantics?:{bindings:[{plan_step_id,target,operation,"
        "argument_sources,unresolved_requirements}],audio_usage_bindings:"
        "[{plan_step_id,argument_path,audio_ref,usage}]}, "
        "tool_selection?:{steps:[{plan_step_id,selected_operation,candidates,"
        "selection_summary}]}, missing_information:[], decision_summary}"
    )


def _memo_qualified_planner_output_contract(
    *,
    phase: Literal["evidence_planning", "execution_planning"],
    state_only: bool,
    evidence_index_format: EvidenceIndexFormat = "expanded",
) -> dict[str, Any]:
    """Compact canonical wire shapes shown to every Planner provider call."""

    decisions = (
        ["complete_from_session_state"]
        if state_only
        else ["request_perception", "execute", "complete_from_session_state", "cannot_plan"]
    )
    return {
        "contract_version": "x-editor.memo-qualified-planner-wire/v1",
        "allowed_decisions": decisions,
        "canonical_shapes": {
            "request_perception": (
                "{decision, selected_base_state_node_id, selected_base_audio_ref, "
                "perception_jobs:[{job_id, scope, perception_operation, primary_audio_ref, "
                "supporting_audio_refs, requirements, expected_evidence_fields, "
                "perception_targets:[{kind,value}], temporal_precision, target_hints, "
                "evidence_questions, observation_need_ids?, recognition_context?, required}], candidate_operations, "
                "missing_information, decision_summary}"
            ),
            "execute": execute_decision_outline(
                "{request,input_audio,steps:[{id,operation,"
                "description,inputs,params,expected_output,depends_on,required}],"
                "output_artifacts,global_params,metadata,missing_inputs}"
            ),
            "complete_from_session_state": (
                "{decision, selected_base_state_node_id, selected_base_audio_ref, "
                "state_completion_kind, changes_satisfied_by_base, decision_summary}"
            ),
            "cannot_plan": (
                "{decision, missing_information, semantic_conflict?, "
                "cannot_plan_reasons:[{code,message,target_ids,operation_ids}], "
                "decision_summary}"
            ),
        },
        "critical_rules": [
            "Return only fields belonging to the selected decision shape.",
            "execute requires at least one real edit step. Selecting an authorized existing state uses complete_from_session_state, never an empty execute plan.",
            "For complete_from_session_state, copy the exact node/audio pair and state_completion_kind from session_completion_choices only after checking the whole request; author changes_satisfied_by_base from the request and memo, not from proof availability alone.",
            "Perception target keys are exactly kind and value.",
            "When a lexical observation needs fresh recognition of disputed wording, author recognition_context with unverified source vocabulary/domain hints. target_hints and evidence_questions alone do not request fresh contextual ASR; they describe the observation objective. Omit recognition_context for ordinary alignment of an already adequate transcript.",
            "Contextual recognition returns source_text and word_timestamps, not semantic classifications. For that job omit perception_targets or use quoted_text for literal candidate words. Keep uncertain context in recognition_context; Reasoner interprets the returned observations. Request other kinds of evidence in separate jobs with their own declared capabilities.",
            "Plan step keys are exactly id, operation, description, inputs, params, expected_output, depends_on, and required.",
            "Place tool arguments inside nested inputs and params objects; never emit step_id or a flattened arguments object.",
            (
                "For an evidence-backed argument source, evidence_id is one admitted evidence ID and evidence_path is the exact corresponding canonical path from evidence_path_sources or evidence_source_index; expand prefix-plus-suffix paths verbatim and never put an evidence ID in evidence_path."
                if evidence_index_format == "prefix-plus-suffix/v1"
                else "For an evidence-backed argument source, evidence_id is one admitted evidence ID and evidence_path is the exact corresponding key from evidence_path_sources; never put an evidence ID in evidence_path."
            ),
            "A runtime-supplied audio ref is deterministic_context or user_request, not perception_evidence merely because Perception observed that audio. user_request and deterministic_context sources must omit evidence_id and evidence_path.",
            "PlanIR input and parameter leaves contain literal argument values, never provenance wrappers.",
            "execution_semantics.bindings contains at most one row per plan_step_id; group that step's argument_sources in the row.",
            "tool_selection is optional audit material. Omit it unless candidate operations were actually compared. If present, every step needs a non-empty candidates list containing selected_operation.",
            "Omit plan.output_artifacts for an ordinary single terminal audio output. If output_artifacts is needed, declare every terminal step and set exactly one artifact primary=true.",
            "cannot_plan reason code is exactly one of missing_evidence, missing_perception_route, unsupported_capability, semantic_ambiguity, conflicting_evidence, invalid_parameter_domain, unsupported_output_topology, or other.",
            "Question marks mark optional fields in this outline and are not part of field names.",
        ],
    }


def _memo_qualified_planner_prompt_projection(
    planner_input: MemoQualifiedPlannerInput,
    *,
    evidence_index_format: EvidenceIndexFormat = "expanded",
    argument_contract_format: ArgumentContractFormat = "expanded",
    retry_guidance_mode: RetryGuidanceMode = "diagnostic",
) -> dict[str, Any]:
    """Keep exact legal domains while removing verbose capability-card prose."""

    payload = planner_input.model_dump(mode="json")
    payload["plan_request_authority"] = {
        "binding_source": "execution_request supplied by the caller",
        "provider_request_role": "descriptive summary, retained for audit",
        "compiler_owned": "the admitted plan.request is bound to execution_request",
        "scope": "does not change or validate the meaning of authored steps and arguments",
    }
    from x_editor.runtime.audio_reference_inventory import audio_reference_inventory
    payload["audio_reference_inventory"] = audio_reference_inventory(
        planner_input.source_audio_refs, planner_input.reference_audio_refs)
    payload.pop("evidence_path_audio_refs", None)
    if evidence_index_format == "prefix-plus-suffix/v1":
        source_index = payload.get("evidence_path_sources", {})
        factored_index = factor_evidence_index(source_index)
        # Lossless representation is an explicit experiment: it can still change
        # model decisions. Small maps retain the shorter expanded representation.
        if len(json.dumps(factored_index)) < len(json.dumps(source_index)):
            payload.pop("evidence_path_sources")
            payload["evidence_source_index"] = factored_index
    payload.pop("future_attempt_guidance", None)
    guidance = (
        planner_input.future_attempt_guidance.prompt_projection(module="planner")
        if planner_input.future_attempt_guidance is not None
        else None
    )
    if guidance is not None:
        payload["future_attempt_guidance_policy"] = {
            "authority": "diagnostic_questions_for_this_new_attempt_only",
            "not_evidence": True,
            "not_an_answer": True,
            "not_a_replacement_plan": True,
            "required_behavior": (
                "Build a fresh PlanIR from the current Reasoner memo, current reviewed "
                "evidence, and current capability contracts. Use the questions only to "
                "inspect likely weak points and preserve the listed invariants. Never cite "
                "diagnostic_evidence_refs as current evidence and never bypass admission or "
                "Verifier requirements."
            ),
        }
        payload["future_attempt_guidance"] = guidance
        if retry_guidance_mode == "actionable/v1":
            payload["future_attempt_guidance_policy"]["required_behavior"] = ACTIONABLE_RETRY_POLICY
    semantic_memo = payload.get("semantic_memo")
    if isinstance(semantic_memo, dict):
        # The exact residual is already carried once as planner_input.execution_request.
        # Keeping a second copy only increases context and invites the provider to echo it.
        semantic_memo.pop("remaining_execution_request", None)
    payload["candidate_operation_contracts"] = (
        project_memo_qualified_candidate_operation_contracts(
            planner_input.candidate_operation_contracts,
            phase=planner_input.phase,
        )
    )
    payload["candidate_operation_contracts"] = [
        group_step_arguments(item) for item in payload["candidate_operation_contracts"]
    ]
    payload["audio_usage_evidence_inventory"] = audio_usage_evidence_inventory(
        audio_refs=[*planner_input.source_audio_refs, *planner_input.reference_audio_refs],
        evidence_path_sources=planner_input.evidence_path_sources,
        evidence_path_audio_refs=planner_input.evidence_path_audio_refs,
    )
    payload["session_completion_proofs"] = [
        {
            "proof_ref": item.proof_ref,
            "proof_kind": item.proof_kind,
            "selected_node_id": item.selected_node_id,
            "selected_audio_ref": item.selected_audio_ref,
            "transition_edge_id": item.transition_edge_id,
        }
        for item in session_completion_proof_inventory(planner_input.session_state)
    ]
    payload["session_completion_choices"] = session_completion_choices(
        planner_input.session_state,
        bound_source_refs=planner_input.source_audio_refs,
    )
    if planner_input.phase == "evidence_planning":
        payload["direct_execution_operation_contracts"] = (
            project_memo_qualified_candidate_operation_contracts(
                [
                    item
                    for item in planner_input.candidate_operation_contracts
                    if _contract_evidence_is_available(item.requirements, planner_input)
                ],
                phase="execution_planning",
            )
        )
        payload["direct_execution_operation_contracts"] = [
            group_step_arguments(item)
            for item in payload["direct_execution_operation_contracts"]
        ]
    if argument_contract_format == "argument-table/v1":
        payload = factor_argument_contracts(payload)
    return payload


def _contract_evidence_is_available(
    requirements: Any,
    planner_input: MemoQualifiedPlannerInput,
) -> bool:
    """Match capability evidence names only to already admitted evidence fields."""

    required = _requirements_index(requirements).get("evidence", [])
    if not required:
        return True
    available_paths = set(planner_input.evidence_path_sources)

    def collect_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                available_paths.add(str(key))
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(planner_input.reviewed_evidence)
    return all(
        any(path == name or path.endswith(f".{name}") for path in available_paths)
        for name in required
    )


def project_memo_qualified_candidate_operation_contracts(
    contracts: list[Any],
    *,
    phase: Literal["evidence_planning", "execution_planning"],
) -> list[dict[str, Any]]:
    """Return the exact compact capability projection visible to Planner."""

    projected_contracts: list[dict[str, Any]] = []
    for item in contracts:
        projected = {
            "operation": item.operation,
            "semantic_effects": list(item.semantic_effects),
            "supported_scopes": list(item.supported_scopes),
            "requirements": copy.deepcopy(item.requirements),
            "preservation_levels": _capability_levels(item.preservation_profile),
            "quality_levels": _capability_levels(item.quality_profile),
            "artifact_risk_levels": _capability_levels(item.artifact_risk_profile),
            "composition_profile": copy.deepcopy(item.composition_profile),
            # Limitations carry semantic non-goals that levels alone cannot
            # express (for example, standalone silence is not an in-speech
            # pause editor). Keep them visible in both planning phases.
            "limitations": list(item.limitations),
        }
        if phase == "execution_planning":
            projected.update(
                {
                    "planner_selectable_arguments": [
                        {
                            "argument_path": argument.argument_path,
                            "required": argument.required,
                            "value_schema": copy.deepcopy(argument.value_schema),
                            "semantic_role": argument.semantic_role,
                            "runtime_value_source": argument.runtime_value_source,
                            "description": argument.description,
                        }
                        for argument in item.arguments
                    ],
                    "framework_owned_arguments": list(item.framework_owned_arguments),
                    "parameter_rules": copy.deepcopy(item.parameter_rules),
                }
            )
        projected_contracts.append(projected)
    return projected_contracts


def _capability_levels(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.get("level") if isinstance(value, dict) else value)
        for key, value in profile.items()
    }


__all__ = [
    "DEFAULT_PLANNER_PROMPT_PROFILE",
    "CandidateProjectionSource",
    "PlannerPromptPackage",
    "PlannerPromptPhase",
    "PlannerPromptProfileName",
    "PromptBudgetReport",
    "build_planner_prompt_package",
    "candidate_capability_projection",
    "compact_operation_capability_index",
    "compact_semantic_effect_capability_index",
    "frozen_edit_intent_from_context",
    "list_planner_prompt_profile_names",
    "memo_qualified_planner_current_call_prompt",
    "memo_qualified_planner_system_prompt",
    "planner_reasoner_turn_contract_outline",
    "project_memo_qualified_candidate_operation_contracts",
]
