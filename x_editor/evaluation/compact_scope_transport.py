"""Scope preparation with shared windows and the same expanded listening contract.

The model still proposes all windows/questions and an independent reviewer checks
coverage. Expansion only resolves explicit table references: no guessed ranges,
repaired IDs, invented questions or relaxed admission.
"""
from copy import deepcopy
from typing import Literal

from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel
from x_editor.evaluation.runtime_scope_contracts import (
    ScopeWindow, ScopeObservationRequest, ScopeProbe, RuntimeScopePlan,
    SCOPE_PLAN_PROMPT, SCOPE_COVERAGE_PROMPT,
)


class CompactProbe(FrozenEvaluationModel):
    question: str = Field(min_length=8, max_length=1400)
    explanation: str = Field(min_length=8, max_length=1200)
    candidate: str = Field(min_length=1)
    references: tuple[str, ...] = Field(default=(), max_length=8)
    signal_measurement: Literal['none', 'aligned_pcm_comparison'] = 'none'


class CompactCriterion(FrozenEvaluationModel):
    criterion_id: str = Field(min_length=1)
    probes: tuple[CompactProbe, ...] = Field(default=(), max_length=12)
    missing_evidence: tuple[str, ...] = ()


class CompactScopePlan(FrozenEvaluationModel):
    windows: dict[str, ScopeWindow] = Field(max_length=240)
    criteria: tuple[CompactCriterion, ...] = Field(min_length=1, max_length=24)
    observation_requests: tuple[ScopeObservationRequest, ...] = Field(default=(), max_length=6)

    @model_validator(mode='after')
    def exact_references(self):
        used = {key for c in self.criteria for p in c.probes
                for key in (p.candidate, *p.references)}
        if used != set(self.windows):
            raise ValueError('window table must contain exactly the referenced windows')
        return self


INSTRUCTION = '''
本轮使用共享窗口表：windows按简短ID保存ScopeWindow；每条probe的candidate与references
只填这些ID。相同版本、时间范围、边界及依据完全相同的窗口只定义一次；不同者不可合并。
保留所有criterion、目标位置、保留成分和接点，不减少问题、不扩大已知证据的证明范围。
question直接问本项原子条件；explanation简述范围依据，一句即可，不重复原题、数值、
完整任务背景或长推理。仍然未知的范围必须留作missing_evidence并按预算请求补感知。
'''
CONCISE_COVERAGE = '''
各题explanation只写一条具体的覆盖依据或缺口，不复述计划、题干和全部数值。
这只缩短说明，仍需检查每题全部条件；不得省略未覆盖条件或未决观察。
'''


def expand_plan(raw):
    compact = CompactScopePlan.model_validate(raw)
    data = compact.model_dump(mode='json')
    windows = data.pop('windows')
    for criterion in data['criteria']:
        for probe in criterion['probes']:
            probe['candidate'] = deepcopy(windows[probe['candidate']])
            probe['references'] = [deepcopy(windows[key]) for key in probe['references']]
    return RuntimeScopePlan.model_validate(data).model_dump(mode='json')


def compact_plan(raw):
    """Inverse for evidence-preservation tests and coverage-review transport."""
    plan = RuntimeScopePlan.model_validate(raw).model_dump(mode='json')
    windows = {}

    def reference(window):
        for key, old in windows.items():
            if old == window:
                return key
        key = f'w{len(windows)}'
        windows[key] = window
        return key

    for criterion in plan['criteria']:
        for probe in criterion['probes']:
            probe['candidate'] = reference(probe['candidate'])
            probe['references'] = [reference(w) for w in probe['references']]
    return CompactScopePlan.model_validate({'windows': windows, **plan}).model_dump(mode='json')


def prepare_request(prompt, payload):
    payload = deepcopy(payload)
    if prompt == SCOPE_PLAN_PROMPT:
        payload['schema'] = CompactScopePlan.model_json_schema()
        return prompt + INSTRUCTION, payload
    if prompt == SCOPE_COVERAGE_PROMPT:
        payload['plan'] = compact_plan(payload['plan'])
        return prompt + INSTRUCTION + CONCISE_COVERAGE, payload
    raise ValueError('compact scopes apply only to plan and coverage preparation')
