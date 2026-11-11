"""Prepare real audio crops before listening; never ask the judge to invent scope.

Scope proposals and observations remain fallible. This adapter verifies file and
coordinate integrity, not semantic authorization or complete task coverage.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Literal

import soundfile as sf
from pydantic import Field, model_validator
from scipy.io import wavfile

from x_editor.evaluation.edit_region_scope import RegionFocus, RegionalObservation, CheckedAudioRegion
from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel
from x_editor.evaluation.iterative_runtime import sha256, write_record
from x_editor.evaluation.refinement_contracts import UtilityAnswer
from x_editor.evaluation.terminal_observation_review import ObservedUtilityAnswer, validate_terminal_references


class ListeningWindow(FrozenEvaluationModel):
    audio_ref: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(gt=0, strict=True)
    boundary_frame: int | None = Field(default=None, ge=0, strict=True)
    basis_evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_frame <= self.start_frame:
            raise ValueError("listening window must contain audio")
        if self.boundary_frame is not None and not self.start_frame < self.boundary_frame < self.end_frame:
            raise ValueError("boundary context must include both sides")
        return self


class ListeningProbe(FrozenEvaluationModel):
    probe_id: str = Field(min_length=1, max_length=160)
    parent_criterion_id: str = Field(min_length=1, max_length=128)
    focus: RegionFocus
    question: str = Field(min_length=8, max_length=1400)
    scope_status: Literal["ready_for_listening", "needs_grounding"]
    scope_explanation: str = Field(min_length=8, max_length=1200)
    candidate: ListeningWindow
    references: tuple[ListeningWindow, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def scope(self):
        if self.focus == "preserved_region" and not self.references:
            raise ValueError("preservation needs a source comparison")
        if self.focus == "boundary_transition" and self.candidate.boundary_frame is None:
            raise ValueError("a boundary probe needs its candidate junction")
        return self


LISTENING_PACKET_PROMPT = """本次是多模态局部听评，不是规划审查或重新定位任务。
附件已经按范围准备端提供的窗口真实裁出。只回答当前 question：
edited_region 检查本处授权修改；preserved_region 对照源和候选的本处内容/属性；
boundary_transition 检查标出的接点两侧，不用整句清晰代替接点判断。
每个附件从局部0秒播放，clip_inventory同时给出原文件时间、局部接点位置和裁片范围。
is_parent_start/end说明附件边缘是否也是原文件边缘。否则首尾突然截断可能来自本次
裁片，不能算作原产物损伤；boundary_transition只评指定内部接点，不用裁片边缘代替。
这些坐标说明附件来自哪里，不证明语义目标真的在这里。若听到的内容与定位描述不符，
或窗口不足以判断，就unknown并说明应补查什么；不得猜一个新时间后直接判通过。
不需要输出时间戳、改写范围或推理工具调用。只描述本题听到的差异再给等级。
保留区的词还在不等于其应保留的响度/说话人/背景属性都未改变，按本题具体属性回答。
用户允许的静音、停顿、音色变化不能作为未授权损伤。源中已有缺陷不自动归罪于编辑。
可选signal_facts是这些裁片的物理测量，不是目标答案；不能由波形差异直接断言语义失败。
3=本题全部满足；2=必要条件满足且仅有轻微缺陷；1=实质缺失或损伤；0=未实现/相反；
unknown时grade=null并填写missing_evidence。确定判断必须引用候选附件；保留判断也
引用实际对比的源附件。局部结论不能认证未播放的区域。返回实际schema对象。
需要比较源与候选的题，observed_condition先分别描述各附件里实际听到的相关事实，
用各自audio_ref标明，再说明相同和差异。不能只复述question中的期望内容就宣称保留。
确定的比较判断须在evidence_ids引用实际比较的全部附件；听不清就保留未知。
"""


def prepare_listening_packet(probe, assets, evidence_catalog, output):
    """Crop exact version-bound windows without normalization, mixing or fades."""
    if probe.scope_status != "ready_for_listening":
        raise ValueError("scope needs grounding before quality judgment")
    windows = [*(('source', w) for w in probe.references), ('candidate', probe.candidate)]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    inventory, paths = [], []
    for index, (role, window) in enumerate(windows):
        if set(window.basis_evidence_ids) - set(evidence_catalog):
            raise ValueError("listening window cites unavailable grounding evidence")
        asset = assets.get(window.audio_ref)
        if not asset or asset['sha256'] != window.sha256:
            raise ValueError("listening window belongs to another audio version")
        if (asset.get('role') == 'candidate') != (role == 'candidate'):
            raise ValueError("listening window changes an audio's source/candidate role")
        path = Path(asset['path'])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != window.sha256:
            raise ValueError("audio changed before crop preparation")
        samples, rate = sf.read(io.BytesIO(data), always_2d=True, dtype='float64')
        if window.end_frame > len(samples):
            raise ValueError("listening window exceeds bound audio")
        clip_ref = f"clip_{index}"
        destination = root / f"{clip_ref}.wav"
        # Retain samples/channels and avoid time-dependent PEAK metadata in
        # libsndfile's float WAV writer, so identical windows have stable bytes.
        wavfile.write(destination, rate, samples[window.start_frame:window.end_frame])
        paths.append(str(destination))
        inventory.append({
            'audio_ref': clip_ref, 'role': role, 'attachment_number': index + 1,
            'sha256': sha256(destination), 'duration_sec': (window.end_frame-window.start_frame)/rate,
            'parent_audio_ref': window.audio_ref, 'parent_sha256': window.sha256,
            'is_parent_start':window.start_frame == 0, 'is_parent_end':window.end_frame == len(samples),
            'parent_start_sec': window.start_frame/rate, 'parent_end_sec': window.end_frame/rate,
            'boundary_local_sec': None if window.boundary_frame is None else (window.boundary_frame-window.start_frame)/rate,
            'boundary_parent_sec': None if window.boundary_frame is None else window.boundary_frame/rate,
            'basis_evidence_ids': list(window.basis_evidence_ids),
        })
    packet = {
        'protocol': 'x-editor.prepared-listening/v1', 'question_id': probe.probe_id,
        'question': probe.question, 'focus': probe.focus, 'clip_inventory': inventory,
        'scope_explanation': probe.scope_explanation,
        'scope_evidence': {k: evidence_catalog[k] for w in [*probe.references, probe.candidate] for k in w.basis_evidence_ids},
        'coverage': 'Only these supplied windows. Scope proposals are not semantic ground truth.',
    }
    write_record(root / 'packet.json', packet)
    return packet, paths


def assess_listening_packet(probe, packet, paths, call, output, *, include_scope_explanation=True,
                            revision_attempts=0, judgment_protocol='legacy', semantic_call=None):
    """Judge local content and translate its citations back to the parent versions."""
    if type(revision_attempts) is not int or revision_attempts not in (0, 1):
        raise ValueError('listening answer revisions must be zero or one')
    inventory = packet['clip_inventory']
    if judgment_protocol in {'evidence_ordinal', 'numeric_polarity'}:
        from x_editor.evaluation.evidence_ordinal_review import observe_audio, judge_evidence
        kwargs = {}
        if judgment_protocol == 'numeric_polarity':
            from x_editor.evaluation.numeric_polarity_review import judge_evidence
            if semantic_call is None:
                raise ValueError('numeric polarity requires opposite-question authoring')
            kwargs['semantic_call'] = semantic_call
        observations = observe_audio(inventory, paths, call, output/'observations')
        result = judge_evidence(probe.question, inventory, paths, observations, call,
                                output/'ordinal', signal_facts=packet.get('signal_facts'), **kwargs)
        value = UtilityAnswer(question_id=probe.probe_id, grade=result['grade'],
            finding=result['finding'], evidence_ids=tuple(dict.fromkeys(i['parent_audio_ref'] for i in inventory)))
        observation = RegionalObservation(criterion_id=probe.probe_id, focus=probe.focus,
            checked_regions=tuple(CheckedAudioRegion(audio_ref=i['parent_audio_ref'],
                start_sec=i['parent_start_sec'], end_sec=i['parent_end_sec'],
                boundary_sec=i['boundary_parent_sec'], observation='Supplied version-bound attachment; judgment recorded separately.')
                for i in inventory),
            unresolved_region_mapping=('Ordinal judgment remains unknown.',) if result['grade'] is None else ())
        write_record(output/'answer.json', {'answer': value.model_dump(mode='json'),
            'regional_observation': observation.model_dump(mode='json'),
            'parent_criterion_id': probe.parent_criterion_id, 'whole_criterion_coverage_certified': False,
            'ordinal_decision': {k:v for k,v in result.items() if k != 'finding'}})
        return value, observation
    if judgment_protocol != 'legacy':
        raise ValueError('unsupported listening judgment protocol')
    schema = ObservedUtilityAnswer.model_json_schema()
    # Candidate hashes and experiment IDs belong in the recording sidecar. They
    # must not alter a local listening question when its audio is identical.
    local_question_id = 'current_question'
    schema['properties']['question_id']['const'] = local_question_id
    schema['properties']['evidence_ids']['items']['enum'] = [i['audio_ref'] for i in inventory]
    visible_fields = {'audio_ref', 'role', 'attachment_number', 'duration_sec',
                      'parent_start_sec', 'parent_end_sec', 'boundary_local_sec',
                      'is_parent_start','is_parent_end'}
    visible = {k:packet[k] for k in ['protocol','question','focus','scope_explanation','coverage']}
    if not include_scope_explanation:
        # Automatic planners may predict quality from tool metadata in this
        # rationale. Keep that speculation in the audit, outside the listening
        # judgment; the atomic question and actual coordinates suffice here.
        visible.pop('scope_explanation')
    visible.update(question_id=local_question_id, schema=schema,
                   clip_inventory=[{k:v for k,v in item.items() if k in visible_fields} for item in inventory])
    # Public history and the grounding catalog belong to scope authoring and its
    # audit. Repeating a whole-file edit instruction can contaminate a question
    # about an explicitly unedited clip. The atomic question carries its terms.
    if 'signal_facts' in packet:
        visible['signal_facts'] = packet['signal_facts']
    for attempt in range(revision_attempts + 1):
        destination = output if attempt == 0 else output / 'revision'
        raw = call(LISTENING_PACKET_PROMPT, visible, paths, destination / 'calls')
        write_record(destination / 'original_answer.json', raw)
        try:
            answer = validate_terminal_references(ObservedUtilityAnswer.model_validate(raw), local_question_id, inventory)
            if answer.grade is not None and probe.focus == 'preserved_region' and any(
                i['audio_ref'] not in answer.evidence_ids for i in inventory if i['role'] == 'source'
            ):
                raise ValueError('preservation grade lacks a source comparison citation')
            break
        except ValueError as exc:
            write_record(destination / 'validation_error.json', {'error': str(exc)})
            if attempt == revision_attempts:
                raise
            # Ask the same judge to answer again with the same audio. Never add
            # citations, keep a grade while repairing its evidence, or tell it
            # that the candidate is correct. The rejected answer stays recorded.
            visible = {**visible, 'response_contract_error': str(exc),
                       'revision_instruction': '上一份回答的结构或引用无效。重新听同一附件并完整回答本题；'
                           '只引用实际比较的附件，无法判断就grade=null。不需维持上一份分数。'}
    cited = [i for i in inventory if i['audio_ref'] in answer.evidence_ids]
    observation = RegionalObservation(
        criterion_id=probe.probe_id, focus=probe.focus,
        checked_regions=tuple(CheckedAudioRegion(
            audio_ref=i['parent_audio_ref'], start_sec=i['parent_start_sec'], end_sec=i['parent_end_sec'],
            boundary_sec=i['boundary_parent_sec'],
            observation=f"Actually supplied {i['role']} window for this local comparison. "
                        "The candidate judgment is recorded separately in the bound question answer.",
        ) for i in cited),
        unresolved_region_mapping=answer.missing_evidence if answer.grade is None else (),
    )
    value = UtilityAnswer(question_id=probe.probe_id, grade=answer.grade,
                          finding=answer.observed_condition,
                          evidence_ids=tuple(dict.fromkeys(i['parent_audio_ref'] for i in cited)))
    write_record(output / 'answer.json', {'answer':value.model_dump(mode='json'),
        'regional_observation':observation.model_dump(mode='json'),
        'parent_criterion_id':probe.parent_criterion_id, 'whole_criterion_coverage_certified':False})
    return value, observation
