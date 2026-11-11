"""Task attributes and target binding, separate from transcript revision.

The audio model can describe existing speakers and select measured segments. It
cannot rewrite the measured words, identities, or boundaries through this API.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel


TASK_GROUNDING_PROMPT = """Listen to the supplied source audio and answer the current
speaker observation request. The speaker inventory contains measured, potentially
imperfect prior evidence, not the desired answer. Speaker IDs are anonymous and
do not encode gender, age, order, or conversational role. Names in the transcript
are not necessarily speakers. Preserve the supplied IDs.

This call supplements task-relevant audible properties and target binding. It
does NOT revise the transcript, speaker attribution, or timestamps. Describe only
properties supported by listening, with the exact supporting segment IDs. Voice
descriptions are perceived acoustic properties, not verified personal identity.
Do not infer a requested property simply because the instruction mentions it.
Words, names, pronouns, kinship terms, and conversational roles are NOT evidence
of a voice's perceived gender, age, emotion, or vocal texture. For these properties,
cite audible characteristics of the relevant voice, not what the person says.
If you cannot hear those characteristics reliably, return unknown. Text may help
identify a quoted utterance or conversational role, but cannot establish a voice
attribute. Apparent pitch alone is not proof of age or personal identity.

If the request selects particular speech, select ALL and ONLY existing complete
segments that satisfy it. Speaker-wide properties and segment-local delivery
(such as whispering or emotion) are different: a property heard once does not
hold for all of that speaker's turns. Whole-clip coverage is not evidence of a
specific target. Keep overlap; do not turn overlapping speakers into one speaker.
If the target needs part of a supplied segment, a missing reference, conflicting
attribution, or uncertain attributes, use unknown and state the specific missing
observation. Do not widen the edit to an entire segment to avoid uncertainty.
If no target selection is requested, use not_requested and return any requested
descriptions. A supported selection must cite segments and explain its evidence.
Unknown is not absence and is not successful grounding. Never invent a time span.

Return only the supplied JSON schema. No new transcript, plan, editing instruction,
or tool call. Keep descriptions and reasons short.
"""


class SpeakerAttributeObservation(StrictModel):
    speaker_id: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=700)
    support_segment_ids: list[str] = Field(min_length=1, max_length=128)


class SpeakerTaskObservation(StrictModel):
    speaker_profiles: list[SpeakerAttributeObservation] = Field(max_length=64)
    selection_status: Literal['supported', 'unknown', 'not_requested']
    selected_segment_ids: list[str] = Field(max_length=256)
    reason: str = Field(min_length=1, max_length=1000)
    unresolved_questions: list[str] = Field(max_length=16)

    @model_validator(mode='after')
    def selection_is_explicit(self):
        if (self.selection_status == 'supported') != bool(self.selected_segment_ids):
            raise ValueError('only a supported target may publish selected segments')
        if self.selection_status == 'supported' and self.unresolved_questions:
            raise ValueError('an unresolved selection must stay unknown')
        if self.selection_status == 'unknown' and not self.unresolved_questions:
            raise ValueError('unknown selection must identify the missing observation')
        if len(self.selected_segment_ids) != len(set(self.selected_segment_ids)):
            raise ValueError('duplicate selected segment')
        return self


def task_payload(result, context):
    report = result['perception_report']
    segments = deepcopy(report.get('speaker_segments') or [])
    if not segments:
        raise ValueError('speaker task grounding requires an observed speaker timeline')
    return {
        'instruction': context['instruction'],
        'observation_needs': context.get('observation_needs') or [],
        'audio_path': context['audio_path'],
        'audio_sha256': hashlib.sha256(Path(context['audio_path']).read_bytes()).hexdigest(),
        'duration_sec': context['duration_sec'],
        'source_text': report.get('source_text'),
        'speaker_segments': [dict(segment_id=f'segment_{i:04d}',
            speaker_id=str(s['speaker_id']), start_sec=s['start_sec'], end_sec=s['end_sec'],
            transcript=s.get('transcript') or s.get('text') or '') for i, s in enumerate(segments)],
        'response_schema': SpeakerTaskObservation.model_json_schema(),
    }


def apply_task_observation(result, payload, raw):
    observation = SpeakerTaskObservation.model_validate(raw)
    by_id = {s['segment_id']: s for s in payload['speaker_segments']}
    for profile in observation.speaker_profiles:
        for ref in profile.support_segment_ids:
            if ref not in by_id or by_id[ref]['speaker_id'] != profile.speaker_id:
                raise ValueError('speaker attribute support must belong to the cited existing speaker')
    if any(ref not in by_id for ref in observation.selected_segment_ids):
        raise ValueError('target references an unobserved segment')
    if len({p.speaker_id for p in observation.speaker_profiles}) != len(observation.speaker_profiles):
        raise ValueError('duplicate speaker profile')
    output = deepcopy(result)
    report = output['perception_report']
    evidence_id = 'speaker-task:' + hashlib.sha256(json.dumps(
        [payload, observation.model_dump()], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report['speaker_profiles'] = [dict(p.model_dump(), source='speaker_task_audio_observation',
        audio_ref_id=payload['audio_path'], source_refs=[evidence_id],
        audio_sha256=payload['audio_sha256']) for p in observation.speaker_profiles]
    # These fields belong to THIS request; never carry an older target selection.
    report['selected_spans'] = []
    for ref in observation.selected_segment_ids:
        s = by_id[ref]
        report['selected_spans'].append(dict(span_id=ref, audio_ref_id=payload['audio_path'],
            start_sec=s['start_sec'], end_sec=s['end_sec'], speaker_id=s['speaker_id'],
            text=s['transcript'], target_id='target_000', semantic_role='edit_target',
            source='speaker_task_audio_observation', source_refs=[evidence_id, ref],
            reason=observation.reason, confidence=None))
    report['selected_speaker_ids'] = list(dict.fromkeys(s['speaker_id'] for s in report['selected_spans']))
    report['target_bindings'] = []
    if observation.selection_status != 'not_requested':
        report['target_bindings'] = [dict(target_id='target_000', target_kind='speaker',
            target_value=payload['instruction'],
            status='resolved' if observation.selection_status == 'supported' else 'unresolved',
            resolved_span_ids=observation.selected_segment_ids,
            resolved_speaker_ids=report['selected_speaker_ids'], resolved_event_ids=[],
            reason=observation.reason, confidence=None, source_refs=[evidence_id])]
    task = dict(observation.model_dump(), evidence_id=evidence_id,
        audio_sha256=payload['audio_sha256'], transcript_revision_performed=False,
        acoustic_accuracy_certified=False)
    report['speaker_task_observation'] = task
    output['speaker_attributed_review']['task_observation'] = task
    report.setdefault('uncertainty', []).extend(observation.unresolved_questions)
    return output


def supplement_speaker_task(result, context, frontend):
    payload = task_payload(result, context)
    output = Path(context['output_dir']) / 'speaker_task_observation'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        raw = frontend.observe_task(payload)
        (output / 'response.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2))
        return apply_task_observation(result, payload, raw)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        (output / 'error.json').write_text(json.dumps(error, ensure_ascii=False))
        # This failed extra observation cannot erase measurements acquired
        # before it. Keep the failure explicit; publish no new target or profile.
        partial = deepcopy(result)
        partial['speaker_attributed_review']['task_observation_error'] = error
        partial['perception_report'].setdefault('uncertainty', []).append(
            'Task-specific speaker observation failed; the requested attributes '
            f'and target remain unverified: {type(exc).__name__}: {exc}')
        return partial
