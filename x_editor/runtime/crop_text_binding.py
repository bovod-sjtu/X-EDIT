"""Bind transcript-conditioned calls to a verified crop and its observations.

This is evidence binding, not recognition or semantic repair. Use a same-version
crop observation, or inherit whole words from its producer's accepted evidence.
The original tag and its parameters must remain unchanged; ambiguous edits need
Planner repair or fresh perception. Global perception state is never overwritten.
"""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

from x_editor.core.tagged_text import render_tagged_source


class CropTextBindingError(ValueError):
    pass


def _same_path(a: Any, b: Any) -> bool:
    return isinstance(a, str) and isinstance(b, str) and Path(a).resolve() == Path(b).resolve()


def _characters(text: str) -> str:
    # Align lexical characters while retaining the authored spelling/format in
    # the actual projected label. No case folding or numerical normalization.
    return ''.join(c for c in text if c.isalnum())


def _project_words(words: list[dict[str, Any]], start: float, end: float,
                   cropped: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(words, list):
        raise CropTextBindingError('crop_text_binding_invalid_words')
    selected = []
    previous_end = -math.inf
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get('text'), str):
            raise CropTextBindingError('crop_text_binding_invalid_words')
        if word.get('is_punctuation'):
            continue
        try:
            word_start, word_end = float(word['start_sec']), float(word['end_sec'])
        except (KeyError, TypeError, ValueError) as exc:
            raise CropTextBindingError('crop_text_binding_invalid_word_coordinates') from exc
        if not all(math.isfinite(value) for value in (word_start, word_end)) or word_end <= word_start:
            raise CropTextBindingError('crop_text_binding_invalid_word_coordinates')
        if word_end <= start + 1e-6 or word_start >= end - 1e-6:
            continue
        if word_start < previous_end - 1e-6:
            raise CropTextBindingError('crop_text_binding_ambiguous_words: unordered/overlapping alignment')
        previous_end = word_end
        if word_start < start - 1e-6 or word_end > end + 1e-6:
            raise CropTextBindingError('crop_text_binding_partial_word: re-observe the crop; do not inherit cut words')
        projected = dict(word, start_sec=max(0.0, word_start - start),
                         end_sec=min(end - start, word_end - start))
        projected['source_word_id'] = projected.pop('word_id', None)
        projected['source_audio_ref_id'] = projected.pop('audio_ref_id', None)
        projected.pop('audio_index', None)
        projected.update(audio_artifact_id=cropped['artifact_id'],
                         timeline_revision=cropped.get('timeline_revision'), timebase='crop_local')
        selected.append(projected)
    if not selected:
        raise CropTextBindingError('crop_text_binding_missing_words: re-observe the actual crop')
    return selected


def _bind_observed_crop(step: Any, state: dict[str, Any], cropped: dict[str, Any],
                        producer: str, primary: dict[str, Any]) -> dict[str, Any] | None:
    binding = state.get('perception_evidence_timebase') or {}
    if (binding.get('observed_artifact_id') != cropped.get('artifact_id')
            or binding.get('observed_timeline_revision') != cropped.get('timeline_revision')
            or not _same_path(binding.get('observed_audio_path'), cropped.get('path'))):
        return None
    report = state.get('perception_report') or {}
    reported_binding = report.get('timebase') or {}
    if reported_binding and any(reported_binding.get(key) != binding.get(key)
                               for key in ('observed_artifact_id', 'observed_timeline_revision')):
        raise CropTextBindingError('crop_text_binding_observation_version_mismatch')
    if reported_binding and not _same_path(reported_binding.get('observed_audio_path'), cropped.get('path')):
        raise CropTextBindingError('crop_text_binding_observation_path_mismatch')
    try:
        duration = float(cropped['duration_sec'])
    except (KeyError, TypeError, ValueError) as exc:
        raise CropTextBindingError('crop_text_binding_requires_crop_geometry') from exc
    if not math.isfinite(duration) or duration <= 0:
        raise CropTextBindingError('crop_text_binding_requires_crop_geometry')
    words = report.get('word_timestamps') or []
    selected = _project_words(words, 0.0, duration, cropped)
    if len(selected) != sum(not word.get('is_punctuation') for word in words):
        raise CropTextBindingError('crop_text_binding_observation_outside_crop')
    wanted = _characters(''.join(word.get('text', '') for word in selected))
    params = step.params
    text, label = params.get('source_text'), params.get('instruction_st')
    lexical = step.operation == 'edit_data_pipeline.text_edit'
    if not isinstance(text, str) or not isinstance(label, str):
        raise CropTextBindingError('crop_text_binding_missing_label: author a crop-local source_text and instruction_st')
    if _characters(text) != wanted or _characters(render_tagged_source(label, lexical=lexical)) != wanted:
        raise CropTextBindingError('crop_text_binding_requires_local_label: use the observed crop text')
    for key in ('text', 'target_text'):
        if key in params and not (lexical and key == 'target_text') and _characters(str(params[key])) != wanted:
            raise CropTextBindingError(f'crop_text_binding_requires_local_{key}')
    return {
        'schema_version': 'x-editor.crop-text-binding/v1', 'evidence_kind': 'observed_crop',
        'producer_step_id': producer, 'input_audio_path': cropped['path'],
        'input_artifact_id': cropped['artifact_id'], 'parent_artifact_id': primary.get('artifact_id'),
        'parent_timeline_revision': primary.get('timeline_revision'), 'source_span': None,
        'observed_artifact_id': binding['observed_artifact_id'],
        'observed_timeline_revision': binding['observed_timeline_revision'],
        'local_word_timestamps': selected, 'params_before': copy.deepcopy(params),
        'params_after': copy.deepcopy(params), 'changed': False,
        'limitation': 'Uses an explicit same-version crop observation; no new recognition or quality certification.',
    }


def bind_crop_text(step: Any, state: dict[str, Any]) -> dict[str, Any] | None:
    """Bind a tagged edit to an accepted crop's observed or inherited words.

    Called after input materialization, before ordinary parameter normalization.
    Returns an audit receipt, or None for non-crops/non-transcript operations.
    """
    if step.operation not in {
        'edit_data_pipeline.pitch_rate_edit', 'edit_data_pipeline.emotion_style_edit',
        'edit_data_pipeline.text_edit', 'edit_data_pipeline.pause_edit',
    }:
        return None
    source = step.inputs.get('input_audio') or step.inputs.get('input_audio_path')
    matches = [(identifier, output) for identifier, output in (state.get('step_outputs') or {}).items()
               if isinstance(output, dict) and _same_path(output.get('output_audio_path'), source)
               and output.get('audio_temporal_transform', {}).get('operation') == 'artifact_ops.extract_spans']
    if not matches:
        return None
    if len(matches) != 1:
        raise CropTextBindingError('crop_text_binding_ambiguous_producer: re-observe the actual input')
    identifier, output = matches[0]
    transform = output['audio_temporal_transform']
    commit = output.get('runtime_artifact_commit') or {}
    context = commit.get('verification_context') or {}
    primary = context.get('primary_input_artifact') or {}
    cropped = context.get('candidate_output_artifact') or {}
    if (commit.get('status') != 'accepted' or not cropped.get('artifact_id')
            or cropped.get('timeline_revision') is None or not _same_path(cropped.get('path'), source)):
        raise CropTextBindingError('crop_text_binding_requires_accepted_crop')
    observed = _bind_observed_crop(step, state, cropped, identifier, primary)
    if observed is not None:
        return observed
    evidence = context.get('evidence_timebase') or {}
    mappings = transform.get('timeline_mappings') or []
    if (commit.get('status') != 'accepted' or transform.get('status') != 'complete'
            or transform.get('derivation') != 'single_extracted_span_compacted_to_output'
            or len(mappings) != 1 or evidence.get('status') not in {
                'current', 'mapped_from_base', 'partially_mapped_from_base'}
            or not primary.get('artifact_id')
            or evidence.get('verifier_artifact_id') != primary['artifact_id']
            or evidence.get('verifier_timeline_revision') != primary.get('timeline_revision')
            or not cropped.get('artifact_id') or not _same_path(cropped.get('path'), source)
            or not _same_path(primary.get('path'), (transform.get('input_audio_paths') or [None])[0])):
        raise CropTextBindingError('crop_text_binding_requires_evidence: need accepted words on the crop input version')
    start = float(mappings[0]['source_span']['start_sec'])
    end = float(mappings[0]['source_span']['end_sec'])
    local = mappings[0]['output_span']
    if (not all(math.isfinite(v) for v in (start, end)) or end <= start
            or abs(float(local['start_sec'])) > 1e-6
            or abs(float(local['end_sec']) - (end-start)) > 1e-6):
        raise CropTextBindingError('crop_text_binding_requires_unit_time_mapping')
    words = evidence.get('word_timestamps') or []
    selected = _project_words(words, start, end, cropped)
    wanted = _characters(''.join(w.get('text', '') for w in selected))
    params = copy.deepcopy(step.params)
    text = params.get('source_text')
    label = params.get('instruction_st')
    if not isinstance(text, str) or not isinstance(label, str):
        raise CropTextBindingError('crop_text_binding_missing_label: author a crop-local source_text and instruction_st')
    lexical = step.operation == 'edit_data_pipeline.text_edit'
    if _characters(render_tagged_source(label, lexical=lexical)) != _characters(text):
        raise CropTextBindingError('crop_text_binding_source_label_mismatch: repair label before scope projection')
    before = copy.deepcopy(params)
    if _characters(text) != wanted:
        # Safe initial case: one non-lexical effect spans precisely every retained
        # word. Never drop a second target outside the crop or guess a repeated
        # phrase's occurrence. Richer labels must be authored locally by Planner.
        pattern = (r'<(?P<tag>pitch|rate|emo|happy|angry|sad|afraid|disgusted|melancholic|surprised|calm)'
                   r'(?=[\s,>])[^>]*>(?P<body>[^<>]+)</(?P=tag)>')
        tags = list(re.finditer(pattern, label))
        all_words = _characters(''.join(w.get('text', '') for w in words))
        if (lexical or step.operation == 'edit_data_pipeline.pause_edit'
                or len(tags) != 1 or _characters(text) != all_words
                or _characters(tags[0]['body']) != wanted
                or '<' in label[:tags[0].start()] + label[tags[0].end():]):
            raise CropTextBindingError('crop_text_binding_requires_local_label: cannot project without dropping or changing an edit')
        tag = tags[0]
        before_crop = ''.join(w.get('text', '') for w in words if float(w['end_sec']) <= start+1e-6)
        if _characters(label[:tag.start()]) != _characters(before_crop):
            raise CropTextBindingError('crop_text_binding_wrong_occurrence: label does not address these retained words')
        params['source_text'] = tag['body']
        params['instruction_st'] = tag.group()
    for key in ('text', 'target_text'):
        if lexical and key == 'target_text':
            continue  # Lexical targets deliberately differ; the tag verifier owns them.
        if key in params and _characters(str(params[key])) != wanted:
            raise CropTextBindingError(f'crop_text_binding_requires_local_{key}')
    receipt = {
        'schema_version': 'x-editor.crop-text-binding/v1',
        'producer_step_id': identifier,
        'input_audio_path': source,
        'input_artifact_id': cropped['artifact_id'],
        'parent_artifact_id': primary['artifact_id'],
        'parent_timeline_revision': primary['timeline_revision'],
        'source_span': dict(start_sec=start, end_sec=end),
        'local_word_timestamps': selected,
        'params_before': before, 'params_after': copy.deepcopy(params),
        'changed': params != before,
        'limitation': 'Inherits supplied alignment uncertainty; no new recognition or quality certification.',
    }
    step.params = params
    return receipt
