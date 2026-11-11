"""Measure model-requested, version-bound clip comparisons without grading.

Equal lengths do not establish correspondence. Gain is only a least-squares
fit to these supplied samples; residual and correspondence caveats travel with
it. No resampling, alignment, normalization or semantic target is inferred.
"""

import numpy as np
import soundfile as sf

from x_editor.evaluation.iterative_runtime import sha256
from x_editor.evaluation.paired_signal_facts import measure_pcm_comparison


def measure_listening_comparisons(packet, paths):
    inventory = packet['clip_inventory']
    if len(inventory) != len(paths):
        raise ValueError('signal measurement needs all bound clips')
    loaded = {}
    for item, path in zip(inventory, paths):
        if sha256(path) != item['sha256']:
            raise ValueError('listening clip changed before measurement')
        samples, rate = sf.read(path, always_2d=True, dtype='float64')
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError('signal measurement needs finite nonempty clips')
        loaded[item['audio_ref']] = samples, rate
    candidates = [item for item in inventory if item['role'] == 'candidate']
    if len(candidates) != 1:
        raise ValueError('signal measurement needs exactly one candidate')
    candidate = candidates[0]
    y, yr = loaded[candidate['audio_ref']]
    pairs = []
    for item in inventory:
        if item['role'] != 'source':
            continue
        x, xr = loaded[item['audio_ref']]
        pair = {'source_audio_ref': item['audio_ref'], 'candidate_audio_ref': candidate['audio_ref'],
                'source_frames': len(x), 'candidate_frames': len(y),
                'source_sample_rate': xr, 'candidate_sample_rate': yr,
                'source_channels': x.shape[1], 'candidate_channels': y.shape[1]}
        if xr != yr or x.shape != y.shape:
            pair.update(comparison_status='unknown', reason='incompatible_sample_rate_frame_count_or_channels')
        else:
            pair.update(measure_pcm_comparison(x, y))
        pairs.append(pair)
    return {'protocol': 'x-editor.listening-signal-facts/v1', 'pairs': pairs,
            'correspondence': 'Unverified model-proposed windows, not an execution or gold mapping.',
            'interpretation': 'The gain is a fitted amplitude coefficient, not loudness or proof of an authorized edit. '
                'Consider fit_residual_rms, polarity and actual content correspondence. Different speech or a time shift '
                'can produce a misleading fitted gain. No target value or pass/fail verdict is supplied.'}
