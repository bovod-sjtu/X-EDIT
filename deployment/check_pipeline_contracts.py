"""Check client payloads against the pinned API models, without model inference.

Run with edit_cpu's interpreter after fetching the pipeline. Temporary audio is
only a transport fixture. This check does not score its sound or editing quality.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from pipeline import ROOT, SPEC, inspect

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / json.loads(SPEC.read_text())['path']))


def main():
    inspect()
    import numpy as np
    import soundfile as sf
    from x_editor.tools.catalog.edit_data_pipeline import client
    from x_editor.tools.catalog.seed_vc.client import SeedVCClient
    from pipeline_service import activate_paths_compat
    from speech_edit.runtime.paths import get_artifact_paths

    activate_paths_compat()
    checks = []
    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    class Captured(Exception):
        pass

    payloads = []
    def capture(url, *, json, **kwargs):
        payloads.append(json)
        raise Captured()

    with tempfile.TemporaryDirectory(prefix='xedit-contract-') as directory:
        audio = Path(directory) / 'input.wav'
        sf.write(audio, np.ones(24000, dtype=np.float32) * .01, 24000)
        for operation, task, instruction, extra in [
            ('text_edit', 'text', '<sub targ="bus">train</sub>', {'max_f5_edit_seconds': 32.0}),
            ('pitch_rate_edit', 'prosody', '<rate, factor=1.25>train</rate>', {'aligner_backend': 'auto'}),
            ('emotion_style_edit', 'emotion', '<happy, level=2>train</happy>', {}),
            ('pause_edit', 'pause', 'train<pause act="ins" level="2"/>', {'stage2_method': 'method_one'}),
        ]:
            model = importlib.import_module(f'speech_edit.pipelines.{task}.service.api_service').ExecuteRequest
            with patch.object(client, 'post_with_retries', capture), patch.dict(os.environ, {
                'X_EDITOR_EDIT_BACKEND': 'edit_data_pipeline',
                'X_EDITOR_ENABLE_F5_TTS': '1', 'F5_SERVICE_URL': 'http://127.0.0.1:1',
            }):
                try:
                    client.EditDataPipelineClient().apply(dict(
                        operation=operation, source_text='train', instruction_st=instruction,
                        input_audio_path=str(audio), output_path=str(Path(directory) / 'output.wav'),
                        evaluate=False, **extra))
                except Captured:
                    pass
            payload = payloads[-1]
            model.model_validate(payload)
            check(operation + ': exact declared fields', not (payload.keys() - model.model_fields.keys()))
            if task == 'emotion':
                check('emotion type/level preserved', payload['instruction_st'] == '<emo type="happy" level="2">train</emo>')
            if task == 'pause':
                check('pause strategy alias preserved', payload['synth_method'] == 'single_region')
        canonical = '<emo type="sad" alpha="0.63">train</emo>'
        check('continuous alpha not quantized', client._canonicalize_emotion_instruction_st_for_service(canonical) == canonical)
        with patch('x_editor.tools.catalog.seed_vc.client.requests.post', capture):
            try:
                SeedVCClient().convert(str(audio), str(audio), str(Path(directory) / 'converted.wav'))
            except Captured:
                pass
        check('SeedVC deterministic default passed explicitly', payloads[-1]['seed'] == 0)
        paths = get_artifact_paths(artifact_dir=directory)
        check('artifact root accepted', paths.resolve('ok.wav').is_relative_to(Path(directory)))
        for value in [ROOT / 'output.wav', Path(directory) / '..' / 'escape.wav']:
            try:
                paths.resolve(value)
            except ValueError:
                checks.append('artifact escape rejected: ' + value.name)
            else:
                raise AssertionError('Artifact containment was bypassed')
    print(json.dumps({'passed': True, 'checks': checks, 'inference_verified': False}, indent=2))


if __name__ == '__main__':
    main()
