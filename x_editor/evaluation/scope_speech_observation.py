"""Independent, output-blind transcription followed by fallible alignment.

Only audio goes to ASR. Only its returned transcript goes to the aligner. User
targets, Actor traces and rubric answers never enter either request. This tool
produces observations, not a verdict or a certified edit boundary.
"""

import base64
import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np
import requests
import soundfile as sf
from pydantic import AnyHttpUrl, Field
from scipy.io import wavfile

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel
from x_editor.evaluation.iterative_runtime import sha256, write_record


class SpeechObservationConfig(FrozenEvaluationModel):
    asr_url: AnyHttpUrl
    alignment_url: AnyHttpUrl
    timeout_sec: float = Field(default=120, gt=0, le=900, allow_inf_nan=False)


class ScopeSpeechObserver:
    """Explicitly configured service access; no bootstrap or fallback routing."""

    description = (
        "独立ASR转写指定音频，再对ASR实际返回的文本进行时间对齐。"
        "返回该版本的词/字时间，均为可错观察；不验证目标、不识别说话人，"
        "不能把对齐词边界当作编辑接点真值。一次观察最多两次服务请求。"
    )

    def __init__(self, config, *, post=requests.post):
        self.config = SpeechObservationConfig.model_validate(config)
        self.post = post

    def _call(self, service, payload, output, audio_hash):
        # Audio bytes are retained in clip.wav. Store every non-audio request
        # field, so one can audit that no expected answer was sent to ASR.
        endpoint = str(getattr(self.config, service + '_url')).rstrip('/')
        route = '/transcribe' if service == 'asr' else '/align'
        write_record(output / 'request.json', {
            'endpoint': endpoint + route, 'audio_sha256': audio_hash,
            'payload_without_audio': {k: v for k, v in payload.items() if k != 'audio'},
        })
        started = time.monotonic()
        try:
            response = self.post(endpoint + route, json=payload, timeout=self.config.timeout_sec)
            response.raise_for_status()
            raw = response.json()
            write_record(output / 'response.json', raw)
            return raw
        finally:
            write_record(output / 'timing.json', {'elapsed_sec': time.monotonic() - started})

    def __call__(self, request, asset, output):
        output = Path(output)
        data = Path(asset['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset['sha256']:
            raise ValueError('audio changed before independent speech observation')
        samples, rate = sf.read(io.BytesIO(data), dtype='float64', always_2d=True)
        end_sec = len(samples) / rate if request.end_sec is None else request.end_sec
        if not 0 <= request.start_sec < end_sec <= len(samples) / rate:
            raise ValueError('speech observation exceeds the bound audio window')
        first, last = math.floor(request.start_sec * rate), min(len(samples), math.ceil(end_sec * rate))
        if not (last > first and np.isfinite(samples[first:last]).all()):
            raise ValueError('speech observation needs finite nonempty audio')
        output.mkdir(parents=True, exist_ok=False)
        clip = output / 'clip.wav'
        wavfile.write(clip, rate, samples[first:last])
        audio = {'data': base64.b64encode(clip.read_bytes()).decode('ascii'), 'format': 'wav'}
        receipt = {
            'protocol': 'x-editor.independent-speech-observation/v1',
            'authority': 'fallible_asr_and_alignment_not_ground_truth',
            'audio_ref': request.audio_ref, 'audio_sha256': asset['sha256'],
            'clip_sha256': sha256(clip), 'start_sec': first / rate, 'end_sec': last / rate,
            'target_text_supplied': False, 'actor_evidence_visible': False,
            'request': request.model_dump(mode='json'), 'service_calls_attempted': 0,
            'status': 'unknown', 'limitations': [
                'An aligned word is not proof it was spoken. Compare conflicting listening observations.',
                'Word boundaries are not certified edit junctions; silence and nonspeech are not covered.',
            ],
        }
        try:
            receipt['service_calls_attempted'] += 1
            transcription = self._call('asr', {'audio': audio}, output / 'asr', receipt['clip_sha256'])
            text, language = transcription.get('text'), transcription.get('language')
            if not isinstance(text, str):
                raise ValueError('ASR returned no textual observation')
            receipt.update(transcript=text, language=language, status='transcribed')
            if not text.strip() or not isinstance(language, str) or not language.strip():
                raise ValueError('no nonempty transcript and explicit observed language for alignment')
            receipt['service_calls_attempted'] += 1
            aligned = self._call('alignment', {'audio': audio, 'text': text, 'language': language,
                                  'word_level': True}, output / 'alignment', receipt['clip_sha256'])
            if aligned.get('status') != 'success' or not aligned.get('alignments'):
                raise ValueError('aligner returned no successful alignment')
            units = []
            for item in aligned['alignments']:
                start, end = float(item['start']), float(item['end'])
                if not (math.isfinite(start) and math.isfinite(end)
                        and 0 <= start <= end <= (last - first) / rate):
                    raise ValueError('alignment falls outside observed audio')
                units.append({**item, 'start_sec': first / rate + start, 'end_sec': first / rate + end})
            receipt.update(status='aligned', units=units, coordinate_system='parent_audio_seconds')
        except Exception as exc:
            # Retain a successful transcript even if alignment fails. A model
            # can use the content without pretending word coordinates exist.
            receipt['error'] = f'{type(exc).__name__}: {exc}'
        write_record(output / 'observation.json', receipt)
        return receipt
