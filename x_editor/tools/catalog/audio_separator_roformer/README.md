# Audio Separator RoFormer Backend

Backend service for vocal/accompaniment separation requests.

This service keeps `audio-separator` and RoFormer dependencies in an isolated
`uv` environment. X-Editor should not import `audio_separator` in the main
framework process.

Default endpoint:

```text
http://127.0.0.1:8042
```

Run:

```bash
cd x_editor/tools/catalog/audio_separator_roformer
./setup.sh
.venv/bin/python service.py --host 127.0.0.1 --port 8042
```

Model selection is service-side and can be overridden per request:

- `model_family=bs_roformer`
- `model_family=melband_roformer`
- explicit `model_filename`

The backend uses `audio-separator` to download/cache the selected model under
its normal model cache. The configured checkpoints separate vocals from
accompaniment; they do not isolate individual overlapping speakers. Additional
checkpoints require their own capability and license review. Use
`speech_enhancement` for single-speaker cleanup.

The default `stem_policy` is `vocals`; `instrumental` selects accompaniment.
The aliases `speech`, `voice` and `foreground` mean vocals, not a speaker ID.
Semantic policies require a unique provider-labeled stem and fail when labels
are missing or ambiguous. Legacy `first`/`last`/`residual` requests select list
positions only and are not exposed as semantic choices in the typed capability
card. An explicit filename substring also requires a unique match. Neither
filename labels nor successful decoding certify perceptual separation quality.

`/health` checks both `audio_separator` and `onnxruntime`; missing ONNX runtime
is reported as `not_ready` before `/separate` runs real inference.
