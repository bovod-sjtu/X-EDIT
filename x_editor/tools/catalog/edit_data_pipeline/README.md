# edit_data_pipeline Tool

This stable tool name now defaults to the separately deployed dots.tts.edit model
for the four tagged-edit operations. Set `edit_backend: edit_data_pipeline` in
the runtime's tools configuration to use the pipeline described below. Direct
callers without that configuration can use `X_EDITOR_EDIT_BACKEND=edit_data_pipeline`.
See [backend setup](../../../../docs/DOTS_TTS_EDIT.md).
Inputs stay compatible; outputs identify the selected backend and model revision.
The service URLs and F5 constraints below apply to the pipeline choice only.

This catalog entry calls the pinned `third_party/dots_tts_edit` pipeline
through separately configured HTTP services. Set `EDIT_DATA_PIPELINE_ROOT`
explicitly for manual service startup.

The integration intentionally calls that repository through its documented HTTP
microservices instead of importing Python modules directly. This keeps X-Editor
independent from the pipeline's uv environments and git submodules.

## MCP Tool

```text
edit_data_pipeline.apply
```

Supported `operation` values:

| Operation | Service | Default URL | Endpoint |
|---|---|---|---|
| `text_edit` | Tagged Text Edit Pipeline | `http://127.0.0.1:8035` | `/execute` |
| `pitch_rate_edit` | Prosody Edit Pipeline | `http://127.0.0.1:8036` | `/execute` |
| `emotion_style_edit` | Emotion/Style Edit Pipeline | `http://127.0.0.1:8037` | `/execute` |
| `pause_edit` | Pause Edit Pipeline | `http://127.0.0.1:8038` | `/execute` |
| `speech_rate` | Speech Rate Conversion | `http://127.0.0.1:8004` | `/stretch` |
| `volume_normalize` | Volume Normalizer | `http://127.0.0.1:8006` | `/normalize` |
| `pitch_shift` | Pitch Conversion | `http://127.0.0.1:8927` | `/shift` |
| `boundary_smooth` | Boundary Pause Smoother | `http://127.0.0.1:8018` | `/smooth` |
| `audio_assembly` | Audio Assembly | `http://127.0.0.1:8021` | `/assemble` |

The default pitch port conflicts with the existing X-Editor ClearerVoice
service default. Override it when both services are running:

```bash
export EDIT_PIPELINE_PITCH_URL=http://127.0.0.1:<your_pitch_port>
```

## Service-Layer Examples

Tagged text edit (`instruction_st` may contain only `<sub>`, `<ins>`, and
`<del>` text-edit labels):

```bash
python examples/run_tool_call.py \
  --tool edit_data_pipeline \
  --args-json '{
    "operation": "text_edit",
    "input_audio_path": "/path/to/input.wav",
    "output_path": "/data/xedit/runs/text_edit.wav",
    "source_text": "开发边界将作为城市发展的刚性约定",
    "instruction_st": "<sub targ=\"城市边界\">开发边界</sub>将作为城市发展的刚性约定",
    "language": "zh",
    "evaluate": false
  }'
```

Pitch or speed edit through the shared prosody pipeline:

```bash
python examples/run_tool_call.py \
  --tool edit_data_pipeline \
  --args-json '{
    "operation": "pitch_rate_edit",
    "input_audio_path": "/path/to/input.wav",
    "output_path": "/data/xedit/runs/rate_edit.wav",
    "source_text": "开发边界将作为城市发展的刚性约定",
    "instruction_st": "<rate, factor=1.2>开发边界</rate>将作为城市发展的刚性约定",
    "language": "zh",
    "evaluate": false
  }'
```

`pitch_rate_edit` accepts only `<pitch, semitones=N>...</pitch>` and
`<rate, factor=F>...</rate>` labels in `instruction_st`; use `<rate>`, not
`<speed>`, and do not send pitch-contour/prosody XML tags.

`text_edit` and `pitch_rate_edit` both call `/execute` and decode
`response["audio"]["data"]` into `output_path`. The wrapper deliberately sends
only service-supported fields; the pipeline services reject unknown fields such
as `operations` or `edit_operations`. `pitch_rate_edit` keeps the stable
X-Editor operation name while targeting the latest `prosody_edit_pipeline`
service; wrapper-only aliases such as `prosody_edit` are normalized to this
operation.

Emotion/style edit through the emotion pipeline service:

```bash
python examples/run_tool_call.py \
  --tool edit_data_pipeline \
  --args-json '{
    "operation": "emotion_style_edit",
    "input_audio_path": "/path/to/input.wav",
    "output_path": "/data/xedit/runs/emotion_edit.wav",
    "source_text": "开发边界将作为城市发展的刚性约定",
    "instruction_st": "<emo type=\"happy\" level=\"2\">开发边界将作为城市发展的刚性约定</emo>",
    "language": "zh",
    "evaluate": false
  }'
```

For `emotion_style_edit`, `instruction_st` must preserve the stripped
`source_text`; only target emotion/style spans should be tagged. Unchanged spans
stay plain. Supported labels are categorical emotion tags such as
`<emo type="happy" level="2">...</emo>` where the emotion is one of `happy`,
`angry`, `sad`, `afraid`, `disgusted`, `melancholic`, `surprised`, or `calm`,
with an explicit supported intensity. A description may accompany the categorical
choice, but a bare `<emo desc="style description">...</emo>` does not specify an
executable control and is rejected instead of defaulting to a different emotion.
Do not route rap, singing, accent/dialect conversion, or channel effects here.

Pause edit through the pause pipeline service:

```bash
python examples/run_tool_call.py \
  --tool edit_data_pipeline \
  --args-json '{
    "operation": "pause_edit",
    "input_audio_path": "/path/to/input.wav",
    "output_path": "/data/xedit/runs/pause_edit.wav",
    "source_text": "Monday, there is gonna be haze, but Tuesday, look for thunderstorms.",
    "instruction_st": "Monday,<pause act=\"ins\" level=\"2\"/> there is gonna be haze, but Tuesday, look for thunderstorms.",
    "language": "en",
    "evaluate": false,
    "persist": true
  }'
```

For `pause_edit`, `instruction_st` is point-based and must strip back to the
same `source_text`. Pause levels map to short, medium, and long insertions.
The X-Editor default URL uses port `8038` so it can run alongside
`emotion_style_edit` on `8037`.

`audio_assembly` does not accept `instruction_st`; use `edit_tagged_text` with
`<target>...</target>` spans plus `target_audio_path` and source/target time
spans. `speech_rate`, `volume_normalize`, `pitch_shift`, and `boundary_smooth`
are parameter-based operations and must not receive XML-style labels.

## Legacy Microservice Example

```bash
python examples/run_tool_call.py \
  --tool edit_data_pipeline \
  --args-json '{
    "operation": "speech_rate",
    "input_audio_path": "/path/to/input.wav",
    "output_path": "/data/xedit/runs/rate.wav",
    "rate": 1.2,
    "method": "wsola+indextts2",
    "text": "full transcript from perception"
  }'
```

For `speech_rate`, the wrapper defaults to `wsola+indextts2`: first WSOLA
stretches the waveform to the target duration, then IndexTTS2 re-synthesizes
the stretched speech for better naturalness. `text` is required for this path.
If `speaker_audio_path` is omitted, the wrapper uses `input_audio_path` as the
speaker reference, matching the edit-data-pipeline rate editor's default usage.

## Service Launch Reference

From the X·Edit repository root, first prepare the environment and model assets
using [deployment instructions](../../../../docs/DEPLOYMENT.md). After configuring
and installing the stack, start the required services and their dependencies:

```bash
python deployment/stack.py --data /data/xedit start edit_text edit_prosody edit_emotion edit_pause
python deployment/stack.py --data /data/xedit export-env
```

Use your own deployment directory in place of `/data/xedit`. Load its generated
client environment into the controller. The stack starts the service adapters in
`deployment/pipeline_service.py`; the old development-branch shell launchers are
not part of this release. Service dependencies and GPU placement are declared in
`deployment/stack.json`. To use an already deployed service, set its explicit
`EDIT_PIPELINE_*_URL` instead.
