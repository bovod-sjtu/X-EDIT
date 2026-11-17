# Public editing pipeline

X·Edit uses the public
[studio-dots-ai/dots.tts.edit-data-pipeline](https://github.com/studio-dots-ai/dots.tts.edit-data-pipeline)
repository at `ba270475fdb3c721db381463d73d3fc43727c554`.
The historical path `third_party/dots_tts_edit` contains this pipeline,
**not** the dots.tts.edit model runtime.

The gitlink records a dependency; source archives and wheels exclude the
checked-out upstream tree. Upstream code, SDKs, patches, and weights retain
their own license terms.

## Fetch only what you need

```bash
python deployment/pipeline.py fetch
python deployment/pipeline.py check
```

The first command fetches only the pinned outer pipeline. For individual managed
SDKs, append names such as `qwen3-asr` or `index-tts`.
`fetch all` fetches the five managed SDKs, **not F5**.
Do not use `git submodule update --init --recursive` as an installation shortcut.

For an existing checkout, inspect local changes before updating the gitlink:

```bash
git submodule sync -- third_party/dots_tts_edit
git submodule update --init -- third_party/dots_tts_edit
python deployment/pipeline.py check
```

The helper refuses a dirty checkout or a different pin instead of resetting it.

## Deploy and select

Use [deployment](DEPLOYMENT.md) to create separate uv environments
for the selected services. `deployment/stack.json` declares their dependency
graph, ports, environment locks, and launch arguments. `bootstrap.py --tools`
uses the same tool selection as the controller.

Set `edit_backend: edit_data_pipeline` in `configs/tools.yaml` and configure:

| Operation | Environment variable | Local default |
| --- | --- | --- |
| Text | `EDIT_PIPELINE_TEXT_TAGGED_URL` | `http://127.0.0.1:8035` |
| Pitch/rate | `EDIT_PIPELINE_PITCH_RATE_URL` | `http://127.0.0.1:8036` |
| Emotion/style | `EDIT_PIPELINE_EMOTION_STYLE_URL` | `http://127.0.0.1:8037` |
| Pause | `EDIT_PIPELINE_PAUSE_EDIT_URL` | `http://127.0.0.1:8038` |

F5 is a separately installed external service, disabled by default. Follow
[external tool setup](EXTERNAL_TOOLS.md) before enabling operations that need it.
Managed startup requires an explicit `--external f5=http://host:port` binding;
it neither downloads nor starts F5. Its model-use restrictions still apply.

## Updated upstream configuration

The public pipeline requires `--pipeline-defaults` and
`--pipeline-parameters` at startup. The stack now passes the upstream service
configuration explicitly:

| Service | Configuration | Service overrides |
| --- | --- | --- |
| Text | `pipeline/text_edit/defaults` | `num_generations: 1` |
| Prosody | `pipeline/prosody_edit/defaults` | none |
| Emotion | `pipeline/emotion_style_edit/defaults` | `max_candidates: 3` |
| Pause | `pipeline/pause_edit/defaults` | `center_regen_second_pass: false` |

Removed startup flags are not sent to the new API. The wrapper verifies the pin
and launches the real upstream module without monkeypatching model or pipeline
code. Git/artifact compatibility now comes from upstream itself.

Keep `ARTIFACT_DIR` outside both source trees. Use a new stack data directory
after updating the manifest; existing deployment records deliberately reject
changed source/configuration. This preserves old runs rather than silently
changing their services.

## Validate

After starting the selected services, load their exported addresses and run:

```bash
source /data/xedit/client.env
.venv/bin/x-edit doctor --config configs/runtime.yaml --live
.venv/bin/x-edit edit --config configs/runtime.yaml \
  --audio /data/audio/source.wav \
  --instruction "Make this sentence sound happier. Keep the words and speaker." \
  --mode forward --output /data/xedit/runs/pipeline-edit
```

`doctor --live` checks connections and interfaces. The edit command runs the
selected workflow; inspect its audio and execution record for actual behavior.
Neither a health check nor a readable waveform proves perceptual quality.
