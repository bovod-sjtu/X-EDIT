# dots.tts.edit model backend

The default backend runs the independently installed dots.tts.edit V1.1 model.
It accepts the same tagged-edit tool inputs as the pipeline; it does not run F5,
pipeline segmentation, or boundary smoothing behind the scenes.

## Versions

- Runtime: [dots.tts](https://github.com/studio-dots-ai/dots.tts) at
  `4cf7d6b13f42066f97784e7d7cf2a9ed5f956997`.
- Weights: [dots-studio/dots.tts.edit](https://huggingface.co/dots-studio/dots.tts.edit)
  at `bef1e97b4d942c62843b92c30a68961d1902b6ec` (V1.1).
- Environment: Python 3.11.15 and `deployment/requirements/dots_tts_edit.lock`.

These are separate source and weight versions. Updating one does not implicitly
update the other. The model card declares `license: other` and refers to the
project; the pinned project README declares Apache-2.0 for code and released
checkpoints. This discrepancy remains in the license inventory for review.
X·Edit does not relicense these assets or attest commercial authorization.

## Install independently

Run from the X·Edit source root. Review upstream terms before obtaining files.
The commands below are explicit operator actions; the controller does not
download or install the model when a request arrives.

```bash
git clone https://github.com/studio-dots-ai/dots.tts.git /data/dots-tts-source
git -C /data/dots-tts-source checkout --detach 4cf7d6b13f42066f97784e7d7cf2a9ed5f956997
uv venv /data/dots-edit-env --python 3.11.15
uv pip sync --python /data/dots-edit-env/bin/python deployment/requirements/dots_tts_edit.lock \
  --index https://pypi.org/simple --default-index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
uv pip install --python /data/dots-edit-env/bin/python --no-deps -e /data/dots-tts-source
uv pip check --python /data/dots-edit-env/bin/python
/data/dots-edit-env/bin/hf download dots-studio/dots.tts.edit \
  --revision bef1e97b4d942c62843b92c30a68961d1902b6ec --local-dir /data/dots-edit-model
/data/dots-edit-env/bin/python deployment/check_dots_runtime.py
```

The runtime check uses synthetic tensors and a stub model. It checks control flow,
not GPU inference. The dependency lock targets this HTTP adapter; the upstream
browser playground is not qualified by these checks.

## Start the model service

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 /data/dots-edit-env/bin/python \
  -m x_editor.tools.catalog.edit_data_pipeline.dots_service \
  --model-path /data/dots-edit-model \
  --model-revision bef1e97b4d942c62843b92c30a68961d1902b6ec \
  --source-revision 4cf7d6b13f42066f97784e7d7cf2a9ed5f956997 \
  --host 127.0.0.1 --port 8049
```

Only an existing local model directory is accepted; loading is offline.
Revision arguments identify the operator's deployment claim, not verification
of arbitrary local files. Keep the download receipt and inspect the actual
installed source.

The service handles one GPU request at a time and returns 429 when busy.
`--optimize` explicitly enables upstream optimization; freeze this setting in
comparisons. For remote access, configure authentication, request limits, and TLS.

## Connect X·Edit

Set `DOTS_TTS_EDIT_URL=http://127.0.0.1:8049` in `.env` and select:

```yaml
schema_version: x-edit.tools/v1
edit_backend: dots_tts_edit
enabled_tools: [artifact_ops, edit_data_pipeline, asr, aligner, omni]
```

Install the selected MCP clients as described in [module deployment](MODULE_DEPLOYMENT.md).
Their environment is separate from the GPU model environment. For the optional
stack launcher, register an existing service with
`--external dots_tts_edit=http://host:8049`; it is not installed or stopped by
the managed stack.

## Interface and limits

| Operation | Tags | Endpoint |
| --- | --- | --- |
| `text_edit` | `sub`, `ins`, `del` | `POST /execute` |
| `emotion_style_edit` | `emo` | `POST /execute` |
| `pitch_rate_edit` | `pitch`, `rate` | `POST /execute` |
| `pause_edit` | `pause` | `POST /execute` |

Audio is sent as base64, not a client-local path. The adapter passes
`instruction_st` to upstream `generate_edit(instruction=...)`. It validates
the source transcript and optional target text; inconsistent inputs return 422
rather than being silently rewritten.

Defaults remain `num_steps=10`, `guidance_scale=1.2`, `use_xvector="auto"`,
`speaker_scale=1.5`, and `seed=42`. The native length limit is 500 generation
positions; source and output share its budget. Over-limit inputs fail rather
than being truncated. Pipeline-only parameters are reported as ignored.

Selecting V1.1 does not expose every upstream playground feature: the framework
retains the four operation contracts above. Other tools such as voice conversion,
separation, and enhancement remain independently configured. There is no automatic
fallback to the pipeline or download of F5 on model failure.

See [validation](VALIDATION.md) for actual operation checks. Successful transport
and a readable waveform are not proof of perceptual quality or a new benchmark score.
