# SeamlessExpressive S2ST Tool

This catalog entry integrates Meta SeamlessExpressive and SeamlessM4T v2 as an
X-Editor speech-to-speech translation tool.

## Capability Boundary

- Primary task: speech-to-speech translation (S2ST).
- Expressive path: `expressivity_predict` with `seamless_expressivity` and
  `vocoder_pretssel`.
- Non-expressive path: `m4t_predict` with `seamlessM4T_v2_large` and
  `vocoder_v2`.
- SeamlessExpressive is used when vocal style, speech rate, or pauses should be
  preserved. This is a model capability claim, not a hard guarantee.
- SeamlessM4T v2 fallback does not preserve vocal style/prosody with the same
  objective; fallback must be explicit via `allow_fallback_to_m4t=true`.

## Access and License Notes

- SeamlessM4T v2 model weights are listed by Meta/Hugging Face under
  CC-BY-NC 4.0.
- SeamlessExpressive artifacts require Meta approval through the Seamless
  downloads request flow and are distributed under the Seamless license.
- Do not commit downloaded model weights or private access tokens.

Sources checked during integration:

- https://github.com/facebookresearch/seamless_communication
- https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md
- https://github.com/facebookresearch/seamless_communication/blob/main/docs/expressive/README.md
- https://huggingface.co/facebook/seamless-m4t-v2-large
- https://huggingface.co/facebook/seamless-expressive
- https://github.com/facebookresearch/seamless_communication/blob/main/SEAMLESS_LICENSE

## Install Wrapper Environment

```bash
cd x_editor/tools/catalog/seamless_expressive_s2st
./setup.sh
```

This installs only the lightweight X-Editor wrapper dependencies.

## Enable Real Seamless Runtime

```bash
cd x_editor/tools/catalog/seamless_expressive_s2st
git clone https://github.com/facebookresearch/seamless_communication.git /path/to/seamless_communication
.venv/bin/python -m pip install -e /path/to/seamless_communication

export SEAMLESS_EXPRESSIVE_MODEL_DIR=/path/to/approved/SeamlessExpressive/model
export SEAMLESS_S2ST_SERVICE_URL=http://127.0.0.1:8039
```

`SEAMLESS_EXPRESSIVE_MODEL_DIR` is required for expressive mode. Without it,
the service reports a clear 503 availability error instead of fabricating an
audio result.

## Run Service

```bash
cd x_editor/tools/catalog/seamless_expressive_s2st
.venv/bin/python service.py --host 127.0.0.1 --port 8039
```

Codex worktree may not contain this ignored `.venv`. The MCP/service command resolver first honors
`X_EDITOR_SEAMLESS_EXPRESSIVE_S2ST_PYTHON`, then `X_EDITOR_TOOL_PYTHON`, and otherwise searches a shared
checkout such as `/path/to/workspace/X-Editor` for the matching tool `.venv/bin/python`.

Health check:

```bash
curl http://127.0.0.1:8039/health
```

## MCP Tool Call Shape

```json
{
  "input_audio_path": "/tmp/input.wav",
  "target_language": "fra",
  "output_path": "/tmp/output.wav",
  "mode": "expressive",
  "preserve_vocal_style": true,
  "preserve_speech_rate": true,
  "preserve_pauses": true,
  "allow_fallback_to_m4t": false
}
```

When the runtime is unavailable, the MCP call fails with the service's
availability payload. This is intentional so planner/executor traces can report
license/access/GPU/setup blockers truthfully.
