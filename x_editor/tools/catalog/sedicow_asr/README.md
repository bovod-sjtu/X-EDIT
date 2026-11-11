# SE-DiCoW Speaker-Attributed ASR Backend

This service isolates SE-DiCoW / TS-ASR-Whisper dependencies from the main
X-Editor runtime. The perception subagent treats it as speaker-attribution
evidence, not final truth.

Speaker IDs are scoped to the provider and recording. Request `speaker_id: null`
to obtain one complete native report. A targeted request must declare
`speaker_namespace: "sedicow_asr"` and use an ID from that recording's native
report. External Diarizen hints do not bind IDs in the official pipeline.
Missing IDs and unknown namespaces return explicit HTTP 422 errors, never an
implicit full transcript. Responses expose `speaker_namespace`,
`selection_status` and the raw native report. The collector uses one full call
and retains external diarization separately, without integer-offset mappings.

Prefer the current Hugging Face model id `BUT-FIT/SE-DiCoW`. The similarly
named `BUT-FIT/SE_DiCoW` repository is an older model card and should only be
used when explicitly requested for comparison.

## Setup

```bash
cd x_editor/tools/catalog/sedicow_asr
./setup.sh
python download_official_assets.py
```

The wrapper is intentionally thin. Real model inference is provided by the
official BUTSpeechFIT DiCoW checkout through `SEDICOW_TRANSCRIBE_COMMAND`.
`download_official_assets.py` clones both official repos and downloads the
default Hugging Face snapshots into ignored local directories:

- `vendor/DiCoW`
- `vendor/TS-ASR-Whisper`
- `checkpoints/hf/BUT-FIT__SE-DiCoW`
- `checkpoints/hf/BUT-FIT__DiCoW_v3_3`
- `checkpoints/hf/BUT-FIT__DiCoW_v3_2`
- `checkpoints/hf/BUT-FIT__diarizen-wavlm-large-s80-md`

These are operator-acquired files, excluded from Git and release packages.
Configure `SEDICOW_REPO` explicitly, or supply your own inference command:

```bash
export SEDICOW_REPO=$PWD/vendor/DiCoW
export SEDICOW_TRANSCRIBE_COMMAND='python '$PWD'/sedicow_infer.py --request-json {request_json}'
uv run --project . python service.py --port 8046
```

For the reproducible release, use `deployment/stack.py` from the repository
root; it configures the pinned source, model cache and isolated interpreter.
For manual deployment set `SEDICOW_REPO`, `DIARIZEN_REPO`,
`SEDICOW_INFERENCE_PYTHON` and the model/cache paths explicitly. Source and cache
locations are not inferred from development worktrees or shared cluster folders.
See [deployment instructions](../../../../docs/DEPLOYMENT.md).

The command receives a JSON request file path through `{request_json}` and must
print a JSON object to stdout. `sedicow_infer.py` emits:

```json
{
  "segments": [
    {"speaker_id": "0", "start_sec": 0.0, "end_sec": 1.2, "text": "hello"}
  ],
  "text": "hello",
  "language": "en"
}
```

If neither an explicit source configuration nor `SEDICOW_TRANSCRIBE_COMMAND` is available,
`/health` reports `not_ready` and requests return a clear error. This keeps
benchmark traces honest when SE-DiCoW weights or enrollment inputs are
unavailable.

## Run

```bash
cd x_editor/tools/catalog/sedicow_asr
uv run --project . python service.py --host 127.0.0.1 --port 8046
```
