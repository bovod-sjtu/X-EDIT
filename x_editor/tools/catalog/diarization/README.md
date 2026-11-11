# X-Editor Diarization Backend

This backend isolates DiariZen / pyannote dependencies from the main X-Editor
framework environment. `SpeechPerceptionClient` calls it over HTTP by default.

## Setup

```bash
cd x_editor/tools/catalog/diarization
./setup.sh
```

`setup.sh` installs the service wrapper with `uv sync --inexact`, then installs DiariZen
inference dependencies and its bundled `pyannote-audio` fork into this
directory's `.venv`. It intentionally does not install DiariZen's full
training/evaluation/notebook requirements.

The isolated env pins `torch==2.8.0` and `torchaudio==2.8.0` because this
DiariZen/pyannote fork expects `torchaudio.AudioMetaData`, which is absent in
newer torchaudio releases.

Set `DIARIZEN_REPO` if the local clone is not at
`/path/to/workspace/DiariZen`.

## Run

```bash
cd x_editor/tools/catalog/diarization
uv run --project . python service.py --port 8041
```

X-Editor uses `X_EDITOR_DIARIZATION_URL`, defaulting to
`http://127.0.0.1:8041`. Local in-process fallback is disabled by default; set
`X_EDITOR_ALLOW_LOCAL_DIARIZATION_FALLBACK=1` only for debugging.
