# Building an accent / dialect reference library

This package ships **code and label definitions only**. It contains no reference
recordings, populated manifests, private dataset lists, or cluster paths. Build
your own library from recordings you may use. The library is optional for ordinary
editing; accent conversion that retrieves a library reference requires it. An
explicit reference supplied by the caller is a separate supported input route.

## What the runtime needs

`runtime/accent_conversion.py` retrieves a reference by canonical accent, with
optional gender, age and language preferences. Seed-VC consumes its audio;
the Qwen3-TTS reference route also needs the **exact words spoken in that clip**.
The library provides reference material; it does not install these services.
Configure them using [deployment instructions](../../docs/DEPLOYMENT.md).

Before import, listen to every clip and check:

- One speaker, a correctly identified accent/dialect, intelligible speech, no
  overlapping voices, prominent BGM, clipping or heavy reverberation.
- Prefer 5–10 seconds of mono WAV/FLAC. This is a selection recommendation, not
  a universal model limit. Preserve the actual sample rate; service adapters
  handle model-specific resampling. Crop/downmix manually when appropriate.
- A verbatim transcript of the **reference**, not the user's desired edited
  text. Do not describe the accent in place of transcribing the speech.
- Known source and usage terms. Record them in `provenance`; importing does not
  grant permission or automatically verify a speaker's consent.

The builder checks files, non-silent finite waveforms, mono channels, duplicate
IDs, supported labels, transcripts, and metadata consistency. **It cannot
certify accent identity, transcription accuracy, usage rights, or output quality.**
Only put approved references in the runtime library: current retrieval scores
quality flags as preferences, rather than treating them as exclusion gates.

## 1. Install the controller and prepare a manifest

From the repository root, after `uv sync --locked`:

```bash
uv run --locked python -m x_editor.accent_library labels
```

Create a directory outside this repository, for example `/data/my-accent-input`,
and put recordings in its `recordings/` folder. Create `references.jsonl` with one
flat JSON object per line. The following is a **format example**, not supplied
audio; replace every value with your reviewed recording's metadata:

```json
{"id":"reference-001","canonical_accent":"British English","reference_audio_path":"recordings/reference-001.wav","transcript":"The actual words spoken in this recording.","gender":"unknown","age_group":"unknown","quality":{"usable_for_vc_reference":true,"usable_for_zeroshot_tts_reference":true},"provenance":{"source_uri":"user-recording:reference-001","license":"Permission for this intended use recorded by the operator"}}
```

Required by the builder: `id`, `canonical_accent`, `reference_audio_path`,
`transcript`, and `provenance.source_uri` / `provenance.license`. Optional fields
include `speaker_id`, `gender`, `age_group`, `language`, `locale`, `region`,
`quality`, and `metadata`. `duration` is in seconds and `sample_rate` in Hz; omit
them to measure from the file, or provide correct values to check. Use the
specific label (e.g. `Shanghai Dialect`), not the broad `Wu Chinese` family.

Relative audio paths resolve **only against the manifest's parent directory**.
Absolute paths are accepted exactly as written. Missing paths are not repaired
by searching another checkout. The runtime schema is also described in
[`accent_dialect_library_schema.yaml`](../tools/accent_dialect_library_schema.yaml).

## 2. Validate and build

```bash
uv run --locked python -m x_editor.accent_library validate \
  --manifest /data/my-accent-input/references.jsonl
uv run --locked python -m x_editor.accent_library build \
  --manifest /data/my-accent-input/references.jsonl \
  --output /data/my-accent-library
```

The output must be a **new directory**. Construction never overwrites an existing
library or changes source recordings. It writes mono PCM-16 WAV copies at the
original sample rate, measures their metadata, records file hashes, and creates:

```text
my-accent-library/
  accent_library.jsonl
  audio/
    00000-<id-hash>.wav
```

The generated manifest uses relative `audio/...` paths. Move the entire folder
together and validate at its destination. No GPU, ASR, Gemini, API key, or private
dataset is needed for these commands. Parquet remains readable for older
libraries if you install a compatible Parquet engine separately; JSONL needs none.

## 3. Connect it to X·Edit

Set this in the controller's environment or its explicitly loaded `.env`:

```bash
export X_EDITOR_ACCENT_LIBRARY_PATH=/data/my-accent-library/accent_library.jsonl
uv run --locked python -m x_editor.accent_library validate \
  --manifest "$X_EDITOR_ACCENT_LIBRARY_PATH"
```

Precedence: an explicit Python `load(path)` argument →
`X_EDITOR_ACCENT_LIBRARY_PATH` →
`X_EDITOR_ACCENT_LIBRARY_DATA_DIR/accent_library.jsonl` →
`${XDG_DATA_HOME:-~/.local/share}/x-edit/accent_library/accent_library.jsonl`.
The package's `data/` directory is never a writable runtime store. An older
Parquet library must be selected explicitly with `X_EDITOR_ACCENT_LIBRARY_PATH`.

Controller and all path-based services must see the **same absolute mounted
directory**. On separate hosts, mount/copy the library to the same path before
starting services. No remote upload or private cluster mount is inferred.

An empty library or absent target accent returns an explicit retrieval error.
Check the configured path, canonical label, reference visibility, and tool-service
readiness. Library validation alone does not verify neural inference; run a short
accent edit and listen to its result after deploying Seed-VC / Qwen3-TTS.

## Optional candidate mining

`python -m x_editor.tools.accent_candidate_selection` is an optional legacy
candidate miner, not required for building a reviewed library. It requires
`pyarrow`, an explicit `--input` or `--config`, and separately configured services
for the checks you enable (Gemini via `GEMINI_API_KEY` and an explicit `X_EDITOR_GEMINI_BASE_URL`
generateContent-compatible gateway using the `api-key` header, ASR via
`--asr-url`, UTMOS via `--utmos-url`; UTMOS expects `POST /predict`). There is no
built-in dataset path or assumed UTMOS endpoint. Use `--help` for supported input
formats and selection thresholds. A `--dry-run` skips quality checks and is **not
reference qualification**.

Mining outputs candidate rows, not a ready-to-use library. Review selected audio,
translate each approved row to the flat schema above, attach source/usage
metadata and a checked transcript, then build. Former private source profiles
have been removed; provide your own dataset configuration outside the repository.

## Keeping user data out of the release

Keep manifests, reference clips, candidate caches and reports outside the source
tree. `data/` is empty except for its README and excluded from package artifacts;
Git ignores other files there. The labels in `labels.py` are code metadata,
not reference recordings. Do not commit datasets, generated library files,
credentials or private service addresses.
