# Resource portability audit — 2026-10-15

This update removes developer-machine assumptions from the final X·Edit release.
It does not change the Reasoner, planning policies, edit selection, or evaluation
algorithms. The anonymous repository and the external pipeline pin are unchanged.

## Corrections

| Area | Previous dependency | Current contract |
| --- | --- | --- |
| Accent source profiles | Three dataset configurations, including private cluster source lists | Removed; operators supply their own reviewed manifest |
| Accent data | Packaged data directory suggested as the library store | No populated records or clips; only a source README remains |
| Accent retrieval | Search across shared checkouts and choose a matching basename | Resolve relative paths against their manifest; absolute paths stay exact |
| Accent construction | Private mining workflow and implicit Parquet dependency | Offline JSONL builder and validator using declared controller dependencies |
| Candidate miner | Implicit dataset paths; UTMOS pointed at the TTS port | Explicit input and UTMOS endpoint; optional Parquet dependency reported clearly |
| Gemini compatibility client | Enterprise gateway as default endpoint | Explicit `base_url` or `X_EDITOR_GEMINI_BASE_URL`; missing endpoint fails before transport |
| Speaker/voice services | Automatic developer-worktree asset discovery | Explicit source, model and interpreter settings; no shared-worktree probing |
| Service launch configuration | Placeholder repository paths and obsolete pipeline launchers | Required environment settings, current local ports, and managed-stack instructions |
| DiariZen setup | Developer path, then unpinned Git install fallback | Explicit source or checkout prepared by the pinned dependency fetcher |
| Packaging | User data could enter a source/package build | Accent data excluded from Git additions, package discovery, wheel data and sdist |

The canonical label definitions remain in `x_editor/accent_library/labels.py`.
The redundant YAML label copy formerly under `data/` has been removed; it was
metadata, not a reference-audio database. No tracked reference waveforms were
present before this update, and no operator-owned dataset was deleted.

## Using an existing installation

- Follow the [accent library guide](../x_editor/accent_library/README.md). Existing
  Parquet libraries must be selected explicitly with
  `X_EDITOR_ACCENT_LIBRARY_PATH`; they are not discovered in old checkouts.
- Move the complete library folder, including audio, then run its validator.
  A stale absolute reference must be corrected in the manifest. It will not be
  redirected to another recording with the same filename.
- Load the managed stack's generated client environment, or supply service
  source/model/interpreter settings explicitly. Shared audio paths must be
  visible at the same absolute mount point to the controller and services.
- The optional legacy Gemini adapter still uses a generateContent payload and
  `api-key` header. Configure a compatible gateway; this change does not claim
  compatibility with every public API. The main Qwen stack is unchanged.

## Verification and limits

Nineteen focused local checks cover construction, relocation, exact path binding,
invalid reference metadata/audio, configuration precedence, absent-endpoint
failure, and explicit speaker-service configuration. Their generated sine-wave
fixtures test file contracts only; they are not accent or speech-quality evidence.
Both wheel and source distributions were inspected for the builder, its guide,
and exclusion of reference data. A populated-data packaging probe additionally
checks that locally created library files cannot enter either artifact.

`python deployment/audit.py` checks source hashes, syntax, the pinned clean
pipeline, service references, absent bundled reference data, and common private
path/endpoint patterns in X·Edit runtime sources. A separate read-only scan of
the pinned pipeline's tracked Python/shell/config files found no matching cluster
mount, shared-worktree, or former gateway strings. This is a scoped static check,
not a guarantee about every upstream SDK or a complete secrets audit.

No model weights were downloaded and no GPU inference or benchmark was rerun
for this update. Library import does not certify accent labels, transcripts,
voice-conversion quality or source permissions. The selected pipeline pin still
requires authorized source access. Current upstream license and revision details
are in [LICENSES.md](LICENSES.md#pipeline-source-access).
