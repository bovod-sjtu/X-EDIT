# Select tools and capabilities

Set `tools: tools.yaml` in `runtime.yaml`. List only tools you have deployed,
configured, and are authorized to use:

```yaml
schema_version: x-edit.tools/v1
edit_backend: dots_tts_edit
enabled_tools:
  - artifact_ops
  - edit_data_pipeline
  - asr
  - aligner
  - omni
```

This minimal selection still needs Planner/Reasoner endpoints and, in revision
mode, judge endpoints. `x-edit init ./configs` exports the same configuration.
The file selects capabilities; it does not download models or start services.

Only enabled operations reach Reasoner, Planner, and the execution registry.
Use the same file with `deployment/bootstrap.py --tools` to provision the
required services. See [deployment](DEPLOYMENT.md) for separate environments.

## Switch editing backends

`edit_data_pipeline` is the compatible tool name for tagged text, emotion,
pitch/rate, and pause editing. Set `edit_backend` to:

- `dots_tts_edit`: the default, independently deployed model.
- `edit_data_pipeline`: the separately deployed editing pipeline.

The input contract stays the same; capability cards are not duplicated.
The file overrides `X_EDITOR_EDIT_BACKEND` for that run and its MCP clients.
Reload configuration and start a new run after switching. Failures do not
silently fall back to another backend.

Numerical pitch/rate/volume, smoothing, and assembly are separate tools, not
automatically replaced by the model. Pipeline operations requiring F5 still
need the explicit [external-tool setup](EXTERNAL_TOOLS.md).

## Tool prerequisites

To use the broader selection, set `tools: tools.full.yaml` and deploy its
dependencies. Merely starting more services does not enable their capabilities.

| Selection | Required services or local implementation |
| --- | --- |
| `artifact_ops` | Local audio operations and FFmpeg; no model service |
| `edit_data_pipeline` | dots.tts.edit, or four tagged-edit pipeline services |
| `edit_pitch_shift`, `edit_speech_rate`, `edit_volume` | `pitch`, `rate`, `volume`; neural methods also need `tts_clone` and `indextts` |
| `edit_boundary_smooth`, `edit_audio_assembly` | `smoother`, `assembly`, alignment, VAD, and operator-installed F5 |
| `qwen3_tts` | Both `tts` (CustomVoice) and `tts_clone` (Base and Tokenizer) |
| `seed_vc` | Reference-conditioned voice conversion service |
| `auk` | AuK editing service |
| `asr` | Qwen3-ASR with the context-aware HTTP adapter |
| `aligner` | `aligner` gateway and `aligner_backend` |
| `omni` | Qwen3-Omni endpoint accepting audio |
| `flexsed` | FlexSED event localization; timed descriptions also need Omni |
| `diarization`, `sedicow_asr`, `moss_transcribe_diarize` | Separate diarization, English speaker-attributed ASR, and Chinese speaker-attributed ASR services |
| `speech_enhancement`, `speech_separation` | `enhancement` and `separation` services |
| `audio_effects`, `spatial_reverb` | Their isolated local MCP environments |

Composite operations need every participating tool. Reference-free voice conversion
needs Qwen3-TTS and Seed-VC; stress editing also needs assembly and ASR/alignment.
Intra-audio voice transfer needs Seed-VC and grounded diarization.
Foreground/background edits need separation and the requested editing tools.
Speaker perception retains its automatic language routing; one speaker service
does not replace the entire set.

Exact rules live in `x_editor/tools/selection.yaml`.
`bootstrap.py --phase plan --tools FILE` prints the dependency set.
Full selection includes F5-dependent smoothing/assembly even with the model backend.
Remove those entries if you do not provide F5.

## Verify the effective selection

```bash
x-edit profile --config configs/runtime.yaml
x-edit-tools --config configs/runtime.yaml --list-capabilities
x-edit doctor --config configs/runtime.yaml
x-edit doctor --config configs/runtime.yaml --live
```

`profile` shows operation cards used by Reasoner and Planner; `x-edit-tools`
shows tool-level cards. `doctor --live` checks selected services and the ASR
context contract. Enabled but unreachable services remain errors. A healthy
port does not prove an edit or composite workflow works; test a real request.

Selection narrows the base profile rather than adding operations outside it.
Unknown names, duplicates, invalid backends, and missing required observation
capabilities fail during configuration. The supplied observation policy needs
ASR, alignment, and Omni: removing them requires a compatible policy.
Nested calls to disabled tools return `tool_not_enabled`.

Each run saves the selection and effective profile in `runtime.json`.
Selection does not change prompts, sampling, or repair algorithms. A bound
development-prior snapshot is retained only when every compared operation remains
enabled; otherwise the snapshot and binding are disabled together with a warning.
Backend changes also require checking whether that evidence applies.
See [tool contracts](TOOL_CONTRACTS.md) for these boundaries.
