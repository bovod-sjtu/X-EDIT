# Release validation — 2026-11-17

This records deployment evidence, not editing-quality scores or legal approval.
Models, audio, detailed logs, and the integration harness are kept outside the
public repository.

## Real editing operations

Both backends were called through the current X·Edit operation normalizer,
real MCP client process, HTTP adapter, and GPU-backed services. No model
responses were mocked in these checks.

The diagnostic recording says “The next train will leave at six fifteen.”
It was generated for service qualification, not taken from a benchmark.

| Request | dots.tts.edit V1.1 | Public pipeline |
| --- | --- | --- |
| Replace “train” with “bus” | Returned audio, 48 kHz | Returned audio, 24 kHz |
| Make the sentence happy | Returned audio, 48 kHz | Returned audio, 24 kHz |
| Raise “six fifteen” by 2 semitones | Returned audio, 48 kHz | Returned audio, 24 kHz |
| Speed up “six fifteen” by a factor of 1.25 | Returned audio, 48 kHz | Returned audio, 24 kHz |
| Insert a level-2 pause after “train” | Returned audio, 48 kHz | Returned audio, 24 kHz |

All ten calls completed and saved nonempty, finite, non-silent waveforms.
This verifies execution and audio transport; it does not establish that every
edit sounds correct or preserves the speaker/background.

### Model deployment

- Runtime: `4cf7d6b`; model revision: `bef1e97b` (full pins in the manifest).
- One NVIDIA L20Z; bfloat16; optimization disabled.
- Parameters unchanged: 10 steps, guidance 1.2, speaker scale 1.5, x-vector
  selection `auto`, seed 42, length limit 500.
- The new upstream source and newly acquired V1.1 checkpoint were used with an
  existing isolated service dependency environment. This was not a fresh-host
  installation of every tool.

### Pipeline deployment

- Public pipeline revision: `ba270475`; source remained unmodified.
- Started 14 run-owned services: four editing APIs, alignment backend/gateway,
  pitch, rate, volume, VAD, smoothing, assembly, IndexTTS, and an independently
  installed F5 service.
- Existing pinned model assets and isolated dependency environments were reused.
  The current public pipeline supplied the orchestration and managed adapters.
- F5 was explicitly enabled for this private diagnostic run using an existing
  operator installation outside the release tree. The release helper did not
  fetch or install its SDK or weights.
- All test-owned processes were stopped after the checks. Other deployments
  were not stopped or modified.

## Interface and source checks

- `deployment/check_pipeline_contracts.py` validates all four request schemas,
  the exact startup arguments and loaded upstream defaults, emotion/pause
  transport, deterministic Seed-VC parameters, and artifact containment.
  Startup dispatch is intercepted in this check; it is not the real-operation
  evidence in the table above.
- `deployment/check_dots_runtime.py` exercises streaming/non-streaming control
  flow with profiling on/off, using a stub model and synthetic tensors.
- Source/content audits retain the no-model, no-audio, no-credential, and
  no-cluster-configuration boundary. The reviewed framework PNG is the only
  permitted binary documentation asset.

## Limits

This release has not rerun MMAE, SpeechEditBench, or an official scorer.
The checks do not qualify a complete cold-host deployment, all optional tools,
TED-backed styles, long recordings, multi-turn workflows, concurrent throughput,
or a prebuilt container. They do not prove bit-identical outputs to earlier
upstream versions. Keep historical benchmark results attached to their original
versions and use a new output directory when validating this release.
