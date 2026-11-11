# Release acceptance

> Historical acceptance records below describe their stated source versions.
> The 2026-10-11 pipeline migration has separate source and live-operation
> evidence in [PIPELINE_MIGRATION.md](PIPELINE_MIGRATION.md).

## Verified baseline

The initial cold-container qualification used CUDA 12.8.1 / Ubuntu 22.04 and
eight 80 GB L20Z GPUs. It installed 13 isolated environments and downloaded 31
pinned model repositories. Twenty-six services handled direct operation probes;
F5 and the alignment gateway handled real requests through the editing pipeline.
See [the recorded scope](COLD_START_VALIDATION.md). Installation repairs were
made during that qualification; it is not evidence that the original archive
installed without intervention.

The separate full-source follow-up submitted on 2026-09-30 used a frozen archive,
fresh environments/model downloads, public service probes and the 24-task
synthetic panel described below. That panel is outside the two-backend
qualification reported here; its quality acceptance must be evaluated separately.

The completed panel has **24/24 task records and 23/24 deliveries**. Background-
preserving speech attenuation failed because no unique primary graph output was
specified. The pause task delivered 0.2 s instead of the requested 0.3 s. See
[sample_validation.json](sample_validation.json). These are execution results
and partial objective checks, not 23 quality passes.

## Repeat from the final source archive

### External backend follow-up (2026-10-01)

IndexTTS and Seed-VC now use official pinned SDK checkouts and separately
licensed adapters. Local GPU regression compared the previous vendored
implementation with the new loader under the same inputs, checkpoints,
environment, device and seeds. Three IndexTTS requests (emotion with token
output, cached continuation, independent segmented emotion) and two Seed-VC
requests (explicit seed 0 and default seed) returned identical decoded audio
sample hashes before and after. Six structural checks passed, covering exact
source compatibility, invalid-source rejection, in-memory imports, path bounds,
deployment wiring and clean upstream checkouts. Both source pins were fetched
from their official repositories using the public dependency installer.

That local comparison reused existing environments/assets. A subsequent fresh
CUDA 12.8.1 / Ubuntu 22.04 deployment on two L20Z GPUs installed both environments,
downloaded all seven required model repositories and passed the same five actual
requests. All five decoded sample hashes matched the local regression; both
upstream checkouts remained clean. The successful worker took 9 minutes 1 second,
excluding queue/image preparation and the earlier failed download attempt.

Cold installation exposed a CUDA-wheel download-host timeout. The installer now
prefers PyPI for equal-version wheels while retaining the tested CUDA builds;
the complete installed package inventories stayed unchanged. See
[backend_cold_validation.json](backend_cold_validation.json).

This follow-up qualifies the two changed backends, not the entire framework or
subjective audio quality. Optional CUDA kernels/acceleration, pitch-conditioned
Seed-VC, and aligner-mediated multi-tag continuation were not exercised. The
source-isolation change does not establish that every backend is available
under Apache-2.0; see each backend's notices.
The machine-readable scope is in [backend_source_validation.json](backend_source_validation.json).

Use a fresh container, new writable data/environment directories, and only the
distributed source. Do not copy a development virtualenv or point to an existing
model server. Follow [DEPLOYMENT.md](DEPLOYMENT.md) for OS installation, pinned
fetch, environment installation, service startup and CUDA checks. Preserve the
archive hash, commands, dependency receipts and failures.

After service startup, the following additional operation probe creates its own
synthetic English speech using the deployed TTS. No private audio is needed:

```bash
/local-ssd/xedit-envs/services/volume/bin/python deployment/qualify_services.py \
  --data /data/xedit --output /data/checks/services-001

/local-ssd/xedit-envs/services/volume/bin/python deployment/qualify.py \
  --data /data/xedit --mode dsp --output /data/checks/dsp-001

source /data/xedit/client.env
uv run --locked x-edit edit --config configs/runtime.yaml \
  --audio /data/checks/services-001/source.wav \
  --instruction 'Lower the recording by 6 dB, preserving its words and duration.' \
  --mode forward --output /data/runs/gain-001
```

The service probe uses a fixed diagnostic sentence and simple operations to
test transport and inference. Its default order creates the Chinese reference
before voice conversion and Chinese speaker transcription. A subset containing
`seed_vc` or `moss` must also include `tts` first. It stores requests, responses,
waveforms, per-operation timing and failures; audio validity is not a quality
score. F5 and gateway dependency calls must also be checked in service logs.

An acceptance run should additionally exercise text replacement, a two-turn
edit using the preceding output, a separate voice-reference asset, and revision
after an actual or controlled execution failure. Controlled fault injection
belongs in an external validation harness, never in the runtime. Keep an
infrastructure failure separate from a delivered but imperfect edit.

## Quality and capacity

The release quality panel is 24 tasks: four each for text, numerical prosody,
emotion/voice attributes, local/speaker editing, multi-asset/background editing,
and multi-turn/revision. Use newly generated or appropriately licensed inputs,
retain every task including failures, and freeze expected requirements before
running. Check delivery, objective effect measurements, non-target preservation,
and artifacts separately. Ten to twelve representative outputs can then receive
human listening. This is a release regression panel, not a new benchmark claim.

Long-input and concurrency qualification remains separate: measure supported
10/30/60/120-second requests and concurrency 1/2/4 on a declared service profile.
Report latency, memory, failure rates and input limits. A clear refusal outside
a tool's declared limits is not an inference success. A complete GPU Docker
build and lower-memory deployment are not yet qualified by the initial test.

## Research reproduction

The source release preserves the editing runtime and its default configuration;
research launchers, benchmark data and private traces are excluded. It does not
claim to reproduce historical paper tables with newer services or boundary
repairs. A paper reproduction bundle needs its own frozen configuration mapping,
dataset acquisition/adapters and independent scorer; see
[BENCHMARK_REPRODUCTION.md](BENCHMARK_REPRODUCTION.md).
