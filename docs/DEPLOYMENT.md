# Install from this source tree

The source distribution includes the controller, deployment helpers and exact
dependency pins. It can fetch edit-data-pipeline and its official SDK submodules
separately; it does not need a sibling development checkout. A **wheel contains
the controller**; use the source archive or Git checkout for GPU service
deployment. The pipeline currently requires authenticated repository access.
Model weights are downloaded separately, not copied into the release.

**The default tagged-edit backend is now the dots.tts.edit model, not this
pipeline stack.** Follow [model deployment and backend selection](DOTS_TTS_EDIT.md)
first. The instructions below provision the alternative pipeline and auxiliary
services; its historical directory name `third_party/dots_tts_edit` is not the
new dots.tts model implementation. Model operation does not require installing F5.

Select deployed tools in `configs/tools.yaml` before running the controller.
The same selection filters Planner/Reasoner operation cards, executor admission
and `doctor --live` checks; see [tool selection](TOOL_SELECTION.md). It declares
installed capabilities, not installation requests. Pass that same file to
`bootstrap.py --tools` to derive the required service closure. See the
[capability parity guide](SERVICE_CAPABILITIES.md) for the full profile, contextual
ASR and speaker services. The `all` commands below are inventory-wide examples,
not requirements of the shipped reduced selection; they require both F5 and dots
external bindings when starting the complete inventory.

For the shipped minimal selection, independently start dots.tts.edit first, then:

```bash
uv run --locked python deployment/bootstrap.py --data /data/xedit-model \
  --phase plan --tools configs/tools.yaml
uv run --locked python deployment/bootstrap.py --data /data/xedit-model \
  --phase all --tools configs/tools.yaml --external dots_tts_edit=http://127.0.0.1:8049
source /data/xedit-model/client.env
uv run --locked x-edit doctor --config configs/runtime.yaml --live
```

Review model licenses, resource placement and the plan before `--phase all`.
This minimal selection needs no F5. It still needs ASR, forced alignment,
Omni and Planner/Reasoner, not only the editing model.

## Environment layout and remote deployment

Start with the Chinese [module deployment guide](MODULE_DEPLOYMENT.md).
New deployments install one uv environment per service under
`<env-root>/services/<service>`, even when services use the same dependency lock.
Selected lightweight MCP clients live under `<env-root>/clients/<tool>`;
the controller stays in its own root `.venv`. Install it using the README first.
Existing deployment configurations retain their recorded layout. Do not mutate
running configurations; create a new data directory when migrating.

`--bind-address` explicitly changes the IPv4 listen address for new deployments.
The default is loopback; use firewall/access controls before exposing a service.
Remote endpoints use `--external NAME=URL`; all path-based services need the same
absolute audio and output paths on both machines. Generated `client.env` contains
local addresses, not guessed remote hostnames.

## External-tool prerequisite

F5 is operator-installed only. Read [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md) first.
Missing required setup fails before fetching other dependencies. SAM and
VoiceSculptor are not supported deployment names, optional group members or
opt-in integrations; only independent extension guidance remains. Their former
assets cannot be fetched through the active dependency manifest.

## Base host or GPU container

Use Linux x86-64, NVIDIA GPUs for neural inference, a compatible host driver,
FFmpeg/ffprobe, sox, libsndfile, Git/Git LFS and build tools. The tested service
runtimes use CUDA 12.8 (AuK uses CUDA 12.6). The controller uses Python 3.11.15;
most editing services use Python 3.10.12. Environments remain separate.

The optional image recipes build a pinned LGPL FFmpeg via `deployment/build_ffmpeg.sh`;
the OS FFmpeg package is not installed by these Dockerfiles. Preserve the complete
`/opt/xedit-ffmpeg` prefix, including corresponding source and license notices.
For manual installation, put `/opt/xedit-ffmpeg/bin` first on `PATH`.
See [dependency remediation](DEPENDENCY_REMEDIATION.md) for verification commands
and the separate model-license inventory. Images are not source-release deliverables.

CUDA 12.8 environment installation prefers PyPI when both official indexes
offer the same version. Best-match resolution still selects the tested `+cu128`
builds from the PyTorch index, including torchao/torchcodec pins without an
explicit local suffix. This avoids routing ordinary CUDA wheels through a
separate NVIDIA host without switching CUDA builds to plain builds.

In a dedicated Ubuntu/Debian container with root access:

```bash
bash deployment/install_os.sh
uv sync --locked --no-dev
```

Some GPU runtimes inject read-only GL driver files into the container. The OS
installer detects these exact mount paths and excludes them from that apt
invocation, preserving the injected driver instead of overwriting it. This avoids
dpkg's cross-device backup error; no persistent package exclusion or driver
unmount is performed. This path has passed on a clean CUDA 12.8.1 container.

Or build `docker build -f deployment/Dockerfile.gpu -t xedit-stack .`.
The GPU Dockerfile is provided but a complete clean Docker build has not been
qualified in this preparation. `deployment/Dockerfile` remains controller-only.
For a managed GPU job, use a CUDA development base image, run the same OS setup while building
it, mount the source and writable data, then use `deployment/service_entrypoint.sh`.
Configure storage mounts and credentials in your deployment environment.

Allow several hundred GB for the full model set, package cache, isolated
Python environments and outputs. A GPU environment can need over 20 GiB during
installation. Small `/tmp` volumes are unsuitable. Prefer local SSD for envs and
UV cache, and a persistent shared mount for models and audio. All path-based
services and the controller must see audio/output files at the same absolute
paths. The default services bind localhost; use an explicit authenticated gateway
and matching shared paths when deploying across hosts.

## Inspect, download, install, start

```bash
# Inspect only; no models are downloaded and no services start.
uv run --locked python deployment/bootstrap.py --data /data/xedit --phase plan all

# Download exact source commits and model revisions.
# For gated weights, configure an approved HF_TOKEN in the runtime environment.
uv run --locked python deployment/bootstrap.py --data /data/xedit --phase fetch --external f5=http://127.0.0.1:8001 --external dots_tts_edit=http://127.0.0.1:8049 all

# Install lightweight MCP clients and isolated backend envs; controller already installed.
uv run --locked python deployment/bootstrap.py --data /data/xedit \
  --env-root /local-ssd/xedit-envs --phase install --external f5=http://127.0.0.1:8001 --external dots_tts_edit=http://127.0.0.1:8049 all

# Review stack.json GPU placement before this step. Start serially and fail visibly.
uv run --locked python deployment/bootstrap.py --data /data/xedit --phase start all
source /data/xedit/client.env
uv run --locked x-edit doctor --config configs/runtime.yaml --live
```

Use the F5 binding only after independently installing an authorized compatible
service; the example does not install it. SAM and VoiceSculptor are not declared
services and cannot be enabled with `--external`. Smaller groups are
`cpu`, `speech`, `models`, `experts`, `editing`, `optional`; individual service
names also work. These are deployment groups, not a silent change to the agent's
allowed tools. Missing enabled services remain failures. `cpu` is useful for
first installing and checking numerical DSP. To expand a configured deployment,
fetch all required assets first and create a new deployment configuration; do
not mutate a running configuration or switch its model pins.

`--gpu SERVICE=VISIBLE_IDS` sets device placement at configuration time, e.g.
`--gpu planner=0,1,2,3 --gpu omni=7`. The default planner has tensor parallelism 4;
changing it requires an explicit manifest change. Default placement is a starting
point for eight large-memory GPUs, not a measured memory guarantee for all models
resident at once. Run optional services on additional GPUs or separate hosts if
needed. `--port-offset N` changes all service ports and generated client bindings.
Record source, model and hardware differences when comparing results.

For a prepared GPU image/source mount:

```bash
export XEDIT_DATA=/data/xedit
bash deployment/service_entrypoint.sh --env-root /local-ssd/xedit-envs --phase install --external f5=http://127.0.0.1:8001 --external dots_tts_edit=http://127.0.0.1:8049 all
bash deployment/service_entrypoint.sh serve all
```

`serve` stays in the foreground, fails if an owned service exits, and stops only
owned services on job termination. A one-shot `start` command by itself is not
a persistent service job. Submit caller inference in the same job or expose
endpoints through an explicitly configured network gateway.

## Separate configuration assets

RoFormer ancillary configuration/metadata is downloaded separately, not included
in Git or packages. `--phase fetch separation` handles this automatically. For a
manual installation use `python deployment/dependencies.py assets roformer`.
Downloads must match the manifest SHA256; changed local files are rejected.
Required asset checks also run before service startup. Downloading an asset does
not waive its original terms.

## Explicit external endpoints or existing weights

To reuse a service you own, configure its URL explicitly; the launcher neither
starts nor stops it:

```bash
python deployment/stack.py --data /data/xedit-custom configure \
  --assets /data/assets.json --env-root /local-ssd/xedit-envs \
  --external planner=http://planner-host:8040 --external omni=http://omni-host:8011
python deployment/stack.py --data /data/xedit-custom plan speech experts
python deployment/stack.py --data /data/xedit-custom install speech experts
python deployment/stack.py --data /data/xedit-custom start speech experts
python deployment/stack.py --data /data/xedit-custom export-env
```

The download helper writes `assets.json` with `{ "models": { "repo/name": {
"path": "/absolute/snapshot", "revision": "exact-sha" } } }`. A path-only map is
also accepted for explicit operator-supplied assets, but does **not** attest their
identity. For shared paths, upstream default-reference loaders are bound to the
pinned snapshot in this deployment's isolated HF cache. Conflicting existing
references/assets are rejected. Service inference uses offline HF loading.

Pins, model-card links and file filters are in `third_party/manifest.json`.
Source fetches preserve upstream submodule commits. The model helper refuses
unresolved revisions and merges successive download receipts without replacing
an existing pin. Source edits or manifest changes require a new deployment
configuration. `status` reports process identity and inference qualification
separately; `stop` stops only processes launched and still owned by that deployment.

## Included service families

| Family | Included implementation / model assets |
| --- | --- |
| Editing pipeline | Pinned external dots.tts.edit text, emotion, prosody, pause, numerical rate/pitch/volume, VAD, smoothing and audio assembly |
| ASR / alignment | Context-ASR gateway + pinned Qwen SDK; Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B |
| Text replacement | Pipeline's F5 HTTP wrapper + official F5 SDK, F5TTS_v1_Base and Vocos |
| Emotion | Pipeline's IndexTTS and TED-TTS adapters and pinned official SDKs; IndexTTS-2 plus MaskGCT, W2v-BERT, CAMPPlus and BigVGAN |
| Voice generation / conversion | Pipeline's Qwen3-TTS and Seed-VC wrappers, official SDKs and pinned model/auxiliary assets |
| Expert observation | FlexSED + CLAP text encoder; SE-DiCoW + DiariZen; MOSS-Transcribe-Diarize |
| General audio editing | Pinned AuK source, AuK checkpoint and Qwen2.5-Omni encoder |
| Additional speech processing | Pinned RoFormer checkpoint/configuration files and ClearVoice checkpoint |
| Reasoner / Planner / judges | Same Qwen3.8-27B and Qwen3-Omni serving versions/arguments in stack.json |

ClearVoice and RoFormer checkpoints are materialized at their upstream expected
paths through explicit links. RoFormer download metadata/configs are frozen with
hashes, avoiding a changing `main` lookup during first inference. Alternative
checkpoints not listed in the manifest require a new declared configuration.
SAM and VoiceSculptor have no code integration or local setup in this release.
Historical identifiers remain in documentation-only extension records; they are
not available to the downloader. See [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md).

For Hub rate limits, set `HF_TOKEN` or `HF_TOKEN_PATH` to your own authorized
credential in the target container; browser login on another host is insufficient.
Keep credentials outside the source, command text, images and public logs. The
download helper uses two workers and bounded backoff for transient HTTP failures;
authorization denial remains an error. Completed model receipts survive a failed
later download so only the failed components need to be retried.

The text-server snapshot has two known **dependency-metadata incompatibilities**:
vLLM 0.19.0 and compressed-tensors 0.14.0.1 declare Transformers <5, whereas the
working source environment uses 5.5.0. We retain the exact running versions.
The installer accepts only these exact diagnostic lines and records them in the
receipt; every other conflict fails installation. This is not a compatibility
certification. Quantized compressed-tensor models are not qualified, and fresh
host Qwen serving must still pass actual inference before acceptance.

The historical SAM decord installation workaround is not part of the current
installer. Its backend environment is operator-owned and is not synchronized by
X·Edit.

## Real-operation acceptance

Health endpoints may be lazy: HTTP 200 does not prove a checkpoint loaded.
Start the dependencies for the chosen probe, then use the newly installed CPU
environment. Starting only `edit_text` is insufficient for the independent ASR
checks in speech qualification:

```bash
uv run --locked python deployment/check_cuda.py --data /data/xedit \
  --output /data/checks/cuda.json all

uv run --locked python deployment/stack.py --data /data/xedit start cpu
# Speech qualification additionally needs ASR and tagged editing dependencies.
uv run --locked python deployment/stack.py --data /data/xedit start asr edit_text

/local-ssd/xedit-envs/services/volume/bin/python deployment/qualify.py \
  --data /data/xedit --mode dsp --output /data/checks/dsp-001

/local-ssd/xedit-envs/services/volume/bin/python deployment/qualify.py \
  --data /data/xedit --mode speech --output /data/checks/speech-001 \
  --audio /data/your-recording.wav --transcript 'YOUR EXACT SOURCE TRANSCRIPT' \
  --tagged 'YOUR SOURCE TRANSCRIPT WITH <sub targ="replacement">original</sub>' \
  --expected-transcript 'YOUR EXPECTED COMPLETE TRANSCRIPT'
```

Speech mode invokes real ASR, alignment, tagged editing, waveform decoding and
independent ASR of the output. The optional expected transcript is compared after
case/punctuation normalization; numeric word forms are not silently rewritten.
Results, timing and hashes are saved even on failure. This check does not certify
speaker preservation, emotion, artifacts or all tools. Finish with representative
real requests for **each exposed capability**, including multi-audio, multi-turn
and failure-driven revision, and inspect the returned audio and runtime records.

The source closure is declared. The selected historical pipeline pin still
requires authorized source access; see [the version-specific license record](LICENSES.md#pipeline-source-access). [PIPELINE_MIGRATION.md](PIPELINE_MIGRATION.md)
records current verification; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) preserves
the historical checks and their original versions.

`deployment/qualify_services.py` extends the DSP/speech check to the other
deployed model services and creates a synthetic diagnostic recording with the
new TTS. See [RELEASE_ACCEPTANCE.md](RELEASE_ACCEPTANCE.md) for the exact commands,
dependency order, evidence requirements and quality/capacity boundaries.


## Cold-host fixes and dependency boundaries

The historical CUDA 12.8 / Ubuntu 22.04 check used eight 80 GB L20Z GPUs. It
predates the current release boundary. F5 has no managed GPU assignment; budget
its independent deployment separately. SAM and VoiceSculptor are not included
services, and historical GPU tests do not establish current support for them. Existing GPU reports do not qualify the new pipeline pin.

Model downloads and service launch both write exact HF cache ref bytes, without
a newline. SE-DiCoW explicitly selects the pinned standalone DiariZen source and
its matching `pyannote/wespeaker-voxceleb-resnet34-LM` embedding checkpoint;
DiCoW's older embedded submodule must not shadow it. The launcher also declares
the inference Python and remote-code cache explicitly.

The Qwen alignment adapter reports unsupported zero-word boundary queries so
the existing pause pipeline can align the two neighboring words. Marked targets
are bound using the full transcript context, including repeated occurrences;
unresolved ambiguous contexts remain errors. Model weights are unchanged.

The RoFormer adapter right-pads inputs shorter than its required inference
chunk, then trims every stem back to the source duration. It records this
preparation and source time-axis mapping. This addresses duration only: callers must still select and verify stem roles;
`first` is not a guarantee of speech. Model failure remains an error.

An exact-ASR transcript assertion may fail on formatting such as numeric versus
spelled-out times. Such failures remain in the reports. Run the operation probe
without `--expected-transcript` when only deployment is being qualified, and
review semantic/audio correctness separately; do not alter references to turn
failed quality checks into passes.

## Pinned pipeline and official SDKs

The active provider-neutral pipeline is the Git submodule at commit
`c36f753a91896e901ba9cf621ac02d6ef507a21d`. The current URL requires authenticated
access; configure a public mirror only after upstream publication. No token is
stored in this repository. `bootstrap.py --phase fetch` initializes the exact
pipeline and only the SDKs required by the selected services.

```bash
python deployment/pipeline.py fetch all
python deployment/pipeline.py check all
python deployment/audit.py
```

`all` includes IndexTTS, TED-TTS, Seed-VC, Qwen3-ASR and Qwen3-TTS, not F5. SDK files
remain unchanged. The obsolete `deployment/backend.py` and duplicate `backends/`
source are removed. IndexTTS and TED use their upstream-maintained, hash-checked
execution checkouts under the deployment cache. SeedVC runs with a writable
working directory whose `checkpoints` link resolves to the pinned model cache.
No SDK is written to or silently replaced.

Four orchestration services now use `speech_edit.pipelines`. Dependency URLs and
output directories are passed explicitly on their command lines. `ARTIFACT_DIR`
is outside both checkouts. `deployment/pipeline_service.py` adapts only Git's
artifact-boundary query for Ubuntu 22.04 Git, which lacks `worktree list -z`;
path containment checks remain active. Newer Git uses the upstream function.

Emotion type/level profiles use IndexTTS and do not require TED. To use the
pipeline's description-based profiles, explicitly start the optional `ted_tts`
service on port 9003. TED reuses IndexTTS
weights but requires its own model process and GPU budget. Duration-conditioned
TED profiles remain unsupported by the pinned pipeline.

Request changes, verification coverage and known limits are recorded in
[PIPELINE_MIGRATION.md](PIPELINE_MIGRATION.md). Configure a new deployment data
directory after this upgrade: older configurations intentionally fail their
manifest check. Existing model assets and compatible locked environments can
be supplied explicitly; do not synchronize an environment serving another job.

Source separation does not remove the custom IndexTTS conditions or SeedVC's
GPL obligations. See [LICENSES.md](LICENSES.md).
