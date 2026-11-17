# Deployment

Install the controller with the [README quick start](../README.md#quick-start).
Use a source checkout for service deployment; a wheel installs the controller.
Select tools in `configs/tools.yaml` before provisioning. Use the same file for
deployment and runtime so the available services match the advertised capabilities.

## Environments and storage

| Component | Environment | Installer |
| --- | --- | --- |
| Controller | `.venv` | `uv sync --locked --no-dev` |
| Selected MCP clients | `<env-root>/clients/<tool>` | `deployment/client_envs.py` or bootstrap |
| Managed services | `<env-root>/services/<service>` | Bootstrap |
| dots.tts.edit and F5 | Operator-chosen, outside this checkout | Independent installation |

Each service has its own uv environment, even when services use the same lock.
The controller uses Python 3.11.15; service versions and locks are specified in
`deployment/stack.json`. Install with uv 0.11.28. Clients use preinstalled
environments without synchronizing dependencies during an edit.

Keep models, audio, logs, results, caches, and environments outside the checkout
wherever configurable. Fetch helpers place external source and service metadata
under `third_party/`; do not stage or redistribute those acquired files.
There are no shared ignore rules: inspect the file list before committing or
archiving. Packaging exclusions remain in `MANIFEST.in` and `pyproject.toml`.

## Host prerequisites

Use Linux x86-64, Git/Git LFS, FFmpeg/ffprobe, SoX, libsndfile, and build tools.
Neural services need NVIDIA GPUs and compatible drivers. Most GPU locks target
CUDA 12.8; AuK uses CUDA 12.6. The full stack is not a single-GPU deployment.
Allow enough space for models and isolated environments; the full stack can need
several hundred GB. Prefer local SSD for environments and shared storage for audio.

On a dedicated Ubuntu/Debian host or container with root access, the optional
`bash deployment/install_os.sh` installs OS dependencies, uv, and Python.
It changes system packages; get administrator approval on shared hosts.
Alternatively, install those prerequisites yourself.

The FFmpeg builder disables GPL/nonfree features and invokes
`deployment/check_ffmpeg.py` to check the binaries and source notices.
This check is part of installation. Put `/opt/xedit-ffmpeg/bin` first on
`PATH` and retain the complete prefix, including corresponding source and
notices, if redistributing that build. No FFmpeg binary or prebuilt image is bundled.

## Plan, fetch, install, start

First independently start [dots.tts.edit](DOTS_TTS_EDIT.md). From the source root:

```bash
.venv/bin/python deployment/bootstrap.py --data /data/xedit --phase plan \
  --tools configs/tools.yaml --external dots_tts_edit=http://127.0.0.1:8049

.venv/bin/python deployment/bootstrap.py --data /data/xedit --phase fetch \
  --tools configs/tools.yaml --external dots_tts_edit=http://127.0.0.1:8049

.venv/bin/python deployment/bootstrap.py --data /data/xedit --phase install \
  --tools configs/tools.yaml --env-root /data/xedit/envs \
  --external dots_tts_edit=http://127.0.0.1:8049

.venv/bin/python deployment/bootstrap.py --data /data/xedit --phase start \
  --tools configs/tools.yaml
source /data/xedit/client.env
.venv/bin/x-edit doctor --config configs/runtime.yaml --live
```

`plan` lists dependencies and GPU placement without downloads or startup.
Review it before `fetch`, which obtains pinned sources and models separately.
Use your own authorized credentials for gated assets. Installation creates the
selected client/service environments. Startup checks assets and interfaces,
including contextual ASR, but does not prove editing quality.

For the alternative backend, select `edit_backend: edit_data_pipeline` and
follow [pipeline setup](PIPELINE.md). Bind independently installed F5 with
`--external f5=http://host:port` for operations needing it. It is never installed
automatically. SAM-Audio and VoiceSculptor are not managed services or built-in tools.
Use a new data directory when changing pins, manifests, bindings, or environment
layout; existing deployment records reject incompatible changes.

## Remote services and GPU placement

`--external NAME=http://host:port` reuses a service you operate without installing,
starting, or stopping it. Supply the same bindings during fetch and initial install;
they are then recorded in the deployment data directory.

`--gpu SERVICE=VISIBLE_IDS` controls local placement at configuration time,
for example `--gpu planner=0,1,2,3 --gpu omni=7`. It does not change tensor
parallelism: the supplied Planner still requires four devices. Default placement
assumes large-memory GPUs, not that every model fits simultaneously.
Run services on separate hosts when needed.

`--bind-address` defaults to loopback. Protect remote endpoints with authentication,
TLS, and network controls. `--port-offset N` shifts local ports. Exported local
URLs are not automatically remote hostnames. Path-based services and the controller
must see audio/output files at the same absolute paths; dots.tts.edit sends base64.

Role endpoints can be separate or shared, independently of tool services:

| Module | Endpoint variable | Model variable |
| --- | --- | --- |
| Reasoner | `XEDIT_REASONER_URL` | `XEDIT_REASONER_MODEL` |
| Planner | `XEDIT_PLANNER_URL` | `XEDIT_PLANNER_MODEL` |
| Audio perception | `XEDIT_PERCEPTION_URL` | `XEDIT_PERCEPTION_MODEL` |
| Semantic judge | `XEDIT_SEMANTIC_JUDGE_URL` | `XEDIT_SEMANTIC_JUDGE_MODEL` |
| Audio judge | `XEDIT_AUDIO_JUDGE_URL` | `XEDIT_AUDIO_JUDGE_MODEL` |

Use OpenAI-compatible base URLs ending in `/v1`. Revision guidance uses the
semantic-judge configuration. Workflow coordination stays in the controller.
Apply role overrides after loading `client.env`. Tool endpoint names are in
`.env.example`.

## Operate and verify

```bash
.venv/bin/python deployment/stack.py --data /data/xedit status
.venv/bin/x-edit profile --config configs/runtime.yaml
.venv/bin/x-edit doctor --config configs/runtime.yaml --live
.venv/bin/python deployment/stack.py --data /data/xedit stop
```

`stop` stops only processes still owned by that deployment, not external services.
Startup is a one-shot command, not a persistent supervisor; use your own process
manager for long-lived deployments. Logs and process records stay in the data
directory. Run a short edit from the README and inspect its audio and trace:
health checks, real editing, and benchmark quality are different checks.
See [tool selection](TOOL_SELECTION.md) and [runtime records](RUNTIME.md).
