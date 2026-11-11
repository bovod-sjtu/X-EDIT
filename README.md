# X·Edit

X·Edit edits speech from natural-language instructions. It combines audio
perception, reasoning, planning, and specialist tools.

For example: “Replace ‘tomorrow’ with ‘Friday’” or “Make this sentence sound
happier without changing the words.” Available edits depend on the tools you
deploy and enable.

This repository contains framework code, original service adapters,
configurations, and documentation. It does not include third-party model code,
weights, datasets, reference recordings, or generated audio.

## How it works

```text
Audio + instruction
  → Perception: transcribe and inspect the audio
  → Reasoner: interpret the request and available evidence
  → Planner: select tools and order the steps
  → Executor: validate inputs, call tools, and save results
  → Edited audio + execution record
```

- **Forward:** one planning and execution attempt, with Reasoner enabled by default.
- **Revision:** adds evaluation and revision, with up to three attempts by default.
- **Tool selection:** only enabled capabilities are shown to the planner and accepted by the executor.
- **Failures:** errors remain visible; the framework does not silently switch editing backends.

## Deployment

The controller, tool clients, and model services use separate Python environments.

| Component | Role | Deployment |
| --- | --- | --- |
| Controller | Perception scheduling, reasoning, planning, execution, and records | One Python 3.11 uv environment |
| MCP clients | HTTP tool calls and local audio processing | One uv environment per client, on the controller host |
| Model and tool services | Model inference and specialist processing | One uv environment per managed service, on the same or separate hosts |

Planner, Reasoner, perception, and evaluation can use separate model endpoints
or share compatible services. The controller still runs the workflow; each
logical module does not need its own HTTP server.

See the [module deployment guide](docs/MODULE_DEPLOYMENT.md) for endpoint
configuration, environment layout, shared files, and network access.

## Quick start

Run these commands from the repository root. Required: Linux, Git, Python
3.11.15, uv 0.11.28, FFmpeg/ffprobe, and libsndfile. Model services also need
compatible hardware, drivers, and separately obtained model files.

See [deployment requirements](docs/DEPLOYMENT.md) for system setup and the
non-GPL FFmpeg build. The complete tool stack is not a single-GPU deployment.

### 1. Install the controller

```bash
python3 -m pip install uv==0.11.28
uv python install 3.11.15
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --locked --no-dev
cp .env.example .env
```

This installs the controller, not the model services.

### 2. Select tools

Edit `configs/tools.yaml`. The default configuration enables local audio
operations, tagged speech editing, ASR, alignment, and Omni perception:

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

The `edit_data_pipeline` tool name is retained for compatibility. Choose its
backend with `edit_backend`:

- `dots_tts_edit` — the independently deployed dots.tts.edit model; default.
- `edit_data_pipeline` — the independently deployed editing pipeline.

Both accept the same editing inputs. There is no automatic fallback between them.

Add other tools from `configs/tools.full.yaml` only after deploying their
services. See [tool selection](docs/TOOL_SELECTION.md) and
[service requirements](docs/SERVICE_CAPABILITIES.md).

### 3. Connect services and install clients

Set the model names and service URLs in `.env`. To deploy services yourself,
follow the [dots.tts.edit setup](docs/DOTS_TTS_EDIT.md) and
[module deployment guide](docs/MODULE_DEPLOYMENT.md).

Install the selected local clients:

```bash
.venv/bin/python deployment/client_envs.py \
  --tools configs/tools.yaml --env-root /data/xedit/envs/clients
```

Set `X_EDITOR_MCP_ENV_ROOT=/data/xedit/envs/clients` in `.env`.
Each client uses its own environment; preinstalled client environments are not
automatically synchronized during execution.

Keep models and run outputs outside the source tree. Across hosts, path-based
services must read and write audio at the **same absolute paths**.

### 4. Check and run

```bash
.venv/bin/x-edit --env-file .env doctor --config configs/runtime.yaml
.venv/bin/x-edit --env-file .env doctor --config configs/runtime.yaml --live

.venv/bin/x-edit --env-file .env edit --config configs/runtime.yaml \
  --audio /data/audio/source.wav \
  --instruction "Make this sentence sound happier. Keep the words and speaker." \
  --mode forward --output /data/xedit/runs/first-edit
```

`doctor` checks configuration and local dependencies. `--live` also checks
service connections and interfaces. Neither verifies model inference or editing
quality; run a real edit with audio you are authorized to use.

Use a new output directory for each run. `result.json` identifies the selected
audio and execution records. A failed delivery returns a nonzero exit status.

Use `--mode revision` to enable evaluation and revision, and configure the
evaluation endpoints. See the [runtime API](docs/RUNTIME.md) for Python,
multi-audio, and multi-turn usage.

The package is `x-edit-audio`, the CLI is `x-edit`, and the Python namespace
is `x_editor`. Wheel installations can export configuration with
`x-edit init ./configs`; deployment scripts are provided in the source repository.

## License and external components

Original code, configuration, and documentation that X·Edit is entitled to
license use [Apache License 2.0](LICENSE). Third-party components, models, data,
and outputs retain their own terms. Independent installation or HTTP access does
not remove those terms or grant commercial model-use rights. Service operators
remain responsible for their own deployments.

- **F5-TTS:** disabled by default, with no automatic installation or download.
  Operations that require it prompt the user to install it separately before
  enabling access. The recorded SDK is MIT-licensed; the recorded official
  `F5TTS_v1_Base` weights are CC-BY-NC-4.0. Other checkpoints require their own
  license review. See [external tool setup](docs/EXTERNAL_TOOLS.md).
- **SAM-Audio and VoiceSculptor-VD:** not included in the built-in toolset or
  deployment code. Only [extension guidance](docs/EXTENSION_GUIDE.md) is provided.
- **edit-data-pipeline:** a pinned external dependency, not bundled source.
  Access to the currently pinned repository requires upstream authorization.
  See [source access](docs/LICENSES.md#pipeline-source-access).
- **FFmpeg:** invoked as an external process; no binary is bundled. The supplied
  build disables GPL and nonfree features. Review any container or binary package
  separately before distribution.

Retain applicable notices when copying or modifying third-party code.
Commercial use of noncommercial models requires separate authorization or a
suitable replacement; excluding weights from a release does not grant that
permission. Exact sources, versions, and terms are listed in the
[third-party notices](third_party/NOTICE.md) and [license inventory](docs/LICENSES.md).

## Documentation

- [Module deployment](docs/MODULE_DEPLOYMENT.md): isolated environments and remote services.
- [Deployment reference](docs/DEPLOYMENT.md): dependencies, startup, and operation checks.
- [Service capabilities](docs/SERVICE_CAPABILITIES.md): tool profiles and prerequisites.
- [Tool contracts](docs/TOOL_CONTRACTS.md): parameters, audio assets, and errors.
- [Release notes](docs/RELEASE_NOTES.md) and [reproducibility](docs/REPRODUCIBILITY.md): changes and validation scope.
- [Release boundary](docs/RELEASE_SCOPE.md): source, models, data, and distribution rules.

Some detailed guides are currently in Chinese. Historical benchmark results
apply to their recorded versions, not automatically to later deployment changes.

Run records contain instructions, audio paths, and model responses. Keep them
private and review them before sharing. This project builds on and extends
dots.tts.edit models and pipeline services; upstream licenses remain unchanged.
