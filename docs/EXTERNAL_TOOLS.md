# Operator-installed tools

F5-TTS is **disabled by default** and must be installed independently. X·Edit
retains its pipeline-facing editing interface but does not download, install or
start the F5 implementation, weights or codecs. Model/source declarations are
kept for disclosure, not as automatic installer inputs.

SAM-Audio and VoiceSculptor are not included integrations. They have no shipped
clients, adapters, catalog entries, registrations, launchers or opt-in switches.
Only [independent extension guidance](EXTENSION_GUIDE.md) remains.

## F5-TTS

When an operation requires F5 and it has not been explicitly enabled, the tool
returns **“请自行下载并安装F5-TTS”** before contacting the editing service. This
applies to text, local prosody, emotion, pause, boundary smoothing and assembly
interfaces. Standalone numerical rate, pitch and volume processing are unaffected.

1. Obtain the [F5 implementation](https://github.com/SWivid/F5-TTS) and
   [selected weights](https://huggingface.co/SWivid/F5-TTS) independently, outside
   the X·Edit checkout. Review the applicable terms for your intended use and
   record the actual source commit, checkpoint revision and auxiliary components.
   The [recorded F5 versions](../third_party/NOTICE.md#recorded-f5-source-and-model)
   are a review reference, not proof of an existing endpoint's model identity.
2. Deploy your own compatible service. A Python package alone is not an HTTP
   service: the pipeline requires its F5 `/sample` contract and `/health` endpoint.
   Consult the pinned pipeline's F5 interface documentation. X·Edit does not
   supply or start that inference backend.
3. Explicitly bind that endpoint when provisioning the pipeline, for example:

```bash
python deployment/bootstrap.py --data /data/xedit-new --phase plan \
  --external f5=http://127.0.0.1:8001 speech
```

Use the same binding for `fetch` and `install` with the new data directory, then
`start`. Generated `client.env` records the explicit opt-in; load it before running
the controller. Already configured deployments reject changed manifests/bindings.

For an independently managed pipeline, configure its F5 URL yourself, then set
both `F5_SERVICE_URL` and `X_EDITOR_ENABLE_F5_TTS=1` in the controller environment.
The first variable documents the operator's F5 endpoint; it does not reconfigure
an already running remote pipeline. Ensure both configurations refer to the same
service. Setting only a URL or only the enable flag is insufficient. Removing the
flag or setting it to `0` disables calls again.

X·Edit does not issue direct F5 inference requests. After explicit setup, its
editing HTTP client can call the separately deployed pipeline, which uses the
operator's F5 service. Setup errors do not trigger an automatic installer.

## Documentation-only extensions

SAM-Audio and VoiceSculptor are absent from the built-in toolset, profiles and
service/download manifests. Former environment variables cannot enable them.
Qwen3-TTS accepts only its native backend or `auto`; an unavailable backend is
rejected, not redirected. Users may independently develop extensions as described
in [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md). No ready-to-run plugin is supplied.
The existing Qwen path is not claimed equivalent to every removed capability.

## Licensing boundary

Apache-2.0 covers X·Edit's original framework and interfaces, **not** external
models or their outputs. F5's MIT source license does not remove the selected
weights' CC-BY-NC-4.0 restriction. SAM has its own agreement and auxiliary
ImageBind restrictions. VoiceSculptor's top-level Apache declaration does not
remove the selected XCodec2 weights' noncommercial restriction. See
[LICENSES.md](LICENSES.md) for exact pins and evidence.

Independent installation and HTTP boundaries do not grant commercial rights or
guarantee legal approval. Commercial use of restricted components requires
appropriate authorization from their rights holders. Other disclosed model-stack
restrictions remain; this is not a claim that the full inference stack is Apache-only.
Service operators retain responsibility for their own deployments, including when
the project team operates a service; asking clients to follow upstream terms does
not transfer that responsibility. See the [source-publication and model-operation
review](OPEN_SOURCE_STATEMENT.zh-CN.md).

## Validation scope

On 2026-10-26, 16 isolated checks passed for tool removal, rejected former
backend selections, native Qwen transport and Qwen-to-SeedVC composition,
F5 setup gates, download/start prevention, DSP availability and the stdio
installation prompt. The surviving Qwen reference-generation branch was also
compared structurally with the baseline and is unchanged. Eleven checks against
the pinned pipeline's native request schemas and artifact boundaries passed.

Validate current source with `python deployment/audit.py` and current source or
packages with `python deployment/release_audit.py`. Offline transport/schema
checks use mock transport and diagnostic fixtures, not real model inference.
They do not establish waveform quality, benchmark parity or commercial clearance.
Historical GPU reports retain their original revisions. This release-boundary
change does not rerun or modify those experiments.
