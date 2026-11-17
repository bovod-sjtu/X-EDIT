# Operator-installed tools

F5-TTS is **disabled by default** and must be installed independently. X·Edit
retains its pipeline-facing editing interface but does not download, install or
start the F5 implementation, weights or codecs. Model/source declarations are
kept for disclosure, not as automatic installer inputs.

SAM-Audio and VoiceSculptor are not included integrations. They have no shipped
clients, adapters, catalog entries, registrations, launchers or opt-in switches.
Only the independent extension guidance below is provided.

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
below. No ready-to-run plugin is supplied.
The existing Qwen path is not claimed equivalent to every removed capability.

| Candidate | Upstream references | Review before developing an extension |
| --- | --- | --- |
| SAM-Audio | [Source](https://github.com/facebookresearch/sam-audio), [official model](https://huggingface.co/facebook/sam-audio-large) | Review the [SAM agreement](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/LICENSE) and auxiliary components, including [ImageBind](https://github.com/facebookresearch/ImageBind/blob/53680b02d7e37b19b124fa37bae4b6c98c38f5be/LICENSE). |
| VoiceSculptor-VD | [Source](https://github.com/ASLP-lab/VoiceSculptor), [model](https://huggingface.co/ASLP-lab/VoiceSculptor-VD) | The top-level Apache declaration does not cover every dependency. The historically used [XCodec2 weights](https://huggingface.co/HKUSTAudio/xcodec2/blob/e412427ed30f0cf9d5e3c95562113deb10a32d03/README.md) declare CC-BY-NC-4.0. |

These are references, not runtime dependencies or approved extension packages.
Historical evidence remains in [the license inventory](../third_party/licenses.json).
The former SAM mirror is not asserted to be identical to official weights.

To develop a separate extension:

1. Review the intended use and all selected source, checkpoint, and auxiliary
   licenses. Obtain additional permission where needed.
2. Keep implementation, downloads, environments, and generated media in your own
   repository and data directory, outside X·Edit's distribution.
3. Define the HTTP contract, failures, timeouts, and output identity. A separator
   must identify target and residual stems; a voice tool must distinguish
   synthesis from editing and preserve unedited words, speakers, and timing.
4. Maintain your own capability card, manual, and MCP wrapper. Generic loaders
   and `ToolExecutor(catalog_dir=...)` accept a separate catalog.
5. Wire that catalog into your application and register handlers with
   `AudioOperationRegistry.register`. Pass the registry to
   `AudioEditExecutor(registry=...)`. An explicit registry replaces default
   registration: include every operation your application needs. This is not
   an automatic plugin installer or stock CLI switch.
6. Test setup failures, disabled tools, output paths, source preservation, and
   real inference separately. Review the complete extension before publishing.

## Additional operator-supplied integrations

These optional catalog interfaces are outside the managed service stack.
Their source and model revisions are not selected by its manifest. Obtain and
review the actual implementations, checkpoints, and auxiliary components yourself.

| Interface | External implementation | Purpose |
| --- | --- | --- |
| `ace_step_music` | [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) | Music and vocal generation/editing |
| `seamless_expressive_s2st` | [Seamless Communication](https://github.com/facebookresearch/seamless_communication) | Expressive speech translation |
| `tse` / `quarkaudio_unise` | [QuarkAudio-UniSE](https://github.com/alibaba/unified-audio/tree/main/QuarkAudio-UniSE), [Spark-TTS assets](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Reference-conditioned target-speaker extraction |

No implementation, checkpoint, or generated recording is bundled for these
interfaces. Keeping an HTTP client does not install or approve a model.

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
not transfer that responsibility.

The source release contains original framework code, configuration, interfaces,
and documentation. It excludes third-party implementations, model weights,
datasets, reference recordings, and generated audio. Downloading, deploying,
using, modifying, and redistributing those assets remain subject to their terms.
Excluding weights or outputs does not establish commercial model-use permission.

Source archives, wheels, containers, preloaded caches, and hosted services need
separate reviews. Copied or modified MIT code must retain its copyright and
permission notice; it is not automatically relicensed as original X·Edit code.
FFmpeg is an external program, not a bundled binary. Its supplied build disables
GPL/nonfree features; other builds must be checked before redistribution.
