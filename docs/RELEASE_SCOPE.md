# Source distribution boundary

Updated 2026-10-26.

X·Edit publishes its original framework code, service interfaces, configurations,
prompts, deployment recipes and documentation under Apache-2.0. Model-backed
operations call separately deployed services. The controller package does not
install model runtimes or download weights.

## Contents of the release

| Included | Obtained or produced separately |
| --- | --- |
| Perception scheduling, task interpretation, planning, execution and session management | Third-party model and SDK source checkouts |
| Assessment orchestration, execution feedback and policy revision | Model weights, tokenizers, codecs and auxiliary checkpoints |
| Original HTTP clients, capability descriptions, validation and service adapters | Datasets, reference recordings and generated audio |
| Configurations, prompts, dependency declarations and version locks | Python dependency wheels and native system libraries |
| Installation recipes, API documentation and sanitized validation summaries | FFmpeg/ffprobe binaries and prebuilt service/container images |

The Git repository contains a pipeline gitlink, its source location and fixed
revision. A gitlink names another repository; its files are not part of the
superproject's source tree. Source archives, wheels and Docker build contexts
exclude the initialized checkout, other downloaded repositories and runtime
assets. Recursive clone and explicit provisioning are separate acquisition
actions governed by the upstream terms.

Original adapters import or call external implementations from the operator's
environment. Any copied or derived third-party material must retain its applicable
license and attribution. An adapter directory is not a blanket Apache grant for
external code. The [license inventory](LICENSES.md) records code and weights
separately, including auxiliary dependencies.
The [third-party notices](../third_party/NOTICE.md) also identify the exact F5
checkpoint and clarify MIT notice retention if a future deliverable includes SDK
code. Independent wrappers are not automatically relicensed by calling a model.

## Using external services

Configure authorized HTTP endpoints in the runtime environment, or explicitly
provision selected services using the source distribution's deployment scripts.
F5-TTS cannot be provisioned by these scripts. It requires independent
installation and explicit opt-in as documented in [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md).
Other fetch/install/start actions are explicit. SAM-Audio and VoiceSculptor are
documentation-only extension candidates: their clients, adapters, tool metadata,
planning options and execution registrations are not part of this release.

F5 remains a disclosed external prerequisite for its editing capabilities. Its
code/weights are not installed as controller dependencies. Removing installation
from this package does not erase that functional prerequisite or grant commercial
rights. Original framework rights and external component rights remain separate.
Noncommercial, custom-license, attribution and copyleft obligations still apply
to independently obtained components and independently developed extensions.
See the [release statement](OPEN_SOURCE_STATEMENT.zh-CN.md).

This boundary addresses what is published, not permission to operate a model.
The actual service operator, including the project team when it hosts inference,
must establish rights for its own use; a notice to users does not transfer that
responsibility. Neither omission of weights nor omission of generated audio
establishes commercial-use permission. Hosted services and any later containers,
preloaded environments or binary packages need their own reviews.

The active pipeline is the public
[studio-dots-ai/dots.tts.edit-data-pipeline](https://github.com/studio-dots-ai/dots.tts.edit-data-pipeline)
revision `ba270475fdb3c721db381463d73d3fc43727c554`. Its root license is Apache-2.0.
The upstream checkout is excluded from source archives and wheels. SDKs, model
weights, and operation permissions remain subject to their own terms.

## FFmpeg

This source release contains no FFmpeg/ffprobe executable or prebuilt image.
Optional image recipes build FFmpeg 8.1.2 from a hash-checked official archive,
with GPL, nonfree and version3 features disabled. They preserve shared libraries,
corresponding source, configuration and LGPL notices. The isolated build has
passed media conversion checks; complete image builds have not been qualified.

Any later binary or image distribution needs inspection of the actual complete
deliverable, including copies installed by base images or Python wheels, and
fulfilment of the applicable LGPL and other component obligations.

## Checking a deliverable

Run from a Git checkout or an extracted source archive:

```bash
python3 deployment/release_audit.py
python3 deployment/release_audit.py \
  --artifact /path/to/x_edit_audio-0.1.0.dev1.tar.gz \
  --artifact /path/to/x_edit_audio-0.1.0.dev1-py3-none-any.whl \
  --output /path/to/release-content-report.json
python3 deployment/license_audit.py --output /path/to/model-stack-review.json
```

The content audit inspects project files and supplied archives for excluded
resources, native binaries, common credential patterns and cluster-specific
configuration. Independently acquired gitlink checkouts and ignored runtime
downloads are outside its project-file scope. The model-stack audit separately
reports component restrictions; a failed stack review is not suppressed by a
successful source-content check. Neither check establishes ownership or legal
approval.

Use the inspected source archive for publication. It excludes Git history.
Changes to the working tree do not remove old content from existing Git history;
do not publish that history without a separate review.
