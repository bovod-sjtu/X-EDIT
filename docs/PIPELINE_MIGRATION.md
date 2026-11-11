# Open-source pipeline integration

## Current revision — 2026-10-26

The active provider-neutral pin is `c36f753a91896e901ba9cf621ac02d6ef507a21d`,
with an Apache-2.0 root license. F5 is operator-installed and disabled by default;
SAM and VoiceSculptor are documentation-only extension candidates, not shipped
tools, clients or adapters. See
[EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md). Configure a new deployment after this
manifest change. No running deployment or historical benchmark snapshot is
modified. Offline transport/schema validation does not certify GPU inference
or equality with previous benchmark results.

## Historical integration — 2026-10-11

Everything below, including GPU results, describes the older pin, not the
current deployment. Historical installation commands are superseded by
[DEPLOYMENT.md](DEPLOYMENT.md).

This change applies to the final, non-anonymous X-Edit project. The historical
anonymous project is not changed. It replaces the local pipeline copy with
`edit-data-pipeline`, branch `prepare-opensource`, pinned at
`2111b99c72258cfefb1920fc1195e6141379236d`.

## Source and deployment boundary

- X-Edit owns the controller, HTTP clients, tool contracts and deployment helpers.
- The pipeline is a Git submodule. Its six official SDK gitlinks, URLs and
  commits are recorded in `third_party/external_pipeline.json`.
- Service wrappers and checked patches live outside those official SDK trees,
  in the pipeline project. X-Edit's duplicate IndexTTS/Seed-VC adapter trees are
  removed. IndexTTS and TED materialize checked execution copies in the runtime
  cache; F5 uses the upstream adapter's in-memory compatibility patch.
- Source packages contain fetch instructions and pins, not external SDK source.
  Generated files, execution copies, weights and audio stay outside the repository.

The selected historical pin requires authorized Git access and predates the
upstream root license file. The upstream project now has an Apache-2.0 license
at a newer revision; [the current access and version record](LICENSES.md#pipeline-source-access)
distinguishes that revision from the tested runtime. SDK-derived modifications
retain their original obligations. Submodules, separate processes
and runtime patches do not remove IndexTTS's custom conditions or Seed-VC's GPL
terms. See [LICENSES.md](LICENSES.md).

## Calling changes

| Boundary | Adaptation |
| --- | --- |
| Four editing APIs | Launch `speech_edit.pipelines.{text,prosody,emotion,pause}.service.api_service`; pass dependency URLs and output directories as CLI arguments. |
| Local emotion | Translate legacy categorical tags into `<emo type="happy" level="2">…</emo>`. Explicit continuous `alpha` is forwarded without quantization. |
| Pause synthesis | Use `synth_method`; translate legacy `method_one` to `single_region` and `method_two` to `dual_boundary`. Preserve `insert_only`. |
| Alignment | Use the supported Qwen3 backend; reject unsupported CTC selection explicitly. |
| Text editing | Keep the controller's segmentation budget local; omit retired dataset/export fields from HTTP payloads and reject explicit requests for those removed features. |
| Seed-VC | Pass seed 0 explicitly, preserving the previous deterministic default. Run official code from an external writable working directory with a link to the pinned model cache. |
| Artifact paths | Keep artifacts outside source checkouts. On Ubuntu 22.04 Git, adapt the unsupported NUL worktree query without disabling path containment checks. |
| TED-TTS | Add an optional independently deployable service for the pipeline's description-based profiles; categorical editing does not start it. This does not add free-form emotion interpretation to X-Edit's existing categorical tool contract. |

The Reasoner, Planner, Forward/revision orchestration and selection algorithms
are unchanged. Tool transport and emotion-intensity fidelity changed, so historical
benchmark scores are not asserted to be identical.

Create a new deployment directory after this upgrade; old configurations reject
the changed manifest. Compatible model receipts and environments can be reused
explicitly. Do not install into an environment serving another running job.

```bash
python deployment/pipeline.py fetch all
python deployment/pipeline.py check all
python deployment/bootstrap.py --data /data/xedit-new --phase plan all
# Continue with fetch, install and start as documented in DEPLOYMENT.md.
/local-ssd/xedit-envs/services/volume/bin/python deployment/check_pipeline_contracts.py
```

## What was verified

The compact machine-readable report is [pipeline_validation.json](pipeline_validation.json).
All real-audio probes below use a short generated speech recording, not a
benchmark. Model weights and compatible locked environments were reused.
New GPU processes loaded IndexTTS, Seed-VC, F5 and Qwen3 ForcedAligner through
the new entrypoints. Existing decision/perception services were used by the
end-to-end agent probes.

| Check | Result | What it establishes |
| --- | --- | --- |
| Pipeline + six SDKs | Exact pins; clean official trees | Declared source closure, including nested submodules and patch checksums |
| Payload/path contracts | 11/11 | Four actual request schemas, emotion level/alpha, pause alias, Seed-VC seed, accepted artifact root and rejected escapes |
| Numerical DSP | 2/2 | Rate 1.25 yields duration ratio 0.8; peak normalization reaches −12 dBFS |
| Text edit | Delivered | `train` → `bus`; separate ASR reads the complete expected sentence |
| Local rate edit | Delivered | Tagged `six fifteen` edited through the new prosody API |
| Local emotion edit | Delivered | Tagged happy/level 2 request reaches IndexTTS and returns decoded audio |
| Pause insertion | Delivered | Requested 0.5 s insertion increases duration by 0.5 s |
| Direct IndexTTS / Seed-VC | 2/2 delivered | New upstream GPU entrypoints perform real inference |
| Agent: text replacement | Delivered | Observation → planning → admitted tool call → readable output |
| Agent: gain, then rate | Both turns delivered | History-based second edit runs on the preceding output |
| Agent: reference voice | Delivered | Separate content/reference assets reach Seed-VC and return audio |

Successful delivery requires a nonempty, decodable, finite waveform; it does
not certify subjective emotion, identity, preservation or artifact quality.
No human listening or full benchmark scoring was performed for this migration.
The three agent probes are Forward checks, not a fresh validation of revision.

The first integration attempts failed at concrete boundaries: old Git rejected
`worktree list -z`; executing API modules through `__main__` prevented Pydantic
from resolving request annotations; an existing healthy port was not an F5
service; and the old emotion tag format was rejected. The outer compatibility
adapter, normal module import, explicit new service endpoints and tag conversion
resolved these failures before the passing probes above.

## Remaining validation and publication work

1. Provide access to the tested pipeline revision or qualify a new accessible
   revision before changing the source pin. External components keep their terms.
2. Run a fresh complete installation of this exact revision in a clean GPU
   host/container. Earlier cold-start reports describe earlier source versions.
3. TED and Qwen3-TTS source/deployment wiring is present, but real inference on
   those new entrypoints was not revalidated here. The new ASR entrypoint also
   needs a fresh probe; independent text verification reused the existing ASR.
4. Check subjective quality, longer recordings, concurrency and policy revision
   separately. Do not infer these from the short delivery probes.
