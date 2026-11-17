# External components and documentation references

The pinned outer pipeline declares Apache-2.0. Its SDKs, model weights and
patches retain their own terms. They are not redistributed by this source package.
`pipeline.py fetch all` acquires five managed SDKs, never F5. F5 must be obtained,
installed and deployed independently; its runtime interface is disabled by default.

## Recorded F5 source and model

These are provenance records, not bundled files, automatic download instructions
or proof of the weights behind an operator's HTTP endpoint. X·Edit pins the outer
pipeline at `ba270475fdb3c721db381463d73d3fc43727c554`; that upstream tree records
F5 in its nested `tools/f5-tts/upstream` gitlink. X·Edit's helpers do not initialize
that F5 checkout. A manually recursive clone is a separate acquisition action.

| Component | Exact recorded version | License evidence |
| --- | --- | --- |
| [F5-TTS SDK](https://github.com/SWivid/F5-TTS) | Git commit `27e20fcf39fbce3f187a61d4a3e92065d2d9b947` | [MIT license at that commit](https://github.com/SWivid/F5-TTS/blob/27e20fcf39fbce3f187a61d4a3e92065d2d9b947/LICENSE) |
| [Official SWivid/F5-TTS weights](https://huggingface.co/SWivid/F5-TTS) | Model `F5TTS_v1_Base`; file `F5TTS_v1_Base/model_1250000.safetensors`; Hub revision `84e5a410d9cead4de2f847e7c9369a6440bdfaca` | [Pinned model card](https://huggingface.co/SWivid/F5-TTS/blob/84e5a410d9cead4de2f847e7c9369a6440bdfaca/README.md) declares [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) |

Source pins come from [external_pipeline.json](external_pipeline.json); the
checkpoint selection comes from [manifest.json](manifest.json). Model-card
evidence and its SHA256 are in [licenses.json](licenses.json). The recorded Hub
revision originated from an existing cache reference, not remote-service
attestation. Operators must verify the actual served checkpoint, source version
and auxiliary components. Other F5 variants, fine-tunes and vocoders require their
own records; neither the MIT SDK nor this model card establishes their terms.

The MIT SDK can coexist with original Apache-2.0 framework code subject to its
conditions. If it is copied, modified or bundled later, retain its copyright
and MIT permission notice in the relevant deliverable; it does not become
X·Edit-owned code. Source archives, wheels, containers, preloaded caches and
hosted services need separate scope reviews. This source release contains no F5
implementation, weights or generated audio and certifies no prebuilt image.

The model's noncommercial condition is not limited to redistribution. Service
operators, including the project team if it hosts inference, remain responsible
for their own use; a user-facing disclaimer cannot transfer that responsibility.
See the [licensing boundary](../docs/EXTERNAL_TOOLS.md#licensing-boundary) for the
separate source-publication and model-operation review boundaries.

## Documentation-only references

SAM-Audio and VoiceSculptor-VD are documentation-only extension candidates, not
active dependencies or shipped integrations. No client, adapter, capability card,
registry entry, backend environment or automatic launcher is included. Their
source/model records are not installer inputs. Historical license evidence is
retained in `licenses.json` for transparency and is marked accordingly.

`src/` and `service_assets/` are acquired separately and excluded from release
archives. No model weights, third-party generated audio or media binaries are
bundled. HTTP, independent installation and extension development do not waive
upstream restrictions or establish commercial authorization. See
[licensing and extension guidance](../docs/EXTERNAL_TOOLS.md).
