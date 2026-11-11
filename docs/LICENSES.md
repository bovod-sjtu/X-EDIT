# Code and model licenses

The [source release boundary](RELEASE_SCOPE.md) covers original framework code,
configuration and documentation. External SDKs, models, datasets and generated
audio are obtained or produced separately. This inventory records declarations
for those runtime dependencies; it does not grant additional rights to them.
The [dependency review](DEPENDENCY_REMEDIATION.md) distinguishes source licenses,
weight licenses, auxiliary dependencies and FFmpeg distribution requirements.

X-Edit's original controller, HTTP clients and deployment code is Apache-2.0.
The pipeline and inference implementations are fetched separately at fixed Git
commits; they are not copied into X-Edit's wheel or source archive. Using a
submodule, HTTP or an outer patch does not waive the dependency's terms.
The [third-party notices](../third_party/NOTICE.md) identify the recorded F5 SDK
and exact checkpoint separately, with pinned license evidence and limits on what
those records establish. Original adapters do not automatically inherit a model's
license merely by calling it; copied or derived material needs separate review.
Source-publication approval does not establish rights to operate a commercial
model service. Operators remain responsible for their own use, not only end users.

## dots.tts.edit

For the default tagged-edit backend, see [dots.tts.edit license evidence](DOTS_TTS_EDIT.md#许可与版本证据).
Its code is Apache-2.0. The pinned project README declares released checkpoints
Apache-2.0, while the exact edit model card says `license: other` and refers back
to that project. Both records and their hashes are retained; the audit requests
scope review instead of presenting the model metadata as an independently
verified unrestricted grant. Source and weights are operator-installed only.

## Qwen models and roles

The Qwen organization page is not a model or a common license for every Qwen
release. The current configuration uses these distinct components:

| Exact model | Role and integration | Pinned revision | License evidence |
| --- | --- | --- | --- |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | The `planner` HTTP service supplies task reasoning, edit planning and the text-only semantic judge used by revision. | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | [Apache-2.0 license file](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE). |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | The `omni` HTTP service supplies audio perception and the audio judge used by revision; its served-model alias is `Qwen3Omni-Instruct`. | `26291f793822fb6be9555850f06dfe95f2d7e695` | The [pinned model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/26291f793822fb6be9555850f06dfe95f2d7e695/README.md) declares `license_name: apache-2.0` under `license: other`; the inventory records this as card evidence, not a separately verified license file. |
| [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | An auxiliary encoder loaded inside the external AuK service, not the controller's planning or judging model. | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` | [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/resolve/f75b40e3da2003cdd6e1829b1f420ca70797c34e/LICENSE), restricted to non-commercial research/evaluation; commercial use requires a separate license from its rights holder. |

Roles follow `configs/forward.json`, `configs/revision.json` and
`deployment/stack.json`. The revisions identify the assets recorded in
`third_party/manifest.json`; they do not attest that an operator's existing
endpoint is serving those exact weights. Verify that identity when deploying.
ASR, forced alignment and TTS use other explicitly listed Qwen3 models below;
they are not aliases for these three models. None of their implementations,
weights or generated audio is included in this source release. External model
terms remain applicable even when only an HTTP interface is distributed.

## Pipeline source access

The active pipeline pin is `c36f753a91896e901ba9cf621ac02d6ef507a21d`. Its
root `LICENSE` declares Apache-2.0; the exact file hash is recorded in
`third_party/external_pipeline.json`. The provider-neutral source was inspected
through authorized Git access on 2026-10-26. No credentials are distributed.

The previous runtime pin was `2111b99c72258cfefb1920fc1195e6141379236d`;
historical GPU validation at that revision does not certify this new pin. The
active authenticated source URL remains in the pipeline manifest. Publish a
publicly accessible source location only after separately verifying access and
the exact commit. Third-party SDK modifications retain their original terms.

F5 is not fetched by X·Edit. Its SDK pin below is retained for provenance; users
obtain it independently. SAM and VoiceSculptor are not included tools or code
integrations; only extension guidance and historical license records remain.
See [runtime opt-in and installation boundary](EXTERNAL_TOOLS.md).

## Pipeline SDK source

The additional operator-supplied ACE-Step, Seamless and QuarkAudio interfaces are
listed in [SERVICE_MATRIX.md](SERVICE_MATRIX.md#additional-operator-supplied-integrations).
They are outside the managed stack and have no selected model/source revision in
its manifest. Review their exact implementation, checkpoint and auxiliary terms
when configuring them; no license conclusion for those unselected assets is
inferred from the retained HTTP interface.

| SDK | Pinned official commit | Code terms / boundary |
| --- | --- | --- |
| IndexTTS | `1eafa935e8887f9ba877f82cf7319fceddfc86cb` | Bilibili custom terms, including source. Patches run in a verified separate checkout; restrictions remain. |
| Seed-VC | `51383efd921027683c89e5348211d93ff12ac2a8` | GPL-3.0. Called by an external service; no inference code copied into the controller. |
| F5-TTS | `27e20fcf39fbce3f187a61d4a3e92065d2d9b947` | MIT source; operator-installed only, not fetched or launched by X·Edit. Selected weights remain CC-BY-NC-4.0. |
| Qwen3-ASR | `c17a131fe028b2e428b6e80a33d30bb4fa57b8df` | Apache-2.0; service adapter imports this source. |
| Qwen3-TTS | `1ab0dd75353392f28a0d05d9ca960c9954b13c83` | Apache-2.0; service adapter imports this source. |
| TED-TTS | `36ffc3e2de346156baa7d60a1749ca4a9365625b` | Outer license: Apache-2.0; also ships IndexTTS model agreements under licenses/. Do not treat the combined backend as unrestricted. |

Full URLs, source pins and patch hashes are available through
[the pipeline manifest](../third_party/external_pipeline.json) and upstream
`tools/*/upstream.json`. Official licenses remain in the initialized submodules.
Other tool/model declarations remain in [the inventory](../third_party/licenses.json).
No weights or generated audio are distributed. F5's MIT code does not remove the
CC-BY-NC conditions on the selected model weights.

## Active and historical model evidence

The table preserves source evidence, not a blanket installer list. SAM, T5,
VoiceSculptor and XCodec2 records are historical/documentation-only extension
references in [extension_references.json](extension_references.json), not active
entries in `third_party/manifest.json`. The corresponding six source records are
also retained solely as historical license evidence. No extension is supplied.

### Model declarations

| Component | Upstream declaration / document | Identity |
| --- | --- | --- |
| ASLP-lab/VoiceSculptor-VD | [apache-2.0](https://huggingface.co/ASLP-lab/VoiceSculptor-VD) | `commit: 6d7a5573960727b00a5b8f3de032a22dfee7f16e` |
| BUT-FIT/SE-DiCoW | [apache-2.0](https://huggingface.co/BUT-FIT/SE-DiCoW) | `commit: 470fce9ffff844dd53a27751cdf6c6df9efecb39` |
| BUT-FIT/diarizen-wavlm-large-s80-md | [cc-by-nc-4.0](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) | `commit: 6285693ddd5b38e8229acb93f864f3d04a82bee1` |
| BUT-FIT/diarizen-wavlm-large-s80-md-v2 | [cc-by-nc-4.0](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) | `commit: f27b9ffbedcf422856d104ecee9b94be37ea578e` |
| HKUSTAudio/xcodec2 | [cc-by-nc-4.0](https://huggingface.co/HKUSTAudio/xcodec2) | `commit: e412427ed30f0cf9d5e3c95562113deb10a32d03` |
| Higobeatz/FlexSED | [mit](https://huggingface.co/Higobeatz/FlexSED) | `commit: eefe52b7ad686a9bc9f1f5dd0803e2c52171e128` |
| IndexTeam/IndexTTS-2 | [LicenseRef-bilibili-Model-Use](https://huggingface.co/IndexTeam/IndexTTS-2/resolve/740dcaff396282ffb241903d150ac011cd4b1ede/LICENSE.txt) | `commit: 740dcaff396282ffb241903d150ac011cd4b1ede` |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | [apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | `commit: 4a1af868018e7974197f4f018730758012b28c27` |
| Plachta/Seed-VC | [gpl-3.0](https://huggingface.co/Plachta/Seed-VC) | `commit: 257283f9f41585055e8f858fba4fd044e5caed6e` |
| Politrees/UVR_resources | [mit](https://huggingface.co/Politrees/UVR_resources) | `commit: 929e057b81aa49bc2e6490bef8671f47b2c120f6` |
| Qwen/Qwen2.5-Omni-3B | [LicenseRef-Qwen-Research](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/resolve/f75b40e3da2003cdd6e1829b1f420ca70797c34e/LICENSE) | `commit: f75b40e3da2003cdd6e1829b1f420ca70797c34e` |
| Qwen/Qwen3-ASR-1.7B | [apache-2.0](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `commit: 7278e1e70fe206f11671096ffdd38061171dd6e5` |
| Qwen/Qwen3-ForcedAligner-0.6B | [apache-2.0](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | `commit: c7cbfc2048c462b0d63a45797104fc9db3ad62b7` |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | [apache-2.0](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `commit: 26291f793822fb6be9555850f06dfe95f2d7e695` |
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | [apache-2.0](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | `commit: fd4b254389122332181a7c3db7f27e918eec64e3` |
| Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | [apache-2.0](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `commit: 0c0e3051f131929182e2c023b9537f8b1c68adfe` |
| Qwen/Qwen3-TTS-Tokenizer-12Hz | [apache-2.0](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) | `commit: 7dd38ad4e9bad454aae9cd937d0cd577604fe229` |
| Qwen/Qwen3.8-27B | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) | `commit: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| SWivid/F5-TTS | [cc-by-nc-4.0](https://huggingface.co/SWivid/F5-TTS) | `commit: 84e5a410d9cead4de2f847e7c9369a6440bdfaca` |
| alibabasglab/MossFormer2_SE_48K | [apache-2.0](https://huggingface.co/alibabasglab/MossFormer2_SE_48K) | `commit: eff8c97925c8bec812af707814b3e5d777fd4503` |
| amphion/MaskGCT | [cc-by-nc-4.0](https://huggingface.co/amphion/MaskGCT) | `commit: 265c6cef07625665d0c28d2faafb1415562379dc` |
| charactr/vocos-mel-24khz | [mit](https://huggingface.co/charactr/vocos-mel-24khz) | `commit: 0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` |
| facebook/w2v-bert-2.0 | [mit](https://huggingface.co/facebook/w2v-bert-2.0) | `commit: da985ba0987f70aaeb84a80f2851cfac8c697a7b` |
| funasr/campplus | [apache-2.0](https://huggingface.co/funasr/campplus) | `commit: e4b6ede7ce16997aff4ae69fbca1f0175e2afede` |
| google-t5/t5-base | [apache-2.0](https://huggingface.co/google-t5/t5-base) | `commit: a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` |
| laion/clap-htsat-unfused | [apache-2.0](https://huggingface.co/laion/clap-htsat-unfused) | `commit: 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` |
| mrfakename/sam-audio-large | [LicenseRef-SAM](https://huggingface.co/mrfakename/sam-audio-large/resolve/497809f191dd4f673a2c93557176387cf2a06cf2/LICENSE) | `commit: 497809f191dd4f673a2c93557176387cf2a06cf2` |
| nvidia/bigvgan_v2_22khz_80band_256x | [MIT](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/resolve/633ff708ed5b74903e86ff1298cf4a98e921c513/LICENSE) | `commit: 633ff708ed5b74903e86ff1298cf4a98e921c513` |
| openai/whisper-small | [apache-2.0](https://huggingface.co/openai/whisper-small) | `commit: 973afd24965f72e36ca33b3055d56a652f456b4d` |
| pyannote/wespeaker-voxceleb-resnet34-LM | [cc-by-4.0](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | `commit: 837717ddb9ff5507820346191109dc79c958d614` |
| tencent/AuK | [MIT](https://huggingface.co/tencent/AuK/resolve/790742b71a4430120daf2b2099192abae449eb9f/LICENSE) | `commit: 790742b71a4430120daf2b2099192abae449eb9f` |

## Separately fetched service metadata

These five files are excluded from Git and source/wheel distributions. Deploying
RoFormer downloads them from the URLs below and verifies each SHA256. Download
does not change their upstream terms or establish checkpoint rights.

| Component | Upstream declaration / document | Identity |
| --- | --- | --- |
| service_assets/roformer/download_checks.json | [Upstream metadata/configuration; no checkpoint license inferred](https://raw.githubusercontent.com/TRvlvr/application_data/3826b05b570dbd4fbedbc807758803b35348ba1b/filelists/download_checks.json) | `license/asset SHA256: d3622e1fa19c161d3cf704927711b453d593a3f1eb0f2e0838c3136907935151` |
| service_assets/roformer/vr_model_data.json | [Upstream metadata/configuration; no checkpoint license inferred](https://raw.githubusercontent.com/TRvlvr/application_data/3826b05b570dbd4fbedbc807758803b35348ba1b/vr_model_data/model_data_new.json) | `license/asset SHA256: 6a2021b9aa355c10f845b68c73944d47bd9e1fdf37b8a584b14ce8f88b09bb2f` |
| service_assets/roformer/mdx_model_data.json | [Upstream metadata/configuration; no checkpoint license inferred](https://raw.githubusercontent.com/TRvlvr/application_data/3826b05b570dbd4fbedbc807758803b35348ba1b/mdx_model_data/model_data_new.json) | `license/asset SHA256: 1aca8f9bcc57233bc714029663a9ec2345d9c7721f91e5e08f46392a879c6a9a` |
| service_assets/roformer/model_bs_roformer_ep_317_sdr_12.9755.yaml | [Upstream metadata/configuration; no checkpoint license inferred](https://raw.githubusercontent.com/TRvlvr/application_data/3826b05b570dbd4fbedbc807758803b35348ba1b/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml) | `license/asset SHA256: 2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f` |
| service_assets/roformer/vocals_mel_band_roformer.yaml | [Upstream metadata/configuration; no checkpoint license inferred](https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/vocals_mel_band_roformer.yaml) | `license/asset SHA256: b958b29c8f7195f0d86bee6759a33980db675c4ecaf2fcaa80fa125828e6cd38` |

## Explicit provenance limits

- Seed-VC: the current runtime fetches the official commit listed above. The exact source revision of the former vendored copy remains unknown; this historical limitation does not apply to the current pinned checkout. Older source-project commits may contain the former SDK copy; this final repository starts a clean history.
- SAM (historical, not currently configured): the former mirror includes the SAM License. Identity with the [official checkpoint](https://huggingface.co/facebook/sam-audio-large) is not asserted. Download availability does not replace the original terms.
- RoFormer: the model collection card declares MIT; checkpoint-specific rights and the separate configuration/metadata terms are not established by that label. Upstream URLs and exact hashes remain recorded.
- IndexTTS-2: the pinned card has no license metadata field, but includes the bilibili agreement; the inventory uses that document.
- Qwen3-Omni: the pinned card declares `license_name: apache-2.0` under `license: other`; this is recorded as card evidence.
- This inventory covers direct model/source pins and included components. Installed Python packages retain their own licenses, available with the deployment environment inventories.

No personal author identity, authentication material, weights or evaluation audio is added to this inventory. Required third-party attribution is retained.

IndexTTS and Seed-VC SDKs are fetched through the pipeline's official gitlinks.
The pipeline's adapters and patches remain subject to applicable upstream
conditions. This is not an Apache-only inference stack. Consult the exact
IndexTTS agreement for its large-organization restrictions and authorization
conditions. Seed-VC's GPLv3 is not a non-commercial license.
