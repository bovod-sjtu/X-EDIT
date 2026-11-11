# Fresh-container validation

> Historical source versions only. SAM and VoiceSculptor are no longer included
> tools or code integrations. These results do not validate the current toolset
> or the provider-neutral pipeline revision. See [release scope](RELEASE_SCOPE.md).

## External SDK follow-up

On 2026-10-01, the IndexTTS and Seed-VC external-source adapters passed a separate
fresh-container qualification using the same CUDA 12.8.1 / Ubuntu 22.04 base
image and two NVIDIA L20Z 80 GB GPUs. The public installer fetched both official
source pins, seven model repositories and new Python environments. IndexTTS
emotion/token synthesis, cached continuation and independent tagged segments,
plus Seed-VC explicit/default seed conversion, all generated valid audio.
All five decoded sample hashes matched the earlier local regression. Official
SDK working trees remained clean after inference; generated normalization
caches were outside them.

The successful worker completed in 541 seconds, including OS installation,
downloads, environment installation, service loading and operation probes,
excluding scheduling/image preparation and the prior failed attempt. A download
timeout from an NVIDIA package host was fixed by preferring PyPI for equal
versions while preserving CUDA local builds. Installed package inventories
remained identical (164 packages for IndexTTS; 261 for the editing GPU environment).
See [the structured result](backend_cold_validation.json) for scope, hashes and
stage times. This follow-up covers these two backends; the broader initial
validation below remains a separate result.

## Environment and scope

On 2026-09-29, the integrated source was installed on a fresh GPU container
using `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`: Ubuntu 22.04, eight NVIDIA L20Z
80 GB GPUs, driver 535.161.08, 96 CPUs and 768 GiB RAM. OS packages, controller,
13 isolated backend environments, pinned source dependencies and public model
snapshots were prepared from the supplied recipes. The model manifest contains
31 repositories, including auxiliary encoders and speaker embeddings.

The source and each correction were frozen before execution. Persistent storage
held downloads, input recordings and receipts; environments were newly created
on node-local disk. Services used node-local endpoints. Mounted development
checkouts and old service environments were not inference dependencies.

The validation iteratively repaired installation and service-boundary failures;
the initial archive did not pass unchanged. All successful corrections are now
included in this source tree. This is a runnable release-candidate acceptance,
not a reproduction of benchmark accuracy.

## Actual checks

| Layer | Operation / result |
| --- | --- |
| OS and controller | Fresh apt/uv installation, FFmpeg/ffprobe and controller construction |
| GPU environments | Actual CUDA matrix operation in all 12 GPU environments; CPU environment installed separately |
| Native dependency | Generated-video decode through SAM's installed decord native library |
| Decision and audio models | Newly started Qwen text server answered a request; Omni processed actual audio |
| Speech editing | ASR, word alignment and F5 tagged train→bus replacement generated readable audio |
| Speech synthesis | Qwen TTS, reference cloning, IndexTTS and VoiceSculptor generated audio |
| Expert observation | FlexSED detection, English SE-DiCoW attributed transcript and Chinese MOSS attributed transcript |
| Voice and audio transformation | Seed-VC reference conversion, AuK whisper edit, SAM target/residual separation and ClearVoice enhancement |
| Composite services | Emotion, prosody, pause insertion, smoothing, assembly, pitch and VAD requests completed |
| Numerical DSP | 1.25× rate produced a 0.8000 duration ratio; −12 dBFS peak normalization measured −11.99995 dBFS |
| Full text-edit Forward | Fresh perception→Reasoner→Graph→tool execution produced edited speech |
| Multi-turn Forward | −6 dB followed by 1.25× speed reused the preceding output; gain and duration were checked numerically |
| Multi-asset Forward | English source and separate reference voice produced a Seed-VC result |

Both BS-RoFormer and MelBand-RoFormer passed on the original 2.376875-second
recording and a 14.26125-second diagnostic repetition. Short-input outputs were
trimmed to the source duration (within output-sample rounding); long-input
outputs were unchanged by the preparation path.

Failure-driven revision also completed: a test-only proxy returned HTTP 503 for
the first attempt's two transport calls. The next outer attempt consumed the
actual error, generated speech, passed the semantic gate and completed numeric
positive/reverse audio review and candidate selection. It selected attempt 2;
review error lists were empty and original evidence remained unchanged. The
proxy is not part of the released runtime. No-delivery attempts did not receive
product listening judgments.

All four end-to-end checks delivered audio. Independent ASR of the simple
Forward returned “Next bus will leave at six fifteen.”, omitting the initial
article; the revision output returned “The next bus will leave at 6:15.”. This
qualifies execution and requested word replacement, not perfect preservation
for every generated output. Numeric-vs-word transcript formatting also remains
visible in the strict transcript qualification record.

These operation checks primarily establish successful execution and readable,
nonempty finite audio. They do not establish subjective emotion, voice identity,
background isolation or artifact-free quality. All input recordings were
development diagnostics; benchmark answers were not used for deployment fixes.

## Failures corrected in the release

| Original failure | General correction |
| --- | --- |
| apt could not replace a read-only injected GL driver file | Exclude only the exact injected mount from that apt invocation; preserve its hash |
| Hub rate limits interrupted fresh downloads | Limited concurrency and bounded transient retry; authorization failures remain visible |
| Offline model lookup missed an existing snapshot | Write HF refs as exact commit bytes without a trailing newline |
| SE-DiCoW lacked a speaker embedding and imported an incompatible older DiariZen | Declare the auxiliary checkpoint and prioritize the pinned standalone source |
| Pause insertion's empty boundary query caused a word-aligner HTTP 500 | Return the explicit unsupported-boundary signal so the existing neighboring-word fallback can run |
| Marked alignment could absorb preceding words or the wrong occurrence | Bind the target using complete marked transcript context; reject unresolved ambiguity |
| RoFormer rejected a source shorter than one inference chunk | Right-pad for inference, trim every output stem to source duration, retain preparation metadata |

For another host, follow [DEPLOYMENT.md](DEPLOYMENT.md), including the actual CUDA
and operation checks. Health-only startup is insufficient. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for source parity, metadata exceptions and
the remaining release/quality boundaries.
