# Service deployment inventory

Controller/tool separation is through HTTP. Each inference environment is isolated. See [pipeline migration](PIPELINE_MIGRATION.md) for current validation and public-access boundaries.

The Environment column names dependency lock families, not shared virtualenvs.
New deployments use `<env-root>/services/<service>` for every managed service;
see [module deployment](MODULE_DEPLOYMENT.md). Use `bootstrap --phase plan --tools`
for the authoritative dependency closure of the selected capabilities.

| Service | Environment | Dependencies | Pipeline SDKs / other sources |
| --- | --- | --- | --- |
| `dots_tts_edit` | operator-installed model environment | — | dots.tts; default tagged-edit backend, see [setup](DOTS_TTS_EDIT.md) |
| `asr` | `asr` | — | X-Edit context-aware HTTP adapter + separately obtained qwen3-asr SDK |
| `aligner_backend` | `edit_gpu` | — | qwen3-asr |
| `aligner` | `edit_cpu` | aligner_backend | pipeline service |
| `f5` | operator-installed only | — | Independently obtained f5-tts |
| `indextts` | `indextts` | — | index-tts |
| `tts` | `edit_gpu` | — | qwen3-tts |
| `tts_clone` | `edit_gpu` | — | qwen3-tts |
| `seed_vc` | `edit_gpu` | — | seed-vc |
| `vad` | `edit_cpu` | — | pipeline service |
| `volume` | `edit_cpu` | — | pipeline service |
| `rate` | `edit_cpu` | — | pipeline service |
| `pitch` | `edit_cpu` | — | pipeline service |
| `smoother` | `edit_cpu` | aligner, f5, vad | pipeline service |
| `assembly` | `edit_cpu` | aligner, smoother | pipeline service |
| `edit_text` | `edit_cpu` | aligner, f5 | pipeline service |
| `edit_prosody` | `edit_cpu` | aligner, f5, pitch, rate | pipeline service |
| `edit_emotion` | `edit_cpu` | indextts, aligner, f5, assembly, vad, smoother | pipeline service |
| `edit_pause` | `edit_cpu` | aligner, smoother | pipeline service |
| `planner` | `model_server` | — | pipeline service |
| `omni` | `model_server` | — | pipeline service |
| `flexsed` | `flexsed` | — | FlexSED |
| `diarization` | `sedicow_asr` | — | DiariZen; independent `/diarize`, not the SE-DiCoW endpoint |
| `sedicow` | `sedicow_asr` | — | DiariZen, DiCoW |
| `moss` | `moss_transcribe_diarize` | — | MOSS-Transcribe-Diarize |
| `auk` | `auk` | — | AuK |
| `enhancement` | `clearer_voice_studio` | — | installed package / X-Edit adapter |
| `separation` | `audio_separator_roformer` | — | installed package / X-Edit adapter |
| `ted_tts` | `indextts` | — | ted-tts |

Total: 29 service interfaces, including operator-installed F5 and dots.tts.edit;
27 locally managed services and 11 managed environments. The separate dots
environment is user-installed. The active provenance inventory has six SDK
gitlinks (five fetched, F5 independently acquired), six other source repositories
and 28 model repositories. Counts are disclosure inventories, not a guarantee
that every model is needed or installed.

SAM and VoiceSculptor are not in this table, the catalog or runtime registry.
Their former exclusive assets moved to documentation-only provenance records.
See [extension guidance](EXTENSION_GUIDE.md) and [F5 setup](EXTERNAL_TOOLS.md).

Use `bootstrap.py --phase plan SERVICE` before downloading. An explicitly configured external service does not install or start a local copy. CPU rate/pitch DSP modes need no neural backend; resynthesis modes need the selected backend. TED-TTS adds no weights but consumes a separate GPU process.

Older cold-start reports retain their original source identity and do not certify this migration. No benchmarks or subjective-quality scores are inferred from service health.

## Additional operator-supplied integrations

The catalog also retains these optional interfaces outside the 29-service
stack. Their model/source revisions are not selected by that stack. Before using
one, obtain its implementation and weights separately, record exact versions and
review their terms. Keeping the interface does not install or approve a model.

| Interface | External implementation | Purpose |
| --- | --- | --- |
| `ace_step_music` | [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) | Music and vocal generation/editing through an operator-provided API |
| `seamless_expressive_s2st` | [Seamless Communication](https://github.com/facebookresearch/seamless_communication) | Expressive speech translation; model and auxiliary vocoder terms are separate |
| `tse` / `quarkaudio_unise` | [QuarkAudio-UniSE](https://github.com/alibaba/unified-audio/tree/main/QuarkAudio-UniSE), with [Spark-TTS assets](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Reference-conditioned target-speaker extraction |

No source checkout, checkpoint or generated recording from these integrations is
bundled. Optional revisions and their transitive components require independent
review; they are not covered by the pinned-stack inference evidence.

See [capability parity and qualification](SERVICE_CAPABILITIES.md) for the mapping
from enabled tools to these service endpoints. `/health` and route presence alone
do not certify contextual ASR, speaker attribution, clone synthesis or composite
editing. The full tool profile does not silently enable documentation-only models.
