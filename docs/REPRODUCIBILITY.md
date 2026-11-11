# Reproducibility status — 2026-09-29

> Historical snapshot (2026-10-08 and earlier). The current pipeline layout and
> verification are in [PIPELINE_MIGRATION.md](PIPELINE_MIGRATION.md). These earlier
> results do not certify the new backend revision.

This integrated source release candidate has passed actual installation and
inference in a fresh GPU container. It no longer requires a sibling private
editing-pipeline checkout or pre-existing model-service environments. The
precise cold-start scope is in [COLD_START_VALIDATION.md](COLD_START_VALIDATION.md);
installation commands and model pins are in [DEPLOYMENT.md](DEPLOYMENT.md).

## Source and behavior

The original source is the `X-Editor-Open` working tree, including uncommitted
fixes. `SOURCE_EXPORT.json` records per-file hashes; a Git commit alone would
not identify that tree. Research workspaces were not cleaned or overwritten.
Tests, experiment launchers, datasets, paper files, run traces and model weights
are excluded from the distribution. Validation tests and private traces are
retained outside the release.

The Reasoner, Planner, perception policies, program compilers, executor and
evaluation/selection algorithms are retained. Required review functions formerly
under `examples/evaluation` are packaged under `x_editor.revision`: 78 relocated
function/class definitions have identical ASTs after import-path normalization.
The source audit identifies 344 unchanged shared Python files and explicitly
records the following shared-file changes:

- `core/config.py`: credentials come from explicit environment/file inputs;
  no implicit searches of sibling projects or shell scripts.
- `tools/mcp/client.py` and `accent_library/store.py`: shared workspace discovery
  requires explicit configuration rather than a guessed cluster username.
- `evaluation/iterative_runtime.py`: import relocation into the package.
- The SE-DiCoW adapter selects the declared standalone DiariZen repository before
  an older embedded submodule and validates that source path.
- The RoFormer adapter prepares short inputs with right padding and trims all
  returned stems back to the original duration, recording the preparation.

The vendored alignment wrapper also reports unsupported empty boundary queries
to the pause pipeline and binds marked query occurrences using full transcript
context. These are service-boundary repairs, not claims of byte-identical
behavior for every request. Their provenance is retained in the vendored source
manifest. Weights, planning prompts and evaluation rules were not changed to
make validation pass.

`api.py`, `cli.py` and deployment helpers compose the existing runtime. Immutable
scenario receipts support revision import; semantic and terminal question
registries use distinct cache namespaces. The default preset is the corrected
explicit-Reasoner/Graph/R2 configuration, identified in `configs/PROVENANCE.json`.
It does not promise reproduction of a historical benchmark score when paired
with newer source, hardware or model servers.

## Verification levels

Earlier preparation passed 82 existing runtime regression checks and six API
checks, a fresh controller install, and wheel imports/Forward construction from
outside the source tree. Those are separate from the subsequent GPU cold start.
The container check used public model downloads, new isolated backend environments and
run-owned services. No development service endpoint or old virtualenv was used
for its actual inference. Failed installation and operation attempts were kept,
then repaired requests were replayed in separate output directories.

Readiness, valid waveforms, objective edit measurements and perceptual quality
are different criteria. A health response is not a model-operation pass; a
readable WAV is not proof of all user requirements. For example, a strict ASR
comparison failed on `6:15` versus `six fifteen` after a real lexical edit.
That failure remains recorded; the expected transcript was not rewritten.

## Remaining boundaries

- The tested deployment is the supplied 28-service profile, on eight 80 GB L20Z
  GPUs with short diagnostic recordings and sequential clients. Unconfigured
  optional catalog backends, long recordings and higher concurrent load are not
  covered by this acceptance.
- This was a fresh-container install, not a separately qualified complete
  `Dockerfile.gpu` build. Arbitrary base images, GPU drivers and GPU placements
  require their own CUDA/model-operation checks.
- Public downloads used pinned revisions. Operator-supplied weight paths still
  require file-identity verification; a path or a declared revision is not proof
  that arbitrary files match it.
- The retained serving snapshot has two narrow Transformers dependency-metadata
  conflicts. Its actual BF16 Qwen inference passed. Quantized compressed-tensor
  models are not qualified. SAM's decord wheel metadata exception additionally
  requires a real native decode check; other conflicts remain errors.
- Model quality, speaker/background preservation and evaluator reliability are
  not certified by deployment tests. Fixed seeds do not promise identical
  neural outputs across hardware, kernels or serving versions.
- Original pipeline orchestration, service adapters and contributor modifications
  are now explicitly covered by Apache-2.0, following contributor authorization.
  Upstream components retain their licenses; see [LICENSES.md](LICENSES.md) for
  code/model distinctions and remaining upstream provenance limits. Model
  weights are not redistributed in this review artifact.

## Observed evaluator limitation

An earlier sequential local check gave both positive and reverse loudness
questions a score of 3 on an objectively verified −6 dB edit. Its direction
disagreement was retained, producing 1.5/3; no scoring rule was changed to force
agreement. The selector retained the delivered waveform without an unsupported
retry. This qualifies routing and record handling, not the accuracy of that
judgment. Keep objective measurements and human listening separate from the
online proxy grades.
