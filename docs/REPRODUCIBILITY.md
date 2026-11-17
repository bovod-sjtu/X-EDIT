# Reproducibility

Record the X·Edit Git revision and any local changes before each experiment.
Do not infer the tested source from a mutable branch name.

## Keep three boundaries separate

- **Source:** framework, prompts, configuration, original adapters, and dependency pins.
- **Deployment:** separately obtained SDKs, checkpoints, per-service uv environments,
  endpoint configuration, and installation receipts.
- **Runs:** input provenance, selected tools/backend, effective model settings,
  instructions, execution traces, generated audio, errors, and evaluation results.

Deployment assets and runs belong outside the source tree. Never copy private
debug launchers, cluster configuration, credentials, or benchmark artifacts into
a release to make an example work.

## Record the configuration that actually ran

Use `third_party/external_pipeline.json` for the pipeline Git revision and SDK
pins; use `third_party/manifest.json` for independently obtained sources and
models. Match these with the installed source, model-download receipt, lock
files, GPU/driver versions, and service startup settings. Declared revisions in
HTTP responses are not independent attestations of local files.

Keep the tool selection and backend choice with each run. Record Reasoner and
revision settings, prompt/configuration revisions, sampling parameters, seeds,
audio preprocessing, and service endpoints. Fixed seeds do not guarantee
byte-identical neural audio across hardware, kernels, or dependency versions.

## Validate at the right level

1. Source/package audits check distribution boundaries and configuration wiring.
2. Client/schema/startup checks verify interfaces without proving inference.
3. Real operation checks execute services and inspect returned audio.
4. Transcript checks, signal measurements, and listening assess edit correctness.
5. Benchmarks use their documented selection, failure handling, and official scorer.

Do not present a healthy endpoint or a readable WAV as a quality score.
Preserve failures and source-audio fallbacks explicitly in evaluation records.
Do not reuse scorer answers as perception or planning inputs.

Current evidence is summarized in [VALIDATION.md](VALIDATION.md). Older GPU
results and scores belong to their original releases; upgrading the public
pipeline or V1.1 weights requires new evidence.
