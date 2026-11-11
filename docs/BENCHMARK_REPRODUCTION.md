# Framework execution and benchmark reproduction

The supplied `configs/` reproduce the release's default explicit-Reasoner,
Graph, expert-evidence, two-additional-observation-round workflow. `PROVENANCE.json`
records its configuration lineage. Forward and sequential revision use the same
runtime. An installation or waveform check does not reproduce a paper score.

Use the public `x-edit run --request ...` scenario interface for benchmark
adapters. Each adapter must provide the source/reference assets and only the
instructions already visible at that turn. For later turns, use the preceding
actual output when required by the benchmark. Keep task IDs, asset hashes,
configuration hashes, raw results and failure records in a separate run folder.
Do not insert benchmark answers or terminal scoring questions into planning.

For MMAE, obtain the dataset and scorer from the
[official project](https://github.com/ddlBoJack/MMAE). Pin their exact versions
and retain the native task population and scoring protocol. Record:

- Framework source identity and configuration, service/weight revisions,
  decoding settings, GPU and serving versions.
- Delivery and fallback policy. A source-audio fallback is not an edited delivery.
- Three valid native judgments per question under the specified majority rule,
  with samples parallelized and questions within a sample scored sequentially.
- Complete evaluation-input hashes and isolated cache namespaces for independent
  scoring batches. Record invalid/missing answers rather than treating them as
  negative votes or complete scoring.
- IFR, CR, EMR and valid-delivery-and-all-checks-pass, using the appropriate
  benchmark denominators. Online revision proxy scores are separate.

When claiming scoring variation, keep execution artifacts fixed across scoring
batches. When claiming execution variation, regenerate the artifacts and identify
the changed run. Fixed seeds alone do not guarantee identical neural outputs
across devices or kernels.

This document specifies the reproduction interface and accounting. The complete
paper-specific adapters, historical frozen snapshots and end-to-end scoring
launcher are not yet included in this clean runtime release. That separate bundle
must be qualified before describing the paper tables as one-command reproducible.
