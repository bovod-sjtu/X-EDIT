# Source distribution checks — 2026-10-22

> Historical record. The 2026-10-26 source boundary removes SAM/VoiceSculptor
> integrations and requires independent F5 setup. The unchanged-code statement
> below applies only to this historical comparison, not the current release.
> See [the current statement](OPEN_SOURCE_STATEMENT.zh-CN.md).

The current distribution boundary is [RELEASE_SCOPE.md](RELEASE_SCOPE.md).
These checks concern the original project files and built packages. External
checkouts, complete model deployments and Git history are separate scopes.

## Preserved behavior

The controller Python files, configurations, managed service graph, model
revisions and six pipeline SDK pins were compared with the pre-change baseline.
They are unchanged. The external pipeline manifest changes only its explanatory
license note. All 26 lockfiles retain identical non-comment dependency
specifications. No replacement model was activated.

The provider-neutral `deployment/service_entrypoint.sh` retains the plan/start/
supervise argument flow. This was tested with a command-recording stub, without
starting services. The optional target-speaker extraction checker now honors
`QUARKAUDIO_UNISE_REPO_DIR` and uses `python3` when no local interpreter exists.
Its external-path check passed with a temporary fixture; no model inference is
implied by that file-presence check.

## Actual package checks

- Built the source archive and controller wheel with `uv build`.
- Inspected both archives for external source, model weights, audio, native
  executables, environment files, common credential patterns and private cluster
  deployment references. No such content was found.
- Copied the project to a temporary directory, inserted representative external
  source, checkpoint configuration, model cache, audio, native-library and private
  environment fixtures, then built both distributions. Both excluded the fixtures.
- Checked that the content scanner rejects model/media/native-binary files,
  credential and infrastructure fixtures, and a path-traversing archive member.
  Ordinary mathematical uses of `pi` remain accepted.
- Rechecked the isolated FFmpeg 8.1.2 prefix: source checksum, shared build,
  required disabled features and license materials passed. This reuses the build
  and conversion evidence recorded in `dependency_remediation_validation.json`.

Source syntax, shell syntax, export hashes and service references also passed.
The content report supplied with the review artifacts identifies the actual
archive and wheel by SHA256. It is separate from the model-stack license report,
which continues to list restricted or unreviewed components.

## Limits

There was no GPU inference, new model download, benchmark rerun, subjective audio
assessment or complete Docker image build in this update. Preserving source and
pins does not certify every runtime environment. Content checks do not establish
copyright ownership, change third-party terms or record legal approval.

The upstream reference now contains an Apache-2.0 license at a newer revision;
the historical runtime pin still requires authorized source access. This update
does not substitute that newer service code. Existing Git history and separately
initialized third-party repositories were not rewritten or republished.
