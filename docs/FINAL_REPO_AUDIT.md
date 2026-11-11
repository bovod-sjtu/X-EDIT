# Final repository audit

> Historical snapshot (2026-10-08 and earlier). The current pipeline layout and
> verification are in [PIPELINE_MIGRATION.md](PIPELINE_MIGRATION.md). These earlier
> results do not certify the new backend revision.

This integrates the release candidate and the qualified IndexTTS/Seed-VC
external-source migration. The previous working tree is preserved. A new Git
history avoids inheriting removed SDK copies, private experiments or credentials.

For anonymous review, distribute the source archive without `.git`. A private
hosting server may require a verified committer identity; do not mirror that
private Git history into an anonymous public repository. Component attribution
must still be retained.

## Historical code and service boundaries

The paths and included SDK material in this table describe the earlier snapshot.
The current release excludes external source; see [RELEASE_SCOPE.md](RELEASE_SCOPE.md).

| Layer | Location and execution | License scope |
| --- | --- | --- |
| Agent | `x_editor/`, presets and CLI; controller environment | Original code: Apache-2.0 |
| Original editing orchestration | `third_party/dots_tts_edit/pipelines` and declared wrappers; separate HTTP services | Original files enumerated in `LICENSE_SCOPE.json`: Apache-2.0 |
| Permissive SDK integrations | Qwen ASR/TTS and F5 in `third_party/dots_tts_edit/tools`; backend environments | Qwen: Apache-2.0; F5: MIT; retained notices |
| IndexTTS / Seed-VC | `backends/` adapters and compatibility edits; official pinned checkouts, separate processes, HTTP clients | Original backend terms remain; not relabeled Apache |
| Other expert/editing dependencies | Public source/model pins and launch recipes; download or explicit existing endpoints | Upstream inventory: `third_party/licenses.json` |
| Models and audio | Not distributed; operator-managed snapshots and output paths | Separate model/input permissions |

IndexTTS/Seed-VC official checkouts are unchanged on disk; exact compatibility
edits are verified and compiled in memory. This does not waive upstream license
conditions. That earlier snapshot retained some permissive SDK source with notices
to preserve tested behavior. The current source package excludes those SDKs.
RoFormer configuration/metadata is fetched and hash-checked during deployment;
its five files are excluded from Git and release packages. Its separate license
scope remains recorded as unresolved. See [LICENSES.md](LICENSES.md).

## Evidence and limits

| Scope | Observed result | Boundary |
| --- | --- | --- |
| Historical cold installation | 28-service profile, 13 environments, 31 model repositories | Installation repaired during acceptance; not every base image tested |
| Historical agent panel | 24 completed, 23 delivered | One graph-output failure; not 23 verified quality successes |
| Numeric edits | −6 dB gain and 1.25× rate measured | A 0.3 s requested pause became 0.2 s |
| External SDK migration | IndexTTS 3/3, Seed-VC 2/2 in a fresh GPU container | Optional acceleration/inference modes not covered |
| Current final-tree checks | [final_validation.json](final_validation.json) | Separate from historical GPU acceptance |

Detailed evidence: [sample panel](sample_validation.json),
[cold GPU scope](COLD_START_VALIDATION.md),
[backend cold test](backend_cold_validation.json).
These are deployment/regression diagnostics, not historical paper-score reproduction.

## Final-integration repairs

- External endpoints now prevent local service/dependency installation and are
  explicitly identified in deployment plans.
- `.env.example` no longer points at a nonexistent legacy pipeline directory;
  managed deployments use generated `client.env` bindings.
- RoFormer metadata/configuration downloads have exact hashes; changed files
  are rejected. The source release no longer redistributes these upstream assets.
- Service/license inventory and sample evidence are consolidated, including failures.
- `deployment/audit.py` checks pins, hashes, service wiring, syntax and backend
  import boundaries without contacting models.

Agent algorithms, prompts, budgets and scoring rules are retained. Installer
changes do not establish new end-to-end quality results.

## Remaining work

1. Reproduce and repair the background-preserving planning failure and exact
   pause-duration mismatch before claiming these editing behaviors fixed.
2. Listen independently to emotion, voice identity, background preservation and
   local transitions; valid waveforms are insufficient.
3. Qualify long audio, concurrency, lower-memory hardware and complete Docker
   builds separately. The full GPU map assumes eight 80 GB GPUs.
4. Benchmark reproduction needs dataset acquisition/adapters and frozen scorer
   mappings; private research launchers are intentionally excluded.
5. Resolve remaining third-party asset licensing for broader redistribution;
   retain component notices and any required upstream permissions.
