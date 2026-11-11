# Release notes

## 2026-11-11 — 独立部署与公开源码整理

- README 改为中文上手说明，先解释框架用途、要部署哪些组件，再介绍配置、运行和许可边界。
- 新建受管部署按服务分别建立 uv 环境；共用锁文件不再意味着共用环境目录。
  控制器、MCP 客户端和 GPU 服务三层分离，旧部署布局不被静默改写。
- MCP 安装按工具选择执行，过滤父进程的 Python/uv 环境覆盖；显式客户端环境目录下禁止
  运行时自动同步，缺失环境直接报错。补齐编辑客户端的 PyYAML 依赖。
- 提供独立客户端初始化检查、IPv4 监听地址配置、各角色端点与跨机器文件共享说明。
  CUDA 自检按实际解释器及设备组合检查，不再用一个同锁环境代表多个独立环境。
- 本地验证：11 项部署隔离回归、31 项既有管线对齐回归通过；完整工具选择中的 9 个
  MCP 子进程在新安装的独立环境中初始化并列出工具，本地音效执行和 F5 禁用提示通过。
  这些检查不加载第三方 GPU 模型，不是全量编辑或评分结果。
- 较早的外部工具回归仍引用已移除的 SAM 客户端、旧服务数量、旧默认后端和旧 ASR
  健康响应，不能原样用作本版全部通过的证据；未为通过旧断言恢复已移除组件。
- 保留第三方许可与服务运营责任声明。公开仓库使用审查后的源码快照，不携带旧仓库
  历史、第三方源码实体、模型、用户音频、实验产物、调试代码或凭据。
  本轮不修改规划/推理算法、提示词、模型参数，不重跑 GPU benchmark。

## 2026-10-30 — service capability parity

- Accept pinned upstream adapters that no longer provide optional `upstream.json`
  metadata. Continue checking the outer revision, clean SDK gitlinks, nested
  repositories and any declared patch hashes; never infer model readiness.
- Restore contextual ASR transport in an original HTTP adapter using the pinned
  external SDK; require schema and active-backend support at startup and in live
  diagnostics. Keep model hints separate from returned transcription.
- Add independent diarization deployment, complete tool-to-service mapping and
  `configs/tools.full.yaml`; `bootstrap --tools` uses the runtime selection.
- Correct rate/pitch token decoding to the Base+Tokenizer endpoint and include
  IndexTTS/clone dependencies. Restore reference Planner/Omni serving capacities.
- Add real contextual ASR and five-mode dots probes, speaker/aligner/DSP probes,
  and explicit qualification coverage. See [service guidance](SERVICE_CAPABILITIES.md).
- Local GPU ASR diagnostic calls pass. Replaying 15 former context failures
  completes all ASR requests, with 12 nonempty and 3 empty transcriptions. This is
  not a complete editing replay, new benchmark score or full-stack GPU approval.
  Frozen experiments and the planning/execution algorithm remain unchanged.

## 2026-10-29 — deployment-selected capabilities

- Add a tools configuration shared by capability discovery, Planner/Reasoner,
  execution guards and deployment checks. Disabled tools and composites with
  missing dependencies are not advertised or started.
- Select dots.tts.edit or edit-data-pipeline explicitly for the same tagged-edit
  interface; carry the choice to MCP children without changing global environment.
- Ship a reduced starting selection, retaining explicit legacy-config behavior.
  Record effective choices in run configuration and omit prior-toolset ranking.
- Local selection and runtime-construction regressions pass. This version has
  not been rerun on GPU benchmarks; previous frozen experiments remain unchanged.

## 2026-10-28 — selectable tagged-edit backend

- Default tagged edits to an independently installed dots.tts.edit model while
  retaining the explicit edit-data-pipeline choice and stable tool inputs.
- Keep endpoint readiness, service startup and actual HTTP dispatch consistent
  with the selected backend. Model failures never silently invoke the pipeline.
- Do not apply F5-only segmentation, re-synthesis or boundary smoothing to the
  model path; retain their explicit installation gate for pipeline operations.
- Add an original offline-loading model service, pinned independent environment,
  source/model identities, distribution boundaries and license evidence. The
  model card's `other` metadata and project Apache declaration both remain visible.
- Five real GPU operations now pass through normalization, MCP and the model HTTP
  service: text, emotion, pitch, rate and pause. Artifact integrity and pinned
  runtime identities pass; this does not establish perceptual editing quality.
  New full benchmark inference and official scoring are pending; earlier reports
  remain tied to their original versions. See [validation scope](DOTS_TTS_EDIT.md).
- Pin the upstream Edit runtime to `8e1f5881b86a679ecdebfcd9d206e502d1b4008e`,
  including its missing streaming-profiler imports. Add a pre-weight-load CPU
  control-flow regression for streaming and non-streaming calls, with profiling
  enabled and disabled. The native non-streaming edit method and model weights
  are unchanged by this upstream fix; browser Playground changes are not bundled.

## 2026-10-22 — source distribution boundary

- Clarified that framework source, original adapters, configurations and documents
  are distributed separately from external SDKs, models, datasets and audio.
- Updated the upstream license observation while retaining the tested pipeline
  and model pins. No model substitution or new inference is claimed.
- Replaced provider-specific deployment references with a portable foreground
  service entrypoint and removed historical infrastructure paths from lockfile
  comments without changing dependency specifications.
- Added source/package content checks and strengthened package exclusions for
  model, media and native binary files. FFmpeg remains an external dependency;
  optional image recipes use the separately checked LGPL build.

## 2026-10-11 — final project adopts prepare-opensource

- Replaced the local dots.tts.edit copy and duplicate IndexTTS/Seed-VC adapters
  with the pinned upstream pipeline and its six official SDK submodules.
- Updated service entrypoints, explicit dependency URLs, emotion and pause
  payloads, Seed-VC seed/cache behavior, and external artifact placement.
- Verified four real editing requests and three end-to-end Forward samples;
  exact source, payload and package checks are recorded in
  [PIPELINE_MIGRATION.md](PIPELINE_MIGRATION.md).
- This is a migration check using existing environments/assets. Earlier cold
  installation results below retain their original scope. Source-access and license statements
  describe the status at that date; current details are in [LICENSES.md](LICENSES.md#pipeline-source-access).

## 2026-09-30 — attribution and repeatable acceptance

- Confirmed Apache-2.0 for original contributor pipeline orchestration, service
  adapters and modifications; retained all upstream component licenses.
- Added a file-scope license map and an inventory of 31 model repositories,
  11 fetched source repositories, six included source components and five
  service metadata/configuration assets. Listed missing provenance explicitly.
- Added a real-operation service probe that generates diagnostic speech from
  the deployed TTS. No private recording is required for these service checks.
- Documented the cold-install, functional-quality and capacity checks separately,
  and specified the boundary between runtime deployment and paper reproduction.
- Removed an obsolete private-source commit identifier and excluded-file list
  from the legacy pipeline descriptor; the shipped source inventory remains.
- No Reasoner, Planner, perception policy, compiler, executor, reward, revision
  or candidate-selection algorithm was changed in this update. Model pins and
  serving configurations remain the same as the preceding release candidate.

The initial cold-container result remains qualified only within its documented
scope. The new final-source run and its quality panel must produce their own
receipts before additional acceptance can be reported.
