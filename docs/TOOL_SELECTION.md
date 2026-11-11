# 按部署选择工具与能力卡

在 `runtime.yaml` 中增加 `tools: tools.yaml`，只把已经安装、配置并有权使用的工具
放入该文件的允许列表。未列出的工具不会进入当前运行的能力卡、Planner/Reasoner
候选操作或执行注册表。配置文件不是安装器，不下载代码或权重，不启动整个服务栈。

```yaml
schema_version: x-edit.tools/v1
edit_backend: dots_tts_edit
enabled_tools:
  - artifact_ops
  - edit_data_pipeline
  - asr
  - aligner
  - omni
```

这是随包提供的精简配置：本地音频操作、四类标签编辑以及识别/对齐/Omni 感知。
仍需部署配置指定的 Planner/Reasoner；开启 revision 时还需要评审接口。
它不代表仅安装 dots.tts.edit 就能运行整个智能体，也不是此前完整实验的工具集。
`x-edit init ./configs` 会复制同一配置，源码安装与 wheel 安装保持一致。

源码包另附 `configs/tools.full.yaml`，wheel 的 `x-edit init` 也会生成 `tools.full.yaml`，
覆盖当前 `forward.json` 的完整 30 项编辑操作
和 6 项感知操作。将 `runtime.yaml` 的 `tools` 改为 `tools.full.yaml` 才会启用，
不能仅部署更多服务而沿用精简允许列表。完整配置还包含依赖用户自装 F5 的平滑/
拼接能力；不需要这些能力时须显式移除相应条目，不静默跳过部署失败。

`bootstrap.py --tools FILE` 使用同一选择文件计算服务及依赖，支持两种编辑后端。
最小配置不会安装其他专家服务；完整配置会额外部署 Qwen3-TTS 两个模式、Seed-VC、
AuK、FlexSED、三类说话人服务、增强、分离及辅助编辑服务。默认 ASR 包含上下文
识别；`doctor --live` 会核验该能力而不只是端口存活。
具体部署及验收命令见 [服务能力对照](SERVICE_CAPABILITIES.md)。

## 选择编辑模型或管线

`edit_data_pipeline` 是保留兼容的逻辑工具名，启用文字、情绪、音高/语速和停顿
标签编辑；`edit_backend` 决定调用 `dots_tts_edit` 模型还是 `edit_data_pipeline`
管线。两者不会作为重复能力卡出现。切换只需修改这一项，输入协议不变。

文件中的后端选择优先于 `X_EDITOR_EDIT_BACKEND`，并传递到对应 MCP 子进程。
不同 `XEdit` 实例使用各自的选择，不通过修改全局环境变量切换。重新加载配置后
开启新运行，不修改已经启动的运行。服务错误不会自动切到另一后端。

管线的独立音量、变调、语速、平滑和拼接服务需分别启用。使用依赖 F5 的管线编辑、
平滑或拼接前，还须完成 [外部工具设置](EXTERNAL_TOOLS.md)。允许列表不替代该
安装/许可检查，也不授予第三方模型的商业使用权。

## 可选条目

| enabled_tools 条目 | 暴露的能力或依赖 |
| --- | --- |
| `artifact_ops` | 本地音量调整、切片、静音、删除、提示音、排序、拼接与重建 |
| `edit_data_pipeline` | 通过所选后端执行四类标签编辑 |
| `edit_pitch_shift`、`edit_speech_rate`、`edit_volume` | 各自独立的管线变调、语速、音量服务 |
| `edit_boundary_smooth`、`edit_audio_assembly` | 管线平滑与编辑拼接；仍需独立 F5 配置 |
| `qwen3_tts`、`seed_vc`、`auk` | 语音生成、参考音色转换、通用音频编辑 |
| `audio_effects`、`spatial_reverb` | 音效、空间/混响处理 |
| `speech_enhancement`、`speech_separation` | 增强、声源分离 |
| `asr`、`aligner` | 语音上下文与词级时间位置 |
| `omni` | 通用音频感知及语义复核 |
| `flexsed` | 音频事件定位；与 Omni 共同支持时间化描述 |
| `diarization`、`sedicow_asr`、`moss_transcribe_diarize` | 当前自动语言路由的说话人分段/转写组合 |

添加 `seed_vc` 会让参考音色转换进入能力卡。仅添加 `qwen3_tts` 不会暴露依赖
Seed-VC 的无参考音色转换；依赖同时具备后才暴露该组合能力。重音编辑还依赖
Qwen3-TTS、Seed-VC、编辑拼接及识别/对齐。前景/背景编辑采用保守的依赖交集。
说话人感知按当前默认自动语言路由要求整组服务，不暗中改变路由。
依赖关系集中在 `x_editor/tools/selection.yaml`，不包含评测样本专用规则。

允许列表只收缩 `forward.json` 原有范围，不自动加入 profile 之外的操作。
组合操作内部若仍试图调用未启用工具，会在启动服务前返回 `tool_not_enabled`。
错误名称、重复条目、非法后端和缺少必要观察能力会在读取配置时失败。默认 profile
的初始语音观察需要 `asr`/`aligner`，通用感知路径需要 `omni`；不能只删这些条目
而保持原观察策略不变。

## 检查最终可见能力

```bash
x-edit profile --config configs/runtime.yaml
x-edit-tools --config configs/runtime.yaml --list-capabilities
x-edit doctor --config configs/runtime.yaml
x-edit doctor --config configs/runtime.yaml --live
```

`profile` 输出工具选择和有效操作列表，这是 Planner/Reasoner 收到的操作能力卡集合。
`x-edit-tools` 显示工具级卡片，详细操作范围以 `profile` 为准。配置工具选择后，
`doctor --live` 只检查已选服务，不因其他可选工具未部署而失败；已启用但不可达的
服务仍明确报错。健康检查成功不证明模型能执行真实编辑。

运行目录的 `runtime.json` 保存具体选择与裁剪后的 profile。配置不修改提示、采样、
修复算法或用户指令，只改变声明可用的能力范围。显式选择不沿用旧工具集的离线
偏好快照，避免把未启用操作或其他后端的方法排序传入 Planner。未指定 `tools`
的历史配置保持原有范围与环境变量选择方式，适用于原样复现。

本次通过本地配置、真实运行时构建、禁用调用拦截和部署检查回归；尚未用该工具选择
版本重新运行 GPU benchmark，也没有新质量评分。已有冻结实验不被修改。
