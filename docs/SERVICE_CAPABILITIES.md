# 服务能力对照与验收

## 默认部署漏掉了什么

历史完整测试给 ASR 增加了上下文输入传递，但此前公开部署仍启动只支持普通识别的
网关。模型本身支持上下文，Planner/Reasoner 也能请求该能力；漏掉的是 HTTP 服务
传递层，而不是少下载了一套 ASR 权重。只检查健康接口、普通转写或其他编辑成功，
无法发现这个缺口。现由自有 `deployment/asr_service.py` 直接调用外部固定版本 SDK，
不修改上游源码、不丢弃 context、不把提示词当成转录结果。

另外，精简 `tools.yaml` 是明确收缩后的部署选择，不是历史完整工具集；以前独立
`diarization` 虽有接口适配代码，却没有纳入 stack。现在补齐独立服务、环境/资产
绑定和客户端地址导出。SE-DiCoW、MOSS 与独立说话人分段各有用途，不能互相顶替。

## 能力与部署对应

`configs/tools.full.yaml` 覆盖当前完整 profile 的 30 项编辑、6 项感知操作。
下表按能力归组，不表示每一行都已在本次重新进行 GPU 验收。

| 能力 / 工具选择 | 必需服务或实现 | 验收重点与边界 |
| --- | --- | --- |
| 普通转写、带上下文再识别 / `asr` | `asr`：Qwen3-ASR-1.7B + 自有 HTTP 层 | schema 声明 context；`/info` 当前 backend 明确支持；真实无/有 context 两次调用 |
| 字词定位 / `aligner` | `aligner` + `aligner_backend`：Qwen3-ForcedAligner | 网关和模型端都验证词级结果；不放宽时间重叠/顺序校验 |
| 音频理解、语义复核 / `omni` | `omni`：Qwen3-Omni；语义复核另需 ASR/对齐 | 真实音频输入；长音频与输出预算仍受上下文长度限制 |
| 规划、任务解释、语义评审 | `planner`：Qwen3.8-27B | `/v1/models` 身份 + JSON/实际工作流；revision 还需配置音频评审 |
| 文字、情绪、音高、语速、停顿标签编辑 / `edit_data_pipeline` | 默认 `dots_tts_edit`；可切四类 pipeline 服务 | 模型五类操作分别验证；相同输入接口不表示编辑质量、随机参数相同 |
| DSP/重合成音高、语速、音量 / `edit_pitch_shift`、`edit_speech_rate`、`edit_volume` | `pitch`、`rate`、`volume`；完整模式另需 `tts_clone`、`indextts` | 工具选择包含神经模式所需依赖；仅部署 CPU 服务不代表开放所有方法 |
| 边界平滑、编辑拼接 / `edit_boundary_smooth`、`edit_audio_assembly` | `smoother`、`assembly` + 对齐、VAD、用户安装的 F5 | dots 模型的标签编辑不自动替代这些辅助接口；缺 F5 明确停止 |
| 自定义音色与参考音色合成 / `qwen3_tts` | `tts` CustomVoice + `tts_clone` Base/Tokenizer | 两个端点、两套模型资产；只有 custom 成功不能证明 clone 可用 |
| 参考音色转换 / `seed_vc` | `seed_vc` | source/reference 音频均需可读，实际返回音频 |
| 无参考音色转换、重音编辑 | Qwen3-TTS、Seed-VC、ASR/对齐；重音还需 assembly | 组合依赖齐全才显示能力；需真实多步运行，单接口成功不算组合验收 |
| 同音频音色迁移 | Seed-VC + `diarization` | 参考片段必须来自有依据的说话人区间 |
| 说话人相关编辑/感知 | `diarization`、`sedicow`、`moss` + ASR/对齐/Omni | 分段、英文带说话人转写、中文带说话人转写各自验收；不改变现有语言路由 |
| 事件定位、带时间描述 / `flexsed` | `flexsed`；时间化描述另需 Omni | 事件时间段与真实音频匹配，不只检查健康状态 |
| 通用音频编辑 / `auk` | `auk`；框架还需 ASR/对齐 | 真实编辑交付，不代表所有指令均满足 |
| 增强、人声/伴奏分离、前景背景编辑 | `enhancement`、`separation`；组合还需编辑/TTS/VC/ASR/对齐 | 按唯一声部标签选轨，不能用文件顺序判断人声；当前分离模型不支持重叠说话人隔离，音质另行验证 |
| 音量、删除、切片、静音、排序、拼接、提示音 | `artifact_ops`，本地音频库/FFmpeg | 输入输出时长与内容位置；无额外模型进程 |
| 音效、混响 | `audio_effects`、`spatial_reverb`，本地 MCP 环境 | 真正启动 MCP 并输出音频；依赖环境不能只测试 Python import |

## 使用同一参数文件部署与运行

先阅读 [部署基础](DEPLOYMENT.md) 和 [dots 模型安装](DOTS_TTS_EDIT.md)。
完整配置的平滑/拼接仍需要按 [F5 外部工具说明](EXTERNAL_TOOLS.md) 自行安装；
框架不自动下载、安装或启动 F5。这里只登记已经合法配置的服务地址：

```bash
uv run --locked python deployment/bootstrap.py --data /data/xedit-full \
  --phase plan --tools configs/tools.full.yaml
uv run --locked python deployment/bootstrap.py --data /data/xedit-full \
  --phase all --tools configs/tools.full.yaml \
  --external dots_tts_edit=http://127.0.0.1:8049 --external f5=http://127.0.0.1:8001
source /data/xedit-full/client.env
```

检查 plan 的服务、模型、环境、GPU 分配和 operator_setup_required 后再执行安装。
`runtime.yaml` 同步设置 `tools: tools.full.yaml`。如需管线后端，将该工具文件的
`edit_backend` 改为 `edit_data_pipeline`，使用**新的部署目录**重算 plan；不再需要
dots 绑定，仍需 F5。不得原地修改正在运行或用于评分的冻结配置。
服务分配中的 GPU 编号是部署起点，不保证全部模型同时驻留时内存足够。

```bash
uv run --locked x-edit profile --config configs/runtime.yaml
uv run --locked x-edit doctor --config configs/runtime.yaml --live
/data/xedit-full/envs/services/volume/bin/python deployment/qualify_services.py \
  --data /data/xedit-full --tools configs/tools.full.yaml --output /data/qualification/full-01
```

若指定了 `--env-root`，换成该目录的解释器。上面的综合检查会通过已部署的 TTS
生成诊断录音；精简部署没有 TTS 时，需用 `--source /data/diagnostic.wav` 提供自己
录制的 `The next train will leave at six fifteen.`。每次使用新的输出目录。
检查保留请求、响应和音频，列出没有直接探测的依赖；`all_passed` 只表示列出的
服务探测全部成功，**不表示复合工具、编辑质量或整个 benchmark 通过**。

## 上下文 ASR 的三层门禁

独立变调/语速服务的 Qwen token 编解码请求现在绑定 `tts_clone`（Base + Tokenizer），
不再绑定以 `--no-tokenizer` 启动的 CustomVoice 服务。对应环境变量是
`EDIT_PIPELINE_QWEN3TTS_TOKENIZER_URL`，默认端口 8019。显式选择这些工具会连同
IndexTTS 与 token 服务一起部署；若仅做 CPU DSP，仍可单独部署 `rate`/`pitch`，
但不能把该部署当作整个工具所有方法都已就绪。

1. `stack.py start` 对本地、已运行和外部 ASR 都核验 OpenAPI `context` 与 `/info`
   当前后端 `supports_context=true`。不合格时停止启动流程；不删除能力声明绕过问题。
2. `doctor --live` 同样检查契约，不发送音频，也不声称已完成真实推理。
3. 综合服务验收包含真实上下文识别，也可单独执行：

```bash
python deployment/check_asr.py --url http://127.0.0.1:8010 \
  --audio /data/my-speech.wav --output /data/qualification/asr-01
```

固定模型仍为 `Qwen/Qwen3-ASR-1.7B` revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`，SDK 使用
`third_party/external_pipeline.json` 中 Qwen3-ASR 的固定外部版本。
不混用旧私有网关和包路径；`/info` 记录实际 SDK 导入位置。默认逐条 Transformers
推理、bfloat16、max_new_tokens=512；无 context 时不传该参数，混合批次逐项保留上下文。
使用已下载的本地权重，不隐式联网取得其他模型。实际模型、SDK、环境与旧实验存在
差异时必须记录，不能仅根据模型名称宣称完全相同。

## 明确保留的边界

目标确认、声部选择与裁片文本使用共同遵循 [工具与证据契约](TOOL_CONTRACTS.md)。
时间戳不自动证明目标身份；可读音频也不自动证明已完成用户要求的编辑。

### 对照配置与实际输入，不只对照工具名称

对比两个部署时，同时保留原始参数文件和运行时实际生效的 profile，核对
`edit_backend`、启用操作、Reasoner 模式、观察预算及模型服务启动参数。还应检查
Planner 实际收到的操作参数契约和感知能力卡，以及服务实际收到的 source_text、
编辑标签和时间范围。工具同名、端口健康或配置中存在某字段，都不证明模型看到了
同样的上下文；外部版本及默认参数变化也可能改变输出。

`advisory_plan` 的独立 Reasoner 在执行前协助判断目标与缺失证据；它不自动接管
每个工具失败。不能把 `reasoner_shadow: disabled` 解释成实际 Reasoner 被关闭，
也不能把提供建议解释成 Planner 已采纳。冻结开发先验的选择规则和各规划模式的
消费边界见 [工具契约](TOOL_CONTRACTS.md)。

声部与说话人是不同能力：有 vocals 不代表已隔离重叠说话人；有说话人区间不代表
已有准确裁片文字。部署验收、语义观察路由、工具输入校验及最终音频质量应分别记录。

### 推理服务容量

部署默认容量恢复为此前完整测试的设置：Planner/Reasoner 共享服务
`max-model-len=262144`、`max-num-seqs=8`、TP=4、GPU memory utilization=0.58；
Omni 为 `max-model-len=32768`、`max-num-seqs=2`、GPU memory utilization=0.90。
此前便携部署的 65536/16384 上下文不是相同服务容量。此更正不调整控制器的提示、
输出 token 预算或模型权重；新机器上仍须验证 KV cache 与实际显存容量是否足够，
不应以降低上下文长度后仍宣称与旧测试相同。长输入加输出预算超过上限仍会失败。

### 不自动恢复的外部内容

- SAM-Audio、VoiceSculptor-VD 仍仅提供扩展指引，不加入代码、工具集或自动安装。
  分离使用当前 RoFormer；音色相关组合使用 Qwen3-TTS/Seed-VC。这不是声称实现与
  旧模型完全等价，也不覆盖任意新音色设计质量。
- F5 仍是外部工具，模型使用条款不因 HTTP 隔离而改变。
- 口音等依赖参考录音的能力需用户准备合法音频库；公开包不包含该库。
- 普通音频 canary 不证明长音频不会触发上下文限制。不为部署问题更改规划、
  感知 token 预算、裁剪/对齐约束或评分口径。
- 部署修复不追改旧实验。新的全量结论必须来自独立的新运行与官方评分。
