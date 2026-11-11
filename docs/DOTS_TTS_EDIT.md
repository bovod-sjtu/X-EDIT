# dots.tts.edit 与管线后端

新版默认用 **dots.tts.edit 模型**执行文字、情绪、音高/语速和停顿编辑。用户可显式
选择原来的 edit-data-pipeline。两者使用同一个工具协议，不需要改用户请求或规划结果：
`edit_data_pipeline.apply` 的稳定名称保留用于兼容，名称不代表当前一定运行管线。

```yaml
schema_version: x-edit.tools/v1
edit_backend: dots_tts_edit
enabled_tools: [artifact_ops, edit_data_pipeline, asr, aligner, omni]
```

通过 `runtime.yaml` 的 `tools: tools.yaml` 引用该文件，并配置 `DOTS_TTS_EDIT_URL`。
切回管线时修改 `edit_backend: edit_data_pipeline`，配置各 `EDIT_PIPELINE_*_URL`，
并完成 [F5 外部安装与显式启用](EXTERNAL_TOOLS.md)。切换后加载新配置并开启新运行。
不含 `tools` 的历史配置仍使用 `X_EDITOR_EDIT_BACKEND`；显式文件优先于环境变量。
工具与能力卡允许列表详见 [按部署选择工具](TOOL_SELECTION.md)。无效值报错，模型离线、
服务错误或超出模型上下文时都不自动切换后端，也不下载 F5。旧版复现必须显式选择管线。

## 保持与改变的接口

| 工具 operation | 共用输入 | 默认模型接口 |
| --- | --- | --- |
| `text_edit` | 输入音频、source_text、含 sub/ins/del 的 instruction_st、输出路径 | POST /execute |
| `pitch_rate_edit` | 相同，标签为 pitch/rate | POST /execute |
| `emotion_style_edit` | 相同，标签为 emo | POST /execute |
| `pause_edit` | 相同，标签为 pause | POST /execute |

请求以 `audio: {data: base64, format: "wav"}` 传输，不把客户端本地路径发送给远端。
服务按内容解码 SoundFile 支持的格式（包括 WAV、FLAC），内部转成浮点 WAV 供模型读取；
不依赖扩展名判断真实格式，不改变采样率或声道数。
适配器将 `instruction_st` 交给上游 `generate_edit(instruction=...)`；旧式情绪标签
只做等价语法转换，不猜测情绪或强度。使用上游解析器检验标签、源文本和可选的目标
文本；不匹配返回 422，不用 target_text 覆盖不一致的编辑指令。

模型默认采样为 `num_steps=10`、`guidance_scale=1.2`、`use_xvector="auto"`、
`speaker_scale=1.5`、`seed=42`，均可显式传入；auto 的语义由上游实现决定。
结果仍写 `output_audio_path`，增加 `backend`、模型/代码 revision 和实际采样参数。
服务收到的 revision 是部署者声明，需与独立下载校验记录核对；HTTP 健康并非推理验收。

F5 专属的片段重合成参数不传给模型，调用结果列出 `ignored_pipeline_parameters`。
模型路径不经过 F5 的 30 秒切片、局部重合成及边界平滑逻辑。默认上游
`max_generate_length=500`，源音频和输出共享模型的长度预算；超限明确失败，不截断
源音频。不能将“接口相同”理解为两种方法的质量、时长保持或支持范围相同。

音量、独立音高变换、WSOLA 语速、声音转换、增强、分离等其余工具不改变实现。
`boundary_smooth` 和 `audio_assembly` 仍是依赖 F5 的管线工具；模型后端不会自动
启用它们。需要这些操作时仍须完成独立安装与许可审查。

## 独立部署

参考上游 [Edit 部署与 Python API](https://github.com/studio-dots-ai/dots.tts/tree/8e1f5881b86a679ecdebfcd9d206e502d1b4008e#-speech-editing-dottsttsedit)。
本项目只提供原创 HTTP 封装，不包含上游实现、权重、样本、生成音频或预装镜像。
请自行下载并安装 dots.tts.edit，先核对下节许可证据。代码与权重放在发布仓库之外，
使用独立 Python 3.11 / CUDA 环境，不并入控制器或 vLLM 环境。

```bash
git clone https://github.com/studio-dots-ai/dots.tts.git /data/dots-tts-source
git -C /data/dots-tts-source checkout --detach 8e1f5881b86a679ecdebfcd9d206e502d1b4008e
uv venv /data/dots-edit-env --python 3.11.15
uv pip sync --python /data/dots-edit-env/bin/python deployment/requirements/dots_tts_edit.lock \
  --index https://pypi.org/simple --default-index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
uv pip install --python /data/dots-edit-env/bin/python --no-deps -e /data/dots-tts-source
uv pip check --python /data/dots-edit-env/bin/python
/data/dots-edit-env/bin/hf download dots-studio/dots.tts.edit \
  --revision e2aea9cab95a802a63f1a491702a1cc8bf796b30 --local-dir /data/dots-edit-model
```

上游推荐的 Gradio 6.17.0 在检查时不可下载；此服务环境使用满足其声明范围的
6.17.3，并单独锁定兼容的 FastAPI，保留其推荐的 Torch/Transformers，不修改模型算法。
安装锁仅覆盖此 HTTP 服务，不保证上游浏览器 Playground 能运行。

本部署固定修复分支的提交 `8e1f5881b86a679ecdebfcd9d206e502d1b4008e`，
不在启动时跟随可变分支。该提交补齐 `generate_stream` 中性能分析上下文的导入；
旧版在带参考音频、关闭 speaker guidance 时，即使不启用性能分析也可能触发
`NameError`。本封装使用非流式 `generate_edit`，该方法、采样默认值、模型权重
及依赖声明未被此提交修改；不能把流式修复描述为本封装已观测到的编辑失败。
分支另含 Playground 下载与混合背景声音展示修改，本项目不复制其前端。

加载权重前，在独立服务环境运行以下 CPU 回归：

```bash
/data/dots-edit-env/bin/python deployment/check_dots_runtime.py
```

检查原生流式、非流式方法在性能分析开启和关闭时的控制流程；模型与输入准备使用
替身，不加载权重，不证明 GPU 推理或编辑质量。其后仍需通过五项真实编辑调用。

从 X·Edit 源码根目录启动（把实际绝对路径代入），服务只接受**已经下载的本地目录**：

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 /data/dots-edit-env/bin/python \
  -m x_editor.tools.catalog.edit_data_pipeline.dots_service \
  --model-path /data/dots-edit-model \
  --model-revision e2aea9cab95a802a63f1a491702a1cc8bf796b30 \
  --source-revision 8e1f5881b86a679ecdebfcd9d206e502d1b4008e \
  --host 127.0.0.1 --port 8049
```

`--optimize` 可显式开启上游优化；比较实验必须冻结同一设置。单服务单 GPU、单请求
推理，忙时返回 429。需用文字、情绪、音高、语速、停顿真实操作验收后再扩大并发。
默认绑定回环地址；远程部署应自行配置访问控制、请求体限制与 TLS。

若使用部署栈，可通过 `--external dots_tts_edit=http://host:8049` 登记已启动的服务。
`stack.py` 不安装、启动或停止该外部实现。`export-env` 会记录当前
`X_EDITOR_EDIT_BACKEND`（未设置时为 dots_tts_edit），供无工具文件的直接调用者使用。
控制器若加载 `tools.yaml`，以文件中的 `edit_backend` 为准，无需同时修改该环境变量。
自动服务准备只探测所选后端；只有用户自己配置 `DOTS_TTS_EDIT_SERVICE_COMMAND`
时才运行该启动命令，不会自动获取源码或模型。

### 控制器与 MCP 启动检查

模型服务环境与控制器的 MCP 工具环境是两套环境。工具配置使用 `uv run` 启动
`mcp_server.py`，因此控制器必须能找到 `uv`；可通过 `X_EDITOR_UV_BIN` 指定其
绝对路径。`X_EDITOR_TOOL_PYTHON` 仅覆盖相对 `.venv` Python 命令，不能替代这里的
`uv` 入口。先验证真实 MCP 子进程能初始化并列出 `edit_data_pipeline.apply`，再加载
模型或提交批量请求；仅直接导入客户端、模拟 HTTP 请求或探测模型健康不覆盖这一步。

使用 `deployment/client_envs.py --tools configs/tools.yaml --env-root /data/xedit/envs/clients`
安装控制器侧工具环境，然后设置 `X_EDITOR_MCP_ENV_ROOT=/data/xedit/envs/clients`。
框架为每个客户端选择独立环境，预装模式使用 `--no-sync`；不再用全局
`UV_PROJECT_ENVIRONMENT` 覆盖所有工具。GPU 模型环境另外安装，详见
[分离部署指南](MODULE_DEPLOYMENT.md)。

## 许可与版本证据

- 代码：上述 Git commit 的 [LICENSE](https://github.com/studio-dots-ai/dots.tts/blob/8e1f5881b86a679ecdebfcd9d206e502d1b4008e/LICENSE) 为 Apache-2.0。
- 模型：固定 `dots-studio/dots.tts.edit` revision，
  [模型卡](https://huggingface.co/dots-studio/dots.tts.edit/blob/e2aea9cab95a802a63f1a491702a1cc8bf796b30/README.md)
  元数据为 `license: other`，正文把许可指向 dots.tts 项目。
- 固定版本 [项目声明](https://github.com/studio-dots-ai/dots.tts/blob/8e1f5881b86a679ecdebfcd9d206e502d1b4008e/README.md#-license)
  写明代码和已发布 checkpoints 使用 Apache-2.0。模型仓库未提供单独 LICENSE 文件。
  这项元数据差异保留在 `third_party/licenses.json`，商业使用的权重授权范围仍需审阅确认，
  不将 `other` 悄悄改写为已独立核实的 Apache 模型许可证。

X·Edit 的 Apache-2.0 不重新授权第三方权重或输出；独立部署不会消除其适用条款。
采用这个默认后端也不代表整个工具栈都是允许无条件商业使用的 Apache-only 组件。

## 复现与指标

新实验独立记录源码、模型 revision/文件校验值、服务环境锁、后端与采样参数。
保持同一份 MMAE speech 426 条和 SpeechEditBench 4700 条选择集，并保留失败条目。
评分使用各 benchmark 的官方实现，不将运行成功率或接口模拟测试当成编辑质量指标。
本次适配尚不构成全量推理或评分完成证明；新轮结果必须等实际推理与评分结束后发布。

### 已完成的模型操作验收（2026-10-28）

上述固定代码与模型版本已在单张 GPU 上完成五项真实调用：文字替换、情绪、音高、
语速、停顿。调用经过参数规范化、真实 MCP 子进程、HTTP 服务和原生模型，均返回
48 kHz 单声道、非空且有限值的音频；输出校验值、后端身份、源码和模型版本、
实际传入的采样参数均已核对。模型保持 bfloat16、optimize=False 和本文默认参数。
MCP 子进程、CUDA 编译与内核、媒体读写检查也均通过，不是仅检查健康接口。

例如，将诊断句中 train 替换为 bus 的请求，现已实际到达模型并保存返回音频，
不再停在工具进程启动阶段。这里验证的是调用完成和产物完整性，尚未通过独立听评
或转录评分确认替换准确性；其他四项也不据此宣称达到编辑质量要求。
五项使用同一条 2.88 秒诊断源音频，不能代表长音频、多轮编辑、全工具栈或吞吐性能。
本轮没有运行两套 benchmark 的新一轮全量实验，也未调用付费评价接口。
