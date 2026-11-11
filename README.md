# X·Edit

**给一段音频，用一句话说明怎么改。X·Edit 负责理解要求、选择工具并执行编辑。**

例如：“把‘明天’改成‘周五’，其他内容不要动”，或者“这句话说得开心一点，但不要改变文字”。
它不是一个包办所有任务的语音模型，而是把感知、推理、规划和专业音频工具组织起来的框架。
能完成哪些编辑，取决于你实际部署并启用了哪些工具。

本仓库提供框架代码、自有服务适配器、配置和部署文档，**不附带第三方模型代码、权重、数据集、参考音频或生成音频**。
安装 Python 包不等于部署完成，也不会自动下载模型。代码包名是 `x-edit-audio`，命令是 `x-edit`，Python 导入名保留 `x_editor`。

## 它怎么工作

```text
音频 + 编辑要求
      ↓
感知：听音频、识别文字、按需定位说话人或事件
      ↓
Reasoner：分析要求与证据 → Planner：安排编辑步骤
      ↓
执行器：检查参数、调用已启用的工具、保存结果
      ↓
输出音频 + 执行记录
```

- `forward`：完成一次理解、规划和执行；默认开启 Reasoner，不做外层评分与反复修订。
- `revision`：在相同执行流程外增加评价与修订，默认最多三次尝试。
- 工具没部署，就不要启用它。未启用的能力不会提供给 Planner/Reasoner，执行器也不会接受相应操作。
- 编辑失败会如实报错，不会偷偷换后端，或拿源音频当作成功编辑的结果。

## 部署时分成三层

| 层 | 做什么 | 放在哪里 |
| --- | --- | --- |
| 控制器 | 理解任务、规划、调度、记录结果 | 一个 Python 3.11 uv 环境；不安装模型推理栈 |
| MCP 工具客户端 | 转发 HTTP 请求；部分工具做本地音频处理 | 控制器机器上，每个客户端一个 uv 环境 |
| 模型与工具服务 | 真正运行文本/音频模型、ASR、编辑或增强 | 独立进程，每个受管服务一个 uv 环境；可部署到不同机器 |

Planner、Reasoner、感知模型和评价模型可以分别配置服务地址，也可以共用兼容的模型服务。
它们仍由控制器编排，并不是每个 Python 类都启动一个 HTTP 服务。
端口、跨机器文件共享和安全要求见 [分离部署指南](docs/MODULE_DEPLOYMENT.md)。

## 先跑起来

以下命令在仓库根目录执行。需要 Linux、Git、Python 3.11.15、uv 0.11.28、FFmpeg/ffprobe 和 libsndfile。
模型服务另需相应 GPU、驱动和已获授权的模型文件；本仓库不保证完整工具栈能放在一张显卡上。
系统依赖与非 GPL FFmpeg 构建见 [部署参考](docs/DEPLOYMENT.md)。

### 1. 安装控制器

```bash
python3 -m pip install uv==0.11.28
uv python install 3.11.15
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --locked --no-dev
cp .env.example .env
```

### 2. 选择你要用的工具

编辑 `configs/tools.yaml`。默认只启用基础音频操作、文字/情绪/韵律/停顿编辑，以及 ASR、对齐和 Omni 感知：

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

`edit_data_pipeline` 是兼容旧接口的工具名，不表示一定运行管线。
`edit_backend: dots_tts_edit` 使用独立的 **dots.tts.edit 模型服务**；改为 `edit_data_pipeline` 则使用管线服务。
两者接收相同的编辑输入，不会同时调用，也不会互相自动兜底。

更多能力按需从 `configs/tools.full.yaml` 选取；不要在服务没部署好时直接启用全部工具。
见 [工具选择](docs/TOOL_SELECTION.md) 和 [各能力需要哪些服务](docs/SERVICE_CAPABILITIES.md)。

### 3. 部署服务并填写地址

有现成服务时，把 `.env` 中的模型地址、模型名和工具地址改成自己的值。
没有服务时，先按 [dots.tts.edit 部署说明](docs/DOTS_TTS_EDIT.md) 启动编辑模型，
再按 [分离部署指南](docs/MODULE_DEPLOYMENT.md) 部署其余已选服务。
模型和运行产物放在仓库外；第三方源码按部署说明独立获取，不随发布提交。

然后安装本机选中的轻量工具客户端：

```bash
.venv/bin/python deployment/client_envs.py \
  --tools configs/tools.yaml --env-root /data/xedit/envs/clients
```

`.env` 中的 `X_EDITOR_MCP_ENV_ROOT` 必须指向同一个目录。
工具客户端不会借用控制器或其他工具的环境；运行时不会在这些预装环境里自动同步依赖。
跨机器调用时，所有基于文件路径的服务必须能以**相同绝对路径**读写音频，不能只改 IP 地址。

### 4. 检查，再编辑

```bash
.venv/bin/x-edit --env-file .env doctor --config configs/runtime.yaml
.venv/bin/x-edit --env-file .env doctor --config configs/runtime.yaml --live

.venv/bin/x-edit --env-file .env edit --config configs/runtime.yaml \
  --audio /data/audio/source.wav \
  --instruction '这句话说得开心一点，保持原来的文字和说话人。' \
  --mode forward --output /data/xedit/runs/first-edit
```

普通 `doctor` 检查配置与本地依赖；`--live` 额外检查服务连接与接口。
**检查通过不等于 GPU 推理或编辑质量通过**，请再用你有权使用的短音频做一次真实编辑。
输出目录必须是新目录；`result.json` 给出选中的音频和执行记录。
没有有效产物时命令返回非零状态，不要仅凭目录或 WAV 文件存在来判断成功。

需要自动评价与修订时，把 `--mode forward` 改成 `--mode revision`，并配置评价模型。
多音频、多轮输入和 Python API 见 [运行接口](docs/RUNTIME.md)、[示例](docs/SAMPLES.md)。
通过 wheel 安装的用户可用 `x-edit init ./configs` 导出配置；部署脚本请从源码仓库运行。

## 开源范围与第三方许可

本项目有权授权的自有代码、配置和文档采用 **Apache License 2.0**，完整条款见 [LICENSE](LICENSE)。
第三方代码、模型、数据和输出不在该授权范围内。自行下载、独立进程或 HTTP 调用都不会消除其许可限制，
也不会自动授予第三方模型的商业使用权。服务运营方仍须为自己的模型用途取得相应许可。

- **F5-TTS**：代码与模型许可分开记录；所记录的 SDK 为 MIT，官方 `F5TTS_v1_Base` 权重为 CC-BY-NC-4.0，不应据此推断其他权重的许可。默认禁用，不自动安装或下载。使用需要它的管线能力前，会提示“请自行下载并安装F5-TTS”；用户独立完成安装、确认用途授权后再显式启用。见 [外部工具说明](docs/EXTERNAL_TOOLS.md)。
- **SAM-Audio、VoiceSculptor-VD**：不在内置工具集或部署代码中；仅提供 [扩展开发说明](docs/EXTENSION_GUIDE.md)。
- **edit-data-pipeline**：是可选的固定版本外部依赖，不随本次源码快照分发；当前固定版本的获取仍需要上游仓库访问权限，不能假定公开用户可以直接下载。见 [源码访问范围](docs/LICENSES.md#pipeline-source-access)。
- **FFmpeg**：作为外部程序调用。源码仓库不附带二进制；提供的构建流程禁用 GPL/nonfree。用户自行制作的镜像、安装包也要单独核查，不能只检查源码包。

复制或修改第三方代码时仍须保留其适用声明。非商用模型用于商业服务需另行取得授权或改用许可适合的组件，
不能仅靠“不分发权重”或 README 声明解决。具体来源、代码提交、模型版本及许可分别见
[第三方声明](third_party/NOTICE.md)、[许可清单](docs/LICENSES.md) 和 [中文开源声明](docs/OPEN_SOURCE_STATEMENT.zh-CN.md)。

## 文档与复现边界

- [分离部署指南](docs/MODULE_DEPLOYMENT.md)：每服务 uv 环境、角色端点、远程工具接入。
- [部署参考](docs/DEPLOYMENT.md)：系统依赖、固定版本、服务启动和实际操作验收。
- [服务能力](docs/SERVICE_CAPABILITIES.md)：默认与完整工具集的区别、语音识别上下文和说话人服务。
- [工具契约](docs/TOOL_CONTRACTS.md)：参数、音频资产和错误边界。
- [版本与验证记录](docs/RELEASE_NOTES.md)、[复现说明](docs/REPRODUCIBILITY.md)：哪些改动已验证，哪些结论只属于历史版本。
- [发布范围](docs/RELEASE_SCOPE.md)：不发布权重、音频、私有实验及基础设施配置；历史实验成绩不是本次部署改动的重新测量结果。

运行记录会包含用户指令、音频路径与模型回答，请放在私有输出目录，分享前检查。
本项目大量使用并扩展 dots.tts.edit 的模型与管线能力；感谢上游项目及各工具作者，第三方授权不因本项目发布而改变。
