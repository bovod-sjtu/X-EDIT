# 分离部署：只装需要的服务，各用各的环境

## 先决定放在哪里

控制器负责调度，不在自身 Python 环境中加载语音大模型。MCP 客户端在控制器机器上运行，
通过 HTTP 调用模型与工具服务。服务可以在同机不同进程，也可以在不同机器。
uv 隔离的是 Python 依赖，不是网络权限、文件权限、显存或操作系统；生产服务仍需要访问控制。

推荐目录如下，代码、环境、模型和产物分开放置：

```text
/opt/x-edit/                         源码与控制器 .venv
/data/xedit/envs/clients/<tool>/      每个 MCP 客户端的 uv 环境
/data/xedit/envs/services/<service>/  每个受管 HTTP 服务的 uv 环境
/data/xedit/cache/                   第三方模型缓存
/data/xedit/logs/                    服务日志
/data/xedit/runs/                    用户任务与产物
/data/audio/                        各机器一致挂载的音频目录
```

不同服务可以使用相同的依赖锁，但不共用环境目录；例如 Planner 与 Omni，文字编辑与停顿编辑，
各自安装、各自启动，不会因升级其中一个环境而覆盖另一个。uv 可以复用下载缓存，不代表共用 site-packages。
已有旧部署配置仍按原有环境布局解析；不要重写正在运行的配置，迁移请新建 `--data` 目录。

## 方案一：连接已经部署好的服务

1. 按 README 安装控制器，在 `configs/tools.yaml` 只启用实际可用的能力。
2. 把 `.env.example` 复制为 `.env`，填写模型和工具的真实地址。
3. 安装选中的 MCP 客户端，不安装 GPU 模型：

```bash
.venv/bin/python deployment/client_envs.py \
  --tools configs/tools.yaml --env-root /data/xedit/envs/clients
```

4. 设置 `.env` 中 `X_EDITOR_MCP_ENV_ROOT=/data/xedit/envs/clients`，执行 README 中的检查与短音频编辑。
   还可以先单独验证 MCP 子进程启动；下面的命令不会调用模型推理：

```bash
X_EDITOR_MCP_ENV_ROOT=/data/xedit/envs/clients \
  .venv/bin/python deployment/check_clients.py --tools configs/tools.yaml
```

安装器按工具自己的 `uv.lock` 同步。没有专用轻量锁的 HTTP 客户端使用控制器的依赖锁，
但仍安装到各自的目录；不会安装该工具的 GPU 推理依赖。
ASR、说话人等由控制器直接通过 HTTP 调用的模块不另建无用的 MCP 进程。
组合能力复用基础工具，不为能力卡虚构服务。

父进程的 `UV_PROJECT_ENVIRONMENT`、`VIRTUAL_ENV`、`PYTHONPATH` 不会把选中的 MCP 客户端拉回另一个环境。
启用 `X_EDITOR_MCP_ENV_ROOT` 后，缺少环境会明确报错，不能退回系统 Python 凑合运行。
uv 启动的客户端使用 `--no-sync`，避免执行时修改依赖。扩展工具显式声明的环境变量仍可使用。

## 方案二：自己部署服务

先阅读 [模型部署](DOTS_TTS_EDIT.md) 并自行准备 dots.tts.edit。再查看其余所选服务的安装计划：

```bash
.venv/bin/python deployment/bootstrap.py --data /data/xedit \
  --phase plan --tools configs/tools.yaml \
  --external dots_tts_edit=http://127.0.0.1:8049
```

`plan` 不下载模型、不占 GPU。确认许可、访问权限、磁盘空间与 GPU 分配后，再依次执行：

```bash
.venv/bin/python deployment/bootstrap.py --data /data/xedit \
  --phase fetch --tools configs/tools.yaml \
  --external dots_tts_edit=http://127.0.0.1:8049
.venv/bin/python deployment/bootstrap.py --data /data/xedit \
  --phase install --env-root /data/xedit/envs --tools configs/tools.yaml \
  --external dots_tts_edit=http://127.0.0.1:8049
.venv/bin/python deployment/bootstrap.py --data /data/xedit \
  --phase start --tools configs/tools.yaml
source /data/xedit/client.env
.venv/bin/x-edit doctor --config configs/runtime.yaml --live
```

受管服务需要的第三方源码由显式 `fetch` 获取到被忽略的依赖目录，不属于公开源码包；模型在 `--data` 下。
即使选择模型编辑后端，辅助服务也可能仍需外部管线源码：当前固定管线版本需要上游访问权限。
没有访问权限时请选择已有兼容 HTTP 服务，不要把下载失败当作安装完成。
F5 不在自动安装流程中；模型后端不需要 F5。管线后端需要用户独立安装并按 [外部工具说明](EXTERNAL_TOOLS.md) 显式启用。

`--external NAME=URL` 可以替换任何已有服务节点。该节点不再自动下载资产、安装环境、启动或停止；
其依赖也不自动部署，由该服务运营者负责。不要用 `all` 代替实际选择，它意味着尝试部署整个服务清单。

`--gpu SERVICE=IDS` 指定设备；`--port-offset N` 整体移动端口。
默认配置的文本服务使用四卡张量并行，请先检查 `deployment/stack.json`，不要照抄到单卡机器。
本轮没有改变模型、推理参数或提示词，分离环境也不代表降低了显存需求。

单独启停已配置的服务：

```bash
.venv/bin/python deployment/stack.py --data /data/xedit start asr
.venv/bin/python deployment/stack.py --data /data/xedit status
.venv/bin/python deployment/stack.py --data /data/xedit stop asr
```

`start` 会启动该服务所需的受管依赖；`stop asr` 只停止自己记录且身份匹配的 ASR 进程，不停外部服务。
长期运行时使用 [部署参考](DEPLOYMENT.md) 中的前台监督入口或自己的服务管理器。

## 各逻辑模块怎样指定服务

| 逻辑模块 | 地址变量 | 模型名变量 | 默认行为 |
| --- | --- | --- | --- |
| Planner | `XEDIT_PLANNER_URL` | `XEDIT_PLANNER_MODEL` | 文本模型，端口 8040 |
| Reasoner | `XEDIT_REASONER_URL` | `XEDIT_REASONER_MODEL` | 默认同一文本服务，独立调用 |
| 感知 | `XEDIT_PERCEPTION_URL` | `XEDIT_PERCEPTION_MODEL` | Omni 服务，端口 8011 |
| 语义评价 | `XEDIT_SEMANTIC_JUDGE_URL` | `XEDIT_SEMANTIC_JUDGE_MODEL` | 默认同一文本服务 |
| 音频评价 | `XEDIT_AUDIO_JUDGE_URL` | `XEDIT_AUDIO_JUDGE_MODEL` | 默认同一 Omni 服务 |

这些地址使用 OpenAI-compatible `/v1` 接口。Guider 与 RubricDesigner 目前复用语义评价配置；
它们不是单独的部署服务。执行器与资产管理在控制器进程内，不需要新开模型端口。
工具 URL 不加 `/v1`；完整变量见 `.env.example` 和服务能力文档。

想让 Reasoner 单独占用模型进程时，在另一机器或新的 `--data` 目录只部署 `planner` 服务模板，
配置独立端口、环境目录和 GPU，然后将 `XEDIT_REASONER_URL` 指向它。
不要为了分开进程改动提示词或模型采样参数，也不要把两个部署指向同一环境目录。

## 跨机器部署注意事项

- 新建受管部署时可传 `--bind-address 0.0.0.0`，默认只监听 `127.0.0.1`。放开监听前先配置防火墙、鉴权网关与加密连接；这些模型服务不是可直接暴露公网的多租户服务。
- `client.env` 是部署机本地地址视图，不会猜测另一台机器可访问的地址。控制器通过 `--external NAME=http://tool-host:PORT` 绑定远端服务，或者在 `.env` 显式填写对应 URL。
- 传路径的接口要求控制器与服务看到相同绝对路径和内容，并具备必要读写权限。建议共享挂载 `/data/audio` 和任务输出目录；模型服务只接收到路径时不会替你上传文件。
- 不同机器不共享 Python 环境；每台机器从自己的锁文件安装。模型与缓存可按存储系统能力复用。
- `.env`、密钥、用户录音、运行记录都不要提交 Git。端口通不代表模型已加载，健康检查通过后仍须真实编辑验收。

## 验证边界

环境路径、选择过滤、安装命令、远程绑定与本地 MCP 子进程可以在不加载模型时验证。
这类验证不能证明显卡可用、第三方模型可加载或编辑质量达标。
GPU 的算子检查与实际操作验收分别使用 `deployment/check_cuda.py` 和工具专用验收入口；
详见 [部署参考](DEPLOYMENT.md#real-operation-acceptance)。历史实验产物留在私有目录，不混入本仓库。
