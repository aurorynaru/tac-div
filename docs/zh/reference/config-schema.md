# 配置文件详解

`config.yaml` 配置文件的详细说明。

---

## 配置结构

```yaml
llm:
  api_key: "your-api-key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"

comfyui:
  comfyui_url: "http://127.0.0.1:8188"
  comfyui_api_key: ""  # ComfyUI API 密钥（可选）
  runninghub_api_key: ""
  runninghub_concurrent_limit: 1  # 并发限制 (1-10)
  runninghub_instance_type: ""  # 实例类型（可选，设为 "plus" 使用 48GB 显存）
  
  image:
    default_workflow: "runninghub/image_flux.json"
    prompt_prefix: "Minimalist illustration style"
  
  video:
    default_workflow: "runninghub/video_wan2.1_fusionx.json"
    prompt_prefix: "Minimalist illustration style"
  
  tts:
    default_workflow: "selfhost/tts_edge.json"

template:
  default_template: "1080x1920/image_default.html"

# 可选的 Cat Factory 配置（显式启用前保持关闭）
cat_factory:
  enabled: false
  llm:
    provider: "openai_compatible"
    model: "<provider-model-id>"
    base_url_env: CAT_FACTORY_LLM_BASE_URL
    api_key_env: CAT_FACTORY_LLM_API_KEY
    timeout_s: 60
    max_retries: 2
    paid: false
  guardrails:
    max_requests_per_job: 0
    max_tokens_per_job: 0
    cost_ceiling_usd: 0.0
    allow_paid_models: false
```

---

## LLM 配置

- `api_key`: API 密钥
- `base_url`: API 服务地址（支持任何 OpenAI 兼容接口）
- `model`: 模型名称

---

## ComfyUI 配置

### 基础配置

- `comfyui_url`: 本地 ComfyUI 地址（默认 `http://127.0.0.1:8188`）
- `comfyui_api_key`: ComfyUI API 密钥（可选，用于 [Comfy Platform](https://platform.comfy.org/profile/api-keys)）

### RunningHub 云端配置

- `runninghub_api_key`: RunningHub API 密钥（使用云端工作流时必填）
- `runninghub_concurrent_limit`: 并发执行限制（1-10，普通会员默认为 1）
- `runninghub_instance_type`: 实例类型（可选）
  - 留空或不设置：使用 24GB 显存机器
  - `"plus"`: 使用 48GB 显存机器（适合大尺寸视频生成）

### 图像配置

- `default_workflow`: 默认图像生成工作流
- `prompt_prefix`: 提示词前缀

### 视频配置

- `default_workflow`: 默认视频生成工作流
  - `runninghub/video_wan2.1_fusionx.json`: 云端工作流（推荐，无需本地环境）
  - `selfhost/video_wan2.1_fusionx.json`: 本地工作流（需要本地 ComfyUI 支持）
- `prompt_prefix`: 视频提示词前缀（用于控制视频生成风格）

### TTS 配置

- `default_workflow`: 默认 TTS 工作流

---

## 模板配置

- `default_template`: 默认帧模板路径（例如 `1080x1920/image_default.html`）

---

## Cat Factory 配置

- `enabled`: 启用 cat factory 功能（布尔值，默认 `false`）。该配置为可选功能，默认禁用。
- `llm.provider`: 逻辑提供商标识（默认 `openai_compatible`）；使用 OpenAI 兼容接口，不按厂商分支处理。
- `llm.model`: 提供商模型标识（默认空值；实时调用前必须设置）。
- `llm.base_url_env`: 保存端点 URL 的环境变量名（默认 `CAT_FACTORY_LLM_BASE_URL`）。
- `llm.api_key_env`: 保存凭据的环境变量名（默认 `CAT_FACTORY_LLM_API_KEY`）。这两个设置存的是**名称**，不是值。
- `llm.timeout_s`: 单次调用超时秒数（默认 `60`）。
- `llm.max_retries`: 每次请求的瞬时故障重试上限（默认 `2`）；JSON 解析/校验另有一次修复重试。
- `llm.paid`: 显式付费标记（默认 `false`）。非本地端点也视为可能产生费用。
- `guardrails.max_requests_per_job`: 每个任务的请求上限（默认 `0`，不限制）。
- `guardrails.max_tokens_per_job`: 每个任务的 token 上限（默认 `0`，不限制）。
- `guardrails.cost_ceiling_usd`: 每个任务的预估费用上限（默认 `0.0`，不限制）。
- `guardrails.allow_paid_models`: 允许付费或非本地配置（默认 `false`）；否则首个请求前停止规划。

端点及凭据的读取顺序：先读取所命名的环境变量；未设置时，读取仓库根目录中私有且被 Git 忽略的 `config.cat_factory.local.yaml`，使用相同名称的顶层键。缺失时直接报错。Cat Factory 解析器不会回退到全局 `llm.api_key` 或 `llm.base_url`。

本仓库当前尚未对真实提供商模型进行端到端验收（**未验证**）：未配置任何提供商模型或端点，因此不应假定该集成可在实时模型上正常工作。

### GPU 编排（`cat_factory.gpu`）

本地单 GPU 图像/视频编排（仅 ComfyUI），由 ComfySupervisor 及其持久化状态机消费。

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `comfy_command` | `list[str]` | `[]` | ComfyUI 启动命令，以参数数组形式给出（可执行文件 + 参数）；绝不是 shell 字符串。 |
| `supervisor_mode` | `"managed"` \| `"attached"` | `"managed"` | `managed` 由本进程启动并监控 ComfyUI；`attached` 连接已在运行的服务。 |
| `generation_timeout_s` | `int` | `900` | 单个生成步骤允许的最长墙钟秒数。 |
| `poll_interval_s` | `float` | `2.0` | 进度/健康检查轮询间隔秒数。 |
| `stale_heartbeat_seconds` | `int` | `300` | 超过该秒数未收到心跳即认为运行中的任务已失联。 |
| `vram_margin_mb` | `int` | `1024` | 在报告的显存总量之下保留的安全余量（MB）。 |
| `ram_margin_mb` | `int` | `2048` | 在报告的系统内存总量之下保留的安全余量（MB）。 |
| `offload_min_ram_mb` | `int` 或 `null` | `null` | 卸载单个模型族所需的实测最低系统内存（MB）；`null` 表示尚未测量。 |
| `oom_max_retries` | `int` | `1` | 显存不足（OOM）失败后的最大自动重试次数。 |
| `memory_release_timeout_s` | `int` | `60` | 等待显存/内存释放的最长秒数，超时则 fail closed。 |
| `state_dir` | `str` | `""` | 存放租约/状态文件的项目相对目录；空值表示使用功能默认位置。 |
| `image_workflow` | `str` | `"workflows/selfhost/image_flux3.json"` | 用于图像生成的项目相对 ComfyUI 工作流。 |
| `reference_workflow` | `str` | `"workflows/selfhost/image_flux3_ref.json"` | 用于参考图条件生成的项目相对 ComfyUI 工作流。 |
| `video_workflow` | `str` | `"workflows/selfhost/video_wan2_2_i2v.json"` | 用于图生视频的项目相对 ComfyUI 工作流。 |

`comfy_command` 是参数数组：直接执行，绝不经过 shell 插值。空值表示“未配置”，此时 GPU 编排会 fail closed，而不会猜测一个可执行文件。`state_dir` 若设置，必须为相对路径（不得为绝对路径，不得含 `..` 穿越段）。

---

## 更多信息

配置文件会自动在首次运行时创建。

