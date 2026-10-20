# Config Schema

Detailed explanation of the `config.yaml` configuration file.

---

## Configuration Structure

```yaml
llm:
  api_key: "your-api-key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"

comfyui:
  comfyui_url: "http://127.0.0.1:8188"
  comfyui_api_key: ""  # ComfyUI API key (optional)
  runninghub_api_key: ""
  runninghub_concurrent_limit: 1  # Concurrent limit (1-10)
  runninghub_instance_type: ""  # Instance type (optional, set to "plus" for 48GB VRAM)
  
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

# Optional Cat Factory profile (disabled until explicitly enabled)
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

## LLM Configuration

- `api_key`: API key
- `base_url`: API service address (supports any OpenAI-compatible interface)
- `model`: Model name

---

## ComfyUI Configuration

### Basic Configuration

- `comfyui_url`: Local ComfyUI address (default `http://127.0.0.1:8188`)
- `comfyui_api_key`: ComfyUI API key (optional, for [Comfy Platform](https://platform.comfy.org/profile/api-keys))

### RunningHub Cloud Configuration

- `runninghub_api_key`: RunningHub API key (required for cloud workflows)
- `runninghub_concurrent_limit`: Concurrent execution limit (1-10, default 1 for regular members)
- `runninghub_instance_type`: Instance type (optional)
  - Empty or unset: Use 24GB VRAM machine
  - `"plus"`: Use 48GB VRAM machine (suitable for large video generation)

### Image Configuration

- `default_workflow`: Default image generation workflow
- `prompt_prefix`: Prompt prefix

### Video Configuration

- `default_workflow`: Default video generation workflow
  - `runninghub/video_wan2.1_fusionx.json`: Cloud workflow (recommended, no local setup required)
  - `selfhost/video_wan2.1_fusionx.json`: Local workflow (requires local ComfyUI support)
- `prompt_prefix`: Video prompt prefix (controls video generation style)

### TTS Configuration

- `default_workflow`: Default TTS workflow

---

## Template Configuration

- `default_template`: Default frame template path (e.g., `1080x1920/image_default.html`)

---

## Cat Factory Configuration

- `enabled`: Enable the cat factory feature (boolean, default `false`). This section is opt-in and disabled by default.
- `llm.provider`: Logical provider ID (default `openai_compatible`); OpenAI-compatible API, with no vendor-specific branch.
- `llm.model`: Provider model ID (default empty; set before a live call).
- `llm.base_url_env`: Name of the environment variable holding the endpoint URL (default `CAT_FACTORY_LLM_BASE_URL`).
- `llm.api_key_env`: Name of the environment variable holding the credential (default `CAT_FACTORY_LLM_API_KEY`). These two settings contain **names**, never values.
- `llm.timeout_s`: Per-call timeout in seconds (default `60`).
- `llm.max_retries`: Maximum transient retries per request (default `2`); JSON parse/validation repair is a separate single retry.
- `llm.paid`: Explicit cost marker (default `false`). A non-local endpoint is also treated as cost-incurring.
- `guardrails.max_requests_per_job`: Per-job request cap (default `0`, unlimited).
- `guardrails.max_tokens_per_job`: Per-job token cap (default `0`, unlimited).
- `guardrails.cost_ceiling_usd`: Per-job estimated cost ceiling (default `0.0`, unlimited).
- `guardrails.allow_paid_models`: Permit paid or non-local profiles (default `false`); otherwise planning stops before the first request.

Credential/endpoint values are read first from the named environment variables, then from top-level keys with those same names in the private, gitignored repository-root `config.cat_factory.local.yaml`. Missing values fail closed. The Cat Factory resolver does not fall back to the global `llm.api_key` or `llm.base_url`.

Live end-to-end acceptance against a real provider model is **unverified** in this checkout: no provider model or endpoint is configured, so the integration must not be assumed to work against a live model.

### GPU Orchestration (`cat_factory.gpu`)

Local single-GPU image/video orchestration (ComfyUI only), consumed by the ComfySupervisor and its durable state machine.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `comfy_command` | `list[str]` | `[]` | ComfyUI launch command as an argument array (executable + args); never a shell string. |
| `supervisor_mode` | `"managed"` \| `"attached"` | `"managed"` | `managed` launches and monitors ComfyUI; `attached` dials an already-running server. |
| `generation_timeout_s` | `int` | `900` | Maximum wall-clock seconds for one generation step. |
| `poll_interval_s` | `float` | `2.0` | Seconds between progress/health polls. |
| `stale_heartbeat_seconds` | `int` | `300` | Seconds without a heartbeat after which a running job is considered stale. |
| `vram_margin_mb` | `int` | `1024` | Safety margin in MB reserved below the reported VRAM total. |
| `ram_margin_mb` | `int` | `2048` | Safety margin in MB reserved below the reported system RAM total. |
| `offload_min_ram_mb` | `int` or `null` | `null` | Measured minimum system RAM in MB required to offload one model family; `null` means not yet measured. |
| `oom_max_retries` | `int` | `1` | Maximum automatic retries after an out-of-memory failure. |
| `memory_release_timeout_s` | `int` | `60` | Maximum seconds to wait for VRAM/RAM release before failing closed. |
| `state_dir` | `str` | `""` | Project-relative directory for lease/state files; empty means the feature default location. |
| `image_workflow` | `str` | `"workflows/selfhost/image_flux3.json"` | Project-relative ComfyUI workflow for image generation. |
| `reference_workflow` | `str` | `"workflows/selfhost/image_flux3_ref.json"` | Project-relative ComfyUI workflow for reference-conditioned image generation. |
| `video_workflow` | `str` | `"workflows/selfhost/video_wan2_2_i2v.json"` | Project-relative ComfyUI workflow for image-to-video generation. |

`comfy_command` is an argument array: it is executed directly (never shell-interpolated). An empty value means "not configured", and GPU orchestration then fails closed rather than guessing an executable. `state_dir`, when set, must be relative (no absolute paths, no `..` traversal).

---

## More Information

The configuration file is automatically created on first run.

