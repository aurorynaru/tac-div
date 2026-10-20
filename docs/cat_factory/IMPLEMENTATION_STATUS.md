# Cat Factory — Implementation Status

## Current status — Phases 1-4 complete; Step 03 library and Step 04 GPU orchestration IMPLEMENTED
- **Phases 1-4: COMPLETE = 29 of 29 tasks** (Phase 1 = 6, Phase 2 = 5, Phase 3 = 8, Phase 4 = 10).
- **Step 03 library: IMPLEMENTED** (unchanged; AC-03.1 BLOCKED / UNVERIFIED, not failed).
- **Step 04 GPU orchestration: IMPLEMENTED and OFFLINE-ACCEPTED.** AC-04.3 LIVE acceptance is **UNVERIFIED** (separate operator authorization required; not run; never recorded as passed).
- **Step 03 library detail.** The opt-in cloud-LLM planning library is implemented: the config schema fields (`CatFactoryLLMConfig` + `CatFactoryGuardrailsConfig`), the five Pydantic v2 planning contract models (`CreativeBrief`, `InspirationCandidate`, `ParodyIdea`, `ShotPlan`, `AudioPlan`) in the `pixelle_video/cat_factory/` package, `llm_profile.py`, `prompts.py`, `planner.py`, `storage.py`, and the offline test suite `tests/cat_factory/`. The end-to-end UI/pipeline is not wired to this library yet.
- **AC-03.1: BLOCKED / UNVERIFIED — explicitly NOT failed.** Reason: no provider model or endpoint is configured in this checkout. The optional live test therefore does not run. No specific model id is configured or asserted.
- Default `cat_factory.enabled=false`; `cat_factory.llm` defaults: `provider=openai_compatible`, `model=""`, `base_url_env=CAT_FACTORY_LLM_BASE_URL`, `api_key_env=CAT_FACTORY_LLM_API_KEY`, `timeout_s=60`, `max_retries=2`, `paid=false`. The configured provider ID is `openai_compatible`; **no model ID is configured**. Never put credential values in this report.
- Default `cat_factory.guardrails`: `max_requests_per_job=0`, `max_tokens_per_job=0`, `cost_ceiling_usd=0.0` (zero means unlimited), `allow_paid_models=false`. A paid-marked or non-local endpoint needs explicit permission before a request.
- Credential/endpoint lookup: the named process environment variables win; otherwise the same top-level names are read from the private repository-root `config.cat_factory.local.yaml` (ignored by `*.local.yaml`). Missing values fail closed. The optional fallback to global `llm.api_key`/`llm.base_url` suggested in the Step 03 spec is deliberately unsupported; the stock `LLMService` has no timeout argument, so the planner applies the timeout around the call.
- Test evidence (2026-10-17): `.venv\\Scripts\\python.exe -m pytest tests/cat_factory -q` → **5 passed, 1 skipped** (live credential unset); `.venv\\Scripts\\ruff.exe check tests/cat_factory` → **all checks passed**; `pre_check_batch` on the three Cat Factory test files → **gates_passed=true**, zero secretscan findings, SAST pass. READ-ONLY OPERATOR commands (run by the operator because the coder shell cannot run git): `git check-ignore -v config.cat_factory.local.yaml` and `git status --short tests/cat_factory`. No network/GPU/provider was contacted by the offline suite.
- Phase 3 task coverage is COMPLETE (tasks 3.1-3.8). Commits: 3.1 `7f43908`, 3.2 `aca0358`, 3.3 `3350a1e`, 3.4 `88a9b35`, 3.5 `b249fec`, 3.6 `58b1a6a`, 3.7 `062b819` / `7e2e505` / `e601261`, 3.8 `6e4a175` / `bfa6c57`. The earlier note that 3.7/3.8 "remain pending" was stale and is superseded here.
- Provider/API incompatibility: only OpenAI-compatible structured-response behavior is exercised offline; provider-specific live behavior has not been established. The global-LLM fallback is not implemented. Step 04 may consume the manifest/checkpoint contract after a caller orchestrates planning and persistence.

## Current status — Step 04 (Phase 4, COMPLETE; offline-accepted, live acceptance UNVERIFIED)
- Phase 4 = **"Local Single-GPU Image/Video Orchestration (ComfyUI only)"**, tasks 4.1-4.10, **COMPLETE = 10 of 10**.
- Architectural target: the **RTX 3090 24GB** is reserved for **local ComfyUI FLUX 3 image generation** and **Wan2.2 GGUF (HighNoise/LowNoise) I2V video generation**, with **concurrency = 1**. Planning runs on a **remote OpenAI-compatible cloud provider**. There is **no TabbyAPI / local-Qwen runtime dependency**.
- Step 04 obsolete-dependency sweep: **ZERO TabbyAPI / local-Qwen runtime lifecycle dependencies found (LIFECYCLE: none found)**; no active integration path correction is required.
- Deliverables (all opt-in; `cat_factory.enabled` still defaults to `false`):
  - `pixelle_video/config/schema.py` — `CatFactoryGpuConfig` (argument-array `comfy_command`, `supervisor_mode`, timeouts, `vram_margin_mb`, `ram_margin_mb`, measured `offload_min_ram_mb`, `oom_max_retries`, `memory_release_timeout_s`, project-relative `state_dir`, and the three workflow paths). Task 4.10.
  - `pixelle_video/cat_factory/gpu/lease.py` — `GpuLeaseManager`: OS-level file lock scoped to the physical GPU id, durable lease record (job id, owner pid, start time, heartbeat, phase, lock generation), explicit stale recovery that names observed evidence and never auto-force-deletes, fail-closed contention. Task 4.2.
  - `pixelle_video/cat_factory/gpu/supervisor.py` — `ComfySupervisor` managed/attached modes: argument-array spawn (`shell=False`), per-job stdout/stderr logs, identity-token-checked termination of the verified owned child tree only (no image-name/global kills), attached mode requires an explicit operator grant and never force-kills, and project-relative state-dir containment. Task 4.3.
  - Startup readiness (task 4.3 remediation): optional `startup_probe` + `startup_timeout_s` on `ComfySupervisor`; a bounded wait that fails closed with a sanitized logged command (command/pid/log paths only — never environment variables or credentials) and cleans up only the identity-verified owned child.
  - `pixelle_video/cat_factory/gpu/monitor.py` + `comfy_client.py` + `mock_comfy.py` — nvidia-smi VRAM sampling with injectable runner, injectable free-RAM reader, measured VRAM-release verification, configured-margin RAM gate (no hardcoded 24576 MB), localhost ComfyUI HTTP surface (`/queue`, `/prompt`, `/history`, `/free`) with a strictly read-only probe, and offline mock adapters. Task 4.4.
  - `pixelle_video/cat_factory/gpu/renderer.py` — durable serial orchestrator enforcing the ordered state machine `IDLE → JOB_ACQUIRE → VRAM_VERIFY → IMAGE_GEN → CHECKPOINT → IMAGE_RELEASE → VRAM_VERIFY → VIDEO_ACQUIRE → VRAM_VERIFY → RAM_VERIFY → VIDEO_GEN → CHECKPOINT → VIDEO_RELEASE → VRAM_VERIFY → CPU_ASSEMBLY → DONE`, concurrency 1, one model family resident, measured release between phases, bounded timeout/retry/OOM handling, checkpointed resume that never re-runs a completed shot, zero LLM calls, and fail-closed consumption of a plan without the `planning_complete` checkpoint. Task 4.5.
  - `pixelle_video/cat_factory/gpu/cli.py` — `probe`, `run --job`, `resume --job`, `status --job`, `release --job --owned-only`, `dry-run`; `dry-run` performs zero destructive calls (no POST, no spawn, no terminate); `release` acts only on an owned lease; messages are non-destructive. Task 4.6.
  - `pixelle_video/cat_factory/gpu/reconcile.py` + `web/utils/streamlit_helpers.py` + `web/app.py` — stock `MediaService` reuse gated on an owned lease, a read-only exclusive-lease probe, and a non-breaking Streamlit warning when a render holds the exclusive GPU lease. Task 4.7.
- **OFFLINE acceptance is PROVEN** (mocked adapters, injectable samplers, `tmp_path`; no GPU, no ComfyUI, no network). Runtime evidence from this session, run strictly one file at a time with the project venv:
  - `tests/cat_factory/test_gpu_lease.py` → 10 passed
  - `tests/cat_factory/test_gpu_supervisor.py` → 26 passed
  - `tests/cat_factory/test_gpu_renderer.py` → 21 passed
  - `tests/cat_factory/test_gpu_cli.py` → 12 passed
  - `tests/cat_factory/test_gpu_live.py` → 1 skipped (opt-in gate)
  - Totals: **69 passed, 1 skipped, 0 failed**. (The earlier 56-passed figure predates the drift-coverage tests added during Phase 4 hardening.)
- Exact operator command to run the offline suite (whole Cat Factory suite): `.venv\Scripts\python.exe -m pytest tests/cat_factory -q`. Live opt-in (read-only probe, no inference): set `CAT_FACTORY_GPU_LIVE=1` first.
- VRAM/RAM measurement basis: nvidia-smi measured GPU name/total/used/free; `cat_factory.gpu.vram_margin_mb` default **1024** and `ram_margin_mb` default **2048**; `offload_min_ram_mb` is a CONFIGURED MEASURED record (None fails closed; it is never computed at runtime); model release is verified by MEASUREMENT rather than trusting a `/free` response; no 24576 MB constant and no per-model footprint are hardcoded.
- Exclusive-ownership guarantee: only one lease holder per physical GPU (OS-level lock + durable record), stale recovery is explicit and evidence-named, and termination is restricted to the identity-verified owned child tree. **No unrelated process is ever terminated**; the offline suite asserts this and asserts attached-mode stop never signals.
- **AC-04.3 LIVE acceptance: UNVERIFIED** — no live GPU generation was authorized or run. It is not failed and must not be recorded as passed.
- Blockers B3/B6 remain open and unresolved: a 480x832 portrait I2V workflow exists (`workflows/selfhost/video_wan2_2_i2v.json`), but the production 1080x1920 output path is unverified and NO 9:16 (1080x1920) ComfyUI video workflow exists (intentionally not created).
- Phase 4 commits: 4.1 `9809f58`, 4.2 `09c33b3` (+`123d181`), 4.10 `3a0bdcb`, 4.3 `ecd7c26` / `20bdcbc` / `2be976b`, 4.4 `3087b2a` / `1d311eb`, 4.5 `ba6ef0a`, 4.6 `9dd45b4` / `9ca4b2e`, 4.7 `8d5f327`, 4.8 `bfd50c3` / `3371a32` / `328bbfd`, 4.3 remediation `382eeae`, 4.8 baselines/remediation `72eee45` / `f573899` / `e1df572`, 4.5 remediation `2a143e7`, 4.8 final coverage `88419e5`.
- Additional offline coverage: `tests/cat_factory/test_gpu_config_schema.py` was added by task 4.10 (test-only, outside its declared `files_touched`; schema-validation coverage).

## 1. Scope
Step 00 = the baseline prerequisite from `plan/00-START-HERE.md`: audit the repository baseline and add an opt-in Cat Factory feature flag that is disabled by default and changes no existing behaviour.

## 2. Verification revision
- Verification revision: `9851dac` (HEAD when this report was last updated, task 2.4).
- Captured with `git rev-parse HEAD` immediately before this report was written.
- IMPORTANT: the recorded revision is the *verification* revision, not a claim about the repository tip. Re-read `git rev-parse HEAD` for the current tip.

## 3. Feature flag design
- At the Step 00 baseline, `CatFactoryConfig(BaseModel)` had one field: `enabled: bool = Field(default=False)`; Step 03 subsequently added the `llm` and `guardrails` subtrees described above.
- Registered on `PixelleVideoConfig` as `cat_factory: CatFactoryConfig = Field(default_factory=CatFactoryConfig)`.
- DEFAULT IS DISABLED: `PixelleVideoConfig().cat_factory.enabled` is `False`; `PixelleVideoConfig.model_validate({}).cat_factory.enabled` is `False`.
- No existing field, default, or validator was modified in Step 00. The Step 03 library does not itself connect the feature flag to the UI/pipeline.

## 4. Files changed (Step 00)
| File | Change | Task | Commit |
|---|---|---|---|
| `pixelle_video/config/schema.py` | Added `CatFactoryConfig` and the `cat_factory` field (+6 lines, 0 deletions) | 1.2 | `e5a8e34` |
| `config.example.yaml` | Added a commented-out `cat_factory:` block with `enabled: false` (+6 lines) | 1.3 | `0537e0a` |
| `docs/en/reference/config-schema.md` | Added `## Cat Factory Configuration` section (+6 lines) | 1.3 | `0537e0a` |
| `docs/zh/reference/config-schema.md` | Added mirrored `## Cat Factory 配置` section (+6 lines) | 1.3 | `0537e0a` |
| `docs/cat_factory/baseline-smoke.json` | Pre-change baseline smoke evidence, 7 checks | 1.1 | `592b31a` |
| `docs/cat_factory/baseline-smoke-post.json` | Post-change baseline smoke evidence + comparison + leaf diff | 1.4 | `488ea58` |

## 5. Verification evidence
- Seven baseline smoke checks were run BEFORE the change (`docs/cat_factory/baseline-smoke.json`) and re-run AFTER the change with identical commands (`docs/cat_factory/baseline-smoke-post.json`).
- All 7 checks PASS in both runs; all 7 comparison rows are `identical: true`.
  1. `python --version` -> Python 3.12.7
  2. `import pixelle_video` -> resolves to `pixelle_video\__init__.py`
  3. `config_manager` -> `ConfigManager`
  4. `PixelleVideoConfig().model_dump()` + `is_llm_configured()` -> dump OK, `False`
  5. AST parse `web/app.py` -> `AST_OK`
  6. AST parse `pixelle_video/service.py` -> `AST_OK`
  7. `Test-Path -LiteralPath tests` -> `False` (tests/ does not exist)
- Configuration defaults are UNCHANGED. Verified by flattening both `config_default_dump` objects to dotted leaf paths: 39 leaves pre-change, 40 post-change, added leaves exactly `["cat_factory.enabled"]`, removed `[]`, changed `[]`. `defaults_unchanged: true`. No pre-existing default value or type differs.
- Note: checks 2-4 emit a stderr WARNING `Config file not found: config.yaml` (config.yaml is absent and gitignored) while still exiting 0; this is expected and is not a failure.
- No TabbyAPI or ComfyUI endpoint was started, stopped, or contacted; no generation was run.

## 6. Known limitations
- The Step 04 GPU path remains opt-in: `cat_factory.enabled` defaults to false. The operator CLI (`pixelle_video/cat_factory/gpu/cli.py`) invokes the renderer for `run`/`resume`; the Streamlit UI/pipeline does not invoke it yet. LIVE acceptance (AC-04.3) is UNVERIFIED. Do not treat the offline suite as live-generation evidence.
- The flag had NO consumer code at Step 00. The Step 03 library still has no production entry-point wiring; toggling `cat_factory.enabled` alone does not run a job. Do not treat "flag read back correctly" as behavioural coverage.
- COMPLETED SPECIFICATIONS vs WORKING APPLICATION FUNCTIONALITY: the `plan/` documents are written specifications. Step 03 now supplies reusable planning/storage modules and Step 04 supplies the offline-accepted GPU orchestrator, but an integrated Cat Factory pipeline has not been implemented. Target stack: RTX 3090 reserved for ComfyUI FLUX3 image + Wan2.2 GGUF I2V; cloud LLM (DeepSeek/GLM) for planning; no TabbyAPI/Qwen production dependency.

## 7. Blockers (carried forward)
- B1: RESOLVED — `plan/06-quality-control.md` was recreated by task 2.4.
- B2: RESOLVED — tests/cat_factory now contains the Step 03 offline suite and the Step 04 offline GPU suite (69 passed + 1 skipped this session).
- B3: A 480x832 portrait Wan2.2 I2V workflow DOES exist at `workflows/selfhost/video_wan2_2_i2v.json`. The remaining unverified item is the production-resolution (1080x1920) output path — NOT workflow existence.
- B4: the config default frame template `1080x1920/default.html` (`pixelle_video/config/schema.py:118`) is ABSENT on disk. `templates/1080x1920` contains `image_default.html`, `static_default.html`, `video_default.html` — no plain `default.html`. `pixelle_video/utils/template_util.py:365-372` migrates `'default.html'` -> `'image_default.html'`; `pixelle_video/config/manager.py:68` logs a fallback. Real-world effect: NEEDS RUNTIME VERIFICATION — do not assume a resolution.
- B5: venv lint is AVAILABLE — `.venv\Scripts\python.exe -m ruff check pixelle_video/cat_factory/gpu tests/cat_factory` reports 3 pre-existing style findings (F401 `mock_comfy.py:32`; E731 `reconcile.py:158`; I001 `tests/cat_factory/test_gpu_config_schema.py:20`); none are functional. The PATH `ruff` binary is absent, so the plugin lint gate soft-skips.
- B6: NO 9:16 (1080x1920) ComfyUI video workflow exists, and creating one was intentionally NOT done. This is a distinct, additional pre-existing blocker; it does not contradict B3 (which records that a 480x832 I2V workflow DOES exist and only the production-resolution output path is unverified).

## 8. Step 03 implementation and Step 04 handoff
- Phase 2 reconciled Step 03 (cloud LLM integration, `plan/03-llm-integration.md`) and Step 04 (local ComfyUI GPU orchestration, `plan/04-gpu-orchestration.md`). **Step 03's reusable library is complete (tasks 3.1-3.8)**; live acceptance (AC-03.1) is BLOCKED / UNVERIFIED as stated above — not failed.
- `cat_factory.enabled` remains `False` by default; no production entry point currently reads it to run a Cat Factory job. Do not describe the library as an integrated working application.
- Step 04 handoff: Phase 4 = "Local Single-GPU Image/Video Orchestration (ComfyUI only)" (tasks 4.1-4.10). Architectural target: RTX 3090 24GB reserved for local ComfyUI FLUX 3 image + Wan2.2 GGUF HighNoise/LowNoise I2V at concurrency 1; planning on a remote OpenAI-compatible cloud provider; no TabbyAPI/local-Qwen runtime dependency. The Step 04 dependency sweep found ZERO TabbyAPI/local-Qwen runtime lifecycle dependencies in active integration paths, so no active integration path correction is required. Step 04's GPU orchestration is IMPLEMENTED (tasks 4.1-4.10) and offline-accepted; AC-04.3 live acceptance is UNVERIFIED.
- Reuse the two baseline files as the pre/post comparison anchor for any later change.
