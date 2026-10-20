# Pixelle-Video — Project Status

**Step:** Step 04 — Local Single-GPU Image/Video Orchestration (ComfyUI only)
**Status:** Phases 1-4 COMPLETE (29 of 29 tasks; Phase 4 = 10 of 10). Step 03 cloud-LLM library IMPLEMENTED with AC-03.1 live acceptance BLOCKED / UNVERIFIED (not failed) pending a configured provider model/endpoint. Step 04 GPU orchestration IMPLEMENTED and OFFLINE-ACCEPTED; AC-04.3 LIVE acceptance remains UNVERIFIED (requires separate operator authorization and has NOT been run).
**Verification revision:** Step 00 baseline at `9851dac`; Step 03 offline checks through `bfa6c57`; Step 04 offline evidence through `3c4efb7` (plus post-ledger `2a143e7`, `88419e5`). Re-read `git rev-parse HEAD` for the current tip.
**Flag state:** `cat_factory.enabled` is DISABLED BY DEFAULT; the production UI/pipeline does not yet invoke the Cat Factory library.

## Progress summary
- **Phases 1-4: COMPLETE = 29 of 29 tasks** (Phase 1 = 6, Phase 2 = 5, Phase 3 = 8, Phase 4 = 10).
- Phase 4 (Step 04) is COMPLETE: 10 of 10 tasks (4.1-4.10).
- AC-03.1 is **BLOCKED / UNVERIFIED — explicitly NOT failed**. Reason: no provider model or endpoint is configured in this checkout. No specific model id is configured, and none is invented here.
- AC-04.3 (live GPU acceptance) is **UNVERIFIED — explicitly NOT passed**. It requires separate operator authorization and a live GPU run; neither has happened in this checkout.

## Phase 3 — Step 03 status (Configurable Cloud LLM Integration) — COMPLETE
- Configured provider ID: `openai_compatible` (default); configured model: **none** (`cat_factory.llm.model=""`). Defaults: `enabled=false`, `base_url_env=CAT_FACTORY_LLM_BASE_URL`, `api_key_env=CAT_FACTORY_LLM_API_KEY`, `timeout_s=60`, `max_retries=2`, `paid=false`; guardrails `max_requests_per_job=0`, `max_tokens_per_job=0`, `cost_ceiling_usd=0.0`, `allow_paid_models=false`.
- Deliverables implemented: opt-in config schema (`CatFactoryLLMConfig` + `CatFactoryGuardrailsConfig`), the `pixelle_video/cat_factory/` package with five Pydantic v2 planning contracts (`CreativeBrief`, `InspirationCandidate`, `ParodyIdea`, `ShotPlan`, `AudioPlan`), `llm_profile.py`, `prompts.py`, `planner.py`, `storage.py`, and the offline test suite `tests/cat_factory/`.
- Credential/endpoint values: environment variables named above take precedence over matching top-level keys in private, gitignored `config.cat_factory.local.yaml` at the repository root. No value is stored in the example or this status record. Missing settings fail closed; the global `llm.api_key`/`llm.base_url` fallback is unsupported. The stock LLM client lacks a timeout argument, so the planner wraps the call.
- Offline evidence (2026-10-17): `.venv\\Scripts\\python.exe -m pytest tests/cat_factory -q` → **5 passed, 1 skipped** (optional live test); `.venv\\Scripts\\ruff.exe check tests/cat_factory` → **all checks passed**; Cat Factory test-file `pre_check_batch` → **passed**, zero secretscan findings. READ-ONLY OPERATOR commands (run by the operator because the coder shell cannot run git): `git check-ignore -v config.cat_factory.local.yaml` confirms the private-config rule; `git status --short tests/cat_factory` exposes both new test files.
- **Acceptance: BLOCKED / UNVERIFIED** for AC-03.1 and provider-specific compatibility until a model, endpoint and credential are configured and the opt-in live test is run. This is NOT a failure. The offline planning/validation contract passed. The reusable manifest/checkpoint writer is present, but no production caller currently orchestrates it with the planner.
- Phase 3 task coverage is COMPLETE (tasks 3.1-3.8). Commits: 3.1 `7f43908`, 3.2 `aca0358`, 3.3 `3350a1e`, 3.4 `88a9b35`, 3.5 `b249fec`, 3.6 `58b1a6a`, 3.7 `062b819` / `7e2e505` / `e601261`, 3.8 `6e4a175` / `bfa6c57`. The earlier note that 3.7/3.8 "remain pending" was stale and is superseded by this entry.

## Phase 4 — Step 04 status (Local Single-GPU Image/Video Orchestration, ComfyUI only) — COMPLETE (offline-accepted; live acceptance UNVERIFIED)
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

## Step 00 Task Tracker

| Task | Description | Status | Commit |
|---|---|---|---|
| 1.1 | Pre-change baseline smoke evidence (7 checks) | DONE | `592b31a` |
| 1.2 | Add opt-in `CatFactoryConfig` flag to config schema | DONE | `e5a8e34` |
| 1.3 | Document the flag (config.example.yaml, en/zh docs) | DONE | `0537e0a` |
| 1.4 | Post-change baseline smoke evidence + comparison + leaf diff | DONE | `488ea58` |
| 1.5 | Step 00 implementation status report | DONE | see HEAD |
| 1.6 | Project status tracker (this file) | DONE | pending commit |

## Verification Summary
- All 7 baseline smoke checks PASS before AND after the change; all 7 comparisons `identical: true`.
- Configuration defaults UNCHANGED: recursive dotted-leaf diff of the config dump is 39 leaves before / 40 after, with exactly ONE added leaf `cat_factory.enabled = false`, and zero removed or changed leaves.
- `cat_factory.enabled` is `False` both from a default instance and from `model_validate({})`.
- Evidence files: `docs/cat_factory/baseline-smoke.json` (pre) and `docs/cat_factory/baseline-smoke-post.json` (post).
- Full detail: `docs/cat_factory/IMPLEMENTATION_STATUS.md`.

## Open Blockers
- B1: RESOLVED — `plan/06-quality-control.md` was recreated by task 2.4.
- B2: RESOLVED — tests/cat_factory now contains the Step 03 offline suite plus the Step 04 offline GPU suite (69 passed + 1 skipped in this session).
- B3: A 480x832 portrait Wan2.2 I2V workflow DOES exist at `workflows/selfhost/video_wan2_2_i2v.json`. The remaining unverified item is the production-resolution (1080x1920) output path — NOT workflow existence.
- B4: the config default frame template `1080x1920/default.html` (`pixelle_video/config/schema.py:118`) is ABSENT on disk. `templates/1080x1920` contains `image_default.html`, `static_default.html`, `video_default.html` — no plain `default.html`. `pixelle_video/utils/template_util.py:365-372` migrates `'default.html'` -> `'image_default.html'`; `pixelle_video/config/manager.py:68` logs a fallback. Real-world effect: NEEDS RUNTIME VERIFICATION — do not assume a resolution.
- B5: venv lint is AVAILABLE — `.venv\Scripts\python.exe -m ruff check pixelle_video/cat_factory/gpu tests/cat_factory` reports 3 pre-existing style findings (F401 `mock_comfy.py:32`; E731 `reconcile.py:158`; I001 `tests/cat_factory/test_gpu_config_schema.py:20`); none are functional. The PATH `ruff` binary is absent, so the plugin lint gate soft-skips.
- B6: NO 9:16 (1080x1920) ComfyUI video workflow exists, and creating one was intentionally NOT done. This is a distinct, additional pre-existing blocker; it does not contradict B3 (which records that a 480x832 I2V workflow DOES exist and only the production-resolution output path is unverified).

## Specification vs Implementation
- COMPLETED SPECIFICATIONS: the `plan/` documents are written specifications, NOT working application functionality.
- WORKING APPLICATION FUNCTIONALITY: reusable Step 03 planning and manifest modules exist, but there is no integrated Cat Factory pipeline or feature-flag consumer yet.
- Target stack: RTX 3090 reserved for ComfyUI FLUX3 image + Wan2.2 GGUF I2V; cloud LLM (DeepSeek/GLM) for planning; no TabbyAPI/Qwen production dependency.

## Next Step
- **Phase 4 / Step 04: COMPLETE (offline-accepted).** Remaining authorized work is explicitly out of scope: AC-03.1 and AC-04.3 are UNVERIFIED pending operator-configured provider/live GPU authorization.
- **Step 03 live acceptance (open):** AC-03.1 is BLOCKED / UNVERIFIED — it needs an explicitly configured provider model/endpoint and credential, then the opt-in live test. It is not failed and no model id is asserted.
- **Historical note (Phase 2 closeout):** the formal `phase_complete(2)` call was blocked by a phase-level agent-participation gate requiring test_engineer participation. All 5 Phase 2 tasks are done, reviewed, drift-verified, and committed (2.1 `11326df`, 2.2 `0fc6000`, 2.3 `1791b81`, 2.4 `ff11009` + `9851dac`, 2.5 `082e774`).
- **Remaining steps after Step 04:** steps 01, 02, 05, 06, 07.
