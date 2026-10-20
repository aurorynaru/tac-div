# Step 04 — Local single-GPU image/video orchestration (ComfyUI only)

**Order:** after Step 03 (`03-llm-integration.md`), before CPU post-processing.
**Objective:** on one 24 GB NVIDIA RTX 3090, own the GPU **exclusively** for the local ComfyUI and sequence the generation phases **serially, concurrency = 1**: FLUX 3 image generation for every shot first, then an image-model release, then Wan2.2 (HighNoise + LowNoise GGUF) image-to-video for the 480×832 portrait workflow.

> **Dependency change:** TabbyAPI / local Qwen is **no longer part of Step 04**. There is **no LLM lifecycle on this host**. The planning LLM runs remotely as a cloud/OpenAI-compatible provider in Step 03 and performs no GPU work here. This runner owns the GPU only for local ComfyUI image/video rendering.

## 1. Position in the flow

Step 04 begins **only after Step 03 has completed** and has:

- persisted the shared, versioned `manifest.json` at `output/cat_factory/<job_id>/manifest.json` (atomic replacement, the single shape defined in [`00-START-HERE.md`](00-START-HERE.md)), and
- emitted the idempotent **`planning_complete` checkpoint**.

The GPU runner **performs zero LLM calls and manages no LLM**. It consumes the saved plan and renders. The `planning_complete` → render handoff is one-directional and read-only:

```text
Step 03 (cloud LLM, no local GPU)                Step 04 (this runner, owns local GPU)
planning_complete  ────────────────────────────▶  read manifest.json (saved plan)
(saved plan, no GPU work)                          render FLUX 3 images -> Wan2.2 I2V
```

If `planning_complete` is absent or the manifest is not in the checkpoint-complete planning state, Step 04 **fails closed** and does not start generation. It never waits on, starts, unloads or restarts any LLM, and it never re-invokes the planner.

## 2. Supervisor = CPU-side runner

The supervisor is a **CPU-side runner in a separate Python process**. It does not run inference in-process; it drives the local ComfyUI over its HTTP surface and owns the GPU through an OS-level lease (Section 8). It sequences, in order:

1. **FLUX 3 image generation** for each shot (`workflows/selfhost/image_flux3.json`, or `image_flux3_ref.json` when a character reference is required).
2. **Image-model release** once all needed keyframes are produced.
3. **Wan2.2 image-to-video** for the portrait workflow (see Section 6).

Only one model family is loaded on the GPU at a time and only one generation job runs at a time: **concurrency = 1**.

## 3. Phase/state machine

Image/short-clip path (all states are ComfyUI/GPU states; **no LLM state exists**):

```text
IDLE
 -> JOB_ACQUIRE   (own the GPU lease)
 -> VRAM_VERIFY
 -> IMAGE_GEN     (FLUX 3, per shot, concurrency=1)
 -> CHECKPOINT
 -> IMAGE_RELEASE
 -> VRAM_VERIFY
 -> VIDEO_ACQUIRE
 -> VIDEO_GEN     (Wan2.2 HighNoise+LowNoise GGUF I2V, 480x832, per shot)
 -> CHECKPOINT
 -> VIDEO_RELEASE
 -> VRAM_VERIFY
 -> CPU_ASSEMBLY
 -> DONE
```

`IMAGE_GEN` and `VIDEO_GEN` loop **per shot** (bounded by the shot count / retry budget); each completed shot is checkpointed before the next begins.

Video path (the Wan2.2 path adds an explicit system-RAM gate before loading the video models):

```text
IDLE
 -> JOB_ACQUIRE -> VRAM_VERIFY
 -> IMAGE_GEN -> CHECKPOINT -> IMAGE_RELEASE -> VRAM_VERIFY
 -> VIDEO_ACQUIRE -> VRAM_VERIFY -> RAM_VERIFY
 -> VIDEO_GEN (Wan2.2 HighNoise+LowNoise GGUF I2V, 480x832, per shot)
 -> CHECKPOINT -> VIDEO_RELEASE -> VRAM_VERIFY
 -> CPU_ASSEMBLY -> DONE
```

`RAM_VERIFY` gates on **system RAM** (not VRAM): require free physical RAM above the workflow's **measured** offload minimum plus margin before loading the quantized/offload video workflow. The value must be measured per workflow; no figure is hardcoded here.

Every state transition must be durable (`manifest.json` / `events.jsonl`), reversible where possible, and fail **closed** (pause if the GPU or RAM is not sufficiently free). Timeouts are configurable. GPU-heavy models run **serially**, concurrency = 1.

## 4. GPU / VRAM gating

- Do **not** start inference when detected free/allocatable VRAM is below the workflow's **measured** minimum plus a safety margin.
- Never treat an optimistic estimate as proof of fit. Measure, then decide.
- After a release request, verify released memory by **measurement**; do not assume that an immediate `POST /free` response means VRAM was actually freed.
- If blocked, offer quantization / offload / a smaller model **with evidence**. Do **not** silently switch to a paid cloud provider.

## 5. System-RAM monitoring

The Wan2.2 A14B GGUF workflow is quantized and CPU-offloads an inactive expert plus the text encoder, so it can consume host RAM beyond VRAM. Gate the video path on free physical RAM above the workflow's **measured** offload minimum plus margin. Do **not** hardcode any specific model's RAM footprint; require the value to be measured for the actual workflow on this host.

Keep the caution: **do not claim Wan2.2 will fit the 3090 before measuring an actual quantized/offload workflow.** Treat out-of-memory as a normal blocked result, not a bug.

## 6. Workflow files

Reference these real, repo-relative workflow files by exact path:

| Workflow | Path | Use |
|---|---|---|
| FLUX 3 image | `workflows/selfhost/image_flux3.json` | Per-shot keyframe generation |
| FLUX 3 reference | `workflows/selfhost/image_flux3_ref.json` | Keyframe generation when a character reference is required |
| Wan2.2 I2V | `workflows/selfhost/video_wan2_2_i2v.json` | Wan2.2 I2V A14B **HighNoise + LowNoise GGUF**, portrait **480×832** (length 81 frames) with the `umt5` CLIP and `wan_2.1_vae` |

The **480×832 portrait Wan2.2 I2V workflow EXISTS** in the repository. The **production-resolution (1080×1920) output path is UNVERIFIED** and is tracked as **blocker B3**: do not claim the production-resolution path works, and do not claim the workflow is missing.

## 7. Sequential load/release

Load only the model family needed for the current phase and release it before loading the next, to minimise peak memory:

- During `IMAGE_GEN` only the FLUX 3 image models are resident.
- `IMAGE_RELEASE` frees the image models **before** `VIDEO_ACQUIRE` loads the Wan2.2 models.
- `VIDEO_RELEASE` frees the video models before `CPU_ASSEMBLY`.
- Single-GPU, concurrency = 1: no two GPU-heavy phases overlap.

## 8. GPU lease and process ownership

`GpuLeaseManager` holds an **OS-level file lock** scoped to the physical GPU ID. The lease records: **job ID, owner PID, start time, heartbeat, phase, and lock generation**. Handle a stale lock with **explicit recovery**, never an automatic force-delete.

`ComfySupervisor` has **two modes**:

- **`managed`** — the runner starts and owns the ComfyUI child process; it saves the exact PID / command / config, can gracefully stop it, wait for termination, and relaunch. Windows process termination is limited to the **verified child process tree**; never `taskkill /IM python.exe` and never globally kill a user's ComfyUI or any unrelated process.
- **`attached`** — a preexisting ComfyUI remains running and is used **only when the operator has explicitly opted into exclusive ownership of that instance**. `attached` mode **never force-kills**; it has no kill fallback.

## 9. Hard constraints (preserved)

- **Never kill unrelated processes.** No global `taskkill`; never terminate a user's services or any process the runner did not start.
- **Exclusive ownership of the local ComfyUI.** Only act on the GPU while holding the lease; if exclusivity cannot be established, pause rather than acting.
- **No unauthorized cloud fallback.** Never silently switch to a paid cloud provider.
- **Single-GPU concurrency = 1.**

## 10. ComfyUI integration

Keep the stock ComfyUI HTTP surface:

- `GET /queue`
- `POST /prompt`
- `GET /history/{prompt_id}`
- `POST /free` with JSON `{"unload_models": true, "free_memory": true}`

The `/free` request requests work through a queue flag; **verify actual VRAM release by measurement** rather than assuming the immediate response means memory is freed.

NVIDIA memory accounting: query `nvidia-smi` (Windows supported with an installed driver); sample per-process memory where available; reserve a safety margin. Avoid hard-coded `24,576 MB` free-VRAM assumptions.

## 11. Implementation tasks

1. In `pixelle_video/cat_factory/gpu/` create `GpuLeaseManager` with an OS-level file lock scoped to physical GPU ID; the lease contains job ID, owner PID, start time, heartbeat, phase and lock generation. Handle stale locks with explicit recovery, never automatic force-delete.
2. Create the `ComfySupervisor` interface with the two modes (`managed`, `attached`) from Section 8.
3. Create **mocked ComfyUI adapters** for offline tests, including timeouts, request errors, partial success and long-running pending jobs. Adapters bind localhost by default; `probe` is read-only.
4. Expose `probe`: record GPU name, total memory, assigned per-process VRAM, ComfyUI queue state, and the required media workflow model files. `probe` never mutates state.
5. `IMAGE_GEN`: for each shot lacking an accepted keyframe, submit the FLUX 3 workflow (`image_flux3.json`, or `image_flux3_ref.json` when a character reference is required) via `POST /prompt`, monitor `GET /queue` and `GET /history/{prompt_id}`, and persist clip paths and errors before releasing.
6. `IMAGE_RELEASE`: prefer a graceful `POST /free` only once the queue is drained; wait until memory release is **measured**.
7. `VIDEO_GEN`: run `video_wan2_2_i2v.json` (Wan2.2 I2V A14B HighNoise + LowNoise GGUF) at 480×832 for each shot, only after `VRAM_VERIFY` and `RAM_VERIFY` pass.
8. Add a VRAM measurement **gate**: do not start video inference if detected free/allocatable VRAM is below the workflow's measured minimum plus margin; do not use optimistic estimates as proof of fit; if blocked, offer quantization/offload/smaller model with evidence and never silently switch to a paid cloud provider.
9. Add checkpoint/restart semantics: on crash, read the manifest and ComfyUI state; revalidate outputs and do not re-run completed shots. An interrupted render is quarantined/retried only within budget; never delete in-use service outputs.
10. Add CLI commands: `probe`, `run --job <id>`, `resume --job <id>`, `status --job <id>`, `release --job <id> --owned-only`, and `dry-run`. `dry-run` must never `POST` to any ComfyUI endpoint, never spawn GPU services, and never terminate processes.
11. Reconcile with stock `MediaService`/ComfyKit: use the existing media generation path where possible, but **only** while the CPU runner owns the GPU lease. Avoid concurrent Streamlit use during an exclusive lease; add a UI warning/guard.
12. Test offline without ComfyUI or a GPU. The runner must still complete a mock render and CPU assembly based on a saved plan.
13. Bounded OOM retries: an OOM marks the shot failed, releases the owned lease, preserves artifacts, and never loops infinitely (retry only within the configured budget).

## 12. Failure scenarios to test offline

| Scenario | Required behavior |
|---|---|
| Another app currently using the GPU | Pause with reasons and process observations; no kills |
| Preexisting ComfyUI the runner does not own | Treat as a blocker unless `attached` exclusive ownership was explicitly opted into; never force-kill |
| `POST /free` returns success but VRAM remains occupied | Timeout and abort the next acquisition |
| ComfyUI queue active | Wait / refuse to load; never discard other users' pending work |
| ComfyUI model OOM | Mark shot failed, release owned lease, preserve artifacts; bounded retry only, no infinite restart |
| Runner crashes after frame 1 | Resume from frame 2 after integrity check; no duplicate output |
| Windows process startup times out | Fail with logged command + sanitized env; no global process kill |
| Free physical RAM below the measured offload minimum | Block the video path with a measured reason; no cloud fallback |

## 13. Acceptance criteria

- **AC-04.1** Offline tests exercise valid transitions, blocked transitions, timeouts and resume — with no GPU, no ComfyUI and no network (mocked adapters).
- **AC-04.2** `dry-run` proves **zero** destructive calls: no `POST` to any ComfyUI endpoint, no GPU service spawned, no process terminated.
- **AC-04.3** A live test (explicitly authorized) acquires the GPU lease, runs the FLUX 3 image phase then the Wan2.2 HighNoise+LowNoise GGUF I2V phase at 480×832, and releases VRAM with a **measured** release (verified by before/after memory sampling, not by the `POST /free` response alone).
- **AC-04.4** No unrelated process is terminated; a preexisting ComfyUI the runner does not own is treated as a blocker, and `attached` mode never force-kills.
- **AC-04.5** One incomplete job can be resumed from the persisted manifest; no duplicate shot output and no lost provenance; completed shots are not re-run.
- **AC-04.6** An OOM during a shot marks that shot failed, releases the owned lease, preserves artifacts, and stops after the bounded retry budget without an infinite loop.

## 14. 3090 test advice

Start with a low-memory image workflow; then a **single** 2–4 s low-resolution image-to-video shot. Record peak VRAM, duration and stability. Wan2.2 model/version, sampler, precision, offload flags, frame count and actual custom-node requirements are *empirical configuration inputs*, not guaranteed by this file. Treat out-of-memory as a normal blocked test result.

## 15. Reconciliation with stock `MediaService`/ComfyKit

Reconcile with the stock `MediaService`/ComfyKit path: use the existing media generation path where possible, but **only** while the CPU runner owns the GPU lease. Avoid concurrent Streamlit button use during an exclusive lease; add a UI warning/guard.

## 16. OpenCode task prompt

> Implement only `04-gpu-orchestration.md` after reviewing the existing repo paths and the stock ComfyUI integration. Build a CPU-only Windows-capable runner and mocked ComfyUI adapters first. Sequence the local ComfyUI phases serially at concurrency=1: FLUX 3 images (`workflows/selfhost/image_flux3.json`, `workflows/selfhost/image_flux3_ref.json`), release, then Wan2.2 HighNoise+LowNoise GGUF I2V (`workflows/selfhost/video_wan2_2_i2v.json`) at 480×832. There is no LLM lifecycle in this runner. Do not run live loading, process termination or GPU-intensive inference without operator opt-in. Never kill unrelated processes or a user's ComfyUI instance, and never silently fall back to a paid cloud provider. Show state-transition tests, dry-run output proving zero destructive calls, and the exact service-ownership strategy. Stop after the step and hand back status.
