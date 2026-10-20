# Step 06 — Automated quality control, targeted retry and review queue

**Order:** after Steps 02–05 (reference-consistent FLUX 3 keyframes, 480×832 Wan2.2 GGUF image-to-video, and the optional audio/caption path), and **before** the Step 07 end-to-end release gate.  
**Objective:** add an automated, **offline-first** quality-control stage that validates each rendered artifact, classifies failures, regenerates **only the failing stage**, keeps every retry bounded and logged, writes a per-job quality report, and routes uncertain or subjective results to a human review queue — reusing the shared manifest contract and the Step 03 cloud-LLM abstraction for the optional vision path.

> **Dependency note:** QC is a **CPU assembly/QC** concern. It loads no GPU model, contacts no network, and invokes no LLM in its **default** path. The optional vision-model analysis path is **OFF by default** and, when authorized, reuses the Step 03 OpenAI-compatible cloud `LLMService` — it does not add a new transport, a local model, or a new credential mechanism.

## 1. Where QC sits in the architecture

QC runs at the **CPU assembly / QC** layer of the three-layer model in [`00-START-HERE.md`](00-START-HERE.md):

```text
Pixelle-Video core + existing pipelines  (unchanged by default)
  └── cat_factory extension (new)
       ├── brief / discovery / scripts / storyboard
       ├── identity / workflow assets / audio decisions / QA   ◀── Step 06 (this spec)
       └── job manifest + resume points
  └── CPU-only external runner / GPU lease controller
       ├── cloud LLM planning (Step 03; remote, holds no local GPU)
       ├── planning_complete checkpoint
       ├── ComfyUI image phase (FLUX 3 loaded, then released)
       ├── ComfyUI video phase (Wan2.2 GGUF I2V 480×832 loaded, then released)
       └── CPU assembly / QC                                    ◀── Step 06 runs here
```

- QC begins **after** the GPU phases have released their models (Step 04 `VIDEO_RELEASE` → `VRAM_VERIFY` → `CPU_ASSEMBLY`). It must be able to run when no GPU is present at all.
- The **default** QC path requires **no GPU and no network**: it uses the local filesystem, `ffprobe`/`ffmpeg` (available on `PATH`), and the persisted manifest only.
- QC **reads** the manifest the earlier steps wrote; a regeneration request is turned into a **targeted, stage-scoped** instruction for the owning step, not a re-plan.

### Suggested (additive) module home

Per `00-START-HERE.md`, QC code lives under the already-proposed `pixelle_video/cat_factory/qc/` (e.g. `ffprobe.py`, `frames.py`, `classify.py`, `report.py`, `vision.py`). This is a **proposed additive path**, documented here only — this step creates no code.

## 2. Manifest integration

QC extends the **single** versioned manifest contract defined in [`00-START-HERE.md`](00-START-HERE.md) at `output/cat_factory/<job_id>/manifest.json` (`schema_version: 1`). It does **not** introduce a second JSON shape. The existing top-level `"qa": {"status": "not_run", "checks": []}` object is filled in place.

### `qa` object shape

```json
{
  "qa": {
    "status": "passed",
    "started_at": "2026-10-15T12:00:00Z",
    "finished_at": "2026-10-15T12:00:07Z",
    "checks": [
      {
        "check_id": "file.playable",
        "stage": "assembly",
        "target": "output/cat_factory/cat-20261015-001/final.mp4",
        "status": "pass",
        "detail": "ffprobe exit 0, valid JSON, container mov/mp4, 8.02s",
        "measured": {"duration_s": 8.02, "size_bytes": 1048576, "width": 480, "height": 832},
        "threshold": {"min_duration_s": 7.0, "max_duration_s": 13.0},
        "tool": "ffprobe"
      }
    ],
    "classification": null,
    "retries": {},
    "vision_analysis": {"status": "skipped", "reason": "not_authorized"},
    "human_review": {"required": false, "decision": null, "decided_at": null}
  }
}
```

### `qa.status` enum

One of: `not_run` | `running` | `passed` | `failed` | `needs_review`.

- `not_run` — the initial value written at planning time.
- `running` — set when QC starts (committed before the first check).
- `passed` — all required checks pass and no check is `needs_review`.
- `failed` — at least one required check failed **and** the bounded retry budget is exhausted.
- `needs_review` — checks are inconclusive, low-confidence, or subjective (e.g. realism, identity continuity), even if structural checks pass.

### Per-check record shape

Every entry in `qa.checks` is exactly:

| Field | Type | Meaning |
|---|---|---|
| `check_id` | str | Stable dotted id, e.g. `duration.total`, `frames.duplicate` |
| `stage` | str | Owning pipeline stage: `planning` \| `keyframe-image` \| `i2v` \| `audio` \| `assembly` |
| `target` | str | Sanitized job-relative path or shot id the check applied to |
| `status` | str | `pass` \| `fail` \| `warn` \| `skipped` \| `error` |
| `detail` | str | Human-readable sanitized explanation (no secrets) |
| `measured` | object | Observed values (durations, sizes, resolutions, counts) |
| `threshold` | object | The configured bound the observation was compared against |
| `tool` | str | `ffprobe` \| `ffmpeg` \| `filesystem` \| `vision` \| `rule` |

### Atomic updates and `schema_version`

- Every manifest write is **atomic** — write to a temp file in the **same directory**, then rename over `manifest.json` (the `00-START-HERE.md` rule). Never partially overwrite a live manifest; a reader must always see a complete document.
- QC **preserves** `schema_version` and all existing top-level fields; it mutates only `qa`, the per-shot `status`, `artifact_paths` (for regenerated artifacts), and appends to `attempts`/`errors`.
- Manifest writes are **idempotent**: re-running QC over the same artifacts produces the same `qa` verdicts without duplicating checks or corrupting prior attempts.

## 3. The checks

Each check writes exactly one per-check record (Section 2). Thresholds come from `cat_factory.qc.*` (Section 10).

### a. Output-file validation

- **Existence & non-zero size** for each artifact named in `artifact_paths` and for the final assembled MP4. A zero-byte or missing file is a `fail`.
- **`ffprobe`-parseable**: run `ffprobe -v error -print_format json -show_format -show_streams <file>`, require **exit code 0** *and* **valid JSON**. A non-zero exit or unparseable output is a `fail` (`tool: ffprobe`).
- **Readable container**: confirm at least one video stream and a recognized container format from the `ffprobe` JSON.

### b. Duration checks

- **Per-shot**: each `shots[].assets` video duration versus the planned `shots[].duration_s`, within `duration_tolerance_s`.
- **Total**: the assembled MP4 duration versus the brief's `duration_s`, within the configured tolerance (a separate total tolerance is allowed).
- Out-of-tolerance duration is a `fail` for the owning `i2v`/`assembly` stage with `measured.duration_s` and `threshold` recorded.

### c. Resolution checks

- Check each per-shot clip and the final MP4 against the requested aspect ratio / resolution (`brief.aspect_ratio`, e.g. 9:16).
- The rendered Wan2.2 I2V output is **480×832**. QC compares the actual artifacts against the **requested** target and records the measured resolution; the **production-resolution 1080×1920 output path is UNVERIFIED (blocker B3)** — QC must **never** claim the 1080×1920 path works. When the requested resolution exceeds the verified render resolution, the resolution check records a `warn` with an explicit detail string naming the unverified production-resolution gap (it does not silently pass).

### d. Frame / visual consistency per shot

- Sample **N frames** per shot (`frame_sample_count`), build a **contact sheet** (an image grid written into the job directory), and detect:
  - **black frames** (mean luma below threshold),
  - **frozen / duplicate frames** (identical or near-identical consecutive samples),
  - **cross-shot identity continuity** (the cat's appearance stays consistent between shots).
- An automatic pass is **NOT sufficient** for subjective realism. Per the Step 07 release gate ([`07-end-to-end-acceptance.md`](07-end-to-end-acceptance.md)), the realism/quality verdict requires a **human visual check and/or an explicitly authorized vision review**. Structural frame checks alone can at most produce `pass`/`warn`; subjective quality is routed to `needs_review` (Section 9) unless an authorized vision review (Section 8) returns a confident verdict.

### e. Audio checks

- Audio-stream presence must be **consistent with `audio_mode`**:
  - `none` → **no** audio stream (a foreign audio stream is a `fail`).
  - `sfx_only` → SFX audio allowed, but **no TTS**; the manifest must show no TTS generation.
  - `caption_only` → captions present, zero audio streams expected.
  - `tts` → narration stream present.
- **Basic clipping check** via `ffmpeg` `astats`/`volumedetect`: flag sustained peak/clip conditions above the configured ceiling.

### f. Rights / provenance record

- Verify the manifest's provenance/licensing fields are present: `inspiration.source_id`/`source_kind`/`rights_notes`, the `cat_character_id`, and an asset license/attribution entry for every reused SFX/media asset.
- A missing provenance or license record is a `fail` (`stage: planning` or `audio`) — an artifact with unrecorded rights must not reach review as "good".

## 4. Failure classification

QC classifies **every** failure into exactly **one** of four classes:

| Class | Meaning | Examples |
|---|---|---|
| `transient` | Temporary/environmental; the same attempt may succeed later | timeout, connection reset, HTTP 429/5xx, ComfyUI queue busy |
| `content` | The artifact is wrong or low quality | black/frozen/duplicate frames, identity drift, duration out of tolerance, audio clipping, wrong resolution |
| `configuration` | The job/config is wrong and retrying changes nothing | missing credentials, bad `base_url`, missing/invalid workflow file, invalid or unknown `audio_mode` value in the manifest, unknown provider |
| `resource` | A bounded local resource is exhausted | GPU/RAM OOM, disk full, GPU lease unavailable |

### Decision table (observed failure → class → permitted action)

| Observed failure | Class | Permitted action |
|---|---|---|
| Timeout / connection reset / 429 / 5xx | `transient` | Backoff retry of the **same stage**, within `max_retries_per_stage` |
| ComfyUI queue busy / model still loading | `transient` | Wait + retry same stage within cap; never discard others' work |
| Black / frozen / duplicate frames | `content` | Regenerate the **`i2v`** stage (or `keyframe-image` if the source keyframe is black) within cap |
| Nowhere-consistent cat identity across shots | `content` | Regenerate **`keyframe-image`** with the fixed `cat_character_id` reference |
| Duration outside tolerance | `content` | Regenerate the owning stage (`i2v` for clips, `assembly` for the total) |
| Resolution below requested (480×832 vs unverified 1080×1920) | `content` | Record `warn`; do **not** silently "fix" — route to review; never claim production resolution works |
| Rendered audio contradicts a **valid, known** `audio_mode` / clipping | `content` | Regenerate the **`audio`** (or `assembly`) stage |
| Missing credentials / bad endpoint / missing workflow | `configuration` | **Stop, fail closed**, actionable error naming the exact key/path; **no retry** |
| GPU/RAM OOM, disk full, lease unavailable | `resource` | Release owned lease, preserve artifacts, pause; bounded retry after the resource recovers; **no cloud fallback** |
| Unclassifiable / repeated failure | `configuration` (fail closed) | Stop the job; do **not** return a partial result as good |

An unclassifiable failure is **fail-closed**: stop rather than guess a class or loop.

## 5. Targeted regeneration (failing stage only)

Regeneration maps a failed check to **only** the stage that produced the failing artifact, reusing the manifest's `artifact_paths` and resume semantics from Steps 03/04:

| Failing artifact / check | Regenerate stage | Inputs reused |
|---|---|---|
| Plan/prompt/schema invalid | `planning` (Step 03) | brief, inspiration, character constraints |
| Keyframe black / identity drift / wrong character | `keyframe-image` (FLUX 3) | shot `image_prompt`, `cat_character_id` reference |
| Clip frames black/frozen/duplicate, clip duration wrong | `i2v` (Wan2.2) | the accepted keyframe + `motion_prompt` |
| Audio missing / contradictory / clipping | `audio` (Step 05) | `audio_mode`, SFX catalog, captions |
| Total duration / concat / container wrong | `assembly` (CPU) | accepted per-shot clips |

Rules:

- **Never re-run a stage that already succeeded.** Completed, accepted artifacts (keyframes, clips) are reused by reference, not regenerated.
- A regeneration writes new `attempts`/`errors` entries and updates `artifact_paths` for the replaced artifact; prior artifacts are preserved (quarantined), never destructively deleted.
- Regeneration is issued as a **stage-scoped instruction** for the owning step; QC itself performs no GPU work.
- If a stage has exhausted its retry budget, the failure is final: set `qa.status` per Section 9 (`failed` or `needs_review`) instead of looping.

## 6. Bounded retries and persistent failure logs

- Per-stage retry caps come from config (`cat_factory.qc.max_retries_per_stage`, optionally per-stage). **No unbounded loops** — a stage that exhausts its cap is final.
- Each retry uses bounded backoff for `transient` failures; `content` retries are limited by the same cap; `configuration` failures are **not retried** at all.
- **Every attempt is appended** to the shared manifest `attempts`/`errors` fields *and* to a **persistent per-job log file** `output/cat_factory/<job_id>/qc.log` (append-only, sanitized — no secrets, no full request bodies). The log survives resume: an interrupted job retains its attempt history.

## 7. Per-job quality report

QC writes a stable, machine-readable report and a human-readable companion into the job output directory:

- `output/cat_factory/<job_id>/quality-report.json` — stable schema: `schema_version`, `job_id`, `generated_at`, `qa_status`, `classification`, `checks[]` (the per-check records), `retries[]` (stage, attempt, class, action, result), `vision_analysis`, and `disposition` (`passed` | `failed` | `needs_review`).
- `output/cat_factory/<job_id>/quality-report.md` — the same content rendered for a human, including measurements, thresholds, per-check verdicts, the classification, the retry log, and the overall disposition.

The report schema is versioned alongside `schema_version` discipline so Step 07 and the review queue can parse it reliably.

## 8. Optional vision-model analysis path

- **OFF by default.** It runs only when both `cat_factory.qc.vision_analysis.enabled` and `cat_factory.qc.vision_analysis.authorized` are `true` **and** the Step 03 profile is configured.
- It **reuses the Step 03 cloud LLM provider abstraction** (stock `LLMService`, config under `cat_factory.llm.*`, credentials from environment variables only) — **not** a new transport, not a local model, and never a second credential mechanism.
- When authorization or credentials are **absent**, it records `status: skipped` (with a reason such as `not_authorized` / `no_credentials`) — **never a silent pass** — and degrades gracefully: the structural checks still run and the job continues.
- It **respects the Step 03 budget guardrails** (`max_requests_per_job`, `max_tokens_per_job`, `cost_ceiling_usd`, `allow_paid_models=false`); a would-be paid call with `allow_paid_models=false` is refused, not silently downgraded.
- It **never uploads media, frames, or secrets** without explicit opt-in. Even when enabled, sending frames for analysis is an explicit, separately acknowledged action; the default is text-only reasoning where possible.
- A confident vision verdict may resolve an otherwise-subjective check; a low-confidence verdict routes the job to the review queue (Section 9).

## 9. Review queue

- Results that are uncertain, **low-confidence**, or subjective are routed to `output/cat_factory/review/<job_id>/`, which receives the final MP4, the QA report (`quality-report.json`/`.md`), the `manifest.json`, and a preview.
- The **human decision is recorded back into the manifest** under `qa.human_review` (`decision`, `decided_at`, optional reviewer note). QC sets `qa.status = needs_review` until a decision lands.
- **No auto-publishing.** The review queue is a human gate; nothing is posted or uploaded. This matches the Step 07 release gate and `06-quality-control.md`'s role as "validation, targeted retry, review queue".

## 10. Config additions (spec only — not implemented here)

Add a `cat_factory.qc` subtree with **safe offline-friendly defaults** (documented for a later step; this step changes no code):

| Key | Type | Default | Purpose |
|---|---|---|---|
| `qc.enabled` | bool | `true` | Run QC after assembly |
| `qc.max_retries_per_stage` | int | `1` | Bounded per-stage retry cap (no unbounded loops) |
| `qc.duration_tolerance_s` | float | `0.5` | Allowed per-shot/total duration delta |
| `qc.resolution_tolerance_px` | int | `0` | Allowed resolution delta (0 = exact) |
| `qc.frame_sample_count` | int | `8` | Frames sampled per shot for the contact sheet |
| `qc.vision_analysis.enabled` | bool | `false` | Enable the optional vision path |
| `qc.vision_analysis.authorized` | bool | `false` | Explicit operator authorization for vision review |
| `qc.review_queue_dir` | str | `output/cat_factory/review` | Review-queue root |

These extend `CatFactoryConfig` (Step 00/03) like the existing `cat_factory.llm.*` / `cat_factory.guardrails.*` blocks; nothing enables a paid or network path by default.

## 11. Offline-testable plan with fixture media

All default QC tests run **offline** — no network, no GPU, no provider. Fixtures are generated locally with `ffmpeg`/`ffprobe` (available on `PATH`) or checked in under `tests/cat_factory/fixtures/`.

### Fixtures

| Fixture | Produced by | Purpose |
|---|---|---|
| tiny valid clip (e.g. 2s, 480×832) | `ffmpeg -f lavfi -i testsrc=...` | Happy-path duration/resolution pass |
| zero-byte file | `fs` write of 0 bytes | `file.exists`/non-zero-size `fail` |
| corrupt file | truncated/invalid bytes | `ffprobe` non-zero exit / unparseable JSON `fail` |
| too-short clip | e.g. 0.5s clip | Duration tolerance `fail` |
| black-frame clip | `ffmpeg -f lavfi -i color=black` | `frames.black` detection |
| duplicate/frozen clip | repeated identical frames | `frames.duplicate` detection |
| silent clip (`audio_mode=none`) | clip with no audio stream | Audio-presence pass |
| foreign-audio clip | audio added to a `none` job | Audio contradiction `fail` |

### Unit-test cases

1. Valid clip → all file/duration/resolution checks `pass`.
2. Zero-byte / missing artifact → `file.*` `fail` with class `content`.
3. Corrupt file → `ffprobe` `fail` (`tool: ffprobe`), no crash.
4. Too-short clip → `duration` `fail` carrying `measured`/`threshold`.
5. Black-frame clip → `frames.black` `fail` → targeted `i2v` regeneration request.
6. Duplicate frames → `frames.duplicate` `fail` → targeted `i2v` regeneration request.
7. Audio contradiction (`none` job with a stream) → `audio` `fail`.
8. Clipping fixture → `audio.clipping` `warn`/`fail`.
9. Retry cap: a stage exceeding `max_retries_per_stage` terminates as final (no loop).
10. Regeneration reuses a **succeeded** stage and never re-runs it.
11. Manifest atomicity: an interrupted write leaves the previous complete manifest intact; `schema_version` unchanged.
12. Report written to the job directory with the stable schema and a `.md` companion.
13. `qa.status` transitions `not_run → running → passed|failed|needs_review` and persists across resume.

### Optional live / vision test

- A live vision test is **skipped when unconfigured** (`vision_analysis.enabled`/`authorized` false or no credentials): it records `status: skipped` and the suite still passes. When configured, it makes a minimal number of calls, respects the Step 03 guardrails, and never prints secrets.

## 12. Acceptance criteria

- **AC-06.1** QC runs fully **offline** (no GPU, no network, no LLM) against the shared `manifest.json` and fills the existing `qa` object without editing unrelated manifest fields.
- **AC-06.2** `ffprobe`-based output validation flags a missing, zero-byte, and corrupt file as `fail`, and passes a valid playable MP4.
- **AC-06.3** Duration (per-shot and total) and resolution checks compare against configured tolerances and record `measured`/`threshold`; the 480×832-vs-unverified-1080×1920 gap is surfaced, never claimed as working.
- **AC-06.4** Per-shot frame/visual checks produce a contact sheet and detect black/frozen/duplicate frames; subjective realism requires human and/or authorized vision review and is never auto-passed.
- **AC-06.5** Audio checks enforce stream presence consistent with `audio_mode` and run a basic clipping check.
- **AC-06.6** Every failure is classified into exactly one of `transient` / `content` / `configuration` / `resource`, and the action taken matches the decision table (configuration failures fail closed with no retry).
- **AC-06.7** A failure triggers regeneration of **only** the failing stage; a succeeded stage is never re-run, and per-stage retries respect the configured cap with no unbounded loop.
- **AC-06.8** Every attempt is appended to the manifest `attempts`/`errors` and to a persistent per-job `qc.log`; the manifest is updated atomically with `schema_version` preserved.
- **AC-06.9** A per-job `quality-report.json` (stable schema) plus `quality-report.md` is written into `output/cat_factory/<job_id>/`.
- **AC-06.10** The optional vision path is OFF by default, reuses the Step 03 cloud-LLM abstraction, records `status: skipped` (never a silent pass) when unauthorized/unconfigured, and respects the Step 03 budget guardrails.
- **AC-06.11** Uncertain/subjective results route to `output/cat_factory/review/<job_id>/` with manifest + QA report + preview; the human decision is recorded back into the manifest and nothing auto-publishes.
- **AC-06.12** The offline fixture tests (Section 11) pass with no network and no GPU, and the optional live/vision test is skipped-when-unconfigured.

## 13. Definition of done and handoff

- Commit only this step's docs/tests and the QC module under `pixelle_video/cat_factory/qc/` plus the `cat_factory.qc` config subtree — no edits to the stock Pixelle pipeline, `LLMService`, GPU controller, or Streamlit UI.
- Record QC config defaults, test evidence, and each AC-06.x disposition in `docs/cat_factory/IMPLEMENTATION_STATUS.md`, with `status: accepted | blocked`.
- Report the exact fixture commands/models used, the measured durations/resolutions, and any blocker (e.g. the unverified 1080×1920 path B3), without secrets.
- Next: **Step 07 — [`07-end-to-end-acceptance.md`](07-end-to-end-acceptance.md)** wires QC into the end-to-end flow (`QC: ffprobe structural pass, frame contact sheet, manual or capability-verified visual check, rights/source record`) and the review handoff. Do not implement publishing, GPU lifecycle, or planning here.

## 14. OpenCode task prompt

> Implement only `06-quality-control.md` in the existing cat extension, after `00-START-HERE.md` and Steps 03–05. Reuse the single shared manifest contract in `output/cat_factory/<job_id>/manifest.json` (`schema_version: 1`) and fill the existing `qa` object atomically (write-temp-then-rename) — do not invent a second JSON shape. Build the QC checks with `ffprobe`/`ffmpeg` only (default path must need no GPU and no network): file validation, duration/resolution tolerances, per-shot contact sheet with black/frozen/duplicate detection, audio-vs-`audio_mode` consistency and clipping, and a rights/provenance record. Classify every failure into exactly one of `transient`/`content`/`configuration`/`resource`; regenerate only the failing stage and never re-run a succeeded one; keep per-stage retries bounded with persistent per-job logs. Write `quality-report.json` + `quality-report.md` into the job directory, route uncertain/subjective results to `output/cat_factory/review/<job_id>/` (no auto-publish), and add the optional vision path (OFF by default; reuse the Step 03 `LLMService`; `status: skipped` when unauthorized; respect the Step 03 budget guardrails). Prove it with the offline ffmpeg fixture tests (no network, no GPU) and the skipped-when-unconfigured vision test. Report changed files, config defaults, test evidence and precise blockers; commit the isolated change once passing.
