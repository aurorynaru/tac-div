# Pixelle-Video → Autonomous Realistic Cat-Parody Factory

**Status:** implementation specification, not implemented code.  
**Target:** stock [`ATH-MaaS/Pixelle-Video`](https://github.com/ATH-MaaS/Pixelle-Video) on Windows with one NVIDIA RTX 3090 (24 GB); a configurable OpenAI-compatible **cloud** LLM for planning (Step 03, [`03-llm-integration.md`](03-llm-integration.md)) and local ComfyUI for rendering (Step 04: FLUX 3 images, then Wan2.2 GGUF image-to-video at 480×832; the production-resolution 1080×1920 output path is unverified). OpenCode is the *development agent*, **not** the runtime supervisor.

## Product definition

Input: a lightweight creative brief (topics, genres, mood, disallowed subjects, optional recurring cat), e.g. `genres=[action, slice_of_life], topics=[kung fu, lazy cats], duration=8–12s, format=9:16`.

Output: a locally generated, **original** short cat parody in a realistic, casually filmed meme style (ordinary settings, plausible tabby fur/anatomy, absurd human-like behavior). Automatic workflow: source research → parody pitches → storyboard → reference-consistent cat frames → animated clips → optional audio/captions → final video → quality report → **human review before publishing**.

The two supplied screenshot examples are **style references only**: tabby-and-white domestic cats, believable fur/light and environments, unusual human-like poses. Do not try to reproduce a specific image, creator watermark, movie clip, soundtrack, or brand packaging verbatim. Allow original comedy inspired by broadly recognizable situations.

## Important findings from stock upstream (checked October 15, 2026)

The following are *inspection starting points*, not a guarantee they match the user's checked-out revision:

| Existing stock functionality | Confirmed area | Implication |
|---|---|---|
| OpenAI SDK compatible LLM and Pydantic-style JSON prompting | `pixelle_video/services/llm_service.py` | Configure the cloud LLM provider profile first; reuse the stock LLM adapter |
| Template-method pipeline | `pixelle_video/pipelines/linear.py` and `standard.py` | Add a dedicated cat-parody entrypoint or subclass; preserve standard generation |
| Storyboard dataclasses | `pixelle_video/models/storyboard.py` | Add backwards-compatible optional fields or separate cat-parody data models |
| TTS executed before video, sets per-frame duration | `pixelle_video/services/frame_processor.py` | Silent video needs explicit duration and no-audio rendering path |
| Video compositor requires audio in common paths | `pixelle_video/services/video.py` | Add silent-image/video helpers or safe optional audio support |
| ComfyUI media service and custom workflow JSONs | `pixelle_video/services/media.py`, `workflows/selfhost/` | Wan2.2 is **not assumed to be preconfigured or 3090-compatible**; validate separately |
| Task output and metadata | `pixelle_video/utils/os_util.py`, `pixelle_video/services/persistence.py` | Reuse task IDs, filenames, persistence rather than parallel ad hoc storage |
| Streamlit UI | `web/app.py` and modules imported from it | Make UI additions last, after headless paths work |

Upstream documents: [repo](https://github.com/ATH-MaaS/Pixelle-Video), [configuration](https://github.com/ATH-MaaS/Pixelle-Video/blob/main/config.example.yaml), [workflows](https://ath-maas.github.io/Pixelle-Video/user-guide/workflows/), [ComfyUI server API](https://github.com/Comfy-Org/ComfyUI/blob/master/server.py).

## Files and execution order

The numbers mirror the six requested features; the **recommended execution order** respects dependencies.

1. **`03-llm-integration.md`** — configure the OpenAI-compatible cloud LLM profile and reliable JSON contract.
2. **`04-gpu-orchestration.md`** — build a CPU-side state machine controlling GPU ownership; NEVER kill unrelated processes.
3. **`01-automatic-meme-discovery.md`** — lawful source discovery and original cat pitches from brief.
4. **`02-cat-character-consistency.md`** — recurring realistic character, custom ComfyUI keyframes and image-to-video.
5. **`05-optional-audio.md`** — captions, SFX and silence without TTS dependency.
6. **`06-quality-control.md`** — validation, targeted retry, review queue.
7. **`07-end-to-end-acceptance.md`** — wire all pieces, failure recovery, smoke test.

Do **not** attempt all documents in one model context. Assign one document per OpenCode coding task, validate, commit, and carry a compact handoff forward.

## Initial prerequisite: repo baseline audit (before Step 03)

Within the user's Pixelle-Video checkout, ask OpenCode to:

1. Record `git rev-parse HEAD`, `git status --short`, Python/uv/FFmpeg, GPU driver, and discovered ComfyUI location, **without exposing keys**. Preserve user changes. Never overwrite existing `config.yaml`.
2. Compare paths above with current checkout. Create `docs/cat_factory/IMPLEMENTATION_STATUS.md` listing confirmed modules, differences, dependency versions, and blockers. Do not invent file paths.
3. Run the stock project’s documented tests or minimal import/CLI smoke checks *without* launching a GPU-intensive model. Record baseline results; do not claim tests pass if missing.
4. Identify how the current UI invokes the pipeline, and where custom workflow parameters go through `MediaService`/ComfyKit. Verify a selected 9:16 video template, or add one only if existing templates are unsuitable.
5. Add a feature flag `cat_factory.enabled: false` (or a separate opt-in config object that fits the live config schema). Do not change the default pipeline, model, keys, or cloud services.
6. Confirm whether the user is using a source checkout (`uv run streamlit run web/app.py`) rather than the bundled Windows release.

**Baseline gate:** all existing stock functionality remains unchanged when cat mode is off.

## Architectural contract (shared across all tasks)

Keep three independent layers:

```text
Pixelle-Video core + existing pipelines  (unchanged by default)
  └── cat_factory extension (new)
       ├── brief / discovery / scripts / storyboard
       ├── identity / workflow assets / audio decisions / QA
       └── job manifest + resume points
  └── CPU-only external runner / GPU lease controller
       ├── cloud LLM planning (Step 03; remote, holds no local GPU)
       ├── planning_complete checkpoint
       ├── ComfyUI image phase (FLUX 3 loaded, then released)
       ├── ComfyUI video phase (Wan2.2 GGUF I2V 480×832 loaded, then released)
       └── CPU assembly / QC
```

Suggested additive paths (adapt only after auditing actual checkout):

```text
pixelle_video/cat_factory/
  models.py                  # Pydantic contract, serialization
  discovery/                 # providers, normalization, ranking
  characters/                # identity packs and reference validation
  adapters/                  # cloud LLM, ComfyUI abstractions
  gpu/                       # Windows-safe external controller
  audio/                     # audio mode and sound postprocessing
  qc/                        # ffprobe, frames, optional VLM checks
  pipeline.py                # cat-mode orchestration
  storage.py                 # per-run durable files
scripts/cat_factory.py      # CPU-side CLI: plan / render / resume / status
config.cat_factory.example.yaml
workflows/selfhost/         # exported real API-format workflows only
tests/cat_factory/          # offline fixtures and unit tests
docs/cat_factory/IMPLEMENTATION_STATUS.md
```

### One versioned manifest per job

Store as `output/cat_factory/<job_id>/manifest.json` (do not commit outputs). **Recommended top-level fields**:

```json
{
  "schema_version": 1,
  "job_id": "cat-20261015-001",
  "state": "planned",
  "brief": {"topics": ["kung fu"], "genres": ["comedy"], "duration_s": 8, "aspect_ratio": "9:16"},
  "inspiration": {"source_id": "example-001", "source_url": null, "source_kind": "classic_trope", "fetched_at": null, "rights_notes": "original parody only"},
  "selected_idea": {"title": "Kung-Fu Cat", "original_premise": "A cat protects a food bowl", "scores": {}},
  "cat_character_id": "tabby-001",
  "audio_mode": "sfx_only",
  "shots": [
    {"shot_id": "s01", "duration_s": 4.0, "caption": "WHO TOUCHED MY BOWL?", "image_prompt": "...", "motion_prompt": "...", "assets": {}, "status": "planned"}
  ],
  "artifact_paths": {},
  "qa": {"status": "not_run", "checks": []},
  "attempts": {},
  "errors": []
}
```

This is a **design contract**, not the current stock Pixelle schema. Use Pydantic validation, atomic file replacement, a version field, sanitized absolute/relative paths and an upgrade strategy. Store all output in per-job directories. Do not treat `output/cat_factory` as a code module.

### Nonnegotiable engineering rules

- One GPU-intensive model family at a time. The RTX 3090 has 24 GB; do not assume Wan2.2 fits until the actual quantized/offload workflow is tested and measured.
- The renderer must continue after the planning LLM call completes. The planning LLM runs remotely (cloud, Step 03) and holds no local GPU; the **CPU-only runner** must not depend on OpenCode’s active inference session to advance state.
- Never release a model with active requests. Drain the local ComfyUI queue before unloading models; a `POST /free` response only flags the request, so verify released VRAM by measurement. Never kill an unrelated ComfyUI instance used by another task.
- No scraping behind logins, paywalls, anti-bot challenges, or site restrictions; no auto-download of copyrighted clips/audio. Respect source terms/rate limits.
- No auto-posting in MVP; outputs land in a human-review folder. Include licenses/attribution of any reused SFX/assets.
- Disable remote media providers by default in the cat factory. Keep all network sources and paid APIs explicit opt-ins with budget limits.
- Retry narrowly, boundedly, and with persistent failure logs; never loop indefinitely to obtain a 'viral' result.
- Tests with mocks/offline fixtures first; live network and GPU tests require explicit operator opt-in. Keep secrets in env/private config; sanitize logs.
- If an upstream module, service API, or workflow differs from this plan, **verify, document, and adapt** rather than blindly applying pseudo-code.

## Definition of complete

A user can supply only topics/genres and obtain a **local MP4 plus the source provenance, cat identity reference, audio mode, and QC report**, repeatably. A deliberately failed GPU stage can resume without repeating research or losing completed clips. Original stock video generation still works with cat mode disabled.

## Copy/paste OpenCode kickoff prompt

> You are extending my existing Pixelle-Video source checkout with an opt-in realistic cat-parody content factory. Read `00-START-HERE.md` and execute the **baseline prerequisite only**. The runtime target is a configurable OpenAI-compatible cloud LLM (Step 03, `03-llm-integration.md`) plus local ComfyUI (Step 04); there is no local LLM process. Inspect the actual code before edits. Preserve all user files/config/secrets and stock behavior. Record repo revision and audit in `docs/cat_factory/IMPLEMENTATION_STATUS.md`; run safe tests; report exact changed paths, commands, results and blockers. Do not start GPU models or implement later steps. Stop after this task and await my next instruction.
