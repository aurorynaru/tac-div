# Step 07 — Wire the modules, prove one genuine cat-meme MP4, and hand over

**Order:** final integration after all six feature tasks.  
**Objective:** a reproducible local *one-shot* finished video, then a reliable fully automatic one-job workflow—without breaking stock Pixelle-Video.

## Preconditions / gates

- `IMPLEMENTATION_STATUS.md` records the current Git revision, changed files, baseline tests, and disposition of each Step 01–06 acceptance criterion.
- User approves one specific cat pack and a local ComfyUI workflow that has actually produced a test frame/clip on the 3090 (or integration is declared blocked, not complete).
- GPU runner validated in dry-run and authorized for controlled live transitions; the planning LLM (Step 03) runs remotely in the cloud and holds no local GPU, so no shared active OpenCode or third-party requests are touched by local ComfyUI model load/release.
- FFmpeg and ffprobe installed; tested output path is writable; API/website data retrieval is opt-in and sources have permission/terms checks.
- Any model license, software install and ComfyUI custom-node requirements have been reviewed by user prior to download/installation.

## MVP integration scenario

Brief:

```yaml
creative_brief:
  topics: ["kung-fu cat", "protecting a food bowl"]
  genres: ["action parody", "deadpan meme"]
  tone: "photoreal, casual phone-camera look, absurd"
  avoid: ["real animal harm", "movie dialogue", "copyrighted soundtracks"]
  target_duration_s: 8
  aspect_ratio: "9:16"
  max_candidates: 5
character_id: "tabby-001"  # replace with an approved real ID
prefer_audio_mode: "sfx_only"
```

**Required final output:** cat-looking cat; 1–2 simple shots in an ordinary room; an original kick-like defensive gesture or reaction without anatomically grotesque movement; caption and licensed SFX; no voice narration; playable vertical MP4. If action quality fails, downgrade choreography *explicitly and report it*, rather than claiming motion transfer succeeded.

## End-to-end test procedure

1. **Offline:** `discover --offline` -> candidate list, original idea + trace.
2. **Offline mocked LLM:** validate scene JSON + character reference + concrete `audio_mode` and total shot durations.
3. **Dry run:** inspect intended LLM/ComfyUI transitions and output paths without model unloads or GPU load.
4. **Live planning (authorized):** the configured cloud LLM finishes and writes an atomic `planning_complete` checkpoint. All in-flight LLM calls are closed before the checkpoint is emitted.
5. **Live render (authorized):** GPU lease acquired; ComfyUI renders the FLUX 3 keyframe then the Wan2.2 GGUF image-to-video workflow at 480×832 for one test clip (the production-resolution 1080×1920 path remains unverified). Verify recorded VRAM and duration; render second only if first passes.
6. **Postprocess (CPU):** captions/SFX according to selected audio mode; FFmpeg joins segments and outputs MP4.
7. **QC:** ffprobe structural pass, frame contact sheet, manual or capability-verified visual check, rights/source record. Bounded retry by failing stage only.
8. **Review:** write `output/cat_factory/review/<job_id>/` with final MP4, QA report, manifest, preview. **Do not auto-publish.**
9. **Resume test:** after first clip is complete, simulate an interruption before next clip; resume and confirm first clip hash unchanged and expensive steps skipped.
10. **Regression:** run stock Pixelle default generation tests/quick smoke check with cat mode disabled.
11. **Security:** verify API keys not present in JSON, commits, console logs or UI settings exports; validate user-supplied paths and source URL handling.
12. **Handoff:** produce `docs/cat_factory/OPERATIONS.md` (service start, Windows process ownership, startup and shutdown, ComfyUI models and load order, cost/time budget, troubleshooting), and `RESULTS.md` with real measurements, screenshots/contact sheets and limitations.

## Test matrix

| Test | Expected result |
|---|---|
| 100% offline | plan, render mock, validate, resume without GPU or Internet |
| Source provider disabled | fixture/classic-trope fallback, clearly labeled |
| Cloud LLM credential missing/invalid or provider unreachable | planning blocked before render; saved jobs preserved |
| Cloud LLM rate-limited/overloaded | bounded retry, then planning blocked; no impact on other sessions; saved jobs preserved |
| ComfyUI unavailable | pause after saved plan; can resume |
| Wan memory out | explicit blocked state and measured cause, no cloud fallback |
| `audio_mode=none` | playable MP4 with no audio stream |
| `audio_mode=sfx_only` | one or more timed licensed effects; no TTS generation |
| Quality failure | at most configured targeted retries; otherwise review queue |
| Second shot resumed | first clip preserved, no redundant GPU work |
| Normal stock Pixelle usage | unchanged with feature flag off |

## User-facing minimal controls

After headless tests are accepted, expose in Pixelle's Streamlit UI:

- a **Cat Parody** opt-in mode;
- topic, genre, mood, duration, output aspect, cat identity;
- discovery providers and permissions, plus "automatic idea pick" or "preview concepts";
- audio mode: auto / none / caption_only / sfx_only / tts;
- dry-run, generate, status, retry failed shot, review video;
- plain-language GPU ownership warning and exact services affected.

Keep the stock app navigation and video pipelines intact. Do not expose the LLM API key or unmanaged process kill controls to UI.

## Release gate (all must pass)

1. One authentic locally generated frame and short MP4 from the selected workflow (not a mock), clearly identified by model/workflow version.
2. At least one end-to-end video with user-expected cat realism after a **human visual check**; automatic pass alone is not sufficient for subjective quality.
3. Safe planning → ComfyUI render handoff via the persisted `planning_complete` checkpoint (no local LLM lifecycle) or an explicitly documented user-controlled fallback; no interrupted OpenCode session.
4. Fully persisted manifests, source links, approved character IDs, audio mode, QA and restartable states.
5. CPU offline tests, source checkout regression tests, license/security review.
6. No unapproved publishing, scraping, paid APIs or destructive process operations.

If any gate fails, status is **BLOCKED** or **ACCEPTED WITH LIMITS**, never falsely `DONE`.

## Final OpenCode integration prompt

> Read `00-START-HERE.md`, `07-end-to-end-acceptance.md`, and all six completed step reports. Audit existing implementation against the shared manifest/schema; repair only cross-module integration issues. Run offline tests, dry-run and regression checks first. Ask authorization before any live GPU switching, model loading or source API calls. Prove one genuine locally generated, visually reviewed cat-parody MP4, save measurements and the operations guide, and produce a clear PASS/BLOCKED acceptance report with exact commands and file paths. Do not publish content.
