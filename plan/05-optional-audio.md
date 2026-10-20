# Step 05 — Optional TTS, silent comedy, captions and sound effects

**Order:** after Steps 02–04 (the local ComfyUI render path: FLUX 3 keyframes, then Wan2.2 GGUF image-to-video at 480×832; the production-resolution 1080×1920 output path is unverified).  
**Objective:** video production must work when **no narration/TTS is requested**. Most cat memes can be silent or use a caption and one funny sound effect.

## Stock code limitation to address

The stock [`FrameProcessor`](https://github.com/ATH-MaaS/Pixelle-Video/blob/main/pixelle_video/services/frame_processor.py) currently invokes TTS at the beginning, sets `frame.duration` from narration audio, feeds that duration to video generation, then calls rendering helpers requiring an audio path. [`VideoService`](https://github.com/ATH-MaaS/Pixelle-Video/blob/main/pixelle_video/services/video.py) includes `create_video_from_image(image, audio, ...)` and `merge_audio_video(video, audio, ...)` without a demonstrated `audio=None` branch. **Skipping TTS without adapting duration/rendering will break or result in invalid timing.**

## Audio modes

Explicit enum per *video*, with per-shot override only when needed:

| Mode | Speech | SFX | Captions | Media duration source |
|---|---|---|---|---|
| `none` | No | No | Optional false (default false) | `shot.duration_s` / actual video duration |
| `caption_only` | No | No | Yes | `shot.duration_s` / actual video duration |
| `sfx_only` | No | Yes | Optional (usually yes) | `shot.duration_s` / actual video duration |
| `tts` | Yes | Optional | Optional (usually yes) | Explicit shot duration; allow narration duration as a constraint, but do not depend entirely on it |

Separate `narration_text`, `caption_text` and `motion_prompt`. The user's text prompt is not automatically spoken. `auto` can be a **planning** choice made by the configured cloud LLM, but the persisted execution value must be one of these four modes.

## Implementation tasks

1. Add `AudioPlan` to cat models: `mode`, `caption_text`, `narration_text`, `sfx_cues` with `time_s`, `asset_id`, `gain_db`, `background_music` (optional license-defined path), and whether to preserve input video audio. Validate no TTS jobs if mode != `tts`.
2. Extend cat storyboard/shot context with **explicit duration** (`duration_s > 0`, bounded), independent of narration. Use source video duration after generation only when policy says preserve/follow source clip. Preserve preexisting fields/defaults for stock callers.
3. Implement dedicated frame/segment rendering path for cat mode or backwards-compatible optional-audio handling in `FrameProcessor` and `VideoService`. For real video clip + no audio: create/render overlaid MP4 **without invoking `merge_audio_video()` with missing audio**. For static image + no audio: use FFmpeg with `-loop 1`, `-t <duration>`, configured FPS/pixel format and no audio input. For input video with native audio: preserve/drop/mix deliberately; don't assume every video has sound.
4. Keep caption overlay independent of speech generation. Verify template `video_*.html` transparency, safe text area, font availability and vertical aspect ratio. Avoid subtitles covering the cat's face/paws. Test multi-line captions and caption-free mode.
5. Sound effects come only from **licensed local assets** (user-supplied, royalty-free with permitted license). Maintain `assets/cat_factory/sfx/catalog.json` with license, attribution, filename and tags. No automatic ripping of copyrighted movie sounds, songs or platform videos.
6. Implement SFX timing/mixing with FFmpeg. Normalize/limit peaks; allow fade and volume adjustments. If no audio is requested, final MP4 can legitimately have **zero audio streams**. Verify silent-file compatibility across target platforms with ffprobe. Don't generate fake blank TTS or unnecessary silence audio as a workaround unless a specific target codec requires it and it is documented.
7. For `tts`, prefer stock TTS service; configure existing Edge-TTS/ComfyUI pathways and ensure speech audio can fit the shot schedule. If narration overruns selected duration, fail or request approved shot length adjustment—do not silently cut dialogue.
8. Add API/CLI/UI audio selection with `auto` (planner picks a stored concrete mode) and manual override. Default to `sfx_only` or `caption_only` for a meme preset after user confirmation; maintain upstream default in non-cat mode.
9. Write metadata fields `audio_mode`, `effective_duration_s`, `audio_assets`, `captions_added`, `has_audio_stream`, and FFmpeg command summary (redacted safe args) into manifest.

## Functional tests (CPU-only using generated color bars / test images)

1. `none`: 5s static cat test image → playable MP4, expected duration and **no audio**, no calls to TTS.
2. `caption_only`: 7s video → visible caption, zero audio streams, no TTS call.
3. `sfx_only`: one SFX at 2s → audible SFX at expected time, no spoken narration.
4. `tts`: mocked 3s narration + 5s video → expected narration mapping; original Pixelle TTS test still passes.
5. native-audio video: preserve if asked, discard when mode `none`, mix only if opted in.
6. ffprobe rejects wrong duration, blank/corrupt output and missing audio in `sfx_only`/`tts`.
7. clip concat: mix 2 silent segments and 1 sound-enabled segment to one valid finished video without unexpected audio drop. 9:16 output.

## Acceptance criteria

- **AC-05.1** A cat video is rendered with no TTS API/model loaded or called.
- **AC-05.2** Each mode yields the exact expected subtitle/audio stream behavior and duration.
- **AC-05.3** Adding TTS remains optional and stock legacy TTS pipeline remains functional.
- **AC-05.4** Licensed source and attribution for every external effect are tracked.
- **AC-05.5** End-to-end silent video (animated real clip, not just static image) works with stable timing.

## OpenCode task prompt

> Implement `05-optional-audio.md` in the existing cat extension. First inspect live `FrameProcessor`, `VideoService`, template and `StoryboardConfig` contracts. Add explicit shot durations and four audio modes without changing stock defaults. Create CPU-only FFmpeg fixture tests for zero-audio MP4, captions, SFX and TTS; then test the cat rendering path with existing generated footage if available. Never substitute copyrighted movie audio. Report changes/tests and stop.
