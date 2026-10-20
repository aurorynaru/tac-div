# Step 02 — Realistic recurring cats + reference-guided video

**Order:** after Step 01 (selected idea), Step 03 (validated plan; see [`03-llm-integration.md`](03-llm-integration.md)) and Step 04 (GPU supervisor).  
**Objective:** make the video look like the user's screenshots: **real tabby-and-white domestic cats**, believable fur, domestic proportions, natural daylight/interior lighting and casually filmed settings. Cats may sit like humans, grip a phone or make an exaggerated kick, but avoid a Pixar/cartoon look.

## Critical feasibility gate

Stock Pixelle documents image/video/ComfyUI workflows, and has a motion-transfer demo, but **does not establish that stock configuration includes a ready-to-run Wan2.2 reference-guided workflow that fits an RTX 3090**. This extension targets the FLUX 3 image workflows (`workflows/selfhost/image_flux3.json`, or `image_flux3_ref.json` when a character reference is required) and the Wan2.2 I2V A14B HighNoise+LowNoise GGUF workflow at 480×832 (`workflows/selfhost/video_wan2_2_i2v.json`); the production-resolution 1080×1920 output path is unverified and must not be claimed to work. Prioritize a measured local 1-shot prototype. Video generation, first-frame guidance, subject reference, motion transfer and identity consistency are **different capabilities**; verify which the installed model and ComfyUI nodes truly support.

Don't promise stable character identity from a text prompt/seed alone. For realism, preserve **reference images and machine-readable traits** and test look consistency. A motion-transfer reference clip must be owned/licensed for use; human motion may not map cleanly to four-legged cats.

## Tasks

1. Create `CharacterIdentity`/`CharacterPack` data model:
   - immutable `character_id`, `version`, `reference_images` (paths + hashes + views), descriptive traits (`tabby-and-white`, fur markings, eye color, nose, ears, belly, body type), `negative_constraints`, `style_notes`, optional seed and approved generation settings.
   - references reside under e.g. `assets/cat_factory/characters/<id>/`, **separate** from transient job output; only files provided/approved for reuse.
   - save image provenance, creation workflow, model checkpoint/license, seed and output dimensions. Store only derived metadata in JSON, binary assets as files.
2. Build a user-facing **reference gallery** and approval step: upload/select cat photo(s) or generate candidates locally, approve a mascot. For initial screenshot look, target a brown/gray tabby + white coat, chubby body, natural paws and expressions. User screenshots define the *style*, not compulsory exact fur patterns or faces.
3. Establish one **baseline image generation** path through existing `MediaService` and the grounded FLUX 3 workflow (`workflows/selfhost/image_flux3.json`, or `image_flux3_ref.json` when a character reference is required), adapting to the local ComfyUI API JSON. Record required checkpoints, custom nodes, sampler, resolution, tensor precision/offload, GPU memory and run time, and measure fit before claiming it.
4. First prove **one usable image** (PNG/JPEG) consistent with reference pack, no deformities. Then prove a **single 2–4s image-to-video** clip with the same cat at the 480×832 portrait resolution (`workflows/selfhost/video_wan2_2_i2v.json`, Wan2.2 I2V A14B HighNoise+LowNoise GGUF). If Wan2.2 cannot fit, mark a blocker with measured VRAM and try a documented quantized/offload workflow or smaller model **only by explicit configuration**, not an unauthorized cloud fallback.
5. Create a workflow adapter `generate_keyframe(shot, character_pack)` and `animate_shot(shot, keyframe, optional_motion_reference)`. Validate ability to pass image inputs to chosen ComfyUI workflow; use its real node IDs/input contract, not fabricated parameter names. Keep the workflow JSON export immutable/versioned (`workflows/selfhost/image_flux3.json`, `image_flux3_ref.json`, `video_wan2_2_i2v.json`).
6. Create prompt builder fields separately: `identity_anchor`, `pose/action`, `location`, `props`, `camera`, `lighting`, `shot_duration_s`, `motion_instruction`, `negative_constraints`. Reuse the identity anchor across shots, vary only scene specifics.
7. Ensure character pack reference data survives job restart; hash image assets and verify presence/integrity before each render. Bounded generation retries must reuse character ID, not silently choose another mascot.
8. Choose storyboard format 1–4 shots and explicit duration; use original settings (kitchen, street, living room). Plan actions appropriate to video capabilities (e.g. seated cat scowls at phone; one dramatic raised-paw pose) before extremely difficult full-body kick choreography.
9. Add scene-level visual checks from Step 06 in a minimal form (basic manual review first): face/stripe identity, fur realism, paws/tail counts, no sudden costume changes, prop continuity, stable background. Store per-shot `reference_match: pass | fail | unverified` without inventing objective confidence if no model exists.
10. Integrate with cat pipeline as an opt-in generator, **not** a rewrite of stock `StandardPipeline`. Consider a `CatParodyPipeline(LinearVideoPipeline)` or additive service if that fits checkout. Preserve `StandardPipeline` behavior.
11. Once one clip works, attempt two separate shots with the same identity. Preserve `shots[].assets.keyframe_path` / `.video_path` in the common job manifest for resume.

## Style spec to attach to all generation prompts

```text
Natural domestic tabby-and-white cat; near-photoreal fur, body, paws, whiskers and
lighting; an everyday real-world environment; absurd human-like expression or
posture but still unmistakably a living cat; casual candid smartphone-video
framing, unpolished comedy, consistent coat markings and chubby silhouette.
Avoid glossy 3D animation, Disney/Pixar appearance, plastic fur, extra limbs,
mutated paws, flickering coat patterns, floating props, text baked into image,
overproduced fantasy visuals, copied logos, and recognizable copyrighted frames.
```

### Example shot prompt fields

```json
{
  "shot_id": "s01",
  "character_id": "tabby-001",
  "duration_s": 4.0,
  "scene": "Quiet living room, low camera angle",
  "action": "Chubby tabby guards a nearly empty food bowl, suspicious stare",
  "camera": "Locked smartphone video, medium-wide",
  "identity_reference": "assets/cat_factory/characters/tabby-001/front.png",
  "caption": "WHO ATE MY DINNER?",
  "audio_mode": "sfx_only"
}
```

This is a **conceptual shot contract**; map to real workflow nodes and the shared `manifest.json` design.

## GPU / motion acceptance

- **AC-02.1:** Repeatable approved cat identity pack stores reference files/traits/hashes and is available across runs.
- **AC-02.2:** One realistic keyframe generated locally from the real workflow; exact settings recorded.
- **AC-02.3:** One measured local 2–4s animation at 480×832 succeeds OR is explicitly **blocked** with OOM/profiler evidence and recommended smaller workflow; the production-resolution 1080×1920 path stays unverified. Do not call it success based on theoretical compatibility.
- **AC-02.4:** One two-shot test retains the same primary cat according to documented visual review; no guaranteed identity threshold without calibrated vision evaluation.
- **AC-02.5:** Completed frames/clips survive a restart and are not rerendered without reason.
- **AC-02.6:** Stock Pixelle generation still works with feature flag off.

## OpenCode task prompt

> Implement `02-cat-character-consistency.md` in Pixelle-Video after reading shared plan/status. First create a reusable realistic cat identity pack and reference-guided image workflow, with offline tests. Then propose one *explicitly authorized* local 3090 image/2–4-second video smoke test. Do not assume stock Wan2.2 support or run huge models automatically. Record model, VRAM peak, duration, failures, workflow JSON and visual evidence. Do not implement discovery/audio/QC beyond interfaces. Stop at the acceptance gate.
