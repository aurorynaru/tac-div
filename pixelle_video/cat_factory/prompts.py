# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-neutral planning prompt templates for the Cat Factory planner.

Step 03 planning prompts are plain-text templates reused unchanged across every
configured LLM. A provider profile is configuration data, not a
template-selection key: there is ONE provider-neutral template set and no
OpenAI/Anthropic/Qwen-specific branches anywhere in this module.

Every template follows the ROLE / INPUTS / RULES / RETURN-ONLY-JSON structure
defined in ``plan/03-llm-integration.md``. The required output keys and their
constraints mirror the ``Field(..., description=...)`` text in
``pixelle_video/cat_factory/models.py`` so the text shown to the model and the
schema it is validated against stay in agreement. Keep this text stable.

This module is pure data: no network, no filesystem access, no API calls, and
no provider SDK imports.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared, provider-neutral building blocks
# ---------------------------------------------------------------------------
# These constants are the single source for the role/rules/output lines that
# every stage template reuses, so the prompt text cannot drift between stages.

PLANNING_ROLE: str = (
    "You are a planner for short, original, realistic cat-comedy parodies."
)

PLANNING_RULES: str = (
    "RULES:\n"
    "  - Use reference metadata only for high-level comic patterns. Do NOT reproduce\n"
    "    copyrighted dialogue, copyrighted audio, exact scene footage, or creator watermarks.\n"
    "  - One visual joke in 6 to 15 seconds across 1 to 4 shots, with realistic tabby\n"
    "    anatomy and absurd but readable action.\n"
    "  - Select audio_mode from: none, caption_only, sfx_only, tts.\n"
    "  - State scene duration, keyframe prompt, motion prompt, caption,\n"
    "    visual constraints, rationale, and source provenance.\n"
)

PLANNING_RETURN_ONLY_JSON: str = (
    "RETURN-ONLY-JSON: Return ONLY a JSON object matching the provided schema. "
    "No prose, no markdown."
)

# Canonical INPUTS line shared by planning stages. Stage templates below may
# narrow it, but the brief/reference/character inputs are always described here.
_BRIEF_INPUTS: str = (
    "CreativeBrief JSON, discovered reference metadata, cat character constraints."
)

# Required output keys per contract. Key names match the model field names and
# their descriptions in ``pixelle_video/cat_factory/models.py`` exactly.
_INSPIRATION_KEYS: str = (
    "source_id, provider, source_kind, title, source_url, short_summary, "
    "published_at, retrieved_at, popularity, evidence_of_trend, comic_mechanism, "
    "cat_parody, rights_notes, engagement_metadata, scores, reason_for_selection"
)
_SHOT_KEYS: str = (
    "shot_id, duration_s, scene, action, camera, image_prompt, motion_prompt, "
    "caption, character_id, identity_reference, audio_mode, negative_constraints, "
    "status, assets, reference_match"
)
_PARODY_IDEA_KEYS: str = (
    "title, source_ids, source_kind, comic_mechanism, original_premise, cat_action, "
    "punchline, shot_count, duration_s, audio_mode, scores, reason_for_selection, shots"
)
_AUDIO_PLAN_KEYS: str = (
    "mode, caption_text, narration_text, sfx_cues, background_music_path, "
    "preserve_source_audio"
)


# ---------------------------------------------------------------------------
# Canonical template (spec form) — ROLE / INPUTS / RULES / RETURN-ONLY-JSON
# ---------------------------------------------------------------------------

PLANNING_PROMPT_TEMPLATE: str = (
    "ROLE: " + PLANNING_ROLE + "\n"
    "INPUTS: " + _BRIEF_INPUTS + "\n"
    + PLANNING_RULES
    + PLANNING_RETURN_ONLY_JSON
)


# ---------------------------------------------------------------------------
# Stage templates (one provider-neutral set; no per-vendor variants)
# ---------------------------------------------------------------------------

INSPIRATION_PROMPT_TEMPLATE: str = (
    "ROLE: " + PLANNING_ROLE + "\n"
    "INPUTS: " + _BRIEF_INPUTS + "\n"
    + PLANNING_RULES
    + "REQUIRED OUTPUT KEYS (InspirationCandidate): " + _INSPIRATION_KEYS + ".\n"
    + PLANNING_RETURN_ONLY_JSON
)

PARODY_IDEA_PROMPT_TEMPLATE: str = (
    "ROLE: " + PLANNING_ROLE + "\n"
    "INPUTS: One selected InspirationCandidate, the CreativeBrief, and cat character "
    "constraints.\n"
    + PLANNING_RULES
    + "REQUIRED OUTPUT KEYS (ParodyIdea): " + _PARODY_IDEA_KEYS + ".\n"
    + "REQUIRED OUTPUT KEYS (ShotPlan, one per shot): " + _SHOT_KEYS + ".\n"
    + PLANNING_RETURN_ONLY_JSON
)

AUDIO_PLAN_PROMPT_TEMPLATE: str = (
    "ROLE: " + PLANNING_ROLE + "\n"
    "INPUTS: One planned ParodyIdea (including its shots) and the CreativeBrief.\n"
    + PLANNING_RULES
    + "REQUIRED OUTPUT KEYS (AudioPlan): " + _AUDIO_PLAN_KEYS + ".\n"
    + PLANNING_RETURN_ONLY_JSON
)


# The single provider-neutral template set. Keys are logical planning stages,
# never provider or vendor identifiers.
PLANNING_PROMPT_TEMPLATES: dict[str, str] = {
    "planning": PLANNING_PROMPT_TEMPLATE,
    "inspiration": INSPIRATION_PROMPT_TEMPLATE,
    "parody_idea": PARODY_IDEA_PROMPT_TEMPLATE,
    "audio_plan": AUDIO_PLAN_PROMPT_TEMPLATE,
}

__all__ = [
    "PLANNING_ROLE",
    "PLANNING_RULES",
    "PLANNING_RETURN_ONLY_JSON",
    "PLANNING_PROMPT_TEMPLATE",
    "INSPIRATION_PROMPT_TEMPLATE",
    "PARODY_IDEA_PROMPT_TEMPLATE",
    "AUDIO_PLAN_PROMPT_TEMPLATE",
    "PLANNING_PROMPT_TEMPLATES",
]
