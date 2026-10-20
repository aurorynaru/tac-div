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

"""Persist the durable Cat Factory planning manifest (task 3.6).

This module owns exactly one responsibility: it builds and atomically writes the
single versioned manifest for a Cat Factory job at
``output/cat_factory/<job_id>/manifest.json``, plus the readers that round-trip
it. It only writes the durable manifest: it performs NO LLM call, NO network
access, and NO GPU work, and it writes no rendered output. The planning LLM phase
belongs to the planner (task 3.5); storage commits the already-quiesced result.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pixelle_video.cat_factory.models import (
    AudioMode,
    CreativeBrief,
    InspirationCandidate,
    ParodyIdea,
    ShotPlan,
)
from pixelle_video.utils.os_util import create_task_id

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_STATE_PLANNED",
    "MANIFEST_FILENAME",
    "CAT_FACTORY_OUTPUT_ROOT",
    "CHECKPOINT_PLANNING_COMPLETE",
    "create_job_id",
    "build_planning_manifest",
    "planning_complete_checkpoint",
    "is_planning_complete",
    "write_manifest",
    "read_manifest",
    "persist_planned_job",
]


# ---------------------------------------------------------------------------
# Manifest contract constants
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_STATE_PLANNED = "planned"
MANIFEST_FILENAME = "manifest.json"

# Mirror the stock ``output/{task_id}`` convention used by
# ``services/persistence.py`` (relative to the current working directory).
CAT_FACTORY_OUTPUT_ROOT = Path("output") / "cat_factory"

# The planning-complete checkpoint name. ``plan/04-gpu-orchestration.md`` does not
# name the concrete checkpoint form, so this module fixes it here: the persisted
# manifest carries a top-level ``checkpoint`` object whose ``name`` is
# ``planning_complete`` and whose ``emitted`` flag is true, alongside the
# ``planned`` state. That pair is what a reader uses to detect the checkpoint.
CHECKPOINT_PLANNING_COMPLETE = "planning_complete"

# Per-shot and QA defaults taken from the shared contract in
# ``plan/00-START-HERE.md``.
SHOT_STATUS_PLANNED = "planned"
QA_STATUS_NOT_RUN = "not_run"


def create_job_id() -> str:
    """Return a fresh Cat Factory job id.

    The stock :func:`pixelle_video.utils.os_util.create_task_id` is reused
    DELIBERATELY so cat jobs share the stock Pixelle task-id shape
    (``{YYYYmmdd_HHMMSS}_{hex4}``). The divergent ``cat-YYYYMMDD-NNN`` scheme
    shown in the plan example is avoided: it is a second id grammar with no
    collision guard, whereas the stock id already carries a per-second random
    suffix.
    """
    return create_task_id()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a Pydantic model or a plain mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _serialize_payload(value: Any, *, field_name: str) -> Any:
    """Serialize a planning contract to JSON-ready data.

    Accepts a Pydantic model (via ``model_dump(mode="json")``) or a plain mapping
    supplied by the caller, so storage never requires the planner's types.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"{field_name} must be a Pydantic model or a mapping, "
        f"got {type(value).__name__}"
    )


def _serialize_records(records: Sequence[Any] | None, *, field_name: str) -> list[dict]:
    """Normalize an attempt/error record sequence to a list of plain dicts."""
    normalized: list[dict] = []
    for record in records or []:
        if hasattr(record, "model_dump"):
            normalized.append(record.model_dump(mode="json"))
        elif isinstance(record, Mapping):
            normalized.append(dict(record))
        else:
            raise TypeError(
                f"{field_name} entries must be mappings or Pydantic models, "
                f"got {type(record).__name__}"
            )
    return normalized


def _audio_mode_value(audio_mode: Any) -> str | None:
    """Serialize an audio mode to its plain string value.

    ``AudioMode`` is a ``str`` enum, so ``.value`` is pinned here because
    ``str(AudioMode.SFX_ONLY)`` would render ``"AudioMode.SFX_ONLY"`` rather than
    the wire value ``"sfx_only"``.
    """
    if audio_mode is None:
        return None
    value = getattr(audio_mode, "value", None)
    if isinstance(value, str):
        return value
    return str(audio_mode)


def _guardrails_payload(guardrails: Any) -> dict:
    """Normalize the per-job guardrail counters to a plain mapping.

    The counters are provided by the caller as a mapping (``JobGuardrails``
    exposes ``to_dict()``); an object exposing ``to_dict()`` is also accepted so
    the caller may pass either form without storage importing the guardrail type.
    """
    if guardrails is None:
        return {}
    if isinstance(guardrails, Mapping):
        return dict(guardrails)
    to_dict = getattr(guardrails, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError(
        "guardrails must be a mapping (or expose to_dict()), "
        f"got {type(guardrails).__name__}"
    )


def _default_qa() -> dict:
    """Return the shared contract's initial QA block."""
    return {"status": QA_STATUS_NOT_RUN, "checks": []}


def _resolve_shots(
    shots: Sequence[ShotPlan | Mapping[str, Any]] | None, selected_idea: Any
) -> list:
    """Return the shot source list for the manifest.

    The planning contract keeps the shot breakdown on the selected idea
    (``ParodyIdea.shots``); fall back to it when the caller omits ``shots``.
    """
    if shots is not None:
        return list(shots)
    idea_shots = _get_field(selected_idea, "shots", None)
    if idea_shots:
        return list(idea_shots)
    return []


def _build_shot_entry(index: int, shot: Any) -> dict:
    """Map one ``ShotPlan`` to its shared-contract manifest entry.

    The per-shot id is derived deterministically from position (``s01``, ``s02``,
    ...) so the manifest is stable and readable regardless of the model's own
    ``shot_id``.
    """
    assets = _get_field(shot, "assets", None)
    status = _get_field(shot, "status", None)
    return {
        "shot_id": f"s{index:02d}",
        "duration_s": _get_field(shot, "duration_s", None),
        "caption": _get_field(shot, "caption", None),
        "image_prompt": _get_field(shot, "image_prompt", ""),
        "motion_prompt": _get_field(shot, "motion_prompt", ""),
        "assets": dict(assets) if assets else {},
        "status": status or SHOT_STATUS_PLANNED,
    }


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_planning_manifest(
    *,
    job_id: str,
    brief: CreativeBrief | Mapping[str, Any],
    inspiration: Sequence[InspirationCandidate | Mapping[str, Any]] | None = None,
    selected_idea: ParodyIdea | Mapping[str, Any],
    cat_character_id: str | None = None,
    audio_mode: AudioMode | str | None = None,
    shots: Sequence[ShotPlan | Mapping[str, Any]] | None = None,
    artifact_paths: Mapping[str, Any] | None = None,
    attempts: Sequence[Mapping[str, Any]] | None = None,
    errors: Sequence[Mapping[str, Any]] | None = None,
    guardrails: Any = None,
) -> dict:
    """Build the shared versioned planning manifest (pure, no I/O).

    Produces the ONE shared JSON contract from ``plan/00-START-HERE.md`` with
    ``state == "planned"``. Keeping the field order aligned with that contract
    matters: it is what readers and later steps index by.

    ``attempts`` is a JSON OBJECT (a mapping, not a list) carrying both the
    planning attempt records under ``attempts.planning`` and the guardrail
    counters under ``attempts.guardrails``, so a resume restores the persisted
    budget instead of resetting it. This stays compatible with the contract's
    ``"attempts": {}``.
    """
    if not job_id:
        raise ValueError("job_id must be a non-empty string")
    if brief is None:
        raise ValueError("brief is required")
    if selected_idea is None:
        raise ValueError("selected_idea is required")

    inspiration_entries = [
        _serialize_payload(item, field_name="inspiration[]")
        for item in (inspiration or [])
    ]
    shot_entries = [
        _build_shot_entry(index, shot)
        for index, shot in enumerate(_resolve_shots(shots, selected_idea), start=1)
    ]

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "job_id": job_id,
        "state": MANIFEST_STATE_PLANNED,
        "brief": _serialize_payload(brief, field_name="brief"),
        "inspiration": inspiration_entries,
        "selected_idea": _serialize_payload(selected_idea, field_name="selected_idea"),
        "cat_character_id": cat_character_id,
        "audio_mode": _audio_mode_value(audio_mode),
        "shots": shot_entries,
        "artifact_paths": dict(artifact_paths) if artifact_paths else {},
        "qa": _default_qa(),
        "attempts": {
            "planning": _serialize_records(attempts, field_name="attempts.planning"),
            "guardrails": _guardrails_payload(guardrails),
        },
        "errors": _serialize_records(errors, field_name="errors"),
    }


# ---------------------------------------------------------------------------
# Planning-complete checkpoint
# ---------------------------------------------------------------------------


def planning_complete_checkpoint(manifest: Mapping[str, Any]) -> dict:
    """Return the canonical ``planning_complete`` checkpoint record.

    If ``manifest`` already carries a planning-complete checkpoint, that stored
    record is returned unchanged; otherwise a fresh record is built. Building is
    only meaningful for a manifest in the planned state, so that state is
    asserted. The concrete form of the checkpoint is fixed here because
    ``plan/04-gpu-orchestration.md`` does not name it.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")

    existing = manifest.get("checkpoint")
    if (
        isinstance(existing, Mapping)
        and existing.get("name") == CHECKPOINT_PLANNING_COMPLETE
    ):
        return dict(existing)

    state = manifest.get("state")
    if state != MANIFEST_STATE_PLANNED:
        raise ValueError(
            "planning_complete checkpoint requires state "
            f"{MANIFEST_STATE_PLANNED!r}, got {state!r}"
        )
    return {
        "name": CHECKPOINT_PLANNING_COMPLETE,
        "emitted": True,
        "emitted_at": _utc_now_iso(),
    }


def is_planning_complete(manifest: Mapping[str, Any]) -> bool:
    """Return True iff ``manifest`` alone proves the planning checkpoint.

    All three conditions must hold: ``state == "planned"``, the top-level
    ``checkpoint.name == "planning_complete"`` and ``checkpoint.emitted is True``.
    """
    if not isinstance(manifest, Mapping):
        return False
    if manifest.get("state") != MANIFEST_STATE_PLANNED:
        return False
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return False
    return (
        checkpoint.get("name") == CHECKPOINT_PLANNING_COMPLETE
        and checkpoint.get("emitted") is True
    )


# ---------------------------------------------------------------------------
# Atomic persistence
# ---------------------------------------------------------------------------


def _resolve_job_dir(job_id: str, output_root: Any = None) -> Path:
    """Resolve ``<root>/<job_id>`` for the manifest job directory."""
    root = Path(output_root) if output_root is not None else CAT_FACTORY_OUTPUT_ROOT
    return root / str(job_id)


def write_manifest(manifest: Mapping[str, Any], *, output_root: Any = None) -> Path:
    """Atomically write ``manifest`` to ``<root>/<job_id>/manifest.json``.

    Writes the JSON to a uniquely named temp file created IN THE SAME DIRECTORY,
    then promotes it with an atomic replace. On ANY failure the temp file is
    removed best-effort and the error is re-raised, so a partial ``manifest.json``
    is never left behind. Re-running for the same job id overwrites the single
    manifest in place, so the write is idempotent (no duplicates, no partials).
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    job_id = manifest.get("job_id")
    if not job_id:
        raise ValueError("manifest must carry a non-empty 'job_id'")

    job_dir = _resolve_job_dir(job_id, output_root)
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / MANIFEST_FILENAME

    # The temp file MUST live in the target directory so the promotion is a
    # same-filesystem rename, which is atomic on Windows and POSIX alike.
    temp_path = job_dir / f"{MANIFEST_FILENAME}.{uuid.uuid4().hex}.tmp"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        # ``os.replace`` (identical to ``Path.replace``) atomically overwrites an
        # existing target. ``os.rename`` is NOT used: on Windows it raises
        # FileExistsError when the target already exists, which would break
        # idempotent re-runs for the same job_id.
        os.replace(temp_path, manifest_path)
    except BaseException:
        # Never leave a partial temp manifest behind on any failure.
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise
    return manifest_path


def read_manifest(job_dir_or_job_id: str | os.PathLike[str]) -> dict:
    """Read a persisted manifest from a job dir, a manifest path, or a job id."""
    candidate = Path(job_dir_or_job_id)
    if candidate.name == MANIFEST_FILENAME:
        manifest_path = candidate
    elif candidate.is_dir() or (
        candidate.parent != Path(".") and candidate.suffix != ".json"
    ):
        manifest_path = candidate / MANIFEST_FILENAME
    else:
        manifest_path = (
            CAT_FACTORY_OUTPUT_ROOT / str(job_dir_or_job_id) / MANIFEST_FILENAME
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"manifest at {manifest_path} is not a JSON object")
    return loaded


def persist_planned_job(
    *,
    brief: CreativeBrief | Mapping[str, Any],
    selected_idea: ParodyIdea | Mapping[str, Any],
    inspiration: Sequence[InspirationCandidate | Mapping[str, Any]] | None = None,
    cat_character_id: str | None = None,
    audio_mode: AudioMode | str | None = None,
    shots: Sequence[ShotPlan | Mapping[str, Any]] | None = None,
    artifact_paths: Mapping[str, Any] | None = None,
    attempts: Sequence[Mapping[str, Any]] | None = None,
    errors: Sequence[Mapping[str, Any]] | None = None,
    guardrails: Any = None,
    job_id: str | None = None,
    output_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Persist a completed planning result as the durable manifest.

    Builds the shared manifest, embeds the ``planning_complete`` checkpoint, and
    commits everything in a SINGLE atomic write, then returns the manifest path.

    Caller contract: the caller must pass the fields of a ``PlanningResult`` whose
    LLM calls have already been awaited and closed. This function never awaits or
    issues an LLM call — it contributes only the commit. Because the checkpoint
    lives in the same atomic write as the manifest body, it becomes observable
    only when that commit lands, and only after the caller has quiesced its LLM
    phase; the quiescence is the caller's responsibility, not storage's.

    The inputs accept either the planning-contract Pydantic models or plain
    mappings. The write is idempotent: a second call for the same job id replaces
    the existing manifest in place.
    """
    manifest = build_planning_manifest(
        job_id=job_id or create_job_id(),
        brief=brief,
        inspiration=inspiration,
        selected_idea=selected_idea,
        cat_character_id=cat_character_id,
        audio_mode=audio_mode,
        shots=shots,
        artifact_paths=artifact_paths,
        attempts=attempts,
        errors=errors,
        guardrails=guardrails,
    )
    # Same single atomic write as the body: a reader never sees a planned
    # manifest lacking its checkpoint, nor a checkpoint without the body.
    manifest["checkpoint"] = planning_complete_checkpoint(manifest)
    return write_manifest(manifest, output_root=output_root)
