# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline tests for the Cat Factory GPU-orchestration config surface.

Covers ``CatFactoryGpuConfig`` (defaults, validators, strictness) and the
``CatFactoryConfig.gpu`` integration point. No GPU, no network, no subprocess.
"""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from pixelle_video.config.schema import (
    CatFactoryConfig,
    CatFactoryGpuConfig,
)


# ---------------------------------------------------------------------------
# 1. Defaults (all 14 fields documented by the Step 04 contract)
# ---------------------------------------------------------------------------


class TestDefaults:
    """Every field must carry the documented conservative default."""

    def test_defaults_exact(self) -> None:
        gpu = CatFactoryGpuConfig()
        assert gpu.comfy_command == []
        assert gpu.supervisor_mode == "managed"
        assert gpu.generation_timeout_s == 900
        assert gpu.poll_interval_s == 2.0
        assert gpu.stale_heartbeat_seconds == 300
        assert gpu.vram_margin_mb == 1024
        assert gpu.ram_margin_mb == 2048
        assert gpu.offload_min_ram_mb is None
        assert gpu.oom_max_retries == 1
        assert gpu.memory_release_timeout_s == 60
        assert gpu.state_dir == ""
        assert gpu.image_workflow == "workflows/selfhost/image_flux3.json"
        assert gpu.reference_workflow == "workflows/selfhost/image_flux3_ref.json"
        assert gpu.video_workflow == "workflows/selfhost/video_wan2_2_i2v.json"

    def test_defaults_are_fourteen_distinct_fields(self) -> None:
        """Guard against a field silently disappearing from the surface."""
        assert set(CatFactoryGpuConfig().model_dump().keys()) == {
            "comfy_command",
            "supervisor_mode",
            "generation_timeout_s",
            "poll_interval_s",
            "stale_heartbeat_seconds",
            "vram_margin_mb",
            "ram_margin_mb",
            "offload_min_ram_mb",
            "oom_max_retries",
            "memory_release_timeout_s",
            "state_dir",
            "image_workflow",
            "reference_workflow",
            "video_workflow",
        }

    def test_empty_command_is_fail_closed_sentinel(self) -> None:
        """An empty launch command means 'not configured' — never a guess."""
        gpu = CatFactoryGpuConfig()
        assert gpu.comfy_command == []
        assert not gpu.comfy_command

    def test_offload_min_ram_mb_is_none_fail_closed(self) -> None:
        """The offload threshold must default to 'not measured' (None)."""
        gpu = CatFactoryGpuConfig()
        assert gpu.offload_min_ram_mb is None


# ---------------------------------------------------------------------------
# 2. supervisor_mode strictness (FR-002: managed | attached)
# ---------------------------------------------------------------------------


class TestSupervisorMode:
    """Only the two FR-002 modes are representable."""

    @pytest.mark.parametrize("mode", ["managed", "attached"])
    def test_valid_modes_accepted(self, mode: str) -> None:
        gpu = CatFactoryGpuConfig(supervisor_mode=mode)
        assert gpu.supervisor_mode == mode

    def test_both_modes_present_in_mirrored_pair(self) -> None:
        managed = CatFactoryGpuConfig(supervisor_mode="managed")
        attached = CatFactoryGpuConfig(supervisor_mode="attached")
        assert managed.supervisor_mode == "managed"
        assert attached.supervisor_mode == "attached"
        assert managed.supervisor_mode != attached.supervisor_mode

    @pytest.mark.parametrize(
        "mode", ["auto", "MANAGED", "Attached", "", "detached", "spawned"]
    )
    def test_invalid_mode_rejected(self, mode: str) -> None:
        with pytest.raises(ValidationError):
            CatFactoryGpuConfig(supervisor_mode=mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. _validate_comfy_command
# ---------------------------------------------------------------------------


class TestComfyCommandValidator:
    """The launch command is an argument array, never a shell string."""

    def test_valid_array_accepted_unchanged(self) -> None:
        command = ["python", "main.py", "--port", "8188"]
        gpu = CatFactoryGpuConfig(comfy_command=command)
        assert gpu.comfy_command == command

    def test_empty_string_element_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            CatFactoryGpuConfig(comfy_command=["python", ""])
        assert "non-empty string" in str(excinfo.value)

    def test_empty_element_in_middle_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CatFactoryGpuConfig(comfy_command=["python", "", "main.py"])

    def test_empty_list_is_not_configured_sentinel(self) -> None:
        gpu = CatFactoryGpuConfig(comfy_command=[])
        assert gpu.comfy_command == []

    def test_bare_string_rejected_not_coerced(self) -> None:
        """A bare string must NOT be silently coerced into a character list."""
        with pytest.raises(ValidationError):
            CatFactoryGpuConfig(comfy_command="python main.py")  # type: ignore[arg-type]

    def test_non_string_element_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CatFactoryGpuConfig(comfy_command=["python", 8188])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 4. _validate_state_dir — the repaired edge-case matrix
# ---------------------------------------------------------------------------

# Each entry is a path that MUST be rejected as absolute or traversing.
_REJECTED_STATE_DIRS = [
    "/abs/path",          # driveless-rooted: the exact previous-review defect
    "C:\\foo",            # Windows drive-absolute (backslash)
    "C:/foo",             # Windows drive-absolute (forward slash)
    "\\\\server\\share",  # UNC path
    "..\\x",              # backslash traversal
    "a/../b",             # forward-slash traversal segment
    "a\\..\\b",           # backslash traversal segment
]

# Each entry is a legitimate project-relative path that MUST be accepted.
_ACCEPTED_STATE_DIRS = [
    "",        # explicit "feature default location" sentinel
    "a/b",     # nested relative path
    "./x",     # explicitly relative path
]


class TestStateDirValidator:
    """state_dir must be empty or a relative, non-traversing path."""

    @pytest.mark.parametrize("value", _REJECTED_STATE_DIRS)
    def test_rejected_paths_raise(self, value: str) -> None:
        with pytest.raises(ValidationError):
            CatFactoryGpuConfig(state_dir=value)

    @pytest.mark.parametrize("value", _ACCEPTED_STATE_DIRS)
    def test_accepted_paths_pass_through_unchanged(self, value: str) -> None:
        gpu = CatFactoryGpuConfig(state_dir=value)
        assert gpu.state_dir == value

    def test_driveless_rooted_absolute_regression(self) -> None:
        """REGRESSION: on Windows ``PureWindowsPath('/abs/path').is_absolute()``
        is False, so the pre-repair validator accepted ``/abs/path``. It must now
        be rejected (the ``win.root`` repair)."""
        with pytest.raises(ValidationError) as excinfo:
            CatFactoryGpuConfig(state_dir="/abs/path")
        assert "project-relative" in str(excinfo.value)

    def test_platform_neutral_absolute_forms_rejected(self) -> None:
        """Build OS-specific absolute forms with pathlib so the assertion holds
        on any host OS, then confirm the REAL class rejects every one."""
        forms = [
            str(pathlib.PurePosixPath("/") / "abs" / "path"),
            str(pathlib.PureWindowsPath("C:/") / "foo"),
            str(pathlib.PureWindowsPath("//server/share") / "x"),
        ]
        anchor = pathlib.Path.cwd().anchor
        if anchor:
            forms.append(str(pathlib.Path(anchor) / "elsewhere"))
        for form in forms:
            with pytest.raises(ValidationError):
                CatFactoryGpuConfig(state_dir=form)

    def test_traversal_segment_after_normalisation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CatFactoryGpuConfig(state_dir="data/../escape")

    def test_double_dot_filename_is_not_a_traversal_segment(self) -> None:
        """Only a whole ``..`` segment is traversal; ``..x`` is a valid name."""
        gpu = CatFactoryGpuConfig(state_dir="a/..x")
        assert gpu.state_dir == "a/..x"


# ---------------------------------------------------------------------------
# 5. Unknown-key strictness (assert what the SOURCE actually configures)
# ---------------------------------------------------------------------------


class TestExtraKeys:
    """schema.py sets no ``model_config`` on these models (verified by grep),
    so pydantic's default ``extra="ignore"`` applies: unknown keys are silently
    DROPPED, not rejected. Assert the actual source behaviour rather than
    assuming ``extra="forbid"``."""

    def test_unknown_key_is_ignored_not_rejected(self) -> None:
        gpu = CatFactoryGpuConfig(**{"comfy_command": [], "unknown_key": 1})
        assert gpu.comfy_command == []
        assert not hasattr(gpu, "unknown_key")

    def test_unknown_key_does_not_appear_in_model_dump(self) -> None:
        gpu = CatFactoryGpuConfig(**{"unknown_key": 1})
        assert "unknown_key" not in gpu.model_dump()

    def test_known_key_alongside_unknown_is_still_applied(self) -> None:
        gpu = CatFactoryGpuConfig(**{"supervisor_mode": "attached", "unknown_key": 1})
        assert gpu.supervisor_mode == "attached"


# ---------------------------------------------------------------------------
# 6. Integration with the parent CatFactoryConfig
# ---------------------------------------------------------------------------


class TestCatFactoryConfigIntegration:
    """The gpu section must default-construct and the flag must stay off."""

    def test_gpu_omitted_defaults_to_gpu_config(self) -> None:
        config = CatFactoryConfig()
        assert isinstance(config.gpu, CatFactoryGpuConfig)
        assert config.gpu.comfy_command == []

    def test_enabled_defaults_false(self) -> None:
        assert CatFactoryConfig().enabled is False

    def test_explicit_gpu_section_accepted(self) -> None:
        config = CatFactoryConfig(gpu={"supervisor_mode": "attached"})
        assert config.gpu.supervisor_mode == "attached"
        assert config.gpu.state_dir == ""

    def test_nested_invalid_state_dir_propagates(self) -> None:
        with pytest.raises(ValidationError):
            CatFactoryConfig(gpu={"state_dir": "/abs/path"})
