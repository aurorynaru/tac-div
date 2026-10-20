# Copyright (C) 2024 Pixelle-Video contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic offline doubles for the local ComfyUI client.

``MockComfyClient`` exposes the same public surface as
:class:`~pixelle_video.cat_factory.gpu.comfy_client.ComfyClient` and never
sleeps, opens a socket, or touches the filesystem beyond read-only
``Path.exists`` checks.  ``MockTransport`` scripts ``(status, body)`` pairs or
exceptions for a real ``ComfyClient(transport=...)`` while recording every
call, which lets orchestration tests cover success, timeout, request errors,
partial success and long-running jobs without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from .comfy_client import ModelFileStatus, ProbeResult, QueueState
    from .monitor import VramSample

__all__ = ["MockComfyClient", "MockTransport"]

_SCENARIOS = frozenset(
    {
        "success",
        "timeout",
        "request_error",
        "partial_success",
        "long_running_pending",
    }
)

ScriptedResponse = tuple[int, str] | BaseException

CallRecord = tuple[str, str, "dict[str, object] | None"]


class MockTransport:
    """Scripted transport for ``ComfyClient`` that records every request."""

    def __init__(
        self,
        responses: Sequence[ScriptedResponse] | None = None,
        *,
        default: ScriptedResponse | None = None,
    ) -> None:
        """Create a transport that consumes scripted responses in order."""
        self._responses = list(responses) if responses is not None else []
        self._default = default
        self.calls: list[CallRecord] = []

    def __call__(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        timeout_s: float,
    ) -> tuple[int, str]:
        """Record one call and return the next scripted response."""
        self.calls.append((method, path, payload))
        if self._responses:
            response = self._responses.pop(0)
        elif self._default is not None:
            response = self._default
        else:
            raise AssertionError(
                f"MockTransport has no scripted response for {method} {path}"
            )
        if isinstance(response, BaseException):
            raise response
        status, body = response
        return int(status), body


class MockComfyClient:
    """Offline ComfyUI client with deterministic per-scenario behavior."""

    def __init__(
        self,
        scenario: str = "success",
        *,
        settle_after_polls: int = 3,
    ) -> None:
        """Create a mock for one validated scenario."""
        if scenario not in _SCENARIOS:
            allowed = ", ".join(sorted(_SCENARIOS))
            raise ValueError(
                f"unknown mock scenario {scenario!r}; expected one of {allowed}"
            )
        if settle_after_polls < 0:
            raise ValueError(
                "settle_after_polls must not be negative, got "
                f"{settle_after_polls}"
            )
        self.scenario = scenario
        self.settle_after_polls = settle_after_polls
        self.calls: list[CallRecord] = []
        self._posted_prompt_id: str | None = None
        self._poll_count = 0

    def _record(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Append one call record."""
        self.calls.append((method, path, payload))

    def post_prompt(
        self,
        workflow: dict[str, object],
        client_id: str | None = None,
    ) -> str:
        """Submit a workflow according to the configured scenario."""
        payload: dict[str, object] = {
            "prompt": workflow,
            "client_id": client_id,
        }
        self._record("POST", "/prompt", payload)

        if self.scenario == "timeout":
            from .comfy_client import ComfyTimeoutError

            raise ComfyTimeoutError("mock: POST /prompt timed out")
        if self.scenario == "request_error":
            from .comfy_client import ComfyRequestError

            raise ComfyRequestError("mock: POST /prompt was rejected")

        prompt_ids = {
            "success": "mock-prompt-1",
            "partial_success": "mock-prompt-2",
            "long_running_pending": "mock-prompt-3",
        }
        prompt_id = prompt_ids[self.scenario]
        self._posted_prompt_id = prompt_id
        self._poll_count = 0
        return prompt_id

    def get_queue(self) -> QueueState:
        """Return a deterministic queue snapshot for the scenario."""
        from .comfy_client import QueueState

        self._record("GET", "/queue")
        prompt_id = self._posted_prompt_id
        if prompt_id is None:
            return QueueState()

        if self.scenario in {"success", "partial_success"}:
            self._poll_count += 1
            if self._poll_count <= 1:
                return QueueState(running=(prompt_id,))
            return QueueState()

        if self.scenario == "long_running_pending":
            self._poll_count += 1
            if self._poll_count <= self.settle_after_polls:
                return QueueState(running=(prompt_id,))
            return QueueState()

        return QueueState()

    def get_history(self, prompt_id: str) -> dict[str, object]:
        """Return a deterministic history entry for the scenario."""
        self._record("GET", f"/history/{prompt_id}")

        if self.scenario == "timeout":
            from .comfy_client import ComfyTimeoutError

            raise ComfyTimeoutError(
                f"mock: GET /history/{prompt_id} timed out"
            )
        if self.scenario == "request_error":
            return {}
        if self.scenario == "long_running_pending":
            self._poll_count += 1
            if self._poll_count <= self.settle_after_polls:
                return {}
            return self._completed_entry(prompt_id)
        if self.scenario == "partial_success":
            return self._partial_entry(prompt_id)
        return self._completed_entry(prompt_id)

    def free(
        self,
        *,
        unload_models: bool = True,
        free_memory: bool = True,
    ) -> bool:
        """Record a ``/free`` request and report acknowledgement."""
        self._record(
            "POST",
            "/free",
            {"unload_models": unload_models, "free_memory": free_memory},
        )
        return True

    def probe(
        self,
        *,
        workflow_paths: Sequence[str | Path] = (),
        vram_sampler: Callable[[], "VramSample"] | None = None,
    ) -> ProbeResult:
        """Return a read-only probe result without parsing workflows."""
        from .comfy_client import ModelFileStatus, ProbeResult

        self._record("GET", "/probe")
        queue = self.get_queue()
        statuses = tuple(
            ModelFileStatus(path=str(raw_path), exists=Path(raw_path).exists())
            for raw_path in workflow_paths
        )
        vram = vram_sampler() if vram_sampler is not None else None
        return ProbeResult(
            endpoint="mock://comfy",
            queue=queue,
            model_files=statuses,
            vram=vram,
        )

    def is_local_endpoint(self) -> bool:
        """The mock always represents a loopback endpoint."""
        return True

    @staticmethod
    def _completed_entry(prompt_id: str) -> dict[str, object]:
        """Build a fully successful history entry with an ``outputs`` map."""
        return {
            "prompt_id": prompt_id,
            "outputs": {
                "9": {
                    "images": [
                        {
                            "filename": "mock-output.png",
                            "subfolder": "",
                            "type": "output",
                        }
                    ],
                    "status": {"status_str": "success", "completed": True},
                }
            },
            "status": {"status_str": "success", "completed": True},
        }

    @staticmethod
    def _partial_entry(prompt_id: str) -> dict[str, object]:
        """Build a history entry with one completed and one failed output."""
        return {
            "prompt_id": prompt_id,
            "outputs": {
                "9": {
                    "images": [
                        {
                            "filename": "mock-partial.png",
                            "subfolder": "",
                            "type": "output",
                        }
                    ],
                    "status": {"status_str": "success", "completed": True},
                },
                "10": {
                    "status": {"status_str": "error", "completed": False},
                },
            },
            "status": {"status_str": "error", "completed": False},
        }
