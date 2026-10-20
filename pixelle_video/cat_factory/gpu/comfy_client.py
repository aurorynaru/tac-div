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

"""Minimal localhost HTTP surface for a stock ComfyUI server.

The client talks only to the ComfyUI endpoints needed for local
orchestration: ``/queue``, ``/prompt``, ``/history/{id}``, ``/free`` and a
read-only ``probe`` that never posts and never writes files.  Every request
goes through an injectable transport so tests can run offline without a
server.

The default transport uses :mod:`urllib.request` and a localhost base URL.
No third-party HTTP or GPU libraries are imported, and importing this module
has no network or filesystem side effects.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Sequence
from urllib.parse import quote, urlparse

if TYPE_CHECKING:
    from .monitor import VramSample

__all__ = [
    "ComfyClient",
    "ComfyClientError",
    "ComfyRequestError",
    "ComfyTimeoutError",
    "ModelFileStatus",
    "ProbeResult",
    "QueueState",
    "Transport",
]

Transport = Callable[
    [str, str, dict[str, object] | None, float],
    tuple[int, str],
]

_MODEL_SUFFIXES = (
    ".safetensors",
    ".gguf",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".sft",
)
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_ERROR_BODY_LIMIT = 300


class ComfyClientError(Exception):
    """Base error for ComfyUI client failures."""


class ComfyRequestError(ComfyClientError):
    """ComfyUI rejected a request or returned an unusable response."""


class ComfyTimeoutError(ComfyClientError):
    """A ComfyUI request exceeded its deadline."""


@dataclass(frozen=True)
class QueueState:
    """Prompt identifiers currently running and pending in ComfyUI."""

    running: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelFileStatus:
    """Existence check for one model file path."""

    path: str
    exists: bool


@dataclass(frozen=True)
class ProbeResult:
    """Read-only snapshot of a ComfyUI endpoint and its model files."""

    endpoint: str
    queue: QueueState
    model_files: tuple[ModelFileStatus, ...] = ()
    vram: "VramSample | None" = None


def _sanitize(text: str) -> str:
    """Collapse whitespace and bound an upstream message for readability."""
    compact = " ".join(text.split())
    if not compact:
        return "<empty body>"
    if len(compact) <= _ERROR_BODY_LIMIT:
        return compact
    return compact[:_ERROR_BODY_LIMIT] + "..."


def _decode_json_object(body: str, operation: str) -> dict[str, object]:
    """Decode a JSON object or raise a sanitized request error."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ComfyRequestError(
            f"{operation} returned malformed JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ComfyRequestError(f"{operation} returned JSON that is not an object")
    return payload


def _summarize_node_errors(node_errors: object) -> str:
    """Describe rejected workflow nodes without dumping the workflow."""
    if isinstance(node_errors, dict) and node_errors:
        node_ids = ", ".join(str(node_id) for node_id in list(node_errors)[:5])
        return f"node_errors for node(s) {node_ids}"
    return "node_errors present"


def _parse_queue_ids(value: object) -> tuple[str, ...]:
    """Extract prompt identifiers from one ComfyUI queue array."""
    if not isinstance(value, (list, tuple)):
        return ()
    prompt_ids: list[str] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            if not item:
                continue
            source = item[1] if len(item) > 1 else item[0]
            prompt_ids.append(str(source))
        else:
            prompt_ids.append(str(item))
    return tuple(prompt_ids)


def _iter_model_filenames(value: object) -> Iterator[str]:
    """Yield model filenames found in workflow node ``inputs`` values."""
    if isinstance(value, dict):
        inputs = value.get("inputs")
        if isinstance(inputs, dict):
            for input_value in inputs.values():
                if isinstance(input_value, str) and input_value.lower().endswith(
                    _MODEL_SUFFIXES
                ):
                    yield input_value
        for nested in value.values():
            yield from _iter_model_filenames(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_model_filenames(nested)


class ComfyClient:
    """Small fail-closed client for a local ComfyUI HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        timeout_s: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        """Create a client bound to a validated http(s) endpoint."""
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ComfyClientError(
                "base_url must use http or https, received "
                f"{parsed.scheme or 'no scheme'!r}"
            )
        if not parsed.hostname:
            raise ComfyClientError(f"base_url has no host: {base_url!r}")
        if timeout_s <= 0:
            raise ComfyClientError(f"timeout_s must be positive, got {timeout_s}")

        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._transport: Transport = (
            transport if transport is not None else self._default_transport
        )

    def _default_transport(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        timeout_s: float,
    ) -> tuple[int, str]:
        """Perform one HTTP request through :mod:`urllib.request`."""
        data: bytes | None = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self._base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
                return int(response.status), body
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except OSError:
                body = ""
            return int(exc.code), body
        except (TimeoutError, socket.timeout) as exc:
            raise ComfyTimeoutError(
                f"{method} {path} timed out after {timeout_s:.1f}s"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise ComfyTimeoutError(
                    f"{method} {path} timed out after {timeout_s:.1f}s"
                ) from exc
            raise ComfyRequestError(
                f"{method} {path} could not reach ComfyUI: {reason}"
            ) from exc
        except OSError as exc:
            raise ComfyRequestError(
                f"{method} {path} failed: {exc}"
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> tuple[int, str]:
        """Run the configured transport, preserving typed client errors."""
        try:
            status, body = self._transport(method, path, payload, self._timeout_s)
        except ComfyClientError:
            raise
        except Exception as exc:
            raise ComfyRequestError(
                f"{method} {path} failed in transport: {exc}"
            ) from exc
        return int(status), body

    def get_queue(self) -> QueueState:
        """Return the running and pending prompt identifiers."""
        status, body = self._request("GET", "/queue")
        if status != 200:
            raise ComfyRequestError(
                f"GET /queue returned HTTP {status}: {_sanitize(body)}"
            )
        payload = _decode_json_object(body, "GET /queue")
        return QueueState(
            running=_parse_queue_ids(payload.get("queue_running")),
            pending=_parse_queue_ids(payload.get("queue_pending")),
        )

    def post_prompt(
        self,
        workflow: dict[str, object],
        client_id: str | None = None,
    ) -> str:
        """Submit a workflow and return its prompt identifier."""
        payload: dict[str, object] = {
            "prompt": workflow,
            "client_id": client_id,
        }
        status, body = self._request("POST", "/prompt", payload)
        if status != 200:
            raise ComfyRequestError(
                f"POST /prompt returned HTTP {status}: {_sanitize(body)}"
            )
        response = _decode_json_object(body, "POST /prompt")
        node_errors = response.get("node_errors")
        if node_errors:
            raise ComfyRequestError(
                "ComfyUI rejected the workflow: "
                f"{_summarize_node_errors(node_errors)}"
            )
        error = response.get("error")
        if error:
            raise ComfyRequestError(
                f"ComfyUI rejected the workflow: {_sanitize(str(error))}"
            )
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyRequestError("POST /prompt returned no prompt_id")
        return prompt_id

    def get_history(self, prompt_id: str) -> dict[str, object]:
        """Return the history entry for ``prompt_id``, or an empty mapping."""
        status, body = self._request(
            "GET", "/history/" + quote(prompt_id, safe="")
        )
        if status != 200:
            raise ComfyRequestError(
                f"GET /history/{prompt_id} returned HTTP {status}: "
                f"{_sanitize(body)}"
            )
        payload = _decode_json_object(body, f"GET /history/{prompt_id}")
        entry = payload.get(prompt_id)
        if isinstance(entry, dict):
            return dict(entry)
        return {}

    def free(
        self,
        *,
        unload_models: bool = True,
        free_memory: bool = True,
    ) -> bool:
        """Ask ComfyUI to unload models; the only mutating request here."""
        payload: dict[str, object] = {
            "unload_models": unload_models,
            "free_memory": free_memory,
        }
        status, body = self._request("POST", "/free", payload)
        if status != 200:
            raise ComfyRequestError(
                f"POST /free returned HTTP {status}: {_sanitize(body)}"
            )
        return True

    def probe(
        self,
        *,
        workflow_paths: Sequence[str | Path] = (),
        vram_sampler: Callable[[], "VramSample"] | None = None,
    ) -> ProbeResult:
        """Inspect the queue, model files and GPU memory without mutating anything.

        GPU name, total memory and per-process VRAM are recorded through the
        optional ``vram_sampler`` (defaulting to ``None``, which keeps this
        client subprocess-free and strictly read-only).

        ``ModelFileStatus.exists`` is a RUNNER-side filesystem check; it does
        NOT prove the model is installed in ComfyUI's own model directory and
        MUST NOT be read as evidence that generation succeeded.
        """
        queue = self.get_queue()
        model_files: list[ModelFileStatus] = []
        for raw_path in workflow_paths:
            workflow_path = Path(raw_path)
            if not workflow_path.exists():
                model_files.append(
                    ModelFileStatus(path=str(workflow_path), exists=False)
                )
                continue
            try:
                document = json.loads(workflow_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                model_files.append(
                    ModelFileStatus(path=str(workflow_path), exists=False)
                )
                continue

            seen: set[str] = set()
            for filename in _iter_model_filenames(document):
                if filename in seen:
                    continue
                seen.add(filename)
                model_files.append(
                    ModelFileStatus(path=filename, exists=Path(filename).exists())
                )
        vram = vram_sampler() if vram_sampler is not None else None
        return ProbeResult(
            endpoint=self._base_url,
            queue=queue,
            model_files=tuple(model_files),
            vram=vram,
        )

    def is_local_endpoint(self) -> bool:
        """Return whether the configured endpoint targets a loopback host."""
        hostname = urlparse(self._base_url).hostname
        if hostname is None:
            return False
        normalized = hostname.lower()
        return normalized in _LOCAL_HOSTS or normalized.startswith("127.")
