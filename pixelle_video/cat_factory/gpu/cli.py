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

"""Safe operator CLI for the Cat Factory local GPU orchestration.

The CLI is intentionally conservative: every subcommand is read-only unless it
explicitly performs a render, and rendering is the only path that can touch
ComfyUI or acquire the GPU lease. ``probe`` and ``dry-run`` are guaranteed to
make no destructive calls (no ``POST /prompt``, no ``POST /free``, no process
spawn, no process termination, no lease acquisition).

Exit codes:

* ``0`` -- success.
* ``2`` -- argument/usage error (raised by :mod:`argparse`).
* ``3`` -- safe refusal or blocked operation (``RenderBlocked``, ``CliError``,
  or ``ComfyClientError`` raised while probing).
* ``1`` -- unexpected runtime failure (``RenderError``, ``OSError``, ...).

Results are written to stdout as JSON; diagnostics are written to stderr.
Environment variables, credentials and prompts are never printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = ["CliError", "build_parser", "main"]

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

DEFAULT_GPU_ID = 0

KEYFRAME_PHASE = "keyframe"
VIDEO_PHASE = "video"

_RELEASE_REFUSAL_REASON = (
    "no lease is owned by this process; refusing to release a lease owned by "
    "another process"
)


class CliError(Exception):
    """A safe-refusal condition that should exit with status ``3``.

    Raised for operator mistakes that are not ``argparse`` usage errors, such
    as a missing or unusable ``cat_factory.gpu`` configuration block.
    """


# ---------------------------------------------------------------------------
# Dependency builders (all project imports are lazy on purpose)
# ---------------------------------------------------------------------------


def _resolve_gpu_config() -> Any:
    """Return ``config_manager.config.cat_factory.gpu``.

    The import is deferred so that ``--help`` and argument parsing work even
    when optional runtime dependencies (for example ``comfykit``) are missing.
    """
    from pixelle_video.config import config_manager

    factory = getattr(config_manager.config, "cat_factory", None)
    gpu_config = getattr(factory, "gpu", None)
    if gpu_config is None:
        raise CliError(
            "cat_factory.gpu is not configured; enable cat_factory in the "
            "application config before using the GPU CLI"
        )
    return gpu_config


def _lock_root(config: Any) -> Path:
    """Return the directory that holds GPU lease lock files."""
    raw_state_dir = getattr(config, "state_dir", "") or ""
    state_dir = Path(raw_state_dir) if raw_state_dir else Path("state") / "cat_factory" / "gpu"
    return state_dir / "lease"


def _build_lease(config: Any, gpu_id: int) -> Any:
    """Build a :class:`GpuLeaseManager` without acquiring anything."""
    from pixelle_video.cat_factory.gpu.lease import GpuLeaseManager

    stale_seconds = getattr(config, "stale_heartbeat_seconds", 300)
    if not isinstance(stale_seconds, int) or stale_seconds <= 0:
        stale_seconds = 300
    return GpuLeaseManager(
        gpu_id,
        _lock_root(config),
        stale_heartbeat_seconds=stale_seconds,
    )


def _build_monitor() -> Any:
    """Build a :class:`GpuMonitor` (construction performs no sampling)."""
    from pixelle_video.cat_factory.gpu.monitor import GpuMonitor

    return GpuMonitor()


def _build_client(endpoint: str | None) -> Any:
    """Build a :class:`ComfyClient` (construction performs no network I/O)."""
    from pixelle_video.cat_factory.gpu.comfy_client import ComfyClient

    if endpoint:
        return ComfyClient(base_url=endpoint)
    return ComfyClient()


def _build_supervisor(job_id: str, config: Any) -> Any:
    """Build a :class:`ComfySupervisor` (construction performs no spawn)."""
    from pixelle_video.cat_factory.gpu.supervisor import ComfySupervisor

    return ComfySupervisor(job_id, gpu_config=config)


def _job_dir(output_root: str | None, job_id: str) -> Path:
    """Return the on-disk directory for ``job_id``."""
    from pixelle_video.cat_factory.storage import CAT_FACTORY_OUTPUT_ROOT

    base = Path(output_root) if output_root else Path(CAT_FACTORY_OUTPUT_ROOT)
    return base / job_id


def _workflow_paths(config: Any) -> tuple[str, ...]:
    """Return the configured workflow paths, skipping unset entries."""
    candidates = (
        getattr(config, "image_workflow", "") or "",
        getattr(config, "reference_workflow", "") or "",
        getattr(config, "video_workflow", "") or "",
    )
    return tuple(str(path) for path in candidates if path)


def _output_root(args: argparse.Namespace) -> Path | None:
    """Normalise the optional ``--output-root`` argument."""
    raw = getattr(args, "output_root", None)
    return Path(raw) if raw else None


# ---------------------------------------------------------------------------
# Manifest helpers (dry-run only; no GPU dependency is imported)
# ---------------------------------------------------------------------------


def _shot_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return shot mappings from a list-shaped planning manifest.

    The renderer only accepts a list-shaped ``shots`` entry, so any other shape
    (including dict-shaped shots and a missing key) is refused here instead of
    reporting a dry-run plan that ``run`` would block.
    """
    shots = manifest.get("shots")
    if not isinstance(shots, list):
        raise CliError(
            "manifest 'shots' must be a list; the renderer refuses "
            f"{type(shots).__name__}-shaped shots"
        )
    return [shot for shot in shots if isinstance(shot, dict)]


def _phase_complete(shot: Mapping[str, Any], phase: str) -> bool:
    """Return ``True`` when ``shot`` has a completed, referenced phase asset."""
    assets = shot.get("assets")
    if not isinstance(assets, dict):
        return False
    status = assets.get(f"{phase}_status")
    reference = assets.get(f"{phase}_ref")
    return status == "completed" and isinstance(reference, str) and bool(reference)


def _count_pending(shots: Sequence[Mapping[str, Any]], phase: str) -> int:
    """Count shots whose ``phase`` asset is not yet complete."""
    return sum(1 for shot in shots if not _phase_complete(shot, phase))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _emit(payload: Mapping[str, Any]) -> None:
    """Write a JSON result to stdout."""
    print(json.dumps(payload, indent=2, default=str))


def _fail(message: str) -> None:
    """Write a diagnostic to stderr."""
    print(f"error: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _handle_probe(args: argparse.Namespace) -> int:
    """Probe ComfyUI and the GPU. Strictly read-only."""
    config = _resolve_gpu_config()
    monitor = _build_monitor()
    client = _build_client(args.endpoint)
    result = client.probe(
        workflow_paths=_workflow_paths(config),
        vram_sampler=monitor.sample_vram,
    )
    _emit(dataclasses.asdict(result))
    return EXIT_SUCCESS


def _handle_run(args: argparse.Namespace) -> int:
    """Render a job end to end (the only command that can spawn ComfyUI)."""
    from pixelle_video.cat_factory.gpu.renderer import GpuRenderer

    config = _resolve_gpu_config()
    renderer = GpuRenderer(
        lease=_build_lease(config, args.gpu_id),
        monitor=_build_monitor(),
        client=_build_client(args.endpoint),
        supervisor=_build_supervisor(args.job, config),
        gpu_config=config,
        output_root=_output_root(args),
    )
    report = renderer.render(args.job)
    _emit(report.to_dict())
    return EXIT_SUCCESS


def _handle_resume(args: argparse.Namespace) -> int:
    """Resume an interrupted render from its last completed checkpoint."""
    from pixelle_video.cat_factory.gpu.renderer import GpuRenderer

    config = _resolve_gpu_config()
    renderer = GpuRenderer(
        lease=_build_lease(config, args.gpu_id),
        monitor=_build_monitor(),
        client=_build_client(args.endpoint),
        supervisor=_build_supervisor(args.job, config),
        gpu_config=config,
        output_root=_output_root(args),
    )
    report = renderer.resume(args.job)
    _emit(report.to_dict())
    return EXIT_SUCCESS


def _handle_status(args: argparse.Namespace) -> int:
    """Report render status without acquiring a lease or contacting ComfyUI."""
    from pixelle_video.cat_factory.gpu.renderer import GpuRenderer

    config = _resolve_gpu_config()
    renderer = GpuRenderer(
        lease=_build_lease(config, args.gpu_id),
        monitor=_build_monitor(),
        client=_build_client(args.endpoint),
        gpu_config=config,
        output_root=_output_root(args),
    )
    report = renderer.status(args.job)
    _emit(report.to_dict())
    return EXIT_SUCCESS


def _handle_release(args: argparse.Namespace) -> int:
    """Release only the lease owned by this process; never force or recover."""
    if not args.owned_only:
        _fail("release requires --owned-only; refusing")
        return EXIT_REFUSED

    config = _resolve_gpu_config()
    lease = _build_lease(config, args.gpu_id)
    if lease.held():
        lease.release()
        _emit({"released": True, "job_id": args.job})
        return EXIT_SUCCESS

    _emit({"released": False, "reason": _RELEASE_REFUSAL_REASON})
    return EXIT_REFUSED


def _handle_dry_run(args: argparse.Namespace) -> int:
    """Explain what a render would do without touching the GPU or ComfyUI.

    This handler imports no ComfyUI client, supervisor, monitor or lease code,
    so it cannot post to ComfyUI, spawn a process, terminate a process, sample
    the GPU or acquire a lease.
    """
    from pixelle_video.cat_factory.storage import (
        is_planning_complete,
        read_manifest,
    )

    config = _resolve_gpu_config()
    job_dir = _job_dir(getattr(args, "output_root", None), args.job)
    manifest = read_manifest(job_dir)
    if not is_planning_complete(manifest):
        _fail(
            f"planning is not complete for job {args.job!r}; refusing to report "
            "a dry-run plan"
        )
        return EXIT_REFUSED

    shots = _shot_entries(manifest)
    payload = {
        "job_id": args.job,
        "would_render": True,
        "image_workflow": getattr(config, "image_workflow", "") or "",
        "reference_workflow": getattr(config, "reference_workflow", "") or "",
        "video_workflow": getattr(config, "video_workflow", "") or "",
        "shots_needing_keyframe": _count_pending(shots, KEYFRAME_PHASE),
        "shots_needing_video": _count_pending(shots, VIDEO_PHASE),
        "destructive_calls": [],
        "will_post": False,
        "will_spawn": False,
        "will_terminate": False,
    }
    _emit(payload)
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Parser and entry point
# ---------------------------------------------------------------------------


def _add_job_argument(parser: argparse.ArgumentParser) -> None:
    """Add the required ``--job`` option to a subparser."""
    parser.add_argument(
        "--job",
        required=True,
        metavar="ID",
        help="job identifier under the Cat Factory output root",
    )


def _add_endpoint_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional ``--endpoint`` option to a subparser."""
    parser.add_argument(
        "--endpoint",
        default=None,
        metavar="URL",
        help="ComfyUI base URL (default: http://127.0.0.1:8188)",
    )


def _add_output_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional ``--output-root`` option to a subparser."""
    parser.add_argument(
        "--output-root",
        default=None,
        metavar="PATH",
        help="override the Cat Factory output root",
    )


def _add_gpu_id_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional ``--gpu-id`` option to a subparser."""
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=DEFAULT_GPU_ID,
        metavar="N",
        help=f"GPU index used for the lease (default: {DEFAULT_GPU_ID})",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Cat Factory GPU CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m pixelle_video.cat_factory.gpu.cli",
        description=(
            "Safe operator CLI for local ComfyUI image/video orchestration. "
            "probe and dry-run never modify state."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe",
        help="probe ComfyUI and GPU availability (read-only)",
        description="Probe ComfyUI, its workflows and VRAM without changing state.",
    )
    _add_endpoint_argument(probe)

    run = subparsers.add_parser(
        "run",
        help="render a job from scratch",
        description="Render a job end to end, acquiring the GPU lease.",
    )
    _add_job_argument(run)
    _add_endpoint_argument(run)
    _add_output_root_argument(run)
    _add_gpu_id_argument(run)

    resume = subparsers.add_parser(
        "resume",
        help="resume an interrupted render",
        description="Resume a job from its last completed checkpoint.",
    )
    _add_job_argument(resume)
    _add_endpoint_argument(resume)
    _add_output_root_argument(resume)
    _add_gpu_id_argument(resume)

    status = subparsers.add_parser(
        "status",
        help="show render status (read-only)",
        description="Report render status without a lease, spawn or ComfyUI call.",
    )
    _add_job_argument(status)
    _add_endpoint_argument(status)
    _add_output_root_argument(status)
    _add_gpu_id_argument(status)

    release = subparsers.add_parser(
        "release",
        help="release the lease owned by this process",
        description="Release only a lease owned by this process; never forces.",
    )
    _add_job_argument(release)
    release.add_argument(
        "--owned-only",
        action="store_true",
        help="required: refuse to touch a lease owned by another process",
    )
    _add_gpu_id_argument(release)

    dry_run = subparsers.add_parser(
        "dry-run",
        help="explain a render with zero destructive calls",
        description=(
            "Report pending shots from the planning manifest without posting, "
            "spawning, terminating or acquiring a lease."
        ),
    )
    _add_job_argument(dry_run)
    _add_output_root_argument(dry_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    ``argparse`` handles usage errors (exit code ``2``). Runtime failures are
    classified into safe refusals (``3``) and unexpected failures (``1``); no
    traceback is printed for expected error types.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from pixelle_video.cat_factory.gpu.comfy_client import ComfyClientError
        from pixelle_video.cat_factory.gpu.renderer import (
            RenderBlocked,
            RenderError,
            RenderStateError,
        )
    except Exception as exc:  # noqa: BLE001 - dependency import must not traceback
        _fail(f"failed to load GPU orchestration dependencies: {exc}")
        return EXIT_FAILURE

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "probe": _handle_probe,
        "run": _handle_run,
        "resume": _handle_resume,
        "status": _handle_status,
        "release": _handle_release,
        "dry-run": _handle_dry_run,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown command: {args.command!r}")

    try:
        return handler(args)
    except (CliError, RenderBlocked, ComfyClientError) as exc:
        _fail(str(exc))
        return EXIT_REFUSED
    except (RenderError, RenderStateError, OSError, ValueError) as exc:
        _fail(str(exc))
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001 - never leak a traceback by default
        _fail(f"unexpected failure: {exc}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
