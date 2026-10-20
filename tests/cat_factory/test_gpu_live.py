r"""Live (opt-in) GPU probe for the local ComfyUI endpoint.

Offline full-suite operator command::

    .venv\Scripts\python.exe -m pytest tests/cat_factory -q

The live probe is disabled unless the operator explicitly opts in::

    set CAT_FACTORY_GPU_LIVE=1
"""

from __future__ import annotations

import os

import pytest

from pixelle_video.cat_factory.gpu.comfy_client import ComfyClient, ProbeResult
from pixelle_video.cat_factory.gpu.monitor import GpuMonitor

pytestmark = pytest.mark.skipif(
    os.environ.get("CAT_FACTORY_GPU_LIVE") != "1",
    reason="live GPU probe disabled; set CAT_FACTORY_GPU_LIVE=1",
)


def test_live_read_only_probe():
    """Probe the live endpoint without submitting any inference request."""
    monitor = GpuMonitor()
    client = ComfyClient()

    result = client.probe(
        workflow_paths=(),
        vram_sampler=monitor.sample_vram,
    )

    assert isinstance(result, ProbeResult)
