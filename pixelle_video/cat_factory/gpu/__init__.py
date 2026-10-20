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

"""Cat Factory local single-GPU orchestration primitives.

This package exposes the GPU lease primitives used to serialize exclusive
access to a single physical GPU across cooperating Cat Factory jobs. It is a
pure-stdlib, fail-closed layer: it never spawns subprocesses, never imports a
GPU runtime, and never terminates processes.
"""

from __future__ import annotations

from .lease import (
    GpuLease,
    GpuLeaseError,
    GpuLeaseLost,
    GpuLeaseManager,
    GpuLeaseNotHeld,
    GpuLeaseUnavailable,
    LeaseRecord,
    StaleAssessment,
)

__all__ = [
    "GpuLease",
    "GpuLeaseError",
    "GpuLeaseLost",
    "GpuLeaseManager",
    "GpuLeaseNotHeld",
    "GpuLeaseUnavailable",
    "LeaseRecord",
    "StaleAssessment",
]
