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

"""
Cat Factory extension package.

Houses the validated planning contract models (and, in later tasks, the
provider resolver, prompt templates, and pipeline) for the opt-in cat-parody
feature.
"""

from pixelle_video.cat_factory.models import (
    AudioPlan,
    CreativeBrief,
    InspirationCandidate,
    ParodyIdea,
    ShotPlan,
)

__all__ = [
    "CreativeBrief",
    "InspirationCandidate",
    "ParodyIdea",
    "ShotPlan",
    "AudioPlan",
]
