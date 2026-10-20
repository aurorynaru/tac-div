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

"""Offline integration tests for the Step 03 Cat Factory cloud-LLM planner.

These five tests are fully hermetic: no network, no GPU, and no provider. They
drive the REAL ``LLMService`` parsing/validation by patching only the client
construction seam (``LLMService._create_client``) via the ``fake_llm`` fixture.

Operator runbook (install the dev extra, then run the suite):

    # uv (the project ships uv.lock; recommended)
    uv pip install -e ".[dev]"
    uv run python -m pytest tests/cat_factory -q

    # plain pip fallback
    python -m pip install -e ".[dev]"
    python -m pytest tests/cat_factory -q

The live (opt-in) test lives in ``tests/cat_factory/test_llm_live.py`` and is
skipped unless its credential env var is set.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import pytest

from pixelle_video.cat_factory.llm_profile import (
    LLMValidationError,
    ParseError,
    ResolvedLLMProfile,
)
from pixelle_video.cat_factory.models import AudioMode, AudioPlan
from pixelle_video.cat_factory.planner import CatFactoryPlanner


async def test_plain_text_response_raises_explicit_parse_failure(
    planner: CatFactoryPlanner,
    fake_llm: Callable[[list[str]], Any],
) -> None:
    """A non-JSON response fails closed: an explicit ``ParseError``, never an object."""
    raw_text = "Sure! Here is a lovely cat video plan. Enjoy watching!"
    client = fake_llm([raw_text])

    with pytest.raises(ParseError):
        await planner.call_structured("Plan the audio.", AudioPlan)

    # Original request plus the single bounded JSON-repair retry; zero partials.
    assert len(client.calls) == 2


async def test_valid_json_response_parses_into_typed_model(
    planner: CatFactoryPlanner,
    llm_profile: ResolvedLLMProfile,
    fake_llm: Callable[[list[str]], Any],
    json_response: Callable[[Mapping[str, Any]], str],
) -> None:
    """A valid JSON response is parsed and validated into the typed contract model."""
    payload = {
        "mode": "sfx_only",
        "sfx_cues": [{"time_s": 1.5, "asset_id": "squeak", "gain_db": -6.0}],
    }
    client = fake_llm([json_response(payload)])

    result = await planner.call_structured("Plan the audio.", AudioPlan)

    assert isinstance(result, AudioPlan)
    assert result.mode is AudioMode.SFX_ONLY
    assert result.sfx_cues[0]["asset_id"] == "squeak"
    # The resolved profile (not any stock default) drove the single request.
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == llm_profile.model
    assert client.client_kwargs[0]["api_key"] == llm_profile.api_key
    assert client.client_kwargs[0]["base_url"] == llm_profile.base_url


async def test_code_fenced_json_response_is_extracted(
    planner: CatFactoryPlanner,
    fake_llm: Callable[[list[str]], Any],
    fenced_json_response: Callable[[Mapping[str, Any]], str],
) -> None:
    """A ```json ... ``` fenced response is extracted by LLMService's code-block path."""
    client = fake_llm([fenced_json_response({"mode": "none"})])

    result = await planner.call_structured("Plan the audio.", AudioPlan)

    assert isinstance(result, AudioPlan)
    assert result.mode is AudioMode.NONE
    assert len(client.calls) == 1


async def test_malformed_json_both_attempts_triggers_exactly_one_repair_retry(
    planner: CatFactoryPlanner,
    fake_llm: Callable[[list[str]], Any],
) -> None:
    """Malformed JSON on both attempts: exactly one repair retry, then an explicit failure."""
    first = '{"mode": "none",}'
    second = '{"mode": "none",]'
    client = fake_llm([first, second])

    with pytest.raises(ParseError) as excinfo:
        await planner.call_structured("Plan the audio.", AudioPlan)

    assert len(client.calls) == 2
    message = str(excinfo.value)
    assert "exactly one" in message
    assert "repair" in message


async def test_schema_invalid_both_attempts_fails_without_inventing_fields(
    planner: CatFactoryPlanner,
    fake_llm: Callable[[list[str]], Any],
    json_response: Callable[[Mapping[str, Any]], str],
) -> None:
    """Schema-invalid responses fail closed after one repair retry, inventing no fields."""
    invalid = json_response({"caption_text": "no mode provided"})  # missing required "mode"
    client = fake_llm([invalid, invalid])

    with pytest.raises(LLMValidationError) as excinfo:
        await planner.call_structured("Plan the audio.", AudioPlan)

    assert len(client.calls) == 2
    assert "No required field was invented" in str(excinfo.value)
