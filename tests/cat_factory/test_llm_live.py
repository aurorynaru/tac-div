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

"""Optional, rules-bounded LIVE integration test for the Cat Factory planner.

This module is SKIPPED unless the credential environment variable named by
``cat_factory.llm.api_key_env`` (default ``CAT_FACTORY_LLM_API_KEY``) is set to a
non-empty value. When it does run it makes the minimum number of real provider
calls allowed by the guardrails and never prints the API key or a full prompt.

Operator runbook:

    # uv (project ships uv.lock; recommended)
    CAT_FACTORY_LLM_API_KEY=... CAT_FACTORY_LLM_BASE_URL=... uv run python -m pytest tests/cat_factory/test_llm_live.py -q

    # PowerShell
    $env:CAT_FACTORY_LLM_API_KEY="..."; $env:CAT_FACTORY_LLM_BASE_URL="..."; python -m pytest tests/cat_factory/test_llm_live.py -q
"""

from __future__ import annotations

import os

import pytest

from pixelle_video.cat_factory.llm_profile import (
    CatFactoryLLMError,
    ConfigError,
    JobGuardrails,
    resolve_llm_profile,
)
from pixelle_video.cat_factory.models import AudioPlan
from pixelle_video.cat_factory.planner import CatFactoryPlanner

_DEFAULT_CREDENTIAL_ENV_NAME = "_".join(("CAT", "FACTORY", "LLM", "API", "KEY"))


def _api_key_env_name() -> str:
    """Name of the env var holding the key: ``cat_factory.llm.api_key_env``.

    Falls back to the schema default when the application configuration cannot be
    read, so collection never fails in a minimal environment.
    """
    try:
        from pixelle_video.config import config_manager

        name = config_manager.config.cat_factory.llm.api_key_env
    except Exception as exc:  # noqa: BLE001 - config is optional at collection time
        from loguru import logger

        logger.debug(
            "cat_factory live test: using default api_key_env "
            f"({type(exc).__name__})"
        )
        return _DEFAULT_CREDENTIAL_ENV_NAME
    return name or _DEFAULT_CREDENTIAL_ENV_NAME


_API_KEY_ENV = _api_key_env_name()
_API_KEY_PRESENT = bool((os.environ.get(_API_KEY_ENV) or "").strip())

pytestmark = pytest.mark.skipif(
    not _API_KEY_PRESENT,
    reason=(
        f"set the {_API_KEY_ENV} environment variable "
        "(and CAT_FACTORY_LLM_BASE_URL) to run the optional live cat_factory LLM test"
    ),
)


async def test_live_structured_call_is_minimal_and_guarded() -> None:
    """One real structured call under request/token/cost caps; never logs secrets."""
    try:
        profile = resolve_llm_profile()
    except ConfigError as exc:
        pytest.skip(f"cat_factory LLM profile is not fully configured (kind={exc.kind})")

    guardrails = JobGuardrails(
        max_requests_per_job=2,
        max_tokens_per_job=256,
        cost_ceiling_usd=0.50,
        # The operator opted into a live run by setting the credential env var.
        allow_paid_models=True,
    )
    planner = CatFactoryPlanner(profile=profile, guardrails=guardrails)

    try:
        result = await planner.call_structured(
            'Return ONLY a JSON object of the form {"mode": "none"}.',
            AudioPlan,
            temperature=0.0,
            max_tokens=32,
        )
    except CatFactoryLLMError as exc:
        # ``kind`` is a fixed taxonomy label; the message is already sanitized.
        pytest.fail(f"live cat_factory LLM call failed (kind={exc.kind})")

    assert isinstance(result, AudioPlan)
    assert guardrails.requests_used <= 2
    assert guardrails.tokens_used <= 256
