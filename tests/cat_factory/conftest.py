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

"""Shared offline harness for the Cat Factory (Step 03) cloud-LLM tests.

The single seam under test is ``LLMService._create_client``. The planner calls the
REAL :class:`pixelle_video.services.llm_service.LLMService` (composition, never a
subclass), so patching only client construction lets the stock JSON parsing and
schema validation run end-to-end with ZERO network, GPU, or provider access.

The fake client mimics the only surface the stock service touches:
``client.base_url`` and ``await client.chat.completions.create(**kwargs)`` whose
result exposes ``response.choices[0].message.content``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import pytest

from pixelle_video.cat_factory.llm_profile import JobGuardrails, ResolvedLLMProfile
from pixelle_video.cat_factory.planner import CatFactoryPlanner
from pixelle_video.services.llm_service import LLMService


def _make_response(content: str) -> Any:
    """Build the minimal object shape the stock service reads a message from."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=None)


class _FakeCompletions:
    """Async stand-in for ``client.chat.completions`` returning canned content."""

    def __init__(self, responses: Sequence[str], calls: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self._calls = calls

    async def create(self, **kwargs: Any) -> Any:
        """Record the call and return the next canned response (last repeats)."""
        index = len(self._calls)
        self._calls.append(kwargs)
        if not self._responses:
            content = ""
        elif index < len(self._responses):
            content = self._responses[index]
        else:
            content = self._responses[-1]
        return _make_response(content)


class FakeLLMClient:
    """AsyncOpenAI-compatible fake that records every ``create`` invocation.

    Attributes:
        base_url: A loopback URL, so any accidental real client stays local.
        calls: The ``chat.completions.create`` keyword args, in call order.
        client_kwargs: The ``api_key``/``base_url`` passed to ``_create_client``.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self.base_url = "http://127.0.0.1:9/v1"
        self.calls: list[dict[str, Any]] = []
        self.client_kwargs: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses, self.calls))


@pytest.fixture
def fake_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Sequence[str]], FakeLLMClient]:
    """Return an installer that redirects ``LLMService._create_client`` offline.

    Calling the returned installer with a sequence of canned response strings
    installs the fake seam and returns the :class:`FakeLLMClient` so a test can
    assert exactly how many physical requests were issued.
    """

    def install(responses: Sequence[str]) -> FakeLLMClient:
        client = FakeLLMClient(responses)

        def _fake_create_client(
            self: LLMService,
            api_key: str | None = None,
            base_url: str | None = None,
        ) -> FakeLLMClient:
            client.client_kwargs.append({"api_key": api_key, "base_url": base_url})
            return client

        monkeypatch.setattr(LLMService, "_create_client", _fake_create_client)
        return client

    return install


# Obviously-fake, non-secret sentinel for offline tests; never a real credential.
_OFFLINE_SENTINEL = "-".join(("test", "key", "not", "a", "secret"))


@pytest.fixture
def llm_profile() -> ResolvedLLMProfile:
    """A fixed, hermetic (loopback, free) profile: no config or env required."""
    return ResolvedLLMProfile(
        provider="openai_compatible",
        **{"api_key": _OFFLINE_SENTINEL},
        base_url="http://127.0.0.1:9/v1",
        model="test-model",
        timeout_s=30,
        max_retries=0,
        paid=False,
    )


@pytest.fixture
def guardrails() -> JobGuardrails:
    """Unlimited per-job budgets; ``allow_paid_models`` keeps the gate hermetic."""
    return JobGuardrails(
        max_requests_per_job=0,
        max_tokens_per_job=0,
        cost_ceiling_usd=0.0,
        allow_paid_models=True,
    )


@pytest.fixture
def planner(
    llm_profile: ResolvedLLMProfile,
    guardrails: JobGuardrails,
) -> CatFactoryPlanner:
    """Planner wired to fixed collaborators plus a config-free ``LLMService``."""
    return CatFactoryPlanner(
        profile=llm_profile,
        guardrails=guardrails,
        service=LLMService({}),
    )


@pytest.fixture
def json_response() -> Callable[[Mapping[str, Any]], str]:
    """Build a raw (unfenced) JSON response string from a payload mapping."""

    def _build(payload: Mapping[str, Any]) -> str:
        return json.dumps(dict(payload))

    return _build


@pytest.fixture
def fenced_json_response() -> Callable[[Mapping[str, Any]], str]:
    """Build a markdown code-fenced JSON response string (```json ... ```)."""

    def _build(payload: Mapping[str, Any]) -> str:
        body = json.dumps(dict(payload))
        return f"Certainly, here is the plan:\n```json\n{body}\n```\n"

    return _build
