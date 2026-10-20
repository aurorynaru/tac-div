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

"""Resolve the cat-factory cloud LLM profile and enforce per-job budgets.

This module resolves the cat-factory cloud profile to the
``(api_key, base_url, model)`` arguments accepted by the stock ``LLMService``,
enforces per-job budget guardrails, and never logs secrets. It is pure stdlib
plus ``pyyaml`` and imports no provider SDK (no ``openai``, ``comfykit``,
``torch``, ``requests``/``httpx``, and no ``pixelle_video.services.llm_service``).
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlsplit

import yaml

if TYPE_CHECKING:
    from pixelle_video.config.schema import CatFactoryGuardrailsConfig


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------

_REDACTION_MARKER = "***REDACTED***"

_SANITIZE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization: <anything to end of line> / Authorization=<anything>
    (
        re.compile(r"(?i)\b(authorization\s*[:=]\s*)([^\r\n]+)"),
        r"\1" + _REDACTION_MARKER,
    ),
    # Bearer <token>
    (
        re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/\-]+=*)"),
        r"\1" + _REDACTION_MARKER,
    ),
    # api_key=... / apikey=... / token=...
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token)\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
        ),
        r"\1=" + _REDACTION_MARKER,
    ),
    # sk- style provider keys
    (
        re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}\b"),
        _REDACTION_MARKER,
    ),
)


def redact_secret(value: str | None) -> str:
    """Return a redaction marker when a secret is present, else an empty string."""
    return _REDACTION_MARKER if value else ""


def sanitize_for_log(text: str) -> str:
    """Scrub credential-shaped substrings so they can never reach a log sink."""
    if not text:
        return ""
    sanitized = str(text)
    for pattern, replacement in _SANITIZE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class CatFactoryLLMError(Exception):
    """Base error for the cat-factory cloud LLM integration.

    The message is always sanitized before it is stored or rendered so that an
    ``api_key`` or ``Authorization`` header can never leak through ``str()`` or
    ``repr()``.
    """

    kind: str = "llm_error"

    def __init__(self, message: str) -> None:
        self.message = sanitize_for_log(str(message))
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind!r}, message={self.message!r})"


class ConfigError(CatFactoryLLMError):
    kind = "config_error"


class AuthError(CatFactoryLLMError):
    kind = "auth_error"


class RateLimitedError(CatFactoryLLMError):
    kind = "rate_limited"


class LLMTimeoutError(CatFactoryLLMError):
    kind = "timeout"


class ProviderError(CatFactoryLLMError):
    kind = "provider_error"


class TransportError(CatFactoryLLMError):
    kind = "transport_error"


class ParseError(CatFactoryLLMError):
    kind = "parse_error"


class LLMValidationError(CatFactoryLLMError):
    kind = "validation_error"


class BudgetExceededError(CatFactoryLLMError):
    kind = "budget_exceeded"


# Only ``ConfigError`` and ``BudgetExceededError`` are raised by this module in
# Step 03; the remaining kinds exist so the request-execution task (3.5) can
# reuse the same taxonomy without redefining it.
ERROR_KINDS: tuple[str, ...] = (
    "config_error",
    "auth_error",
    "rate_limited",
    "timeout",
    "provider_error",
    "transport_error",
    "parse_error",
    "validation_error",
    "budget_exceeded",
)


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

_LOCAL_CONFIG_FILENAME = "config.cat_factory.local.yaml"
_LOCAL_CONFIG_DISPLAY = f"<repo_root>/{_LOCAL_CONFIG_FILENAME}"


@dataclass(frozen=True)
class ResolvedLLMProfile:
    """A fully resolved, non-secret-leaking cloud LLM profile."""

    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_s: int
    max_retries: int
    paid: bool

    def __repr__(self) -> str:
        return (
            "ResolvedLLMProfile("
            f"provider={self.provider!r}, "
            f"api_key={redact_secret(self.api_key)!r}, "
            f"base_url={_host_from_base_url(self.base_url)!r}, "
            f"model={self.model!r}, "
            f"timeout_s={self.timeout_s!r}, "
            f"max_retries={self.max_retries!r}, "
            f"paid={self.paid!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def _repo_root() -> Path:
    """Return the repository root (two parents above this module file)."""
    return Path(__file__).resolve().parents[2]


def _local_config_path() -> Path:
    return _repo_root() / _LOCAL_CONFIG_FILENAME


def _load_local_config() -> Mapping[str, Any]:
    """Load the private local config overlay; absence means an empty mapping."""
    path = _local_config_path()
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"Failed to read local cat-factory config at {path}: "
            f"{type(exc).__name__}."
        ) from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Local cat-factory config at {path} must be a YAML mapping, "
            f"got {type(raw).__name__}."
        )
    return raw


def _resolve_setting(
    *,
    env_name: str,
    local_config: Mapping[str, Any],
    label: str,
) -> str:
    """Resolve a single credential/base-url setting with env-wins precedence.

    Precedence (authoritative spec, plan/03-llm-integration.md):
      1. Environment variable named ``env_name`` (used only when non-empty).
      2. The SAME name as a top-level key of ``config.cat_factory.local.yaml``.

    No other file (``config.yaml``, ``.env``, any other overlay) is consulted.
    """
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip():
        return env_value.strip()

    file_value = local_config.get(env_name)
    if isinstance(file_value, str) and file_value.strip():
        return file_value.strip()

    # Fail closed: never fall through to a dummy key and never switch endpoint.
    raise ConfigError(
        f"Missing {label}: set the environment variable {env_name!r} or add a "
        f"top-level key {env_name!r} to {_LOCAL_CONFIG_DISPLAY} "
        f"(repository-root {_LOCAL_CONFIG_FILENAME})."
    )


def resolve_llm_profile() -> ResolvedLLMProfile:
    """Resolve the cloud LLM profile from config plus the credential sources.

    The returned ``api_key``/``base_url``/``model`` triple is exactly what the
    stock ``LLMService`` accepts. A missing credential fails closed.

    Incompatibility recorded for Step 03: the spec's optional fallback to the
    global ``llm.base_url`` / ``llm.api_key`` values is DELIBERATELY NOT
    IMPLEMENTED here and is treated as an explicit unsupported behavior.
    """
    from pixelle_video.config import config_manager

    cfg = config_manager.config.cat_factory.llm
    local_config = _load_local_config()

    key = _resolve_setting(
        env_name=cfg.api_key_env,
        local_config=local_config,
        label="cat_factory LLM api_key",
    )
    base_url = _resolve_setting(
        env_name=cfg.base_url_env,
        local_config=local_config,
        label="cat_factory LLM base_url",
    )

    model = (cfg.model or "").strip()
    if not model:
        raise ConfigError(
            "Missing cat_factory.llm.model: set it in the project config.yaml "
            "under cat_factory.llm.model."
        )

    return ResolvedLLMProfile(
        provider=cfg.provider,
        api_key=key,
        base_url=base_url,
        model=model,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        paid=bool(cfg.paid),
    )


# ---------------------------------------------------------------------------
# Paid-model predicate
# ---------------------------------------------------------------------------


def _host_is_provably_local(host: str | None) -> bool:
    """Return True only when ``host`` is provably a local/private address."""
    if not host:
        return False
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    normalized = normalized.lower()
    if not normalized:
        return False
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_link_local or ip.is_private:
        return True
    if ip.version == 6 and ip in ipaddress.ip_network("fc00::/7"):
        return True
    return False


def _host_from_base_url(base_url: str | None) -> str | None:
    """Extract the hostname from a base URL, tolerating unparseable input."""
    if not base_url:
        return None
    try:
        return urlsplit(base_url).hostname
    except ValueError:
        return None


def is_cost_incurring(profile: ResolvedLLMProfile) -> bool:
    """Return True when using ``profile`` may incur cost.

    Fail-closed decision: an ABSENT or UNPARSEABLE host is NOT provably local,
    therefore it is treated as cost-incurring.
    """
    if profile.paid:
        return True
    return not _host_is_provably_local(_host_from_base_url(profile.base_url))


def check_profile_allowed(
    profile: ResolvedLLMProfile, guardrails: JobGuardrails
) -> None:
    """Stop before the first request when a cost-incurring profile is barred.

    Only the HOST is reported; any userinfo/credential segment of the URL is
    never included in the message.
    """
    if guardrails.allow_paid_models:
        return
    if not is_cost_incurring(profile):
        return

    host = _host_from_base_url(profile.base_url)
    if profile.paid:
        reason = "cat_factory.llm.paid is true"
    else:
        reason = f"base_url host {host!r} is not provably local"
    raise ConfigError(
        f"Refusing to use a cost-incurring LLM profile: {reason}. "
        "Set cat_factory.guardrails.allow_paid_models to true to permit it. "
        f"(host={host!r}; credentials are never included in this message.)"
    )


# ---------------------------------------------------------------------------
# Per-job budget guardrails
# ---------------------------------------------------------------------------


@dataclass
class JobGuardrails:
    """Per-job request/token/cost budget that round-trips through the manifest.

    A cap of ``0`` (or ``0.0``) means UNLIMITED for that dimension.
    """

    max_requests_per_job: int = 0
    max_tokens_per_job: int = 0
    cost_ceiling_usd: float = 0.0
    allow_paid_models: bool = False
    requests_used: int = 0
    tokens_used: int = 0
    cost_used_usd: float = 0.0

    @classmethod
    def from_config(cls, cfg: CatFactoryGuardrailsConfig) -> JobGuardrails:
        """Build a fresh (zeroed counter) guardrail set from config."""
        return cls(
            max_requests_per_job=int(getattr(cfg, "max_requests_per_job", 0) or 0),
            max_tokens_per_job=int(getattr(cfg, "max_tokens_per_job", 0) or 0),
            cost_ceiling_usd=float(getattr(cfg, "cost_ceiling_usd", 0.0) or 0.0),
            allow_paid_models=bool(getattr(cfg, "allow_paid_models", False)),
        )

    def _enforce_limits(
        self, requests_value: int, tokens_value: int, cost_value: float
    ) -> None:
        if self.max_requests_per_job > 0 and requests_value > self.max_requests_per_job:
            raise BudgetExceededError(
                "Per-job request budget exceeded: limit "
                f"cat_factory.guardrails.max_requests_per_job="
                f"{self.max_requests_per_job}, observed {requests_value} requests. "
                "Increase cat_factory.guardrails.max_requests_per_job to allow more."
            )
        if self.max_tokens_per_job > 0 and tokens_value > self.max_tokens_per_job:
            raise BudgetExceededError(
                "Per-job token budget exceeded: limit "
                f"cat_factory.guardrails.max_tokens_per_job="
                f"{self.max_tokens_per_job}, observed {tokens_value} tokens. "
                "Increase cat_factory.guardrails.max_tokens_per_job to allow more."
            )
        if self.cost_ceiling_usd > 0.0 and cost_value > self.cost_ceiling_usd:
            raise BudgetExceededError(
                "Per-job cost budget exceeded: limit "
                f"cat_factory.guardrails.cost_ceiling_usd="
                f"{self.cost_ceiling_usd}, observed {cost_value:.6f} USD. "
                "Increase cat_factory.guardrails.cost_ceiling_usd to allow more."
            )

    def record_request(
        self, *, tokens: int = 0, estimated_cost_usd: float = 0.0
    ) -> None:
        """Validate projected totals, then commit atomically.

        Raises ``BudgetExceededError`` BEFORE mutating any counter, so a failed
        validation leaves the guardrails untouched.
        """
        projected_requests = self.requests_used + 1
        projected_tokens = self.tokens_used + max(int(tokens), 0)
        projected_cost = self.cost_used_usd + max(float(estimated_cost_usd), 0.0)
        self._enforce_limits(projected_requests, projected_tokens, projected_cost)
        self.requests_used = projected_requests
        self.tokens_used = projected_tokens
        self.cost_used_usd = projected_cost

    def check(self) -> None:
        """Raise ``BudgetExceededError`` when any cap is already exceeded."""
        self._enforce_limits(self.requests_used, self.tokens_used, self.cost_used_usd)

    def to_dict(self) -> dict:
        """Serialize all fields (caps plus counters) for manifest persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobGuardrails:
        """Rebuild guardrails from a manifest mapping.

        Unknown keys are ignored and missing keys fall back to defaults so a
        resume can restore prior counters without resetting the budget.
        """
        if data is None:
            return cls()

        def as_int(key: str, default: int) -> int:
            try:
                return int(data.get(key, default))
            except (TypeError, ValueError):
                return default

        def as_float(key: str, default: float) -> float:
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            max_requests_per_job=as_int("max_requests_per_job", 0),
            max_tokens_per_job=as_int("max_tokens_per_job", 0),
            cost_ceiling_usd=as_float("cost_ceiling_usd", 0.0),
            allow_paid_models=bool(data.get("allow_paid_models", False)),
            requests_used=as_int("requests_used", 0),
            tokens_used=as_int("tokens_used", 0),
            cost_used_usd=as_float("cost_used_usd", 0.0),
        )


def resolve_job_guardrails() -> JobGuardrails:
    """Build per-job guardrails from the live cat_factory configuration."""
    from pixelle_video.config import config_manager

    cfg = config_manager.config.cat_factory.guardrails
    return JobGuardrails.from_config(cfg)
