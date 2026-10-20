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

"""Cat Factory cloud-LLM planning service (Step 03).

This module is the only place in the Cat Factory extension that calls an LLM.
It reuses the stock :class:`pixelle_video.services.llm_service.LLMService`
**unchanged** (via composition, never subclassing or copying its internals) and
layers on the Step 03 planning contract:

- credentials, ``base_url`` and ``model`` come from the task 3.3 resolver
  ``resolve_llm_profile()`` and are passed as explicit ``api_key`` /
  ``base_url`` / ``model`` arguments, so the client's ``"dummy-key"`` fallback
  is never reached;
- structured output is requested with ``response_type=<Pydantic model>`` and
  validated as a typed model (never a raw dict);
- exactly ONE bounded JSON-repair retry re-sends the ORIGINAL schema (the same
  ``response_type``) and never relaxes it or invents required fields;
- only transient failures (timeout, connection reset, HTTP 429/5xx) are retried,
  with bounded backoff derived from ``cat_factory.llm.max_retries``;
- the per-call ``cat_factory.llm.timeout_s`` is enforced here (the stock call
  exposes no timeout argument) and the in-flight call stays cancellable;
- every failure is classified into exactly one of the nine error kinds from
  ``pixelle_video.cat_factory.llm_profile`` with a sanitized message;
- per-job budgets are enforced through the reused ``JobGuardrails`` and a
  cost-incurring profile is refused before the first request when
  ``allow_paid_models`` is false.

This module deliberately does NOT write the manifest (that is task 3.6): it
returns a :class:`PlanningResult` for the caller to persist.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    NoReturn,
    Sequence,
    TypeVar,
)

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from pixelle_video.cat_factory.llm_profile import (
    AuthError,
    BudgetExceededError,
    CatFactoryLLMError,
    ConfigError,
    JobGuardrails,
    LLMTimeoutError,
    LLMValidationError,
    ParseError,
    ProviderError,
    RateLimitedError,
    ResolvedLLMProfile,
    TransportError,
    check_profile_allowed,
    resolve_job_guardrails,
    resolve_llm_profile,
    sanitize_for_log,
)
from pixelle_video.cat_factory.models import (
    AudioPlan,
    CreativeBrief,
    InspirationCandidate,
    ParodyIdea,
)
from pixelle_video.cat_factory.prompts import (
    AUDIO_PLAN_PROMPT_TEMPLATE,
    INSPIRATION_PROMPT_TEMPLATE,
    PARODY_IDEA_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from pixelle_video.services.llm_service import LLMService


T = TypeVar("T", bound=BaseModel)

# Bounded backoff for transient retries: 0.5s, 1.0s, 2.0s, ... capped at 8s.
_BACKOFF_BASE_S: float = 0.5
_BACKOFF_MAX_S: float = 8.0

# Substrings (matched against the exception MRO class-name chain, lowercased)
# that identify transport-level failures we may retry.
_TRANSPORT_MARKERS: tuple[str, ...] = (
    "connection",
    "connecterror",
    "apiconnection",
    "remoteprotocol",
    "readerror",
    "transport",
    "network",
)


class _InspirationBatch(BaseModel):
    """Private response envelope for the inspiration stage.

    The inspiration stage asks the model for several candidates in one request,
    so it needs a list-shaped structured-output target. This envelope is a
    transport shape, not a planning contract; the individual candidates are the
    reused :class:`InspirationCandidate` models.
    """

    candidates: list[InspirationCandidate] = Field(
        default_factory=list,
        description="Ordered InspirationCandidate objects, most promising first.",
    )


class _SchemaRepairRequired(Exception):
    """Internal signal that a parse/validation failure allows one repair retry.

    Deliberately NOT part of the public nine-kind taxonomy: it means "issue the
    single bounded JSON-repair retry", not "the job failed".
    """

    def __init__(self, kind: str, reason: str, original: BaseException) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.original = original


@dataclass
class PlanningResult:
    """Outcome of a completed Cat Factory planning run.

    Carries the validated planning contracts plus the shared guardrail counters
    and the per-attempt/per-error trace. The caller (task 3.6) persists these to
    the manifest; this module never writes it.
    """

    brief: CreativeBrief
    inspiration: list[InspirationCandidate]
    selected_idea: ParodyIdea
    audio_plan: AudioPlan
    guardrails: JobGuardrails
    attempts: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure introspection helpers
# ---------------------------------------------------------------------------


def _exception_name_chain(exc: BaseException) -> str:
    """Return the class-name chain of ``exc`` (its MRO), space separated."""
    return " ".join(cls.__name__ for cls in type(exc).__mro__)


def _extract_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction that avoids importing any provider SDK."""
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    legacy = getattr(exc, "status", None)
    if isinstance(legacy, int):
        return legacy
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_transient_exception(exc: BaseException) -> bool:
    """Return True only for clearly transient failures (timeout/reset/429/5xx)."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    status = _extract_status_code(exc)
    if status == 429:
        return True
    if status is not None and status >= 400:
        # Deterministic 4xx (bad request, auth, invalid model) is NOT retried;
        # only 5xx is transient.
        return 500 <= status < 600
    lowered = _exception_name_chain(exc).lower()
    if "timeout" in lowered:
        return True
    return any(marker in lowered for marker in _TRANSPORT_MARKERS)


def _format_validation_error(exc: ValidationError) -> str:
    """Name the failing field(s) and why, without echoing model-controlled input."""
    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = error.get("loc") or ()
        field_name = ".".join(str(part) for part in location) if location else "<root>"
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{field_name} ({message})")
    if not parts:
        return "the response did not match the required schema"
    return "; ".join(parts)


def _schema_failure_from(exc: BaseException) -> _SchemaRepairRequired | None:
    """Map a parse/validation exception to a repair signal, or None if unrelated."""
    if isinstance(exc, ValidationError):
        return _SchemaRepairRequired(
            kind="validation_error",
            reason=_format_validation_error(exc),
            original=exc,
        )
    if isinstance(exc, json.JSONDecodeError):
        return _SchemaRepairRequired(
            kind="parse_error",
            reason="the response was not valid JSON",
            original=exc,
        )
    if isinstance(exc, ValueError):
        # The stock parser raises a plain ValueError when the content cannot be
        # parsed as the requested typed object.
        return _SchemaRepairRequired(
            kind="parse_error",
            reason="the response could not be parsed as the required typed object",
            original=exc,
        )
    return None


def _extract_usage_tokens(usage: Any) -> int | None:
    """Read ``total_tokens`` from a provider usage object/mapping when present."""
    if usage is None:
        return None
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int):
        return total
    if isinstance(usage, dict):
        value = usage.get("total_tokens")
        if isinstance(value, int):
            return value
    return None


# ---------------------------------------------------------------------------
# Candidate selection (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _candidate_score(candidate: InspirationCandidate) -> float:
    """Sum the numeric score components so selection is deterministic and offline."""
    return sum(
        float(value)
        for value in candidate.scores.values()
        if isinstance(value, (int, float))
    )


def _select_candidate(
    candidates: Sequence[InspirationCandidate],
) -> InspirationCandidate:
    """Select the highest-scoring candidate, stable on ties (first wins)."""
    best = candidates[0]
    best_score = _candidate_score(best)
    for candidate in candidates[1:]:
        score = _candidate_score(candidate)
        if score > best_score:
            best, best_score = candidate, score
    return best


# ---------------------------------------------------------------------------
# Prompt composition (template + JSON inputs; provider-neutral)
# ---------------------------------------------------------------------------


def _compose_inspiration_prompt(
    *,
    brief_json: str,
    reference_metadata: Sequence[Mapping[str, Any]] | None,
    max_candidates: int,
    cat_character_id: str | None,
) -> str:
    """Build the inspiration-stage prompt from the shared template plus inputs."""
    references = json.dumps(list(reference_metadata or []), ensure_ascii=False)
    character = cat_character_id or "(none; propose a new recurring cat character id)"
    limit = max(int(max_candidates), 1)
    return (
        INSPIRATION_PROMPT_TEMPLATE
        + "\n\nINPUTS:\n"
        + f"CreativeBrief JSON:\n{brief_json}\n"
        + f"Discovered reference metadata JSON:\n{references}\n"
        + f"cat_character_id: {character}\n"
        + "Return a JSON object with a single key 'candidates' holding an array of "
        + f"at most {limit} InspirationCandidate objects, ordered best first."
    )


def _compose_parody_idea_prompt(
    *,
    brief_json: str,
    selected: InspirationCandidate,
    cat_character_id: str | None,
) -> str:
    """Build the parody-idea prompt from one selected candidate and the brief."""
    character = cat_character_id or "(none; propose a new recurring cat character id)"
    return (
        PARODY_IDEA_PROMPT_TEMPLATE
        + "\n\nINPUTS:\n"
        + f"CreativeBrief JSON:\n{brief_json}\n"
        + f"Selected InspirationCandidate JSON:\n{selected.model_dump_json(indent=2)}\n"
        + f"cat_character_id: {character}"
    )


def _compose_audio_plan_prompt(
    *,
    brief_json: str,
    selected_idea: ParodyIdea,
) -> str:
    """Build the audio-plan prompt from the planned idea and the brief."""
    return (
        AUDIO_PLAN_PROMPT_TEMPLATE
        + "\n\nINPUTS:\n"
        + f"CreativeBrief JSON:\n{brief_json}\n"
        + f"Planned ParodyIdea JSON:\n{selected_idea.model_dump_json(indent=2)}"
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class CatFactoryPlanner:
    """Provider-agnostic planner built on the stock ``LLMService``.

    Collaborators are injectable so offline unit tests can supply a mocked
    service, a fixed profile, and fixed guardrails. When omitted, the
    task 3.3 resolver builds them from configuration on first use.
    """

    def __init__(
        self,
        profile: ResolvedLLMProfile | None = None,
        guardrails: JobGuardrails | None = None,
        service: LLMService | None = None,
    ) -> None:
        self._profile = profile
        self._guardrails = guardrails
        self._service = service
        self._attempts: list[dict] = []
        self._errors: list[dict] = []
        self._call_index = 0

    # -- public API ---------------------------------------------------------

    async def call_structured(
        self,
        prompt: str,
        response_type: type[T],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stage: str = "planning",
    ) -> T:
        """Call the reused ``LLMService`` for a typed, schema-validated response.

        Encapsulates guardrail pre-flight, the per-call timeout, transient-only
        retries, the single original-schema JSON-repair retry, and error
        classification. Raises a ``CatFactoryLLMError`` subclass on failure so
        the job fails closed.
        """
        self._ensure_ready()
        try:
            return await self._execute_with_transient_retries(
                prompt,
                response_type,
                temperature=temperature,
                max_tokens=max_tokens,
                stage=stage,
            )
        except asyncio.CancelledError:
            # Never swallow cancellation: the caller may be aborting the job.
            raise
        except _SchemaRepairRequired as first:
            repair_prompt = self._build_repair_prompt(prompt, first)
            logger.debug(
                sanitize_for_log(
                    "cat_factory LLM structured-output failure "
                    f"({first.kind}); issuing exactly one original-schema repair retry"
                )
            )
            try:
                return await self._execute_with_transient_retries(
                    repair_prompt,
                    response_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stage=f"{stage}:repair",
                )
            except asyncio.CancelledError:
                raise
            except _SchemaRepairRequired as second:
                error = self._final_schema_error(response_type, second)
                self._record_error(error.kind, str(error))
                raise error from second.original

    async def plan(
        self,
        brief: CreativeBrief,
        *,
        cat_character_id: str | None = None,
        reference_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> PlanningResult:
        """Run the three-stage planning flow and return the validated result.

        This module never writes the manifest; the caller persists the returned
        :class:`PlanningResult`. Any failure raises a classified
        ``CatFactoryLLMError`` (fail closed) rather than returning a partial plan.
        """
        self._reset_run()
        self._ensure_ready()
        if self._guardrails is None:
            self._raise_with_record(
                ConfigError("Cat Factory guardrails were not resolved.")
            )

        brief_json = brief.model_dump_json(indent=2)

        inspiration_prompt = _compose_inspiration_prompt(
            brief_json=brief_json,
            reference_metadata=reference_metadata,
            max_candidates=brief.max_candidates,
            cat_character_id=cat_character_id,
        )
        batch = await self.call_structured(
            inspiration_prompt,
            _InspirationBatch,
            stage="inspiration",
        )
        candidates = list(batch.candidates)
        if not candidates:
            self._raise_with_record(
                ProviderError(
                    "The inspiration stage returned no candidates; failing closed "
                    "rather than planning from an empty candidate set."
                )
            )

        selected = _select_candidate(candidates)
        idea_prompt = _compose_parody_idea_prompt(
            brief_json=brief_json,
            selected=selected,
            cat_character_id=cat_character_id,
        )
        selected_idea = await self.call_structured(
            idea_prompt,
            ParodyIdea,
            stage="parody_idea",
        )

        audio_prompt = _compose_audio_plan_prompt(
            brief_json=brief_json,
            selected_idea=selected_idea,
        )
        audio_plan = await self.call_structured(
            audio_prompt,
            AudioPlan,
            stage="audio_plan",
        )

        return PlanningResult(
            brief=brief,
            inspiration=candidates,
            selected_idea=selected_idea,
            audio_plan=audio_plan,
            guardrails=self._guardrails,
            attempts=list(self._attempts),
            errors=list(self._errors),
        )

    # -- readiness / guardrails --------------------------------------------

    def _reset_run(self) -> None:
        """Clear per-run traces so a reused planner starts from a clean slate."""
        self._attempts = []
        self._errors = []
        self._call_index = 0

    def _ensure_ready(self) -> None:
        """Resolve missing collaborators (lazily, avoiding import side effects)."""
        try:
            if self._profile is None:
                self._profile = resolve_llm_profile()
            if self._guardrails is None:
                self._guardrails = resolve_job_guardrails()
            if self._service is None:
                from pixelle_video.config import config_manager
                from pixelle_video.services.llm_service import LLMService

                self._service = LLMService(config_manager.config.model_dump())
        except ConfigError as exc:
            self._record_error(exc.kind, str(exc))
            raise
        self._enforce_paid_gate()

    def _enforce_paid_gate(self) -> None:
        """Refuse a cost-incurring profile before the first request when barred."""
        try:
            check_profile_allowed(self._profile, self._guardrails)
        except ConfigError as exc:
            self._record_error(exc.kind, str(exc))
            raise

    # -- request execution --------------------------------------------------

    async def _invoke_once(
        self,
        prompt: str,
        response_type: type[T],
        *,
        temperature: float,
        max_tokens: int,
        stage: str,
    ) -> T:
        """Issue exactly one physical LLM request under guardrails and timeout."""
        if (
            self._profile is None
            or self._guardrails is None
            or self._service is None
        ):
            self._raise_with_record(
                ConfigError("Cat Factory planner collaborators were not resolved.")
            )

        projected_tokens = max(int(max_tokens), 0)
        # Enforce request/token caps BEFORE the call using the request's
        # max_tokens as a conservative projection.
        self._guardrails.record_request(tokens=projected_tokens)

        logger.debug(
            sanitize_for_log(
                "cat_factory LLM request: "
                f"provider={self._profile.provider}, model={self._profile.model}, "
                f"stage={stage}, timeout_s={self._profile.timeout_s}, "
                f"max_tokens={max_tokens}"
            )
        )

        call = self._service(
            prompt,
            api_key=self._profile.api_key,
            base_url=self._profile.base_url,
            model=self._profile.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_type=response_type,
        )

        timeout_s = self._profile.timeout_s
        if timeout_s and float(timeout_s) > 0:
            result = await asyncio.wait_for(call, timeout=float(timeout_s))
        else:
            result = await call

        self._reconcile_usage(result, projected_tokens)
        return result

    async def _execute_with_transient_retries(
        self,
        prompt: str,
        response_type: type[T],
        *,
        temperature: float,
        max_tokens: int,
        stage: str,
    ) -> T:
        """Run ``_invoke_once`` with transient-only bounded retries.

        Parse/validation failures are re-raised as ``_SchemaRepairRequired``
        without consuming the transient retry budget, so the caller can issue
        the single JSON-repair retry instead.
        """
        max_retries = max(int(self._profile.max_retries), 0)
        total_attempts = max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                result = await self._invoke_once(
                    prompt,
                    response_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stage=stage,
                )
            except asyncio.CancelledError:
                raise
            except CatFactoryLLMError as exc:
                # Our own taxonomy (BudgetExceededError, ConfigError, ...) is
                # authoritative and must never be retried or reclassified.
                self._record_attempt(
                    stage=stage,
                    response_type=response_type,
                    outcome="error",
                    kind=exc.kind,
                    message=str(exc),
                )
                self._record_error(exc.kind, str(exc))
                raise
            except Exception as exc:
                schema_failure = _schema_failure_from(exc)
                if schema_failure is not None:
                    self._record_attempt(
                        stage=stage,
                        response_type=response_type,
                        outcome="error",
                        kind=schema_failure.kind,
                        message=schema_failure.reason,
                    )
                    self._record_error(schema_failure.kind, schema_failure.reason)
                    raise schema_failure from exc

                classified = self._classify_exception(exc)
                self._record_attempt(
                    stage=stage,
                    response_type=response_type,
                    outcome="error",
                    kind=classified.kind,
                    message=str(classified),
                )
                if _is_transient_exception(exc) and attempt < total_attempts:
                    logger.debug(
                        sanitize_for_log(
                            "cat_factory LLM transient failure "
                            f"({classified.kind}); retrying after backoff "
                            f"(attempt {attempt}/{total_attempts})"
                        )
                    )
                    await asyncio.sleep(self._backoff_delay(attempt))
                    continue
                self._record_error(classified.kind, str(classified))
                raise classified from exc
            else:
                self._record_attempt(
                    stage=stage,
                    response_type=response_type,
                    outcome="ok",
                )
                return result

        # Fail closed if the loop ever falls through without returning/raising.
        self._raise_with_record(
            ProviderError("cat_factory LLM request did not complete after retries.")
        )

    def _reconcile_usage(self, result: Any, projected_tokens: int) -> None:
        """Add any token usage the provider exposed beyond the projection, then check."""
        if self._guardrails is None:
            return
        actual_tokens = _extract_usage_tokens(getattr(result, "usage", None))
        if actual_tokens is None or actual_tokens <= projected_tokens:
            return
        self._guardrails.tokens_used += actual_tokens - projected_tokens
        self._guardrails.check()

    # -- classification -----------------------------------------------------

    def _classify_exception(self, exc: BaseException) -> CatFactoryLLMError:
        """Map any failure to exactly one of the nine sanitized error kinds."""
        provider = self._profile.provider if self._profile else "unknown"
        model = self._profile.model if self._profile else "unknown"
        status = _extract_status_code(exc)
        lowered = _exception_name_chain(exc).lower()

        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in lowered:
            timeout_s = self._profile.timeout_s if self._profile else "unknown"
            return LLMTimeoutError(
                f"LLM request timed out after {timeout_s}s "
                f"(provider={provider}, model={model})."
            )
        if (
            status in (401, 403)
            or "authentication" in lowered
            or "permissiondenied" in lowered
        ):
            return AuthError(
                f"LLM provider rejected the credentials (HTTP {status}); "
                f"verify the configured api_key for provider={provider}."
            )
        if status == 429 or "ratelimit" in lowered:
            return RateLimitedError(
                f"LLM provider rate limit reached (HTTP {status}) for model={model}."
            )
        if status is not None and status >= 400:
            return ProviderError(
                f"LLM provider returned HTTP {status} for "
                f"provider={provider}, model={model}."
            )
        if any(marker in lowered for marker in _TRANSPORT_MARKERS):
            return TransportError(
                f"LLM transport error for provider={provider}: "
                f"{type(exc).__name__}."
            )
        return ProviderError(
            f"Unclassified LLM failure for provider={provider}: {type(exc).__name__}."
        )

    def _final_schema_error(
        self,
        response_type: type[BaseModel],
        failure: _SchemaRepairRequired,
    ) -> CatFactoryLLMError:
        """Build the explicit failure raised after the single repair retry fails."""
        name = getattr(response_type, "__name__", str(response_type))
        if failure.kind == "validation_error":
            return LLMValidationError(
                f"Structured output failed schema validation for {name} after exactly "
                f"one bounded JSON-repair retry; failing field(s): {failure.reason}. "
                "No required field was invented."
            )
        return ParseError(
            f"Structured output could not be parsed as {name} after exactly one "
            f"bounded JSON-repair retry: {failure.reason}."
        )

    def _build_repair_prompt(self, prompt: str, failure: _SchemaRepairRequired) -> str:
        """Append a repair directive WITHOUT relaxing the original schema.

        The stock service re-appends the JSON schema for the same ``response_type``,
        so this only nudges the model to fix the named field(s).
        """
        reason = sanitize_for_log(failure.reason)
        return (
            prompt
            + "\n\nREPAIR: Your previous response was rejected by the required schema "
            f"({failure.kind}). Reason: {reason}. "
            "Return ONLY a JSON object that matches the SAME schema exactly, with every "
            "required field present and correctly typed. Do not add commentary or markdown."
        )

    # -- trace bookkeeping --------------------------------------------------

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff derived from the retry count, capped for safety."""
        delay = _BACKOFF_BASE_S * (2 ** max(attempt - 1, 0))
        return min(delay, _BACKOFF_MAX_S)

    def _record_attempt(
        self,
        *,
        stage: str,
        response_type: type[BaseModel],
        outcome: str,
        kind: str | None = None,
        message: str | None = None,
    ) -> None:
        """Record one physical request in the run's attempts trace (sanitized)."""
        self._call_index += 1
        self._attempts.append(
            {
                "index": self._call_index,
                "stage": stage,
                "response_type": getattr(response_type, "__name__", str(response_type)),
                "outcome": outcome,
                "kind": kind,
                "message": sanitize_for_log(message) if message else None,
            }
        )

    def _record_error(self, kind: str, message: str) -> dict:
        """Record one classified failure in the run's errors list (sanitized)."""
        entry = {"kind": kind, "message": sanitize_for_log(message)}
        self._errors.append(entry)
        return entry

    def _raise_with_record(self, error: CatFactoryLLMError) -> NoReturn:
        """Record a classified failure and raise it (fail closed)."""
        self._record_error(error.kind, str(error))
        raise error


__all__ = [
    "CatFactoryPlanner",
    "PlanningResult",
]
