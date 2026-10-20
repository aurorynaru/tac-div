# Step 03 — Configurable Cloud LLM Integration

**Order:** implement first after `00-START-HERE.md` baseline.
**Objective:** make the Cat Factory's planning LLM a **configurable, OpenAI-compatible cloud provider** with a validated structured-JSON contract, bounded retries, budget guardrails, and a durable `planning_complete` handoff — while reusing the stock Pixelle LLM stack unchanged.

> **Dependency change:** TabbyAPI / local Qwen is **no longer a runtime dependency of this step**. Any OpenAI-compatible endpoint (hosted or self-hosted) may be configured; the default profile targets a hosted cloud provider. GPU ownership and any local model lifecycle belong to Step 04 and are out of scope here.

## Why this is mostly configuration

Inspected stock paths (verify against the actual checkout before editing):

- [`pixelle_video/services/llm_service.py`](../../pixelle_video/services/llm_service.py) already implements `LLMService` on the OpenAI SDK (`AsyncOpenAI`). It supports `base_url`, `model`, and `response_type` (Pydantic structured output via a JSON-schema instruction appended to the prompt, then parsed).
- [`pixelle_video/config/schema.py`](../../pixelle_video/config/schema.py) already has `llm.api_key`, `llm.base_url`, `llm.model` and the opt-in `cat_factory.enabled` flag added in Step 00.

Therefore Step 03 **does not fork the OpenAI SDK internals**. It configures `LLMService` for a cloud provider and adds a thin, provider-agnostic profile layer plus the Cat Factory's own schema/prompt/tests. If `LLMService` already satisfies a requirement, use it directly and do not duplicate its JSON-schema prompting or parsing.

## Provider abstraction

The provider is selected **by configuration**, not by code branches hardcoded per vendor.

- Reuse stock `LLMService` for every call. Do **not** subclass or copy `_create_client`, `_call_with_structured_output`, `_get_json_schema_instruction`, or `_parse_response_as_model`.
- A provider is a named profile resolving to: `provider` id, `base_url`, `model`, `timeout_s`, `max_retries`.
- Only introduce a thin provider/profile resolver (in `pixelle_video/cat_factory/`) **if** `LLMService`'s dynamic `config_manager`-based lookup cannot express the Cat Factory profile. The resolver maps a profile to the `(api_key, base_url, model)` arguments that are already accepted by `LLMService.__call__`, so no new transport code is added.
- Any OpenAI-compatible provider is accepted on equal terms: a profile is configuration data, not a supported-vendor allowlist.

### Suggested profile shape

```yaml
cat_factory:
  enabled: false
  llm:
    provider: "openai_compatible"   # logical id, not a vendor branch
    model: "<provider model id>"
    base_url_env: CAT_FACTORY_LLM_BASE_URL   # env var NAME, not the value
    api_key_env: CAT_FACTORY_LLM_API_KEY     # env var NAME, not the value
    timeout_s: 60
    max_retries: 2
```

`base_url_env` / `api_key_env` hold the **name** of an environment variable (or a key into a private, untracked config file), never the secret itself. A profile may also point at the existing `llm.base_url` / `llm.api_key` for users who configure one global endpoint.

## Credentials

- Credentials come **only** from environment variables (e.g. `CAT_FACTORY_LLM_API_KEY`, `CAT_FACTORY_LLM_BASE_URL`) or a **private, untracked** config file that is git-ignored.
- Never hardcode a key, never commit a key, never echo a key. `.gitignore` must cover the private config path; the example file contains placeholders only.
- **Sanitize logs.** Never log the API key, the `Authorization` header, or full request bodies that may embed secrets. Log the provider id and model name only; redact anything that could carry a credential.
- Missing credentials for a configured profile is a **fail-closed actionable configuration error** naming the expected env var — never a silent fallback to an unauthenticated or different endpoint.

## Cat Factory config additions

Extend `CatFactoryConfig` (or the Cat Factory config subtree) with an `llm` profile and explicit budget guardrails. Defaults are safe and **local/offline-friendly**; nothing paid turns on by default.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `llm.provider` | str | `"openai_compatible"` | Logical provider/profile id |
| `llm.model` | str | `""` | Model id requested from the provider |
| `llm.base_url_env` | str | `"CAT_FACTORY_LLM_BASE_URL"` | Env var name holding the base URL |
| `llm.api_key_env` | str | `"CAT_FACTORY_LLM_API_KEY"` | Env var name holding the API key |
| `llm.timeout_s` | int/float | `60` | Per-call timeout |
| `llm.max_retries` | int | `2` | Bounded transient retries |
| `guardrails.max_requests_per_job` | int | `0` (unlimited) | Hard cap on LLM requests per job |
| `guardrails.max_tokens_per_job` | int | `0` (unlimited) | Hard cap on tokens per job |
| `guardrails.cost_ceiling_usd` | float | `0.0` (unlimited) | Soft budget ceiling (job aborts when exceeded) |
| `guardrails.allow_paid_models` | bool | `false` | Must be `true` before any paid model is used |

Rules:

- There is **no silent switch to a paid model.** If a profile would incur cost and `allow_paid_models` is false, the job stops before the first request with an actionable message.
- On a budget breach (`max_requests_per_job`, `max_tokens_per_job`, or `cost_ceiling_usd`), stop and raise an actionable error that names the exceeded limit, the observed value, and how to raise it in config. Do not continue, and do not downgrade/switch providers silently.
- Guardrail counters are per job and persisted with the manifest so a resume does not reset the budget.

## Structured JSON contract

Define the planning Pydantic models in `pixelle_video/cat_factory/models.py`:

- `CreativeBrief`
- `InspirationCandidate`
- `ParodyIdea`
- `ShotPlan`
- `AudioPlan`

Rules:

- Call the reused `LLMService` with `response_type=<Model>` so structured output is validated as a Pydantic model. Never accept a raw dict as a substitute for the typed model.
- **Exactly ONE** JSON-repair retry on parse/validation failure, re-sending the **original schema** (same `response_type`). Never relax the schema for the retry.
- After the single repair attempt fails, raise an **explicit failure with a logged, sanitized reason** (which field/why). Never invent required fields, never fabricate values to satisfy the schema, never loop.
- Enforce the Cat Factory limits from the plan: cap duration, shot count, prompt length, and total candidate count as field constraints/validators.
- Field-level `Field(..., description=...)` hints are the schema the model is shown; keep them stable so the prompt and validator agree.

## Planning prompt templates

Planning prompts are **templates reused across providers**, structured as: ROLE / INPUTS / RULES / RETURN-ONLY-JSON.

```text
ROLE: You are a planner for short, original, realistic cat-comedy parodies.
INPUTS: CreativeBrief JSON, discovered reference metadata, cat character constraints.
RULES:
  - Use references only for high-level comic patterns. Do NOT reproduce
    copyrighted dialogue, copyrighted audio, exact scene footage, or creator watermarks.
  - One visual joke in 6–15 seconds, 1–4 shots, realistic tabby anatomy,
    absurd but readable action.
  - Select audio_mode from: none, caption_only, sfx_only, tts.
  - State scene duration, keyframe prompt, motion prompt, caption,
    visual constraints, rationale, and source provenance.
OUTPUT: Return ONLY a JSON object matching the provided schema. No prose, no markdown.
```

Templates live beside the models (e.g. a template module under `pixelle_video/cat_factory/`) and are provider-neutral — do not branch template text by provider id.

## Retry, timeout and error handling

- **Transient-only bounded retries with backoff.** Retry only clearly transient failures (timeouts, connection resets, HTTP 429/5xx). Do not retry deterministic 4xx (bad request, auth, invalid model) or schema-validation failures — those use the single JSON-repair path above, not the transient retry budget.
- **Per-call timeout** from `llm.timeout_s`, passed through to the client/request.
- **Cancellation:** an in-flight call must be cancellable (async task/`asyncio` cancellation) so planning can be aborted cleanly before writing a manifest.
- **Explicit error taxonomy.** Distinguish at minimum: `config_error` (missing key/endpoint), `auth_error`, `rate_limited`, `timeout`, `provider_error`, `transport_error`, `parse_error`, `validation_error`, `budget_exceeded`. Each carries a sanitized message and is recorded in the job's `errors` list.
- Fail closed: an unclassifiable or repeated failure stops the job rather than returning a partial/invalid plan.

## planning_complete handoff

- Planning writes the shared versioned manifest to `output/cat_factory/<job_id>/manifest.json` using **atomic replacement** (write-temp-then-rename), following the single manifest contract in [`00-START-HERE.md`](00-START-HERE.md). Do **not** invent a second JSON shape; extend the shared fields.
- Planning sets `state` to `planned`, fills `brief`, `inspiration`, `selected_idea`, `cat_character_id`, `audio_mode`, `shots`, and `artifact_paths`, and records its LLM attempts/counters under the shared `attempts`/`errors` fields.
- After the manifest commit, emit a **`planning_complete` checkpoint** — the durable handoff consumed by **Step 04** (`04-gpu-orchestration.md`). All in-flight LLM calls must be awaited/closed before the checkpoint is emitted, so Step 04 observes a quiescent LLM phase.
- The checkpoint must be idempotent: re-running planning for a completed job does not corrupt or duplicate the manifest, and Step 04 can rely on `state`/checkpoint presence to know planning finished.

## Tests / acceptance criteria

Offline unit tests (mocked OpenAI-compatible responses, **no network**):

1. Plain text response (no JSON) → explicit parse failure, no silent object.
2. Valid JSON response → parsed and validated typed model.
3. Code-fenced JSON response → parsed per `LLMService`'s code-block extraction.
4. Malformed JSON response → one repair retry, then explicit failure with logged reason.
5. Schema-invalid JSON response (missing/wrong fields) → one repair retry, then explicit failure; **no invented fields**.

Optional live integration test:

- A rules-bounded live test requiring an env-var key; it is **skipped when the key is unset**. It must respect the guardrails (request/token/cost caps), make a minimal number of calls, and never print secrets or full prompts.

### Acceptance criteria

- **AC-03.1** A planning call succeeds against a **configured OpenAI-compatible cloud provider** whose `base_url`/`model`/credentials come from config + environment (no TabbyAPI or local Qwen required).
- **AC-03.2** Structured output is validated as a Pydantic model; malformed and schema-invalid responses fail clearly after **exactly one** bounded JSON-repair retry, and no required field is ever invented.
- **AC-03.3** Planning produces a persisted, versioned manifest (shared `00-START-HERE.md` shape) and emits a `planning_complete` checkpoint; no subsequent phase needs the LLM for basic rendering.
- **AC-03.4** Credentials exist only in environment variables or private untracked config; no secret is logged or committed (sanitized logs verified).
- **AC-03.5** Cat mode is opt-in; the configured provider profile is data-driven, no silent switch to a paid model occurs, and a budget breach stops with an actionable error.
- **AC-03.6** The five offline tests run with no network, no GPU, and no provider service available.

## Definition of done and handoff

- Commit only this step's docs/tests and the minimal Cat Factory config/models/templates changes; no edits to `LLMService` internals.
- Record config defaults, the configured provider id/model (never the key), and test evidence in `docs/cat_factory/IMPLEMENTATION_STATUS.md`, with `status: accepted | blocked`.
- Report the exact env-var names expected for credentials, plus any provider/API incompatibility, **without** copying secrets.
- Next: **Step 04 — `04-gpu-orchestration.md`** consumes the `planning_complete` checkpoint. Do not implement GPU lifecycle, process control, or local model management here.

## OpenCode task prompt

> Implement only `03-llm-integration.md` in the stock Pixelle-Video checkout, respecting `00-START-HERE.md`. Reuse the stock `LLMService` unchanged; do not fork its OpenAI SDK internals. Add the Cat Factory config profile + guardrails, the Pydantic planning models, provider-neutral prompt templates, bounded retry/timeout/error taxonomy, the `planning_complete` handoff, and the five offline tests plus the skipped-when-unset live test. Do not create or modify any `.py` file beyond what this spec lists in the Cat Factory extension, do not touch GPU/ComfyUI/TabbyAPI, and do not log or commit secrets. Report changed files, config defaults, the env-var names, test evidence and precise blockers; commit the isolated change once passing.
