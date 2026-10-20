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
Configuration schema with Pydantic models

Single source of truth for all configuration defaults and validation.
"""
import pathlib
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLM configuration"""
    api_key: str = Field(default="", description="LLM API Key")
    base_url: str = Field(default="", description="LLM API Base URL")
    model: str = Field(default="", description="LLM Model Name")


class APIProviderCommonConfig(BaseModel):
    """Common API provider settings"""
    print_model_input: bool = Field(default=False, description="Print provider request parameters for debugging")
    local_proxy: str = Field(default="", description="Local HTTP proxy for providers that need it")


class APIKeyProviderConfig(BaseModel):
    """Provider settings with API key and optional base URL"""
    api_key: str = Field(default="", description="Provider API Key")
    base_url: str = Field(default="", description="Provider API Base URL")
    use_proxy: bool = Field(default=False, description="Route provider requests through common local proxy")


class AccessSecretProviderConfig(BaseModel):
    """Provider settings with access key / secret key credentials"""
    base_url: str = Field(default="", description="Provider API Base URL")
    access_key: str = Field(default="", description="Provider Access Key")
    secret_key: str = Field(default="", description="Provider Secret Key")
    use_proxy: bool = Field(default=False, description="Route provider requests through common local proxy")


class APIProvidersConfig(BaseModel):
    """Direct model provider API configuration"""
    common: APIProviderCommonConfig = Field(default_factory=APIProviderCommonConfig)
    openai: APIKeyProviderConfig = Field(default_factory=APIKeyProviderConfig)
    dashscope: APIKeyProviderConfig = Field(default_factory=APIKeyProviderConfig)
    deepseek: APIKeyProviderConfig = Field(default_factory=APIKeyProviderConfig)
    gemini: APIKeyProviderConfig = Field(default_factory=APIKeyProviderConfig)
    ark: APIKeyProviderConfig = Field(default_factory=APIKeyProviderConfig)
    kling: AccessSecretProviderConfig = Field(default_factory=AccessSecretProviderConfig)


class TTSLocalConfig(BaseModel):
    """Local TTS configuration (Edge TTS)"""
    voice: str = Field(default="zh-CN-YunjianNeural", description="Edge TTS voice ID")
    speed: float = Field(default=1.2, ge=0.5, le=2.0, description="Speech speed multiplier (0.5-2.0)")


class TTSComfyUIConfig(BaseModel):
    """ComfyUI TTS configuration"""
    default_workflow: Optional[str] = Field(default=None, description="Default TTS workflow (optional)")


class TTSSubConfig(BaseModel):
    """TTS-specific configuration (under comfyui.tts)"""
    inference_mode: str = Field(default="local", description="TTS inference mode: 'local' or 'comfyui'")
    local: TTSLocalConfig = Field(default_factory=TTSLocalConfig, description="Local TTS (Edge TTS) configuration")
    comfyui: TTSComfyUIConfig = Field(default_factory=TTSComfyUIConfig, description="ComfyUI TTS configuration")
    
    # Backward compatibility: keep default_workflow at top level
    @property
    def default_workflow(self) -> Optional[str]:
        """Get default workflow (for backward compatibility)"""
        return self.comfyui.default_workflow


class ImageSubConfig(BaseModel):
    """Image-specific configuration (under comfyui.image)"""
    default_workflow: Optional[str] = Field(default=None, description="Default image workflow (optional)")
    prompt_prefix: str = Field(
        default="Minimalist black-and-white matchstick figure style illustration, clean lines, simple sketch style",
        description="Prompt prefix for all image generation"
    )


class VideoSubConfig(BaseModel):
    """Video-specific configuration (under comfyui.video)"""
    default_workflow: Optional[str] = Field(default=None, description="Default video workflow (optional)")
    prompt_prefix: str = Field(
        default="Minimalist black-and-white matchstick figure style illustration, clean lines, simple sketch style",
        description="Prompt prefix for all video generation"
    )


class ComfyUIConfig(BaseModel):
    """ComfyUI configuration (includes global settings and service-specific configs)"""
    comfyui_url: str = Field(default="http://127.0.0.1:8188", description="ComfyUI Server URL")
    comfyui_api_key: Optional[str] = Field(default=None, description="ComfyUI API Key (optional)")
    runninghub_api_key: Optional[str] = Field(default=None, description="RunningHub API Key (optional)")
    runninghub_concurrent_limit: int = Field(default=1, ge=1, le=10, description="RunningHub concurrent execution limit (1-10)")
    runninghub_instance_type: Optional[str] = Field(default=None, description="RunningHub instance type (optional, set to 'plus' for 48GB VRAM)")
    tts: TTSSubConfig = Field(default_factory=TTSSubConfig, description="TTS-specific configuration")
    image: ImageSubConfig = Field(default_factory=ImageSubConfig, description="Image-specific configuration")
    video: VideoSubConfig = Field(default_factory=VideoSubConfig, description="Video-specific configuration")


class TemplateConfig(BaseModel):
    """Template configuration"""
    default_template: str = Field(
        default="1080x1920/default.html",
        description="Default frame template path"
    )


class CatFactoryLLMConfig(BaseModel):
    """Cloud LLM provider profile for the cat factory feature.

    The existing config models define no ``model_config``, so pydantic's
    default ``extra="ignore"`` silently drops unknown YAML keys. Every profile
    key must therefore be explicitly modelled here, otherwise a configured
    ``cat_factory.llm`` block would be silently unreadable.
    """
    provider: str = Field(default="openai_compatible", description="LLM provider profile identifier")
    model: str = Field(default="", description="Model name to request from the provider")
    base_url_env: str = Field(default="CAT_FACTORY_LLM_BASE_URL", description="Name of the environment variable (or private local config key) holding the LLM base URL, never the URL value itself")
    api_key_env: str = Field(default="CAT_FACTORY_LLM_API_KEY", description="Name of the environment variable (or private local config key) holding the LLM API key, never the secret value itself")
    timeout_s: int = Field(default=60, description="Per-request timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum number of retry attempts per request")
    paid: bool = Field(default=False, description="Explicit cost marker: True when this profile incurs paid provider usage")


class CatFactoryGuardrailsConfig(BaseModel):
    """Budget guardrails for the cat factory feature"""
    max_requests_per_job: int = Field(default=0, description="Maximum provider requests allowed per job (0 = unlimited)")
    max_tokens_per_job: int = Field(default=0, description="Maximum tokens allowed per job (0 = unlimited)")
    cost_ceiling_usd: float = Field(default=0.0, description="Maximum estimated cost in USD per job (0.0 = unlimited)")
    allow_paid_models: bool = Field(default=False, description="Whether profiles marked paid=True are permitted")


class CatFactoryGpuConfig(BaseModel):
    """Local single-GPU image/video orchestration configuration (ComfyUI only).

    Consumed by the ComfySupervisor and its durable state machine. All values
    are conservative defaults: an empty ``comfy_command`` disables GPU
    orchestration entirely (fail closed) rather than guessing an executable.
    """
    comfy_command: list[str] = Field(default_factory=list, description="ComfyUI launch argument array (executable + args); never a shell string; empty list means 'not configured' and downstream code must fail closed")
    supervisor_mode: Literal["managed", "attached"] = Field(default="managed", description="Supervisor mode: 'managed' launches/monitors ComfyUI, 'attached' dials an already-running server")
    generation_timeout_s: int = Field(default=900, ge=1, description="Maximum wall-clock seconds allowed for a single generation step")
    poll_interval_s: float = Field(default=2.0, gt=0, description="Seconds between progress/health polls")
    stale_heartbeat_seconds: int = Field(default=300, ge=10, description="Seconds without a heartbeat after which a running job is considered stale")
    vram_margin_mb: int = Field(default=1024, ge=0, description="Safety margin in MB reserved below the reported VRAM total")
    ram_margin_mb: int = Field(default=2048, ge=0, description="Safety margin in MB reserved below the reported system RAM total")
    offload_min_ram_mb: Optional[int] = Field(default=None, ge=0, description="Measured minimum system RAM in MB required to offload one model family; None means not yet measured")
    oom_max_retries: int = Field(default=1, ge=0, description="Maximum automatic retries after an out-of-memory failure")
    memory_release_timeout_s: int = Field(default=60, ge=1, description="Maximum seconds to wait for VRAM/RAM release before failing closed")
    state_dir: str = Field(default="", description="Project-relative directory for lease/state files; empty means the feature default location")
    image_workflow: str = Field(default="workflows/selfhost/image_flux3.json", description="Project-relative ComfyUI workflow for image generation")
    reference_workflow: str = Field(default="workflows/selfhost/image_flux3_ref.json", description="Project-relative ComfyUI workflow for reference-conditioned image generation")
    video_workflow: str = Field(default="workflows/selfhost/video_wan2_2_i2v.json", description="Project-relative ComfyUI workflow for image-to-video generation")

    @field_validator("comfy_command")
    @classmethod
    def _validate_comfy_command(cls, value: list[str]) -> list[str]:
        """Validate the ComfyUI launch command is a non-empty-element argument array.

        The value is executed as an ARGUMENT ARRAY (``exec`` style) and is never
        shell-interpolated. An EMPTY list is the explicit "not configured"
        sentinel; downstream code MUST fail closed on it and must not silently
        substitute a guessed executable.
        """
        for element in value:
            if not isinstance(element, str) or element == "":
                raise ValueError(
                    "comfy_command is an argument array (executable + args) and each element must be a non-empty string; it is never shell-interpolated"
                )
        return value

    @field_validator("state_dir")
    @classmethod
    def _validate_state_dir(cls, value: str) -> str:
        """Validate the state directory is empty or a relative, non-traversing path.

        Absolute paths (POSIX absolute, Windows drive-absolute, and driveless-rooted
        paths such as ``/abs`` or ``\\abs``) and any ``..`` traversal segment are
        rejected so no hardcoded machine path can enter the config surface. A
        driveless-rooted path is rejected on ``win.root`` because
        ``PureWindowsPath("/abs/path").is_absolute()`` is False yet
        ``PureWindowsPath("/abs/path").root == "/"`` (truthy), while legitimate
        relative paths keep ``root == ""`` (e.g. ``a/b``, ``./x``).
        """
        if value == "":
            return value
        win = pathlib.PureWindowsPath(value)
        if pathlib.Path(value).is_absolute() or win.is_absolute() or win.root:
            raise ValueError("state_dir must be a project-relative path; absolute paths are not allowed")
        parts = value.replace("\\", "/").split("/")
        if ".." in parts:
            raise ValueError("state_dir must not contain '..' traversal segments")
        return value


class CatFactoryConfig(BaseModel):
    """Cat factory opt-in feature flag configuration"""
    enabled: bool = Field(default=False)
    llm: CatFactoryLLMConfig = Field(default_factory=CatFactoryLLMConfig)
    guardrails: CatFactoryGuardrailsConfig = Field(default_factory=CatFactoryGuardrailsConfig)
    gpu: CatFactoryGpuConfig = Field(default_factory=CatFactoryGpuConfig)


class PixelleVideoConfig(BaseModel):
    """Pixelle-Video main configuration"""
    project_name: str = Field(default="Pixelle-Video", description="Project name")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api_providers: APIProvidersConfig = Field(default_factory=APIProvidersConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    cat_factory: CatFactoryConfig = Field(default_factory=CatFactoryConfig)
    
    def is_llm_configured(self) -> bool:
        """Check if LLM is properly configured"""
        return bool(
            self.llm.api_key and self.llm.api_key.strip() and
            self.llm.base_url and self.llm.base_url.strip() and
            self.llm.model and self.llm.model.strip()
        )
    
    def validate_required(self) -> bool:
        """Validate required configuration"""
        return self.is_llm_configured()
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for backward compatibility)"""
        return self.model_dump()
