from typing import Any
from pydantic import ValidationError
from dialtone.types import (
    ProviderConfig,
    Dials,
    RouterModelConfig,
    FallbackConfig,
    ToolsConfig,
    LLM,
)


class DialtoneBase:
    def validate_inputs(
        self,
        provider_config: ProviderConfig | dict[str, Any],
        dials: Dials | dict[str, Any] = Dials(),
        router_model_config: RouterModelConfig | dict[str, Any] = RouterModelConfig(),
        fallback_config: FallbackConfig | dict[str, Any] = FallbackConfig(),
        tools_config: ToolsConfig | dict[str, Any] = ToolsConfig(),
    ):
        try:
            if isinstance(provider_config, dict):
                provider_config = ProviderConfig(**provider_config)
        except ValidationError as e:
            raise ValidationError(f"Invalid provider_config: {e}")

        try:
            if isinstance(dials, dict):
                dials = Dials(**dials)
        except ValidationError as e:
            raise ValidationError(f"Invalid dials: {e}")

        try:
            if isinstance(router_model_config, dict):
                router_model_config = RouterModelConfig(**router_model_config)
        except ValidationError as e:
            raise ValidationError(f"Invalid router_model_config: {e}")

        try:
            if router_model_config.include_models:
                router_model_config.include_models = [
                    LLM(model) if isinstance(model, str) else model
                    for model in router_model_config.include_models
                ]
        except ValueError as e:
            raise ValidationError(f"Invalid include_models: {e}")

        try:
            if router_model_config.exclude_models:
                router_model_config.exclude_models = [
                    LLM(model) if isinstance(model, str) else model
                    for model in router_model_config.exclude_models
                ]
        except ValueError as e:
            raise ValidationError(f"Invalid exclude_models: {e}")

        try:
            if isinstance(fallback_config, dict):
                fallback_config = FallbackConfig(**fallback_config)
        except ValidationError as e:
            raise ValidationError(f"Invalid fallback_config: {e}")

        try:
            if isinstance(tools_config, dict):
                tools_config = ToolsConfig(**tools_config)
        except ValidationError as e:
            raise ValidationError(f"Invalid tools_config: {e}")
