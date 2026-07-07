"""Shared resolver for OpenAI-compatible model providers."""

from dataclasses import dataclass
import os

import openai


GENERIC_PREFIX = "openai-compatible/"
QWEN_PREFIX = "qwen/"
QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class OpenAICompatibleModel:
    provider: str
    model: str
    api_key: str
    base_url: str


def is_openai_compatible_model(model: str | None) -> bool:
    if not model:
        return False
    return model.startswith(GENERIC_PREFIX) or model.startswith(QWEN_PREFIX)


def _strip_required_prefix(model: str, prefix: str) -> str:
    stripped = model[len(prefix) :]
    if not stripped:
        raise ValueError(f"Model '{model}' must include a provider-native model name.")
    return stripped


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} environment variable must be set.")
    return value


def resolve_openai_compatible_model(model: str) -> OpenAICompatibleModel:
    if model.startswith(QWEN_PREFIX):
        return OpenAICompatibleModel(
            provider="qwen",
            model=_strip_required_prefix(model, QWEN_PREFIX),
            api_key=_required_env("DASHSCOPE_API_KEY"),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", QWEN_DEFAULT_BASE_URL),
        )

    if model.startswith(GENERIC_PREFIX):
        return OpenAICompatibleModel(
            provider="openai-compatible",
            model=_strip_required_prefix(model, GENERIC_PREFIX),
            api_key=_required_env("OPENAI_COMPATIBLE_API_KEY"),
            base_url=_required_env("OPENAI_COMPATIBLE_BASE_URL"),
        )

    raise ValueError(f"Model '{model}' is not an OpenAI-compatible provider alias.")


def create_openai_compatible_client(
    model: str,
    *,
    max_retries: int = 2,
) -> tuple[openai.OpenAI, str]:
    resolved = resolve_openai_compatible_model(model)
    return (
        openai.OpenAI(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            max_retries=max_retries,
        ),
        resolved.model,
    )
