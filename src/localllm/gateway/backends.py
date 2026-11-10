"""Backend clients used by the gateway. Remote calls read keys from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from localllm.data.prompts import parse_user_message, whitespace_tokenize
from localllm.eval.harness import OpenAICompatibleBackend, TinyBackend
from localllm.eval.metrics import openrouter_cost_usd

__all__ = [
    "Completion",
    "TinyBackend",
    "OpenAICompatibleBackend",
    "FailingBackend",
    "ScriptedBackend",
    "build_gateway_backends",
]


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float | None
    model: str


class FailingBackend:
    """Stand-in for a down server."""

    model = "down"

    def __init__(self, message: str = "backend down") -> None:
        self.message = message

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        raise ConnectionError(self.message)


class ScriptedBackend:
    model = "scripted"

    def __init__(self, sql: str, model: str = "scripted") -> None:
        self.sql = sql
        self.model = model
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        self.messages.append(messages)
        prompt = "\n".join(item["content"] for item in messages)
        return Completion(
            text=self.sql,
            input_tokens=len(whitespace_tokenize(prompt)),
            output_tokens=len(whitespace_tokenize(self.sql)),
            latency_s=0.0,
            cost_usd=0.0,
            model=self.model,
        )


def adapt_tiny() -> TinyBackend:
    return TinyBackend()


def build_gateway_backends(settings) -> dict:
    """Map public backend names to clients. Missing URLs are omitted."""
    backends: dict = {"tiny": TinyBackend()}
    if settings.vllm_base_url:
        backends["base"] = OpenAICompatibleBackend(
            settings.vllm_base_url,
            settings.models.get("base", "Qwen/Qwen2.5-1.5B-Instruct"),
            api_key=os.environ.get("VLLM_API_KEY"),
            gpu_usd_per_hour=settings.gpu_usd_per_hour,
            name="vllm",
        )
        backends["ft-lora"] = OpenAICompatibleBackend(
            settings.vllm_base_url,
            settings.models.get("ft-lora", "spider-sql"),
            api_key=os.environ.get("VLLM_API_KEY"),
            gpu_usd_per_hour=settings.gpu_usd_per_hour,
            name="vllm",
        )
        backends["ft-awq"] = OpenAICompatibleBackend(
            settings.vllm_base_url,
            settings.models.get("ft-awq", "spider-sql-awq"),
            api_key=os.environ.get("VLLM_API_KEY"),
            gpu_usd_per_hour=settings.gpu_usd_per_hour,
            name="vllm",
        )
    if settings.llamacpp_base_url:
        backends["ft-gguf"] = OpenAICompatibleBackend(
            settings.llamacpp_base_url,
            settings.models.get("ft-gguf", "spider-sql-q4"),
            name="llamacpp",
        )
    if settings.openrouter_api_key:
        backends["gpt-4o-mini"] = OpenAICompatibleBackend(
            "https://openrouter.ai/api/v1",
            settings.models.get("gpt-4o-mini", "openai/gpt-4o-mini"),
            api_key=settings.openrouter_api_key,
            input_usd_per_million=settings.openrouter_input_per_million,
            output_usd_per_million=settings.openrouter_output_per_million,
            name="openrouter",
        )
    return backends


def estimate_openrouter_cost(input_tokens: int, output_tokens: int, settings) -> float:
    return openrouter_cost_usd(
        input_tokens,
        output_tokens,
        settings.openrouter_input_per_million,
        settings.openrouter_output_per_million,
    )


def schema_and_question(messages: list[dict[str, str]]) -> tuple[str, str]:
    return parse_user_message(messages)
