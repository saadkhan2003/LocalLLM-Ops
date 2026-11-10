"""OpenRouter request body and the cost fields on a chat completion.

Cost comes from the response usage object. The local price table is only a
fallback when those fields are absent, and the summary says which one was used.
"""

from __future__ import annotations

import os

from localllm.eval.metrics import openrouter_cost_usd

REASONING_EFFORTS = ("minimal", "low", "medium", "high")


def default_reasoning_effort(model: str, explicit: str | None) -> str | None:
    """gpt-5* defaults to minimal so a baseline does not buy a long trace.

    gpt-4o-mini is not a reasoning model. An effort is attached only when you
    pass one. ``minimal`` and ``low`` are the two settings this project uses.
    """
    if explicit:
        if explicit not in REASONING_EFFORTS:
            raise SystemExit(f"reasoning effort must be one of {', '.join(REASONING_EFFORTS)}")
        return explicit
    if "gpt-5" in model:
        return "minimal"
    return None


def chat_request_body(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    reasoning_effort: str | None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
        body["reasoning_effort"] = reasoning_effort
    return body


def cost_from_usage(usage: dict | None) -> tuple[float | None, str]:
    """Return ``(usd, source)`` from OpenRouter's usage object.

    Preferred field is ``usage.cost``. Otherwise sum the numeric ``*cost``
    entries under ``usage.cost_details``.
    """
    if not usage:
        return None, "missing"
    raw = usage.get("cost")
    if isinstance(raw, int | float):
        return float(raw), "openrouter_usage.cost"
    details = usage.get("cost_details") or {}
    if isinstance(details, dict):
        if isinstance(details.get("upstream_inference_cost"), int | float):
            return float(details["upstream_inference_cost"]), "openrouter_usage.cost_details"
        parts = [
            float(value)
            for key, value in details.items()
            if isinstance(value, int | float) and "cost" in key
        ]
        if parts:
            return sum(parts), "openrouter_usage.cost_details"
    return None, "missing"


def resolve_cost(
    usage: dict | None,
    *,
    input_usd_per_million: float | None,
    output_usd_per_million: float | None,
) -> tuple[int, int, float | None, str]:
    usage = usage or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    cost, source = cost_from_usage(usage)
    if cost is None and input_usd_per_million is not None and output_usd_per_million is not None:
        cost = openrouter_cost_usd(
            input_tokens,
            output_tokens,
            input_usd_per_million,
            output_usd_per_million,
        )
        source = "price_table"
    return input_tokens, output_tokens, cost, source


def api_key_from_env() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment. It is not read from source.")
    return key
