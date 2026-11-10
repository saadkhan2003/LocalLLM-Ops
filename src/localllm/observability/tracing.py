"""Trace a gateway call. Langfuse is used when both keys are set and the SDK imports."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("localllm.trace")


class NullTracer:
    """Structured log when Langfuse is not configured. Keeps local runs secret-free."""

    def trace(self, payload: dict[str, Any]) -> None:
        logger.info("%s", json.dumps({"event": "sql_trace", **payload}, default=str))


class LangfuseTracer:
    def __init__(self, client) -> None:
        self.client = client

    def trace(self, payload: dict[str, Any]) -> None:
        trace = self.client.trace(
            name="text-to-sql",
            input={"messages": payload.get("messages")},
            output={"sql": payload.get("sql"), "valid": payload.get("valid")},
            metadata=payload,
            tags=[str(payload.get("backend", ""))],
        )
        trace.generation(
            name=str(payload.get("model") or payload.get("backend")),
            model=payload.get("model"),
            input=payload.get("messages"),
            output=payload.get("sql"),
            usage={
                "input": payload.get("input_tokens") or 0,
                "output": payload.get("output_tokens") or 0,
            },
            metadata={
                "latency_ms": payload.get("latency_ms"),
                "cost_usd": payload.get("cost_usd"),
                "valid": payload.get("valid"),
                "rejection_reason": payload.get("rejection_reason"),
                "fallback": payload.get("fallback"),
            },
        )


def build_tracer() -> NullTracer | LangfuseTracer:
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        return NullTracer()
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("LANGFUSE keys are set but the langfuse package is not installed")
        return NullTracer()
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    client = Langfuse(public_key=public, secret_key=secret, host=host)
    return LangfuseTracer(client)
