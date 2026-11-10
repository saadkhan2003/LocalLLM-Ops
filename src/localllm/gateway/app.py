"""Text-to-SQL gateway: auth, rate limit, SQL allow-list, optional execution."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from localllm import __version__
from localllm.data.fixtures import fixture_examples, materialize_fixture
from localllm.data.prompts import build_messages
from localllm.data.schema import resolve_sqlite, schema_from_sqlite, truncate_schema
from localllm.eval.execution import execute_sql
from localllm.gateway.page import PAGE_HTML
from localllm.gateway.sql_guard import SqlRejected, validate_select_only
from localllm.observability.tracing import build_tracer

logger = logging.getLogger("localllm.gateway")

SQL_REQUESTS = Counter(
    "localllm_sql_requests_total",
    "SQL gateway requests",
    ["backend", "status"],
)
SQL_LATENCY = Histogram(
    "localllm_sql_latency_seconds",
    "End-to-end gateway latency",
    ["backend"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)


class SqlRequest(BaseModel):
    question: str = Field(min_length=1)
    schema_text: str | None = Field(default=None, alias="schema")
    db_id: str | None = None
    backend: str = "tiny"
    execute: bool = False

    model_config = {"populate_by_name": True}


class SqlResponse(BaseModel):
    sql: str
    valid: bool
    rejection_reason: str | None = None
    columns: list[str] | None = None
    rows: list[list] | None = None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None
    backend: str
    model: str
    fallback: bool = False
    schema_truncated: bool = False


@dataclass
class Settings:
    api_key: str | None = None
    rate_limit: int = 60
    rate_window_s: float = 60.0
    database_dir: Path | None = None
    max_schema_chars: int = 6000
    sql_timeout_s: float = 5.0
    fallback_backend: str | None = None
    sample_log_path: Path | None = None
    sample_rate: float = 0.0
    vllm_base_url: str | None = None
    llamacpp_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_input_per_million: float = 0.15
    openrouter_output_per_million: float = 0.60
    gpu_usd_per_hour: float | None = None
    models: dict[str, str] = field(default_factory=dict)


def settings_from_env() -> Settings:
    def _path(name: str) -> Path | None:
        raw = os.environ.get(name)
        return Path(raw) if raw else None

    def _float(name: str) -> float | None:
        raw = os.environ.get(name)
        return float(raw) if raw else None

    return Settings(
        api_key=os.environ.get("GATEWAY_API_KEY") or None,
        rate_limit=int(os.environ.get("GATEWAY_RATE_LIMIT", "60")),
        rate_window_s=float(os.environ.get("GATEWAY_RATE_WINDOW_S", "60")),
        database_dir=_path("LOCALLLM_DATABASE_DIR"),
        max_schema_chars=int(os.environ.get("LOCALLLM_MAX_SCHEMA_CHARS", "6000")),
        sql_timeout_s=float(os.environ.get("LOCALLLM_SQL_TIMEOUT_S", "5")),
        fallback_backend=os.environ.get("GATEWAY_FALLBACK_BACKEND") or None,
        sample_log_path=_path("GATEWAY_SAMPLE_LOG"),
        sample_rate=float(os.environ.get("GATEWAY_SAMPLE_RATE", "0")),
        vllm_base_url=os.environ.get("VLLM_BASE_URL"),
        llamacpp_base_url=os.environ.get("LLAMACPP_BASE_URL"),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        openrouter_input_per_million=float(
            os.environ.get("OPENROUTER_INPUT_USD_PER_MILLION", "0.15")
        ),
        openrouter_output_per_million=float(
            os.environ.get("OPENROUTER_OUTPUT_USD_PER_MILLION", "0.60")
        ),
        gpu_usd_per_hour=_float("LOCALLLM_GPU_USD_PER_HOUR"),
        models={
            "base": os.environ.get("VLLM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            "ft-lora": os.environ.get("VLLM_LORA_MODEL", "spider-sql"),
            "ft-awq": os.environ.get("VLLM_AWQ_MODEL", "spider-sql-awq"),
            "ft-gguf": os.environ.get("GGUF_MODEL", "spider-sql-q4"),
            "gpt-4o-mini": os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        },
    )


class RateLimiter:
    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        moment = time.monotonic() if now is None else now
        bucket = [stamp for stamp in self._hits[key] if moment - stamp < self.window_s]
        if len(bucket) >= self.limit:
            self._hits[key] = bucket
            return False
        bucket.append(moment)
        self._hits[key] = bucket
        return True


def create_app(settings: Settings | None = None, backends: dict | None = None) -> FastAPI:
    settings = settings or settings_from_env()
    if backends is None:
        from localllm.gateway.backends import build_gateway_backends

        backends = build_gateway_backends(settings)
    tracer = build_tracer()
    limiter = RateLimiter(settings.rate_limit, settings.rate_window_s)
    if settings.database_dir is None:
        settings.database_dir = materialize_fixture(Path("data/fixtures")) / "database"

    app = FastAPI(title="LocalLLM Ops", version=__version__)
    app.state.settings = settings
    app.state.backends = backends
    app.state.limiter = limiter
    app.state.tracer = tracer
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        logging.getLogger("localllm.access").info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        )
        return response

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__, "auth": bool(settings.api_key)}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE_HTML

    @app.get("/v1/examples")
    def examples() -> dict:
        payload = []
        for row in fixture_examples():
            db_path = resolve_sqlite(settings.database_dir, row["db_id"])
            schema = schema_from_sqlite(db_path) if db_path else ""
            payload.append(
                {
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "schema": schema,
                    "difficulty": row["difficulty"],
                }
            )
        return {"examples": payload, "backends": sorted(backends)}

    @app.post("/v1/sql")
    def sql_endpoint(body: SqlRequest, request: Request):
        denied = _authorize(request, settings, limiter)
        if denied is not None:
            return denied
        status, payload = _handle(body, settings, backends, tracer)
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/sql/stream")
    def sql_stream(body: SqlRequest, request: Request):
        denied = _authorize(request, settings, limiter)
        if denied is not None:
            return denied
        status, payload = _handle(body, settings, backends, tracer)

        def chunks():
            sql_text = payload.get("sql") or ""
            # One token-sized chunk so a load test can mark time to first byte.
            yield f"data: {json.dumps({'delta': sql_text})}\n\n"
            yield f"data: {json.dumps({'done': True, 'status': status, 'body': payload})}\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream", status_code=status)

    if not settings.api_key:
        logger.warning("GATEWAY_API_KEY is empty; /v1/sql auth is disabled")
    return app


def _authorize(request: Request, settings: Settings, limiter: RateLimiter):
    if settings.api_key:
        header = request.headers.get("authorization", "")
        token = (
            header.removeprefix("Bearer ").strip() if header.lower().startswith("bearer ") else ""
        )
        supplied = token or request.headers.get("x-api-key", "")
        if supplied != settings.api_key:
            return JSONResponse(status_code=401, content={"detail": "invalid API key"})
        identity = supplied
    else:
        identity = request.client.host if request.client else "local"
    if not limiter.allow(identity):
        return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
    return None


def _handle(body: SqlRequest, settings: Settings, backends: dict, tracer) -> tuple[int, dict]:
    started = time.perf_counter()
    if body.backend not in backends:
        SQL_REQUESTS.labels(body.backend, "error").inc()
        return 404, {"detail": f"unknown backend {body.backend}", "available": sorted(backends)}
    schema, truncated, error = _resolve_schema(body, settings)
    if error:
        SQL_REQUESTS.labels(body.backend, "error").inc()
        return 422, {"detail": error}
    messages = build_messages(schema, body.question)
    used_backend = body.backend
    fallback = False
    try:
        completion = backends[used_backend].complete(messages)
    except Exception as exc:
        fallback_name = settings.fallback_backend
        if fallback_name and fallback_name in backends and fallback_name != used_backend:
            try:
                completion = backends[fallback_name].complete(messages)
                used_backend = fallback_name
                fallback = True
            except Exception as fallback_exc:
                SQL_REQUESTS.labels(body.backend, "error").inc()
                return 503, {"detail": f"backend down: {exc}; fallback failed: {fallback_exc}"}
        else:
            SQL_REQUESTS.labels(body.backend, "error").inc()
            return 503, {"detail": f"backend down: {exc}"}
    try:
        sql = validate_select_only(completion.text)
        valid = True
        reason = None
        status = 200
        status_label = "ok"
    except SqlRejected as exc:
        sql = completion.text
        valid = False
        reason = exc.reason
        status = 422
        status_label = "rejected"
    columns = None
    rows = None
    db_path = resolve_sqlite(settings.database_dir, body.db_id) if body.db_id else None
    if valid and body.execute:
        if db_path is None:
            SQL_REQUESTS.labels(used_backend, "error").inc()
            return 422, {"detail": "execute requires a known db_id"}
        try:
            raw_rows = execute_sql(db_path, sql, timeout_s=settings.sql_timeout_s)
            rows = [list(row) for row in raw_rows]
            columns = _column_names(db_path, sql)
        except Exception as exc:
            valid = False
            reason = f"execution failed: {exc}"
            status = 422
            status_label = "rejected"
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    SQL_REQUESTS.labels(used_backend, status_label).inc()
    SQL_LATENCY.labels(used_backend).observe(latency_ms / 1000)
    payload = SqlResponse(
        sql=sql,
        valid=valid,
        rejection_reason=reason,
        columns=columns,
        rows=rows,
        latency_ms=latency_ms,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=completion.cost_usd,
        backend=used_backend,
        model=completion.model,
        fallback=fallback,
        schema_truncated=truncated,
    ).model_dump()
    trace = {
        "backend": used_backend,
        "model": completion.model,
        "messages": messages,
        "sql": sql,
        "valid": valid,
        "rejection_reason": reason,
        "latency_ms": latency_ms,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "cost_usd": completion.cost_usd,
        "fallback": fallback,
        "schema_truncated": truncated,
    }
    tracer.trace(trace)
    _maybe_sample(settings, trace)
    return status, payload


def _resolve_schema(body: SqlRequest, settings: Settings) -> tuple[str, bool, str | None]:
    schema = body.schema_text or ""
    if body.db_id and not schema:
        db_path = resolve_sqlite(settings.database_dir, body.db_id)
        if db_path is None:
            return "", False, f"unknown db_id {body.db_id}"
        schema = schema_from_sqlite(db_path)
    if not schema.strip():
        return "", False, "schema is empty"
    schema, truncated = truncate_schema(schema, settings.max_schema_chars)
    return schema, truncated, None


def _column_names(db_path: Path, sql: str) -> list[str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(sql)
        return [item[0] for item in cursor.description or []]
    finally:
        conn.close()


def _maybe_sample(settings: Settings, trace: dict) -> None:
    if settings.sample_log_path is None or settings.sample_rate <= 0:
        return
    if random.random() > settings.sample_rate:
        return
    settings.sample_log_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.sample_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, default=str) + "\n")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_load_dotenv()
app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "localllm.gateway.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8747"))
    )


if __name__ == "__main__":
    main()
