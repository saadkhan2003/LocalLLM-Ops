"""Run a backend over a split and write execution accuracy, latency, and cost."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from localllm.data.fixtures import materialize_fixture
from localllm.data.prompts import build_messages, whitespace_tokenize
from localllm.data.schema import resolve_sqlite
from localllm.eval.execution import exact_match, execution_match, strip_sql
from localllm.eval.metrics import self_hosted_cost_usd, summarize, write_summary_csv
from localllm.eval.tiny_model import TinySqlModel

REQUIRED_FIELDS = ("db_id", "schema", "question", "gold_sql")


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float | None
    model: str
    token_count_method: str
    cost_source: str = ""


class TinyBackend:
    name = "tiny"
    model = "tiny-keyword-sql"
    token_count_method = "whitespace"

    def __init__(self) -> None:
        self._model = TinySqlModel()

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        started = time.perf_counter()
        user = next(item["content"] for item in messages if item["role"] == "user")
        schema, question = user.split("\n\nQuestion: ", 1)
        schema = schema.removeprefix("Schema:\n")
        sql = self._model.generate(schema, question)
        latency = time.perf_counter() - started
        prompt = "\n".join(item["content"] for item in messages)
        return Completion(
            text=sql,
            input_tokens=len(whitespace_tokenize(prompt)),
            output_tokens=len(whitespace_tokenize(sql)),
            latency_s=latency,
            cost_usd=0.0,
            model=self.model,
            token_count_method="whitespace",
        )


class MockGoldBackend:
    """Returns the gold SQL. Used to prove the scorer in CI. Not a model score."""

    name = "mock"
    model = "mock-gold"
    token_count_method = "whitespace"

    def complete(self, messages: list[dict[str, str]], gold_sql: str = "") -> Completion:
        started = time.perf_counter()
        latency = time.perf_counter() - started
        return Completion(
            text=gold_sql,
            input_tokens=1,
            output_tokens=len(whitespace_tokenize(gold_sql)),
            latency_s=latency,
            cost_usd=0.0,
            model=self.model,
            token_count_method="whitespace",
        )


class OpenAICompatibleBackend:
    """Client for vLLM, llama-server, or OpenRouter. Not used by the CPU smoke test."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        input_usd_per_million: float | None = None,
        output_usd_per_million: float | None = None,
        gpu_usd_per_hour: float | None = None,
        name: str = "vllm",
        reasoning_effort: str | None = None,
        extra_body: dict | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million
        self.gpu_usd_per_hour = gpu_usd_per_hour
        self.name = name
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or {}
        self.token_count_method = "api_usage"

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        import httpx

        from localllm.eval.openrouter import chat_request_body, resolve_cost

        url = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = chat_request_body(self.model, messages, self.temperature, self.reasoning_effort)
        payload.update(self.extra_body)
        started = time.perf_counter()
        response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        latency = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"] or ""
        input_tokens, output_tokens, cost, source = resolve_cost(
            body.get("usage"),
            input_usd_per_million=self.input_usd_per_million,
            output_usd_per_million=self.output_usd_per_million,
        )
        if cost is None and self.gpu_usd_per_hour is not None:
            cost = self_hosted_cost_usd(self.gpu_usd_per_hour, latency)
            source = "gpu_hour"
        return Completion(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            cost_usd=cost,
            model=self.model,
            token_count_method="api_usage" if body.get("usage") else "missing",
            cost_source=source,
        )


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"{path} is missing {missing}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def run_harness(
    *,
    backend_name: str,
    rows: list[dict],
    database_dir: Path,
    out_dir: Path,
    run_id: str,
    split: str = "test",
    temperature: float = 0.0,
    timeout_s: float = 10.0,
    model: str | None = None,
    backend=None,
) -> dict:
    if not rows:
        raise ValueError("no examples")
    backend = backend or build_backend(backend_name, model=model, temperature=temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        db_path = resolve_sqlite(database_dir, row["db_id"])
        messages = build_messages(row["schema"], row["question"])
        if isinstance(backend, MockGoldBackend):
            completion = backend.complete(messages, gold_sql=row["gold_sql"])
        else:
            completion = backend.complete(messages)
        predicted = strip_sql(completion.text)
        matched = False
        error = "missing_database"
        if db_path is not None:
            matched, error = execution_match(
                db_path, predicted, row["gold_sql"], timeout_s=timeout_s
            )
        invalid = error not in {None, "result_mismatch"}
        records.append(
            {
                "db_id": row["db_id"],
                "question": row["question"],
                "gold_sql": row["gold_sql"],
                "pred_sql": predicted,
                "execution_accuracy": 1 if matched else 0,
                "exact_match": 1 if exact_match(predicted, row["gold_sql"]) else 0,
                "invalid_sql": 1 if invalid else 0,
                "error": error,
                "latency_ms": round(completion.latency_s * 1000, 3),
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "cost_usd": completion.cost_usd,
                "difficulty": row.get("difficulty", ""),
                "model": completion.model,
            }
        )
    jsonl_path = out_dir / f"{run_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    token_method = getattr(backend, "token_count_method", None) or (
        "whitespace" if records[0]["model"] in {"tiny-keyword-sql", "mock-gold"} else "api_usage"
    )
    summary = summarize(
        records,
        {
            "run_id": run_id,
            "backend": backend_name,
            "model": records[0]["model"],
            "split": split,
            "temperature": temperature,
            "token_count_method": token_method,
        },
    )
    summary_path = out_dir / f"{run_id}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(out_dir / "summary.csv", summary)
    print(
        f"run={run_id} n={summary['n']} ex={summary['execution_accuracy']:.4f} "
        f"em={summary['exact_match']:.4f} invalid={summary['invalid_sql_rate']:.4f} "
        f"-> {jsonl_path}"
    )
    return summary


def build_backend(name: str, model: str | None, temperature: float):
    if name == "tiny":
        return TinyBackend()
    if name == "mock":
        return MockGoldBackend()
    if name == "vllm":
        base = os.environ.get("VLLM_BASE_URL")
        if not base:
            raise SystemExit("Set VLLM_BASE_URL to the vLLM OpenAI base URL.")
        return OpenAICompatibleBackend(
            base,
            model or os.environ.get("VLLM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            api_key=os.environ.get("VLLM_API_KEY"),
            temperature=temperature,
            gpu_usd_per_hour=_optional_float("LOCALLLM_GPU_USD_PER_HOUR"),
            name="vllm",
        )
    if name == "llamacpp":
        base = os.environ.get("LLAMACPP_BASE_URL")
        if not base:
            raise SystemExit("Set LLAMACPP_BASE_URL to the llama-server OpenAI base URL.")
        return OpenAICompatibleBackend(
            base,
            model or os.environ.get("GGUF_MODEL", "spider-sql"),
            temperature=temperature,
            name="llamacpp",
        )
    if name == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("Set OPENROUTER_API_KEY. The baseline is not run in CI.")
        return OpenAICompatibleBackend(
            "https://openrouter.ai/api/v1",
            model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            api_key=key,
            temperature=temperature,
            input_usd_per_million=float(os.environ.get("OPENROUTER_INPUT_USD_PER_MILLION", "0.15")),
            output_usd_per_million=float(
                os.environ.get("OPENROUTER_OUTPUT_USD_PER_MILLION", "0.60")
            ),
            name="openrouter",
        )
    raise SystemExit(f"unknown backend {name}")


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a text-to-SQL backend.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--backend", choices=["tiny", "mock", "vllm", "llamacpp", "openrouter"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--database-dir", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.config:
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        for key, value in loaded.items():
            if getattr(args, key, None) in (None, parser.get_default(key)):
                setattr(args, key, value)
    args.out = Path(args.out)
    if args.data is not None:
        args.data = Path(args.data)
    if args.database_dir is not None:
        args.database_dir = Path(args.database_dir)
    if args.backend is None:
        raise SystemExit("--backend is required")
    use_fixture = args.fixture or args.data is None and args.backend in {"tiny", "mock"}
    if use_fixture:
        root = materialize_fixture(Path("data/fixtures"))
        data_path = root / "test.jsonl"
        database_dir = root / "database"
    else:
        if args.data is None:
            raise SystemExit("--data is required for this backend (or pass --fixture)")
        data_path = args.data
        database_dir = args.database_dir or data_path.parent / "database"
    rows = load_jsonl(data_path, args.limit)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_harness(
        backend_name=args.backend,
        rows=rows,
        database_dir=database_dir,
        out_dir=args.out,
        run_id=run_id,
        split=args.split,
        temperature=args.temperature,
        timeout_s=args.timeout,
        model=args.model,
    )


if __name__ == "__main__":
    main()
