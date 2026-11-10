"""OpenRouter text-to-SQL baseline with a hard dollar cap.

Reads ``OPENROUTER_API_KEY`` from the environment. Writes
``results/final/quality.csv`` and ``results/final/cost.csv`` plus a jsonl of
the rows that actually ran. Langfuse tracing turns on only when both
``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set. The host defaults
to https://cloud.langfuse.com.

The budget check happens before each request. After the first response, the
next request is skipped when ``spent + last_cost`` would pass ``--max-usd``.
The process exits 0 and keeps the rows already scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

from localllm.data.prompts import build_messages
from localllm.data.schema import resolve_sqlite
from localllm.eval.execution import exact_match, execution_match, strip_sql
from localllm.eval.harness import Completion, OpenAICompatibleBackend, load_jsonl
from localllm.eval.metrics import summarize
from localllm.eval.openrouter import api_key_from_env, default_reasoning_effort
from localllm.observability.tracing import NullTracer, build_tracer

FINAL_QUALITY_FIELDS = [
    "model",
    "execution_accuracy",
    "exact_match",
    "invalid_sql_percent",
    "n",
    "notes",
]
FINAL_COST_FIELDS = [
    "model",
    "n",
    "total_cost_usd",
    "cost_per_1k_usd",
    "max_usd",
    "stopped_reason",
    "cost_source",
]


def sample_rows(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Deterministic subset. Same seed and file contents always pick the same rows."""
    ordered = sorted(rows, key=lambda row: (row["db_id"], row["question"], row.get("gold_sql", "")))
    random.Random(seed).shuffle(ordered)
    if limit is None:
        return ordered
    if limit < 1:
        raise SystemExit("--limit must be at least 1")
    return ordered[:limit]


def should_stop_for_budget(spent: float, last_cost: float | None, max_usd: float | None) -> bool:
    if max_usd is None:
        return False
    if spent >= max_usd:
        return True
    return last_cost is not None and spent + last_cost > max_usd


def run_baseline(
    *,
    rows: list[dict],
    database_dir: Path,
    backend,
    out_dir: Path,
    model: str,
    max_usd: float | None,
    seed: int,
    temperature: float = 0.0,
    timeout_s: float = 30.0,
    reasoning_effort: str | None = None,
    tracer=None,
) -> dict:
    if not rows:
        raise SystemExit("no examples to score")
    out_dir.mkdir(parents=True, exist_ok=True)
    tracer = tracer if tracer is not None else build_tracer()
    records: list[dict] = []
    spent = 0.0
    last_cost: float | None = None
    stopped = ""
    for row in rows:
        if should_stop_for_budget(spent, last_cost, max_usd):
            stopped = "budget"
            break
        db_path = resolve_sqlite(database_dir, row["db_id"])
        messages = build_messages(row["schema"], row["question"])
        completion: Completion = backend.complete(messages)
        cost = completion.cost_usd
        if cost is not None:
            spent += cost
            last_cost = cost
        predicted = strip_sql(completion.text)
        matched = False
        error = "missing_database"
        if db_path is not None:
            matched, error = execution_match(
                db_path, predicted, row["gold_sql"], timeout_s=timeout_s
            )
        invalid = error not in {None, "result_mismatch"}
        record = {
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
            "cost_usd": cost,
            "cost_source": completion.cost_source,
            "difficulty": row.get("difficulty", ""),
            "model": model,
        }
        records.append(record)
        _trace(tracer, record, messages, reasoning_effort)
        if max_usd is not None and spent > max_usd:
            # The request already happened. Do not start another one.
            stopped = "budget"
            break
    if not records:
        raise SystemExit(
            f"no requests sent. spent={spent:.6f} max_usd={max_usd}. Raise --max-usd above zero."
        )
    slug = _slug(model)
    jsonl_path = out_dir / f"{slug}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = summarize(
        records,
        {
            "run_id": slug,
            "backend": "openrouter",
            "model": model,
            "split": "test",
            "temperature": temperature,
            "token_count_method": "api_usage",
        },
    )
    sources = sorted({record["cost_source"] for record in records if record["cost_source"]})
    summary.update(
        {
            "seed": seed,
            "max_usd": max_usd,
            "total_cost_usd": spent,
            "stopped_reason": stopped,
            "reasoning_effort": reasoning_effort or "",
            "cost_source": ",".join(sources),
            "langfuse": not isinstance(tracer, NullTracer),
        }
    )
    (out_dir / f"{slug}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _upsert_csv(
        out_dir / "quality.csv",
        FINAL_QUALITY_FIELDS,
        "model",
        {
            "model": model,
            "execution_accuracy": f"{summary['execution_accuracy']:.6f}",
            "exact_match": f"{summary['exact_match']:.6f}",
            "invalid_sql_percent": f"{summary['invalid_sql_rate'] * 100:.4f}",
            "n": summary["n"],
            "notes": stopped or "openrouter",
        },
    )
    per_1k = summary["cost_per_1k_usd"]
    _upsert_csv(
        out_dir / "cost.csv",
        FINAL_COST_FIELDS,
        "model",
        {
            "model": model,
            "n": summary["n"],
            "total_cost_usd": f"{spent:.8f}",
            "cost_per_1k_usd": "" if per_1k is None else f"{per_1k:.8f}",
            "max_usd": "" if max_usd is None else f"{max_usd:.8f}",
            "stopped_reason": stopped,
            "cost_source": summary["cost_source"],
        },
    )
    _flush(tracer)
    print(
        f"model={model} n={summary['n']} ex={summary['execution_accuracy']:.4f} "
        f"cost_usd={spent:.6f} stopped={stopped or 'no'} -> {jsonl_path}"
    )
    return summary


def _trace(tracer, record: dict, messages: list[dict], reasoning_effort: str | None) -> None:
    try:
        tracer.trace(
            {
                "backend": "openrouter",
                "model": record["model"],
                "messages": messages,
                "sql": record["pred_sql"],
                "valid": record["invalid_sql"] == 0,
                "execution_accuracy": record["execution_accuracy"],
                "latency_ms": record["latency_ms"],
                "input_tokens": record["input_tokens"],
                "output_tokens": record["output_tokens"],
                "cost_usd": record["cost_usd"],
                "cost_source": record["cost_source"],
                "reasoning_effort": reasoning_effort or "",
                "fallback": False,
            }
        )
    except Exception as exc:
        print(f"langfuse trace failed: {exc}")


def _flush(tracer) -> None:
    flush = getattr(tracer, "flush", None)
    if flush is None:
        return
    try:
        flush()
    except Exception as exc:
        print(f"langfuse flush failed: {exc}")


def _slug(model: str) -> str:
    return model.replace("/", "-").replace(" ", "-")


def _upsert_csv(path: Path, fields: list[str], key: str, row: dict) -> None:
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = [item for item in csv.DictReader(handle) if item.get(key) != row[key]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in existing:
            writer.writerow({name: item.get(name, "") for name in fields})
        writer.writerow({name: row.get(name, "") for name in fields})


def build_openrouter_backend(model: str, reasoning_effort: str | None, timeout_s: float):
    return OpenAICompatibleBackend(
        "https://openrouter.ai/api/v1",
        model,
        api_key=api_key_from_env(),
        temperature=0.0,
        timeout_s=timeout_s,
        input_usd_per_million=float(os.environ.get("OPENROUTER_INPUT_USD_PER_MILLION", "0.15")),
        output_usd_per_million=float(os.environ.get("OPENROUTER_OUTPUT_USD_PER_MILLION", "0.60")),
        reasoning_effort=reasoning_effort,
        name="openrouter",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter Spider baseline with a dollar cap.")
    parser.add_argument("--model", required=True, help="openai/gpt-4o-mini or openai/gpt-5-mini")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-usd", type=float, default=None)
    parser.add_argument("--data", type=Path, default=Path("data/spider/test.jsonl"))
    parser.add_argument("--database-dir", type=Path, default=Path("data/spider/database"))
    parser.add_argument("--out", type=Path, default=Path("results/final"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sql-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.max_usd is not None and args.max_usd < 0:
        raise SystemExit("--max-usd must be >= 0")
    effort = default_reasoning_effort(args.model, args.reasoning_effort)
    rows = sample_rows(load_jsonl(args.data, None), args.limit, args.seed)
    backend = build_openrouter_backend(args.model, effort, args.timeout)
    run_baseline(
        rows=rows,
        database_dir=args.database_dir,
        backend=backend,
        out_dir=args.out,
        model=args.model,
        max_usd=args.max_usd,
        seed=args.seed,
        timeout_s=args.sql_timeout,
        reasoning_effort=effort,
    )


if __name__ == "__main__":
    main()
