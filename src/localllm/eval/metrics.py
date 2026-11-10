"""Latency, cost, and summary rows for an eval run."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def openrouter_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million: float = 0.15,
    output_usd_per_million: float = 0.60,
) -> float:
    return (input_tokens / 1_000_000) * input_usd_per_million + (
        output_tokens / 1_000_000
    ) * output_usd_per_million


def self_hosted_cost_usd(gpu_usd_per_hour: float, wall_time_s: float) -> float:
    if gpu_usd_per_hour < 0 or wall_time_s < 0:
        raise ValueError("cost inputs must be non-negative")
    return gpu_usd_per_hour * (wall_time_s / 3600)


def cost_per_1k_queries(gpu_usd_per_hour: float, requests_per_second: float) -> float:
    """GPU price divided by throughput, scaled to 1,000 queries.

    ``cost_per_1k = gpu_usd_per_hour / requests_per_second / 3600 * 1000``
    """
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    if gpu_usd_per_hour < 0:
        raise ValueError("gpu price must be non-negative")
    return gpu_usd_per_hour / requests_per_second / 3600 * 1000


def break_even_requests_per_second(gpu_usd_per_hour: float, api_cost_per_1k: float) -> float:
    """Throughput where self-hosting and the API cost the same per 1,000 queries."""
    if gpu_usd_per_hour <= 0 or api_cost_per_1k <= 0:
        raise ValueError("prices must be positive")
    return (gpu_usd_per_hour * 1000) / (api_cost_per_1k * 3600)


def summarize(rows: list[dict], meta: dict) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("no rows to summarize")
    ex_hits = sum(1 for row in rows if row.get("execution_accuracy") == 1)
    em_hits = sum(1 for row in rows if row.get("exact_match") == 1)
    invalid = sum(1 for row in rows if row.get("invalid_sql"))
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    costs = [float(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None]
    by_difficulty: dict[str, dict[str, int]] = {}
    for row in rows:
        label = row.get("difficulty") or ""
        if not label:
            continue
        bucket = by_difficulty.setdefault(label, {"n": 0, "ex": 0})
        bucket["n"] += 1
        bucket["ex"] += 1 if row.get("execution_accuracy") == 1 else 0
    summary = {
        "run_id": meta["run_id"],
        "backend": meta["backend"],
        "model": meta.get("model", ""),
        "split": meta.get("split", "test"),
        "n": n,
        "execution_accuracy": ex_hits / n,
        "exact_match": em_hits / n,
        "invalid_sql_rate": invalid / n,
        "latency_p50_ms": percentile(latencies, 50) if latencies else None,
        "latency_p95_ms": percentile(latencies, 95) if latencies else None,
        "cost_per_1k_usd": (sum(costs) / len(costs) * 1000) if costs else None,
        "temperature": meta.get("temperature", 0),
        "token_count_method": meta.get("token_count_method", ""),
        "exact_match_definition": "whitespace_normalized_string",
        "execution_definition": "test_suite_bag_semantics",
        "difficulty": {
            name: {"n": bucket["n"], "execution_accuracy": bucket["ex"] / bucket["n"]}
            for name, bucket in sorted(by_difficulty.items())
        },
    }
    return summary


SUMMARY_FIELDS = [
    "run_id",
    "backend",
    "model",
    "split",
    "n",
    "execution_accuracy",
    "exact_match",
    "invalid_sql_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_1k_usd",
    "temperature",
    "token_count_method",
]


def write_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = [
                row for row in csv.DictReader(handle) if row.get("run_id") != summary["run_id"]
            ]
    rows = existing + [{key: summary.get(key, "") for key in SUMMARY_FIELDS}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
