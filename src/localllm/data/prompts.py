"""One chat prompt for training, evaluation, and serving."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path

from localllm.data.fixtures import materialize_fixture
from localllm.data.schema import truncate_schema

SYSTEM_PROMPT = "You write SQLite SQL. Return only SQL."

Tokenize = Callable[[str], list[str]]


def build_messages(schema: str, question: str, sql: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(schema, question)},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql.strip()})
    return messages


def user_content(schema: str, question: str) -> str:
    return f"Schema:\n{schema}\n\nQuestion: {question}"


def render_prompt(schema: str, question: str, sql: str | None = None) -> str:
    """Stable rendering used by the exact-string unit test.

    Training applies the tokenizer chat template to :func:`build_messages`.
    This renderer is the contract for tests and for logs.
    """
    return "\n".join(f"{message['role']}: {message['content']}" for message in build_messages(schema, question, sql))


def parse_user_message(messages: list[dict[str, str]]) -> tuple[str, str]:
    user = next(message["content"] for message in messages if message["role"] == "user")
    if not user.startswith("Schema:\n") or "\n\nQuestion: " not in user:
        raise ValueError("user message is not in the LocalLLM Ops prompt format")
    schema, question = user[len("Schema:\n") :].split("\n\nQuestion: ", 1)
    return schema, question


def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def token_length_report(texts: Iterable[str], tokenize: Tokenize | None = None) -> dict[str, float]:
    tokenize = tokenize or whitespace_tokenize
    lengths = [len(tokenize(text)) for text in texts]
    if not lengths:
        raise ValueError("no texts to measure")
    ordered = sorted(lengths)
    return {
        "n": float(len(ordered)),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def prompt_texts_from_jsonl(path: Path, max_schema_chars: int = 6000) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            schema, _truncated = truncate_schema(row["schema"], max_schema_chars)
            texts.append(render_prompt(schema, row["question"], row.get("gold_sql")))
    return texts


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (percentile / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Report prompt token lengths.")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--max-schema-chars", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=1024)
    args = parser.parse_args()
    if args.fixture or args.jsonl is None:
        root = materialize_fixture(Path("data/fixtures"))
        path = root / "test.jsonl"
    else:
        path = args.jsonl
    texts = prompt_texts_from_jsonl(path, args.max_schema_chars)
    report = token_length_report(texts)
    report["tokenizer"] = "whitespace"
    report["fits_limit"] = report["p95"] <= args.limit
    print(json.dumps(report, indent=2))
    if not report["fits_limit"]:
        raise SystemExit(f"p95 {report['p95']} exceeds {args.limit} tokens")


if __name__ == "__main__":
    main()
