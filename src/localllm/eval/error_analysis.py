"""Group harness failures with a short heuristic label. Does not invent examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def classify(gold_sql: str, pred_sql: str, error: str | None) -> str:
    if error and error not in {"result_mismatch", None, ""}:
        return f"invalid SQL ({error})"
    gold_joins = len(re.findall(r"\bjoin\b", gold_sql, flags=re.IGNORECASE))
    pred_joins = len(re.findall(r"\bjoin\b", pred_sql, flags=re.IGNORECASE))
    if gold_joins != pred_joins:
        return "wrong join"
    gold_cols = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", gold_sql))
    pred_cols = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", pred_sql))
    if gold_cols - pred_cols:
        return "wrong column or missing predicate"
    return "wrong predicate or aggregation"


def write_report(jsonl: Path, dest: Path, limit: int) -> int:
    failures = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("execution_accuracy") == 1:
                continue
            row["reason"] = classify(
                row.get("gold_sql", ""), row.get("pred_sql", ""), row.get("error")
            )
            failures.append(row)
    chosen = failures[:limit]
    lines = [
        "# Error analysis",
        "",
        f"Source: `{jsonl}`. {len(failures)} failures, showing {len(chosen)}.",
        "Labels are heuristics (join count, token overlap), not a human review.",
        "",
    ]
    for index, row in enumerate(chosen, start=1):
        lines.append(f"## {index}. {row['reason']}")
        lines.append("")
        lines.append(f"- Question: {row.get('question', '')}")
        lines.append(f"- Gold: `{row.get('gold_sql', '')}`")
        lines.append(f"- Predicted: `{row.get('pred_sql', '')}`")
        lines.append("")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a short failure report from a harness jsonl."
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    count = write_report(args.run, args.out, args.limit)
    print(f"wrote {count} failures to {args.out}")


if __name__ == "__main__":
    main()
