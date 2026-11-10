"""Render README result tables from results/final. Missing files stay pending.

OpenRouter rows are filled from quality.csv, cost.csv, and the summary JSON.
Qwen, serving, and self-hosted cost stay pending until those runs exist.
The renderer does not invent a score when the file is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Display name, results/final model id, note used only while that id is missing.
QUALITY_ROWS = [
    ("Qwen2.5-1.5B base", None, "zero-shot"),
    ("+ LoRA", None, ""),
    ("+ QLoRA", None, ""),
    ("+ LoRA, AWQ W4A16", None, ""),
    ("+ LoRA, GGUF Q4_K_M", None, ""),
    ("GPT-4o-mini (OpenRouter)", "openai/gpt-4o-mini", "not measured"),
    ("GPT-5-mini (OpenRouter)", "openai/gpt-5-mini", "not measured"),
]

# These two summaries are the 10 Nov 2026 full-split runs. The date is attached
# only when the file still matches that execution accuracy, so a later run does
# not keep a stale date.
NOV10_EX = {
    "openai/gpt-4o-mini": 0.7620889748549323,
    "openai/gpt-5-mini": 0.7562862669245648,
}

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_summaries(final_dir: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if not final_dir.exists():
        return found
    for path in sorted(final_dir.glob("*.summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model")
        if model:
            found[str(model)] = payload
    return found


def _is_nov10(model: str, summary: dict | None) -> bool:
    if not summary:
        return False
    expected = NOV10_EX.get(model)
    if expected is None:
        return False
    try:
        return abs(float(summary["execution_accuracy"]) - expected) < 1e-12
    except (KeyError, TypeError, ValueError):
        return False


def _quality_note(model: str, summary: dict | None) -> str:
    dated = _is_nov10(model, summary)
    parts = ["measured 10 Nov 2026" if dated else "measured", "OpenRouter", "full test split"]
    if summary:
        if summary.get("n"):
            parts.append(f"n={summary['n']}")
        if summary.get("seed") is not None:
            parts.append(f"seed {summary['seed']}")
        if summary.get("temperature") is not None:
            temperature = summary["temperature"]
            if isinstance(temperature, float) and temperature.is_integer():
                temperature = int(temperature)
            parts.append(f"temperature {temperature}")
        effort = summary.get("reasoning_effort") or ""
        if effort:
            parts.append(f"reasoning {effort}")
        if summary.get("latency_p50_ms") is not None and summary.get("latency_p95_ms") is not None:
            p50 = round(float(summary["latency_p50_ms"]))
            p95 = round(float(summary["latency_p95_ms"]))
            parts.append(f"p50/p95 {p50}/{p95} ms")
    return ", ".join(parts)


def _quality_table(final_dir: Path) -> str:
    by_model = {
        row["model"]: row for row in _read_csv(final_dir / "quality.csv") if row.get("model")
    }
    summaries = _read_summaries(final_dir)
    lines = [
        "| Model | Exec. accuracy (EX) | Exact match | Invalid SQL % | Notes |",
        "|---|---|---|---|---|",
    ]
    for display, model_id, pending_note in QUALITY_ROWS:
        measured = by_model.get(model_id) if model_id else None
        if not measured:
            lines.append(f"| {display} | pending | pending | pending | {pending_note} |")
            continue
        ex = f"{float(measured['execution_accuracy']):.3f}"
        em = f"{float(measured['exact_match']):.3f}"
        invalid = f"{float(measured['invalid_sql_percent']):.2f}%"
        note = _quality_note(model_id or "", summaries.get(model_id or ""))
        lines.append(f"| {display} | {ex} | {em} | {invalid} | {note} |")
    return "\n".join(lines)


def _pending_serving() -> str:
    return "\n".join(
        [
            "| Variant | Disk | GPU memory | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 | Tokens/s | Cold start |",
            "|---|---|---|---|---|---|---|---|---|",
            "| FP16 | pending | pending | pending | pending | pending | pending | pending | pending |",
            "| AWQ W4A16 | pending | pending | pending | pending | pending | pending | pending | pending |",
            "| GGUF Q8_0 | pending | pending | pending | pending | pending | pending | pending | pending |",
            "| GGUF Q4_K_M | pending | pending | pending | pending | pending | pending | pending | pending |",
        ]
    )


def _usd(value: str, digits: int = 3) -> str:
    return f"${float(value):.{digits}f}"


def _cost_table(final_dir: Path) -> str:
    rows = {row["model"]: row for row in _read_csv(final_dir / "cost.csv") if row.get("model")}
    summaries = _read_summaries(final_dir)
    lines = [
        "| Item | Value |",
        "|---|---|",
        "| Training GPU-hours | pending |",
        "| Training cost | pending |",
        "| Self-hosted cost per 1k queries | pending |",
    ]
    labels = [
        ("openai/gpt-4o-mini", "GPT-4o-mini cost per 1k queries"),
        ("openai/gpt-5-mini", "GPT-5-mini cost per 1k queries"),
    ]
    for model_id, label in labels:
        row = rows.get(model_id)
        if not row or not row.get("cost_per_1k_usd"):
            lines.append(f"| {label} | pending |")
            continue
        summary = summaries.get(model_id)
        dated = _is_nov10(model_id, summary)
        when = "measured 10 Nov 2026, " if dated else ""
        effort = ""
        if summary and summary.get("reasoning_effort"):
            effort = f", reasoning {summary['reasoning_effort']}"
        total = _usd(row["total_cost_usd"])
        per_1k = _usd(row["cost_per_1k_usd"])
        source = row.get("cost_source") or "usage.cost"
        source = source.removeprefix("openrouter_")
        lines.append(
            f"| {label} | {per_1k} ({when}OpenRouter {source}{effort}, total {total}, n={row['n']}) |"
        )
    lines.append("| Break-even requests/s | pending |")
    return "\n".join(lines)


def _footer(final_dir: Path) -> str:
    measured = bool(_read_csv(final_dir / "quality.csv"))
    if not measured:
        return (
            "Numbers stay `pending` until `results/final/*.csv` exists. "
            "Do not type measured scores by hand. Run `make results-table` after a real eval."
        )
    return (
        "Exact match is a whitespace-normalized string comparison, so it stays low when the SQL "
        "text differs and the query still returns the same rows. Execution accuracy is the main metric. "
        "Qwen, serving, and self-hosted cost stay `pending` until the trained adapter is evaluated. "
        "Per-row JSONL files are not committed."
    )


def render_block(final_dir: Path) -> str:
    return "\n\n".join(
        [
            "### Quality (Spider test split, 1,034 questions)",
            _quality_table(final_dir),
            "### Serving",
            _pending_serving(),
            "### Cost",
            _cost_table(final_dir),
            _footer(final_dir),
        ]
    )


def update_readme(readme: Path, final_dir: Path) -> None:
    text = readme.read_text(encoding="utf-8")
    block = render_block(final_dir)
    if START not in text or END not in text:
        raise SystemExit(f"{readme} is missing {START} / {END}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{block}\n{END}{after}", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh README result tables.")
    parser.add_argument("--final", type=Path, default=Path("results/final"))
    parser.add_argument("--readme", type=Path, default=None)
    args = parser.parse_args()
    block = render_block(args.final)
    if args.readme:
        update_readme(args.readme, args.final)
        print(f"updated {args.readme}")
    else:
        print(block)


if __name__ == "__main__":
    main()
