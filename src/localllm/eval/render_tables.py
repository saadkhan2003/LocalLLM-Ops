"""Render README result tables from results/final. Missing files stay pending."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

QUALITY_MODELS = [
    "Qwen2.5-1.5B base",
    "+ LoRA",
    "+ QLoRA",
    "+ LoRA, AWQ W4A16",
    "+ LoRA, GGUF Q4_K_M",
    "GPT-4o-mini (OpenRouter)",
]
QUALITY_NOTES = [
    "zero-shot",
    "",
    "",
    "",
    "",
    "OpenRouter list price, not a measured score",
]

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _pending_quality() -> str:
    lines = [
        "| Model | Exec. accuracy (EX) | Exact match | Invalid SQL % | Notes |",
        "|---|---|---|---|---|",
    ]
    for model, note in zip(QUALITY_MODELS, QUALITY_NOTES, strict=True):
        lines.append(f"| {model} | pending | pending | pending | {note} |")
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


def _pending_cost() -> str:
    return "\n".join(
        [
            "| Item | Value |",
            "|---|---|",
            "| Training GPU-hours | pending |",
            "| Training cost | pending |",
            "| Self-hosted cost per 1k queries | pending |",
            "| GPT-4o-mini cost per 1k queries | pending |",
            "| Break-even requests/s | pending |",
        ]
    )


def _from_csv(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return None
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_block(final_dir: Path) -> str:
    quality = _from_csv(final_dir / "quality.csv") or _pending_quality()
    serving = _from_csv(final_dir / "serving.csv") or _pending_serving()
    cost = _from_csv(final_dir / "cost.csv") or _pending_cost()
    return "\n\n".join(
        [
            "### Quality (Spider test split, 1,034 questions)",
            quality,
            "### Serving",
            serving,
            "### Cost",
            cost,
            "Numbers stay `pending` until `results/final/*.csv` exists. "
            "Do not type measured scores by hand. Run `make results-table` after a real eval.",
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
