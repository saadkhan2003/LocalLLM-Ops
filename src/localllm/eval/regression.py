"""Fail when execution accuracy drops more than a set tolerance from a baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def assert_no_regression(current_ex: float, baseline_ex: float, max_drop: float = 0.03) -> float:
    drop = baseline_ex - current_ex
    if drop > max_drop + 1e-12:
        raise SystemExit(
            f"EX dropped by {drop:.4f} from {baseline_ex:.4f} to {current_ex:.4f} "
            f"(allowed drop {max_drop:.4f})"
        )
    return drop


def read_ex(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "execution_accuracy" not in payload:
        raise SystemExit(f"{path} has no execution_accuracy")
    return float(payload["execution_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare an eval summary with a stored baseline.")
    parser.add_argument("--current", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--endpoint", default=None, help="Live gateway URL. Unused when empty.")
    parser.add_argument("--max-drop", type=float, default=0.03)
    args = parser.parse_args()
    endpoint = args.endpoint if args.endpoint is not None else os.environ.get("EVAL_ENDPOINT", "")
    if endpoint and not args.current:
        if not args.baseline or not args.baseline.exists():
            raise SystemExit(
                "EVAL_ENDPOINT is set but the baseline file is missing. "
                "Refusing to invent a baseline. Save a real summary JSON first."
            )
        raise SystemExit(
            "Live re-eval is not run from this environment. On your machine: "
            "python -m localllm.eval.harness --backend vllm --data data/spider/test.jsonl "
            "--limit 100 --run-id nightly && "
            "python -m localllm.eval.regression --current results/nightly.summary.json "
            f"--baseline {args.baseline}"
        )
    if endpoint == "" and args.current is None and args.baseline is None:
        print(
            "LIVE EVAL SKIPPED: set EVAL_ENDPOINT and a baseline file to enforce the nightly check"
        )
        return
    if args.current is None or args.baseline is None:
        raise SystemExit("--current and --baseline are required for an offline comparison")
    drop = assert_no_regression(read_ex(args.current), read_ex(args.baseline), args.max_drop)
    print(f"ok drop={drop:.4f}")


if __name__ == "__main__":
    main()
