"""AWQ W4A16 quantization with llm-compressor.

The recipe matches the current llm-compressor AWQ example: ``AWQModifier``
followed by ``QuantizationModifier`` with scheme ``W4A16_ASYM``, ignoring
``lm_head``. ``--dry-run`` prints the recipe and the calibration prompts
without importing torch or llm-compressor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localllm.data.prompts import build_messages, render_prompt

SCHEME = "W4A16_ASYM"


def awq_recipe() -> list[dict]:
    return [
        {"modifier": "AWQModifier", "duo_scaling": "both", "targets": ["Linear"]},
        {
            "modifier": "QuantizationModifier",
            "ignore": ["lm_head"],
            "scheme": SCHEME,
            "targets": ["Linear"],
        },
    ]


def calibration_prompts(path: Path, samples: int) -> list[str]:
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            build_messages(row["schema"], row["question"])
            prompts.append(render_prompt(row["schema"], row["question"]))
            if len(prompts) >= samples:
                break
    if len(prompts) < min(samples, 1):
        raise ValueError(f"no calibration rows in {path}")
    return prompts


def dry_run(train_jsonl: Path, samples: int, model: str, output_dir: Path) -> dict:
    prompts = calibration_prompts(train_jsonl, samples) if train_jsonl.exists() else []
    payload = {
        "model": model,
        "scheme": SCHEME,
        "ignore": ["lm_head"],
        "num_calibration_samples": samples,
        "prompts_loaded": len(prompts),
        "recipe": awq_recipe(),
        "output_dir": str(output_dir),
        "note": "Dry run only. Pass --run on a GPU with the llmcompressor package.",
    }
    print(json.dumps(payload, indent=2))
    return payload


def run_awq(
    model: str, train_jsonl: Path, output_dir: Path, samples: int, max_seq_length: int
) -> None:
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier
    except ImportError:
        from llmcompressor.modifiers.awq import AWQModifier

    prompts = calibration_prompts(train_jsonl, samples)
    dataset = Dataset.from_dict({"text": prompts})
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    loaded = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme=SCHEME, targets=["Linear"]),
    ]
    oneshot(
        model=loaded,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=min(samples, len(prompts)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ W4A16 via llm-compressor.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/spider/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/awq"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run_awq(args.model, args.train_jsonl, args.output_dir, args.samples, args.max_seq_length)
        return
    dry_run(args.train_jsonl, args.samples, args.model, args.output_dir)


if __name__ == "__main__":
    main()
