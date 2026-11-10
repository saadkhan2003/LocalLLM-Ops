"""TRL SFTTrainer entrypoint for LoRA and QLoRA.

``--dry-run`` checks the config, builds the chat messages, and writes a run
manifest. It does not import torch. The real trainer runs only when you pass
``--run`` on a GPU machine. See RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localllm.train.config import TrainConfig, git_commit, load_train_config
from localllm.train.formatting import load_sft_rows


def write_manifest(config: TrainConfig, out_dir: Path, *, dry_run: bool, train_rows: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run": dry_run,
        "git_commit": git_commit(),
        "train_rows": train_rows,
        "params": config.flat_params(),
        "assistant_only_loss": config.assistant_only_loss,
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def dry_run(config: TrainConfig, train_path: Path | None, max_steps: int | None) -> Path:
    if max_steps is not None:
        config.max_steps = max_steps
        config.validate()
    rows = []
    if train_path and train_path.exists():
        rows = load_sft_rows(train_path, config.max_train_examples)
    out = Path(config.output_dir)
    path = write_manifest(config, out, dry_run=True, train_rows=len(rows))
    print(f"dry-run manifest {path} rows={len(rows)} commit={git_commit()}")
    return path


def run_trainer(config: TrainConfig) -> None:
    """Import the GPU stack and train. Called only from ``--run``."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_rows = load_sft_rows(Path(config.train_jsonl), config.max_train_examples)
    eval_rows = []
    val_path = Path(config.val_jsonl)
    if val_path.exists():
        eval_rows = load_sft_rows(val_path, None)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = None
    torch_dtype = torch.bfloat16 if config.bf16 else torch.float16
    if config.load_in_4bit:
        compute = getattr(torch, config.bnb_4bit_compute_dtype)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    args_kwargs = dict(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=config.eval_steps,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to=["mlflow"],
        assistant_only_loss=True,
        packing=False,
        dataset_text_field=None,
    )
    if config.max_steps > 0:
        args_kwargs["max_steps"] = config.max_steps
    try:
        sft_args = SFTConfig(max_seq_length=config.max_seq_length, **args_kwargs)
    except TypeError:
        sft_args = SFTConfig(max_length=config.max_seq_length, **args_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_rows,
        eval_dataset=eval_rows or None,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    write_manifest(config, Path(config.output_dir), dry_run=False, train_rows=len(train_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA / QLoRA SFT for Spider SQL.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run", action="store_true", help="Actually train. Needs a GPU and the train extra."
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    args = parser.parse_args()
    config = load_train_config(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.run and args.dry_run:
        raise SystemExit("pass only one of --run and --dry-run")
    if args.run:
        run_trainer(config)
        return
    train_path = args.train_jsonl or Path(config.train_jsonl)
    dry_run(config, train_path, args.max_steps)


if __name__ == "__main__":
    main()
