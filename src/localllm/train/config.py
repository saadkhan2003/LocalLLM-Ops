"""YAML config for LoRA and QLoRA. Importing this module does not import torch."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    method: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))
    learning_rate: float = 2.0e-4
    num_train_epochs: float = 3
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    max_steps: int = -1
    seed: int = 42
    assistant_only_loss: bool = True
    train_jsonl: str = "data/spider/train.jsonl"
    val_jsonl: str = "data/spider/val.jsonl"
    max_train_examples: int | None = None
    output_dir: str = "outputs/qlora"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "localllm-ops"
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"

    def validate(self) -> None:
        if self.method not in {"lora", "qlora"}:
            raise ValueError("method must be lora or qlora")
        if self.lora_r < 1:
            raise ValueError("lora r must be >= 1")
        if self.lora_alpha < 1:
            raise ValueError("lora alpha must be >= 1")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora dropout must be in [0, 1)")
        if self.max_seq_length < 32:
            raise ValueError("max_seq_length is too small")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.method == "qlora" and not self.load_in_4bit:
            raise ValueError("QLoRA requires load_in_4bit: true")
        if self.method == "lora" and self.load_in_4bit:
            raise ValueError("LoRA config must not set load_in_4bit")
        if self.bf16 and self.fp16:
            raise ValueError("choose bf16 or fp16, not both")
        if not self.assistant_only_loss:
            raise ValueError("assistant_only_loss must stay on so prompt tokens are masked")
        if not self.target_modules:
            raise ValueError("target_modules is empty")

    def flat_params(self) -> dict:
        return asdict(self)


def load_train_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    lora = raw.get("lora") or {}
    train = raw.get("train") or {}
    data = raw.get("data") or {}
    mlflow = raw.get("mlflow") or {}
    qlora = raw.get("qlora") or {}
    method = raw.get("method", "lora")
    config = TrainConfig(
        model_name=raw.get("model_name", TrainConfig.model_name),
        method=method,
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", TARGET_MODULES)),
        learning_rate=float(train.get("learning_rate", 2.0e-4)),
        num_train_epochs=float(train.get("num_train_epochs", 3)),
        max_seq_length=int(train.get("max_seq_length", 1024)),
        per_device_train_batch_size=int(train.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 16)),
        warmup_ratio=float(train.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(train.get("lr_scheduler_type", "cosine")),
        bf16=bool(train.get("bf16", False)),
        fp16=bool(train.get("fp16", True)),
        gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
        logging_steps=int(train.get("logging_steps", 10)),
        save_steps=int(train.get("save_steps", 200)),
        eval_steps=int(train.get("eval_steps", 200)),
        max_steps=int(train.get("max_steps", -1)),
        seed=int(train.get("seed", 42)),
        assistant_only_loss=bool(train.get("assistant_only_loss", True)),
        train_jsonl=str(data.get("train_jsonl", "data/spider/train.jsonl")),
        val_jsonl=str(data.get("val_jsonl", "data/spider/val.jsonl")),
        max_train_examples=data.get("max_train_examples"),
        output_dir=str(raw.get("output_dir", "outputs/run")),
        mlflow_tracking_uri=str(mlflow.get("tracking_uri", "file:./mlruns")),
        mlflow_experiment=str(mlflow.get("experiment_name", "localllm-ops")),
        load_in_4bit=bool(qlora.get("load_in_4bit", method == "qlora")),
        bnb_4bit_quant_type=str(qlora.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qlora.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=str(qlora.get("bnb_4bit_compute_dtype", "float16")),
    )
    config.validate()
    return config


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
