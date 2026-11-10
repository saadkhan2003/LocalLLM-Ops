"""QLoRA training on Modal. This file is not executed in CI.

Saad runs it after `modal setup`. Tokens come from the Modal secret named
`huggingface` (env `HF_TOKEN`) and from the Modal CLI login, never from source.

    modal run deploy/modal_train.py --max-steps 50
    modal run --detach deploy/modal_train.py --max-steps -1
"""

from __future__ import annotations

import modal

app = modal.App("localllm-ops-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "pyyaml",
        "mlflow",
    )
    .add_local_dir("src/localllm", remote_path="/root/localllm")
    .add_local_dir("configs", remote_path="/root/configs")
    .env({"PYTHONPATH": "/root"})
)

weights = modal.Volume.from_name("localllm-weights", create_if_missing=True)
data_volume = modal.Volume.from_name("localllm-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=6 * 60 * 60,
    volumes={"/vol/weights": weights, "/vol/data": data_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=1,
)
def train(config_name: str = "train_qlora_t4.yaml", max_steps: int = 50) -> str:
    import os
    from pathlib import Path

    os.environ.setdefault("HF_HOME", "/vol/weights/hf")
    from localllm.train.config import load_train_config
    from localllm.train.sft import run_trainer

    config = load_train_config(Path("/root/configs") / config_name)
    config.output_dir = "/vol/weights/spider-qlora"
    if Path("/vol/data/train.jsonl").exists():
        config.train_jsonl = "/vol/data/train.jsonl"
        config.val_jsonl = "/vol/data/val.jsonl"
    if max_steps > 0:
        config.max_steps = max_steps
    run_trainer(config)
    weights.commit()
    return config.output_dir


@app.local_entrypoint()
def main(config_name: str = "train_qlora_t4.yaml", max_steps: int = 50) -> None:
    print(train.remote(config_name, max_steps))
