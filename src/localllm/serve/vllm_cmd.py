"""Build the vLLM OpenAI server command. The binary is not invoked here."""

from __future__ import annotations

import argparse
import shlex


def build_vllm_command(
    model: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str = "auto",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 32,
    quantization: str | None = None,
    lora_modules: str | None = None,
    enable_prefix_caching: bool = True,
    served_model_name: str | None = None,
) -> list[str]:
    if not model:
        raise ValueError("model is required")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    command = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    if served_model_name:
        command += ["--served-model-name", served_model_name]
    if enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if quantization:
        command += ["--quantization", quantization]
    if lora_modules:
        command += ["--enable-lora", "--lora-modules", lora_modules]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a vLLM serve command.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--lora-modules", default=None)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    command = build_vllm_command(
        args.model,
        port=args.port,
        quantization=args.quantization,
        lora_modules=args.lora_modules,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    print(shlex.join(command))


if __name__ == "__main__":
    main()
