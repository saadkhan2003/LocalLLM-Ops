"""vLLM OpenAI server on Modal. Scale to zero. Not deployed from CI.

    modal deploy deploy/modal_vllm.py

Weights are cached on the `localllm-weights` volume. The Hugging Face token is
read from the Modal secret `huggingface`. Set a workspace spend limit before
deploying. GPU time is billed. Idle containers scale to zero
(`min_containers=0`, `scaledown_window=120`).
"""

from __future__ import annotations

import modal

app = modal.App("localllm-ops-vllm")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm")
    .env({"HF_HOME": "/weights/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
)

weights = modal.Volume.from_name("localllm-weights", create_if_missing=True)
hf_cache = modal.Volume.from_name("localllm-hf-cache", create_if_missing=True)

VLLM_PORT = 8000
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@app.server(
    image=vllm_image,
    gpu="L4",
    scaledown_window=120,
    min_containers=0,
    max_containers=1,
    timeout=10 * 60,
    startup_timeout=15 * 60,
    port=VLLM_PORT,
    volumes={"/weights": weights, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
class Server:
    @modal.enter()
    def start(self) -> None:
        import subprocess

        command = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "auto",
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            "32",
            "--enable-prefix-caching",
            "--served-model-name",
            MODEL_NAME,
        ]
        # Swap in a merged or AWQ checkpoint from the volume when it exists.
        merged = "/weights/spider-merged"
        awq = "/weights/spider-awq"
        import os

        if os.path.isdir(awq):
            command[2] = awq
            command += ["--quantization", "compressed-tensors"]
        elif os.path.isdir(merged):
            command[2] = merged
        lora = "/weights/spider-qlora"
        if os.path.isdir(lora):
            command += ["--enable-lora", "--lora-modules", f"spider-sql={lora}"]
        self.process = subprocess.Popen(command)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


@app.local_entrypoint()
def main() -> None:
    print("Deploy with: modal deploy deploy/modal_vllm.py")
    print("The server scales to zero after 120s idle. Cap is 1 container.")
