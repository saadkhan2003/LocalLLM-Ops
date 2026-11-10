#!/usr/bin/env bash
# Convert a Hugging Face model directory to GGUF and quantize Q4_K_M and Q8_0.
# Requires a local llama.cpp checkout. This script does not download weights.
set -euo pipefail

MODEL_DIR="${1:-}"
OUT_DIR="${2:-outputs/gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

if [[ -z "$MODEL_DIR" ]]; then
  echo "usage: quant/gguf.sh MODEL_DIR [OUT_DIR]" >&2
  echo "Set LLAMA_CPP_DIR to your llama.cpp checkout (default: ~/llama.cpp)." >&2
  exit 2
fi

CONVERT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [[ ! -x "$QUANTIZE" && -x "$LLAMA_CPP_DIR/llama-quantize" ]]; then
  QUANTIZE="$LLAMA_CPP_DIR/llama-quantize"
fi

if [[ ! -f "$CONVERT" ]]; then
  echo "missing $CONVERT" >&2
  echo "git clone https://github.com/ggml-org/llama.cpp \"$LLAMA_CPP_DIR\" && cmake -S \"$LLAMA_CPP_DIR\" -B \"$LLAMA_CPP_DIR/build\" && cmake --build \"$LLAMA_CPP_DIR/build\" --target llama-quantize -j" >&2
  exit 2
fi
if [[ ! -x "$QUANTIZE" ]]; then
  echo "missing llama-quantize at $QUANTIZE" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
F16="$OUT_DIR/model-f16.gguf"
python "$CONVERT" "$MODEL_DIR" --outfile "$F16"
"$QUANTIZE" "$F16" "$OUT_DIR/model-Q4_K_M.gguf" Q4_K_M
"$QUANTIZE" "$F16" "$OUT_DIR/model-Q8_0.gguf" Q8_0
echo "wrote $OUT_DIR/model-Q4_K_M.gguf and $OUT_DIR/model-Q8_0.gguf"
echo "CPU serve: llama-server -m $OUT_DIR/model-Q4_K_M.gguf --host 0.0.0.0 --port 8081"
