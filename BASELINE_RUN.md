# Baseline run

Commands for a plain Linux machine with Python 3 and [uv](https://docs.astral.sh/uv/). This downloads the official Spider SQLite databases and scores OpenRouter models. It spends money only when you run the baseline commands below.

Do not put keys in this file or in the shell history you paste anywhere. Export them in the same terminal.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone <this-repo> localllm-ops
cd localllm-ops
uv sync --extra dev
```

## 1. Spider questions and SQLite

GNU make rejects unknown options, so `make data --require-sqlite` is not a valid make invocation. The `data` recipe already passes `--download-sqlite` and `--require-sqlite`.

```bash
make data
```

The same work without make:

```bash
uv run python -m localllm.data.prep_spider \
  --out data/spider \
  --download-sqlite \
  --require-sqlite
```

Check the manifest:

```bash
uv run python -c "import json; m=json.load(open('data/spider/manifest.json')); print(m['counts']); print(m['sqlite_missing'])"
```

Expected counts are train 6650, val 350, test 1034. `sqlite_missing` must be `[]`. The zip is about 206 MB and stays under `data/`, which is gitignored.

## 2. OpenRouter baseline

```bash
export OPENROUTER_API_KEY="..."
```

The key is read from the environment. A run with no key exits before any request.

Start with 20 questions and a five-cent cap. The subset is a shuffle of the test file with seed 42, so the same `--limit` and `--seed` pick the same rows. The script stops before the next request when `spent + last request cost` would pass `--max-usd`. It writes the rows it already scored and exits 0.

```bash
uv run python -m localllm.eval.baseline \
  --model openai/gpt-4o-mini \
  --limit 20 \
  --seed 42 \
  --max-usd 0.05 \
  --data data/spider/test.jsonl \
  --database-dir data/spider/database \
  --out results/final
```

GPT-5-mini defaults to reasoning effort `minimal` when the flag is omitted, so the first call does not buy a long trace. `low` is explicit:

```bash
uv run python -m localllm.eval.baseline \
  --model openai/gpt-5-mini \
  --reasoning-effort minimal \
  --limit 20 \
  --seed 42 \
  --max-usd 0.10 \
  --out results/final

uv run python -m localllm.eval.baseline \
  --model openai/gpt-5-mini \
  --reasoning-effort low \
  --limit 20 \
  --seed 42 \
  --max-usd 0.10 \
  --out results/final
```

`openai/gpt-4o-mini` does not send a reasoning field unless you pass `--reasoning-effort`.

Cost is `usage.cost` from the OpenRouter response. If that field is missing, the script sums numeric `usage.cost_details` keys that contain `cost`. The price table in `.env.example` (`OPENROUTER_INPUT_USD_PER_MILLION`, `OPENROUTER_OUTPUT_USD_PER_MILLION`) is used only when the response has no cost, and `cost_source` in the CSV says which one was used.

Outputs:

- `results/final/quality.csv`
- `results/final/cost.csv`
- `results/final/<model>.jsonl`
- `results/final/<model>.summary.json`

A later run of the same model replaces that model's row. Other models in the CSV stay. `make results-table` fills the README from `results/final/` and leaves Qwen pending when those rows are absent. The 10 Nov 2026 OpenRouter run is already in that directory. Do not commit the per-row JSONL. Do not type scores in by hand.

Raise `--limit` and `--max-usd` only after the 20-row summary looks right. Omit `--limit` for the full 1,034-question test split, and keep a `--max-usd` you are willing to spend.

## Optional Langfuse

Tracing stays off when the keys are unset. With both keys set, each example is a trace and the client is flushed at the end. A Langfuse error is printed and does not fail the run. Host is `https://cloud.langfuse.com` unless you override it.

```bash
uv sync --extra observe
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
uv run --extra observe python -m localllm.eval.baseline \
  --model openai/gpt-4o-mini \
  --limit 20 \
  --seed 42 \
  --max-usd 0.05
```
