# RUNBOOK

What you run yourself. Nothing in this file has been executed against a paid GPU or a paid API from the machine that built the repo. Do not paste tokens into source files. Put them in `.env`, Kaggle secrets, Colab secrets, or GitHub Actions secrets.

Prices below were checked on 10 Nov 2026. Open the provider page before you start a job.

## Accounts and keys

| Account | Why | Cost | Where the key goes |
|---|---|---|---|
| Hugging Face | download Qwen, push the adapter | free | `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` |
| Kaggle | free T4 for QLoRA | free weekly GPU quota | Kaggle secret `HF_TOKEN` |
| Google Colab | same T4 path | free tier disconnects | Colab secret `HF_TOKEN` |
| Modal | train or serve if you want scale-to-zero | starter credit, then per-second GPU | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| OpenRouter | GPT-4o-mini baseline | about a dollar or less for the full test split if prices hold | `OPENROUTER_API_KEY` |
| Langfuse Cloud | traces | free tier, check current limits | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| GitHub | CI secrets | free | repo secrets listed in the workflows |

Create a Modal secret named `huggingface` with `HF_TOKEN`. The deploy scripts call `modal.Secret.from_name("huggingface")`. Set a workspace spend limit in the Modal dashboard before `modal deploy`.

```bash
cp .env.example .env
# fill the values, then:
set -a && source .env && set +a
uv sync --extra dev
make ci
```

`make ci` must pass before you spend GPU time. It does not call OpenRouter, Hugging Face, or Modal.

## 1. Spider data

`make data` streams the questions and downloads the official SQLite zip (Google Drive id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`, linked from https://yale-lily.github.io/spider). GNU make does not accept extra flags, so the recipe itself passes `--download-sqlite --require-sqlite`.

```bash
uv sync --extra dev
make data
# same command, if you are not using make:
uv run python -m localllm.data.prep_spider \
  --out data/spider --download-sqlite --require-sqlite
```

`data/spider/manifest.json` should show train 6650, val 350, test 1034, and `"sqlite_missing": []`. The test file is the Hugging Face validation split. The zip and the database directory stay in `data/` and are not committed.

`make data-small` is the 200-row prompt-only subset. It does not download SQLite.

Optional tokenizer check, after `uv sync --extra train` (this downloads the tokenizer, not the full model weights if you only load the tokenizer):

```bash
uv run --extra train python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
uv run python -m localllm.data.prompts --jsonl data/spider/train.jsonl
```

The CI tokenizer is whitespace. Re-check that p95 is under 1,024 Qwen tokens before a long train. If it is not, lower `max_schema_chars` in prep and rebuild the jsonl.

## 2. QLoRA on a free T4 (Kaggle)

Account: **notsaaadkhan**. The notebook trains. It is not a 50-step smoke. `configs/train_qlora_kaggle.yaml` runs 2 epochs over the full 6,650 train rows (batch 1, gradient accumulation 16, about 416 steps per epoch, about 830 steps). `max_steps: 900` is a ceiling so a slow step still ends inside four hours. That ceiling is not a measured duration. Stop the Kaggle session yourself if the timer passes 4 hours.

Qwen2.5-1.5B-Instruct is public. `HF_TOKEN` is optional. Do not print it.

1. Sign in to https://www.kaggle.com as **notsaaadkhan**.
2. On your machine, zip the repo without the large or local directories:

```bash
git archive --format=zip --output localllm-ops.zip HEAD
```

3. Kaggle → Datasets → New Dataset. Upload `localllm-ops.zip`. Set the title to `localllm-ops`. Set the visibility to Private. Create.
4. Kaggle → Code → New Notebook.
5. Right-hand panel → Add Data → Your Datasets → `localllm-ops`.
6. Notebook options → Accelerator → GPU T4. Internet → On.
7. Optional: Add-ons → Secrets → add `HF_TOKEN`. Leave it blank if you do not have one.
8. File → Import Notebook → upload `notebooks/train_kaggle.ipynb` from the zip, or paste that file over the empty notebook.
9. Run all. The first code cell finds `/kaggle/input/localllm-ops` or `/kaggle/input/datasets/notsaaadkhan/localllm-ops`. The prep cell downloads Spider and asserts train 6650, val 350, test 1034, and an empty `sqlite_missing` list. The last cell runs:

```bash
python -m localllm.train.sft --config configs/train_qlora_kaggle.yaml --run
```

10. When it finishes, download `outputs/qlora-kaggle` from the notebook output (Save Version → Output, or the file browser). Record the wall time you saw on the Kaggle meter in `NOTES.md`. Do not invent it.

`configs/train_qlora_t4.yaml` remains the 50-step CPU dry-run config. Do not use it for this session.

Push the adapter only if you want it online:

```bash
huggingface-cli login
huggingface-cli upload notsaaadkhan/qwen2.5-1.5b-spider-qlora outputs/qlora-kaggle
```

## 3. QLoRA on Colab

Runtime → T4 GPU. `notebooks/train_colab.ipynb` is the 50-step smoke. For the 2-epoch run, use the Kaggle notebook and `configs/train_qlora_kaggle.yaml`. Secrets: `HF_TOKEN` is optional because the base model is public.

## 4. Train on Modal (spends credit)

```bash
pip install modal
modal setup
modal secret create huggingface HF_TOKEN="$HF_TOKEN"
# put train.jsonl and val.jsonl on the volume first, or let the T4 config
# read data/spider_small that you uploaded with modal volume
modal run deploy/modal_train.py --max-steps 50
```

Detach a longer run only after the 50-step job prints a loss and the volume has a checkpoint:

```bash
modal run --detach deploy/modal_train.py --config-name train_qlora.yaml --max-steps -1
```

Stop it from the Modal dashboard if you walk away. `max_containers` on the train function is 1.

## 5. Serve with vLLM on Modal (spends credit)

```bash
modal deploy deploy/modal_vllm.py
```

The app name is `localllm-ops-vllm`. `min_containers=0` and `scaledown_window=120` mean idle cost goes to zero after two minutes. `max_containers=1`. Confirm both in the dashboard. Measure cold start with:

```bash
curl -sS "$MODAL_URL/v1/models"
```

Put the URL in `VLLM_BASE_URL` (include `/v1` if you point the harness at it; the Modal server itself is the vLLM port).

Local GPU, if you have one:

```bash
uv run python -m localllm.serve.vllm_cmd --model Qwen/Qwen2.5-1.5B-Instruct
# then run the printed command inside the vLLM environment
```

LoRA:

```bash
uv run python -m localllm.serve.vllm_cmd \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --lora-modules spider-sql=outputs/qlora-t4
```

## 6. Evaluate

Mock and tiny (no key, no GPU), already what CI runs:

```bash
uv run python -m localllm.eval.harness --backend mock --limit 20 --run-id ci-mock --out results/ci
uv run python -m localllm.eval.harness --backend tiny --limit 20 --run-id ci-tiny --out results/ci
```

Base model against a live vLLM:

```bash
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export LOCALLLM_GPU_USD_PER_HOUR="0.80"   # only if this is the price you are paying
uv run python -m localllm.eval.harness \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/spider/test.jsonl \
  --database-dir data/spider/database \
  --temperature 0 \
  --run-id base-qwen \
  --out results
```

Use `--limit 20` the first time. The full 1,034 is the number that goes in the README, and only after you have checked a sample by hand.

OpenRouter baselines (GPT-4o-mini and GPT-5-mini) use `python -m localllm.eval.baseline`. Exact commands, the dollar cap, and Langfuse are in `BASELINE_RUN.md`. The script writes `results/final/quality.csv` and `results/final/cost.csv` from `usage.cost`. Start with `--limit 20` and a small `--max-usd`.

Copy ten failures after that run:

```bash
uv run python -m localllm.eval.error_analysis \
  --run results/final/openai-gpt-4o-mini.jsonl \
  --out results/error_analysis.md \
  --limit 10
```

Then edit the one-line reasons yourself. The heuristic label is a starting point, not the write-up.

When a full run looks right, write `results/baselines/live_ex.json`:

```json
{"execution_accuracy": 0.0}
```

Use the measured value. Nightly CI (`.github/workflows/nightly_eval.yml`) fails if a later 100-example run drops more than 0.03 from that file. Add GitHub secrets `EVAL_ENDPOINT` or `VLLM_BASE_URL` only when the server exists.

## 7. AWQ

On a GPU machine, after the merged or base weights are on disk:

```bash
uv pip install llmcompressor transformers torch datasets
uv run python -m localllm.quant.awq --dry-run --train-jsonl data/spider/train.jsonl --samples 256
uv run python -m localllm.quant.awq --run \
  --model outputs/merged \
  --train-jsonl data/spider/train.jsonl \
  --samples 256 \
  --output-dir outputs/awq
```

Record the directory size (`du -sh outputs/awq`). Load it in vLLM with `--quantization compressed-tensors` (or `awq` if the saved config says so). Re-run the harness. Write the row into `results/final/quality.csv` yourself from the summary JSON.

## 8. GGUF

```bash
git clone https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build"
cmake --build "$HOME/llama.cpp/build" --target llama-quantize -j
bash src/localllm/quant/gguf.sh outputs/merged outputs/gguf
"$HOME/llama.cpp/build/bin/llama-server" -m outputs/gguf/model-Q4_K_M.gguf --host 127.0.0.1 --port 8081
export LLAMACPP_BASE_URL="http://127.0.0.1:8081/v1"
uv run python -m localllm.eval.harness --backend llamacpp --data data/spider/test.jsonl \
  --database-dir data/spider/database --limit 20 --run-id gguf-q4-dev --out results
```

Repeat for `model-Q8_0.gguf`. For CPU tokens/s, note the llama-server log on a single request. That number is the "runs on a laptop" figure. Do not estimate it.

## 9. Gateway in front of a real backend

```bash
export GATEWAY_API_KEY="$(openssl rand -hex 24)"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export GATEWAY_FALLBACK_BACKEND="gpt-4o-mini"   # only if OPENROUTER_API_KEY is set
export LOCALLLM_DATABASE_DIR="data/spider/database"
uv run uvicorn localllm.gateway.app:app --host 127.0.0.1 --port 8747
```

```bash
curl -sS http://127.0.0.1:8747/healthz
curl -sS http://127.0.0.1:8747/v1/sql \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "content-type: application/json" \
  -d '{"question":"How many students are there?","db_id":"school","backend":"tiny","execute":true}'
```

`school` exists only in the built-in fixture. For Spider, pass `"schema"` or a real `db_id` present under `LOCALLLM_DATABASE_DIR`.

Docker image:

```bash
docker build -f deploy/docker/Dockerfile.gateway -t localllm-gateway .
docker run --rm -p 8747:8747 --env-file .env localllm-gateway
```

The GitHub workflow `Gateway image` pushes to GHCR on `main` when gateway files change. `Modal deploy` runs on tags `v*` and needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

## 10. Load test

Warm the server first (a few curl calls). Then, against the GPU endpoint, not the tiny model:

```bash
export LOCUST_HOST="https://your-modal-host"
export LOCUST_BACKEND="base"
export GATEWAY_API_KEY="..."
# On the gateway process, set GATEWAY_RATE_LIMIT=0 for the load test
# (0 disables the in-memory limit) or set it above the rate you intend to send.
# one level at a time, 3 minutes, after the server is warm
for users in 1 4 8 16 32 64; do
  uv run locust -f loadtest/locustfile.py --headless \
    -u "$users" -r "$users" -t 3m \
    --host "$LOCUST_HOST" \
    --csv "results/loadtest-${users}"
done
```

`make loadtest` is the short local version (tiny backend, about 20 seconds) so you can see the CSV format. Do not put those rows in the README serving table.

Find the concurrency where p95 jumps. That is the knee. Write it in the README in your own sentence after you have the CSV.

## 11. Prometheus, Grafana, Langfuse

```bash
docker compose -f deploy/monitoring/docker-compose.yml up -d
```

Prometheus scrapes the gateway on port 8747 and vLLM on port 8000 (`deploy/monitoring/prometheus.yml`). Grafana is on port 3001. The dashboard `LocalLLM Ops` is provisioned from `deploy/monitoring/grafana/dashboards/localllm.json`. Alert rules: gateway p95 above 2 seconds, or error rate above 2 percent (`deploy/monitoring/alerts.yml`).

Screenshot the dashboard during a real Locust run and save it under `docs/`. That file is intentionally absent until the run happens.

Langfuse:

```bash
uv sync --extra observe
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

Restart the gateway. One `/v1/sql` call should show up as a trace with backend, model, prompt, SQL, validation, latency, tokens, and cost. Save a view that compares backends. If the keys are empty, the gateway logs the same fields as JSON and does not call Langfuse.

## 12. Publish the numbers

1. Copy summary numbers into `results/final/quality.csv`, `serving.csv`, and `cost.csv`.
2. `make results-table` rewrites the README block between the HTML comments.
3. Run `notebooks/analysis.ipynb` and commit the charts it writes.
4. Update `NOTES.md` in your own words, including whether the one change you tried (rank, epochs, foreign keys, sample rows) helped.
5. Model card on Hugging Face: data, license, metrics, limitations, contamination note.

## What CI already proved without you

- ruff and pytest
- execution match, including a second SQLite file in the same directory
- a 20-example mock backend with execution accuracy 1.0
- a 20-example keyword backend with a non-perfect execution accuracy (the SQL really runs)
- gateway rejection of `DROP`, `UPDATE`, `DELETE`, `INSERT`, multiple statements, and prompt text that asks the model to drop a table
- read-only execution, SQL timeout, auth, rate limit, fallback when the primary backend raises
- train config load and a dry-run manifest with the git commit
- AWQ recipe JSON and the vLLM command line
