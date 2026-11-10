# NOTES

Build decisions for this checkout. The interview section at the bottom is blank on purpose: write those answers yourself before this goes on a CV.

## Why the harness does not vendor `test-suite-sql-eval`

The official repo's execution metric is: run gold and prediction on every SQLite file in the database directory, compare bags of rows, allow a column permutation, and treat row order as significant when the gold SQL contains `ORDER BY`. `DISTINCT` is stripped unless `--keep_distinct` is set. `--plug_value` is off by default.

`src/localllm/eval/execution.py` follows that. Two differences are documented in the module:

- Column-permutation search is deterministic. The official code samples rows with `random.choice` once a result has more than three columns, so the same pair can theoretically flicker. This project enumerates permutations up to six columns.
- Exact match in the summary is whitespace-normalized string equality. Official exact match is clause-level and needs Spider's SQL parser (`process_sql.py`). The summary field `exact_match_definition` says so. Do not quote it as the Spider leaderboard exact-match number.

## Why prompts can be built before the SQLite zip exists

`xlangai/spider` on Hugging Face is questions and SQL only (7,000 train, 1,034 validation). `make data` downloads the official zip, drops the macOS `._*` sidecars, and lays out `data/spider/database/<db_id>/<db_id>.sqlite`. When that file exists, the schema is read from SQLite. `manifest.json` lists `sqlite_missing`. `--require-sqlite` refuses to finish while any example database is missing.

The 5% validation holdout is shuffled with seed 42 after sorting, so a re-download does not reshuffle. The HF validation split is written to `test.jsonl` and is not concatenated into train.

## Why `tiny` exists

CI and the demo page have to run on CPU with no keys. `TinySqlModel` is a keyword-to-SQL function with a fixed execution accuracy on a 20-question fixture that this repo creates. It is named `tiny-keyword-sql` in the summary so it cannot be confused with Qwen. The README quality table does not include that score.

## Why training imports torch only under `--run`

`localllm.train.config` and `formatting` are pure Python so unit tests and `make train-dry` work in CI. `--run` imports TRL, PEFT, and bitsandbytes. The T4 config is QLoRA, NF4, fp16, batch 1, gradient accumulation 16, max sequence 1024, 50 steps. That is a smoke length, not the result you publish. Assistant tokens are the only trained span (`assistant_only_loss`, and `mask_prompt_labels` is the tested implementation of that mask).

## Why the gateway fails closed on SQL

`sqlglot` must see exactly one statement, and that statement must be `SELECT`, `UNION`, `INTERSECT`, or `EXCEPT`. A semicolon plus a second statement is rejected, including `SELECT 1; DROP TABLE`. A string literal that happens to contain the word DROP is allowed. Execution opens SQLite read-only and sets `PRAGMA query_only`. Timeouts use the SQLite progress handler.

Auth is off when `GATEWAY_API_KEY` is empty, and the process logs that. Set the key before any port is exposed. Rate limiting is in-memory, per key or per client host. It will not coordinate across multiple gateway processes.

If the chosen backend raises, and `GATEWAY_FALLBACK_BACKEND` names another configured backend, the response sets `fallback: true`. With Langfuse keys and the `langfuse` package, that flag is on the trace. Without keys, the same payload is a JSON log line.

## Why Modal code is not deployed from this environment

`deploy/modal_train.py` and `deploy/modal_vllm.py` target the current Modal API (`App`, `Image.uv_pip_install`, `@app.function`, `@app.server`, `min_containers=0`, `scaledown_window`, `max_containers=1`, `Secret.from_name`). Deploying them creates resources and can bill a workspace. `RUNBOOK.md` is the command list. CI only deploys on a `v*` tag, and only if the Modal secrets exist in GitHub.

## Interview answers

Write these in your own words. If you cannot, study that part before the project is on your CV.

1. LoRA vs QLoRA. What `r` and `alpha` mean, which modules you targeted, why NF4 saves memory, and what it costs.
2. SFT. Chat templates, loss masking on prompt tokens, learning rate, and what overfitting looked like on your val curve.
3. Execution accuracy vs exact match. Why two different SQL strings can both be right, and how the test-suite evaluator uses more than one database file.
4. vLLM. PagedAttention, continuous batching, KV cache, and why throughput rises while latency gets worse as concurrency grows.
5. TTFT vs time per output token. p50 vs p95, and why p95 is the number you would put in an SLO.
6. Quantization. Weight-only INT4 (W4A16), what the calibration set is doing in AWQ, GGUF K-quants, and when you would serve on CPU.
7. Cost. `$/hour` divided by requests per hour, and the traffic level where the API is cheaper than the GPU you measured.
8. Observability. A trace vs a metric vs a log, and what you would page on.
9. Security. Prompt injection against text-to-SQL, a read-only database connection, and an allow-list of statements.
