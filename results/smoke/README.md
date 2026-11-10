# CPU fixture smoke

These files are the keyword backend (`tiny-keyword-sql`) on the 20-question fixture built by `localllm.data.fixtures`. They are not Qwen, LoRA, or GPT-4o-mini scores.

Measured by `python -m localllm.eval.harness --backend tiny --limit 20 --run-id ci-tiny`:

- execution accuracy 0.55
- exact match 0.55 (whitespace-normalized string match)
- invalid SQL rate 0

Re-run with `make smoke-eval`. The harness writes `results/ci/` (gitignored). This directory is the copy of that tiny run.

A separate localhost check of `POST /v1/sql` with this same backend, execute on, 40 requests: p50 2.15 ms, p95 3.05 ms. That is gateway plus keyword SQL, not vLLM.
