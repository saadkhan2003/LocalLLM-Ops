# results/final

Measured OpenRouter baselines from 10 Nov 2026 are checked in:

- `quality.csv` — execution accuracy, exact match, invalid SQL percent, n=1034
- `cost.csv` — total USD and USD per 1k from `usage.cost`
- `openai-gpt-4o-mini.summary.json`
- `openai-gpt-5-mini.summary.json` — reasoning effort `minimal`

`make results-table` rewrites the README block from these files. Qwen rows stay pending until a trained adapter is scored. Serving and self-hosted cost stay pending until a GPU serve run exists.

Per-row `*.jsonl` files are gitignored. Do not add them.
