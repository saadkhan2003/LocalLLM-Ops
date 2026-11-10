# results/final

Put measured CSV files here after a real GPU or OpenRouter run. `make results-table` reads them and rewrites the README block.

Expected files, written by you from harness and load-test output (do not invent rows):

- `quality.csv` — one row per model: model, execution accuracy, exact match, invalid SQL percent, notes
- `serving.csv` — disk, GPU memory, TTFT, end-to-end latency, tokens/s, cold start
- `cost.csv` — training hours, training cost, cost per 1k queries, break-even rate

Until those files exist, the README tables stay `pending`.
