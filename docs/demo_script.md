# Two-minute demo script

The OpenRouter baselines are measured (10 Nov 2026, Spider test split, n=1,034, seed 42, temperature 0). GPT-4o-mini execution accuracy is 0.762. GPT-5-mini with reasoning effort minimal is 0.756. Exact match is about 0.06 because it is a string comparison. Use execution accuracy when you say the number.

Do not record this until the fine-tuned adapter is served. The local tiny-model page is a keyword stand-in. Do not present it as Qwen.

## What you can say today

- The harness scored both OpenRouter models on all 1,034 test questions.
- GPT-4o-mini cost $0.061 total, $0.059 per 1k queries. GPT-5-mini cost $0.178 total, $0.172 per 1k.
- Latency was about 1.1 s p50 and 1.9 s p95 for GPT-4o-mini, and about 1.2 s / 2.0 s for GPT-5-mini.
- Qwen, LoRA, QLoRA, AWQ, and GGUF rows are still pending. Kaggle training has not run.

## Shot list, after the adapter and the Modal endpoint exist

- 0:00–0:15 — One sentence: English question in, SQL out. Quote the execution-accuracy number from the README, including the OpenRouter comparison.
- 0:15–0:50 — Demo UI: the same question to the base model, the fine-tuned model, and GPT-4o-mini. Show the SQL and the rows.
- 0:50–1:20 — Grafana while Locust is running against the served model.
- 1:20–1:40 — One Langfuse trace: backend, SQL, validation, tokens, cost. This needs `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.
- 1:40–2:00 — The results table. Say that exact match is a string match and execution accuracy is the metric. Close on measured cost per 1k, and leave self-hosted cost out until that cell is filled.
