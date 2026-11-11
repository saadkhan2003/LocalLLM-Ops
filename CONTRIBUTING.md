# Contributing

Contributions should improve reproducibility, evaluation correctness, or the
text-to-SQL request path. This project is in development; a training or deployment
script is not evidence that a model has been trained or a service deployed.

## Development

Use a Codespace or Linux cloud workspace with Python 3.11+, uv, and Make. Keep
model downloads, datasets, and GPU jobs out of the laptop environment if following
the cloud-first workflow.

```bash
uv sync --extra dev
make ci
```

The CPU checks do not require real API credentials or GPU model weights. For a
behavior change, add a focused regression test, then run the complete CPU checks.
Do not remove safeguards or weaken a test just to obtain a passing result.

## Pull requests

Explain the problem, the changed behavior, and the commands actually executed.
Separate passed, failed, skipped, and unverified checks. Keep scope narrow and
avoid formatting unrelated files.

For evaluation changes, document the dataset split, sample count, seed, prompt,
backend/model, scoring protocol, and any cost assumptions. Keep synthetic-fixture
results separate from model benchmarks. Never tune on the evaluation split or
fill a pending result without a measured artifact.

The README results block is generated between `RESULTS:START` and `RESULTS:END`.
Update the source summaries/CSVs in `results/final/` and run `make results-table`;
do not hand-edit generated scores. Exclude private row-level logs, secrets, model
weights, and database archives from commits.

## Operational boundaries

Do not enable deployment workflows, invoke paid APIs, or start billable GPUs as
part of routine tests. GitHub Actions is currently disabled. Review credentials,
permissions, limits, and provider costs before intentionally running live work.

For sensitive issues, follow [SECURITY.md](SECURITY.md), not a public issue.
