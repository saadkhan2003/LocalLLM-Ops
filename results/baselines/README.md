# Baselines

The OpenRouter full-split scores live in `results/final/`. `live_ex.json` is created by you after the first served-model run you trust. Do not copy the GPT-4o-mini score into that file. The nightly job compares a later vLLM sample with this file.

```json
{"execution_accuracy": 0.0}
```

Replace `0.0` with the measured execution accuracy. The nightly workflow
(`.github/workflows/nightly_eval.yml`) exits non-zero when a later 100-example
run drops more than 0.03 absolute from that file.

Do not commit a guessed number. Leave this file as documentation until the run exists.
