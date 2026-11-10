# Baselines

`live_ex.json` is created by you after the first full test-split run you trust.

```json
{"execution_accuracy": 0.0}
```

Replace `0.0` with the measured execution accuracy. The nightly workflow
(`.github/workflows/nightly_eval.yml`) exits non-zero when a later 100-example
run drops more than 0.03 absolute from that file.

Do not commit a guessed number. Leave this file as documentation until the run exists.
