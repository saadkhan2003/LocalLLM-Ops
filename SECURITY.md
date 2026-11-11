# Security policy

LocalLLM Ops is an in-development engineering project, not a production-audited
database access service. The SQL allow-list and read-only SQLite execution are
defense layers, not permission to expose confidential databases to arbitrary
requests. There is no supported stable-release security-maintenance schedule yet.

## Report privately

Email **msaad.official6@gmail.com** with the repository name, affected commit,
reproduction steps, expected impact, and a minimal example using synthetic data.
Do not publish exploit details, credentials, private schemas, or query logs in a
public issue. No response-time guarantee or bug-bounty program is offered.

## Safe operation

- Keep development ports private. Set `GATEWAY_API_KEY` before broader exposure.
- Restrict database files to data you are authorized to query; isolate the
  gateway from unrelated data and services.
- Keep `.env`, tokens, private datasets, and sensitive request traces out of Git.
  Redact secrets before sharing logs or screenshots.
- Treat model output as untrusted, even when it passes syntax validation.
- Configure provider spending limits and review deployment triggers explicitly.
- If a credential is exposed, revoke or rotate it first. Removing it from a
  current file does not remove it from Git history.

See [RUNBOOK.md](RUNBOOK.md) for configuration and the README for current limits.
