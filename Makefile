UV ?= uv
PORT ?= 8747
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: help lint format test smoke-eval ci data data-small train-dry eval serve loadtest \
	results-table awq-dry token-stats

help:
	@echo "LocalLLM Ops"
	@echo "  make lint            ruff check + format check"
	@echo "  make test            pytest"
	@echo "  make smoke-eval      CPU eval on the 20-example fixture (mock + tiny)"
	@echo "  make ci              lint + test + smoke-eval"
	@echo "  make data-small      stream a 200-row Spider subset (network, no GPU)"
	@echo "  make data            stream the full Spider splits (network, no GPU)"
	@echo "  make train-dry       validate a train config without loading a model"
	@echo "  make serve           gateway on port $(PORT)"
	@echo "  make loadtest        short Locust run against LOCUST_HOST or localhost:$(PORT)"
	@echo "  make results-table   refresh the placeholder results tables"

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest

smoke-eval:
	$(UV) run python -m localllm.eval.harness --backend mock --limit 20 --run-id ci-mock --out results/ci
	$(UV) run python -m localllm.eval.harness --backend tiny --limit 20 --run-id ci-tiny --out results/ci

ci: lint test smoke-eval

data:
	$(UV) run python -m localllm.data.prep_spider --out data/spider

data-small:
	$(UV) run python -m localllm.data.prep_spider --out data/spider_small --limit 200

train-dry:
	$(UV) run python -m localllm.train.sft --config configs/train_qlora_t4.yaml --dry-run --max-steps 50

eval:
	$(UV) run python -m localllm.eval.harness --config configs/eval.yaml

serve:
	$(UV) run uvicorn localllm.gateway.app:app --host 0.0.0.0 --port $(PORT)

loadtest:
	@echo "The gateway must already be running. Set GATEWAY_RATE_LIMIT=0 on that process for a real ramp."
	$(UV) run locust -f loadtest/locustfile.py --headless -u $${USERS:-4} -r $${SPAWN:-2} \
		-t $${DURATION:-20s} --host $${LOCUST_HOST:-http://127.0.0.1:$(PORT)} \
		--csv results/loadtest --html results/loadtest.html

results-table:
	$(UV) run python -m localllm.eval.render_tables --readme README.md

awq-dry:
	$(UV) run python -m localllm.quant.awq --dry-run --train-jsonl tests/fixtures/prompts.jsonl --samples 4

token-stats:
	$(UV) run python -m localllm.data.prompts --jsonl data/spider/train.jsonl || \
		$(UV) run python -m localllm.data.prompts --fixture
