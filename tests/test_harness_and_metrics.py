import json

from localllm.data.fixtures import fixture_examples, materialize_fixture
from localllm.eval.harness import MockGoldBackend, TinyBackend, load_jsonl, run_harness
from localllm.eval.metrics import (
    break_even_requests_per_second,
    cost_per_1k_queries,
    openrouter_cost_usd,
    percentile,
    self_hosted_cost_usd,
)
from localllm.eval.tiny_model import TinySqlModel


def test_percentile_and_cost_formulas():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert openrouter_cost_usd(1_000_000, 1_000_000, 0.15, 0.60) == 0.75
    assert self_hosted_cost_usd(1.0, 3600) == 1.0
    assert cost_per_1k_queries(3.6, 1.0) == 1.0
    # 3.6 USD/hour at 1 request/s is 3.6/3600 per request, times 1000.
    rate = break_even_requests_per_second(3.6, 1.0)
    assert abs(cost_per_1k_queries(3.6, rate) - 1.0) < 1e-9


def test_mock_backend_is_perfect_and_deterministic(tmp_path):
    root = materialize_fixture(tmp_path / "fix")
    rows = load_jsonl(root / "test.jsonl", 20)
    summary = run_harness(
        backend_name="mock",
        rows=rows,
        database_dir=root / "database",
        out_dir=tmp_path / "results",
        run_id="mock",
        temperature=0,
    )
    again = run_harness(
        backend_name="mock",
        rows=rows,
        database_dir=root / "database",
        out_dir=tmp_path / "results",
        run_id="mock",
        temperature=0,
    )
    assert summary["n"] == 20
    assert summary["execution_accuracy"] == 1
    assert summary["invalid_sql_rate"] == 0
    assert summary["temperature"] == 0
    left = (tmp_path / "results" / "mock.jsonl").read_text(encoding="utf-8")
    assert again["execution_accuracy"] == 1
    assert left == (tmp_path / "results" / "mock.jsonl").read_text(encoding="utf-8")
    csv_text = (tmp_path / "results" / "summary.csv").read_text(encoding="utf-8")
    assert "execution_accuracy" in csv_text
    assert csv_text.strip().count("\n") == 1


def test_tiny_cpu_smoke_has_a_real_execution_score(tmp_path):
    root = materialize_fixture(tmp_path / "fix")
    rows = load_jsonl(root / "test.jsonl", 20)
    summary = run_harness(
        backend_name="tiny",
        rows=rows,
        database_dir=root / "database",
        out_dir=tmp_path / "results",
        run_id="tiny",
    )
    assert summary["n"] == 20
    assert summary["execution_accuracy"] == 0.55
    assert summary["exact_match"] == 0.55
    assert summary["invalid_sql_rate"] == 0
    assert summary["model"] == "tiny-keyword-sql"
    assert "easy" in summary["difficulty"]


def test_tiny_model_known_queries():
    schema = "Table students\n  name TEXT\n  age INTEGER"
    model = TinySqlModel()
    assert model.generate(schema, "How many students are older than 18?") == (
        "SELECT COUNT(*) FROM students WHERE age > 18"
    )
    assert (
        model.generate(schema, "What are the names of the students?") == "SELECT name FROM students"
    )


def test_backends_do_not_require_network():
    assert (
        TinyBackend()
        .complete(
            [
                {
                    "role": "user",
                    "content": "Schema:\nTable t\n  id INTEGER\n\nQuestion: How many t are there?",
                }
            ]
        )
        .text.startswith("SELECT")
    )
    gold = MockGoldBackend().complete([], gold_sql="SELECT 1")
    assert gold.text == "SELECT 1"
    assert len(fixture_examples()) == 20


def test_recorded_row_shape(tmp_path):
    root = materialize_fixture(tmp_path / "fix")
    rows = load_jsonl(root / "test.jsonl", 1)
    run_harness(
        backend_name="tiny",
        rows=rows,
        database_dir=root / "database",
        out_dir=tmp_path / "results",
        run_id="one",
    )
    row = json.loads((tmp_path / "results" / "one.jsonl").read_text(encoding="utf-8"))
    for key in ("pred_sql", "execution_accuracy", "latency_ms", "input_tokens", "cost_usd"):
        assert key in row
