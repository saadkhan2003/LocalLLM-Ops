import json
import zipfile
from pathlib import Path

import pytest

from localllm.data.fixtures import materialize_fixture
from localllm.data.spider_sqlite import parse_drive_confirm, place_sqlite_tree
from localllm.eval.baseline import run_baseline, sample_rows, should_stop_for_budget
from localllm.eval.harness import Completion, load_jsonl
from localllm.eval.openrouter import (
    chat_request_body,
    cost_from_usage,
    default_reasoning_effort,
    resolve_cost,
)
from localllm.observability.tracing import NullTracer
from localllm.train.config import load_train_config


class _Meter:
    def __init__(self, costs: list[float]) -> None:
        self.costs = costs
        self.calls = 0

    def complete(self, messages):
        cost = self.costs[self.calls]
        self.calls += 1
        return Completion(
            text="SELECT 1",
            input_tokens=100,
            output_tokens=2,
            latency_s=0.01,
            cost_usd=cost,
            model="openai/gpt-4o-mini",
            token_count_method="api_usage",
            cost_source="openrouter_usage.cost",
        )


class _RecordingTracer:
    def __init__(self) -> None:
        self.events = []
        self.flushed = False

    def trace(self, payload):
        self.events.append(payload)

    def flush(self):
        self.flushed = True


def test_openrouter_usage_cost_beats_the_price_table():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "cost": 0.002}
    cost, source = cost_from_usage(usage)
    assert cost == 0.002
    assert source == "openrouter_usage.cost"
    tokens_in, tokens_out, resolved, resolved_source = resolve_cost(
        usage, input_usd_per_million=0.15, output_usd_per_million=0.60
    )
    assert (tokens_in, tokens_out, resolved, resolved_source) == (
        1_000_000,
        1_000_000,
        0.002,
        "openrouter_usage.cost",
    )
    details = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "cost_details": {
            "upstream_inference_prompt_cost": 0.01,
            "upstream_inference_completions_cost": 0.02,
        },
    }
    assert cost_from_usage(details) == (0.03, "openrouter_usage.cost_details")
    _, _, fallback, source = resolve_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        input_usd_per_million=0.15,
        output_usd_per_million=0.60,
    )
    assert fallback == 0.15
    assert source == "price_table"


def test_reasoning_effort_defaults():
    assert default_reasoning_effort("openai/gpt-5-mini", None) == "minimal"
    assert default_reasoning_effort("openai/gpt-5-mini", "low") == "low"
    assert default_reasoning_effort("openai/gpt-4o-mini", None) is None
    body = chat_request_body("openai/gpt-5-mini", [], 0, "minimal")
    assert body["reasoning"] == {"effort": "minimal"}
    assert body["reasoning_effort"] == "minimal"
    plain = chat_request_body("openai/gpt-4o-mini", [], 0, None)
    assert "reasoning" not in plain


def test_sample_rows_is_stable_for_a_seed():
    rows = [
        {"db_id": "b", "question": "q2", "gold_sql": "SELECT 2"},
        {"db_id": "a", "question": "q1", "gold_sql": "SELECT 1"},
        {"db_id": "c", "question": "q3", "gold_sql": "SELECT 3"},
    ]
    assert sample_rows(rows, 2, 42) == sample_rows(rows, 2, 42)
    assert sample_rows(rows, 2, 42) != sample_rows(rows, 2, 7)
    assert len(sample_rows(rows, None, 42)) == 3


def test_budget_guard_stops_before_the_next_call(tmp_path: Path):
    assert should_stop_for_budget(0.04, 0.04, 0.05)
    assert not should_stop_for_budget(0.0, None, 0.05)
    assert should_stop_for_budget(0.0, None, 0.0)
    root = materialize_fixture(tmp_path / "fix")
    rows = load_jsonl(root / "test.jsonl", None)
    backend = _Meter([0.04, 0.04, 0.04])
    tracer = _RecordingTracer()
    summary = run_baseline(
        rows=rows,
        database_dir=root / "database",
        backend=backend,
        out_dir=tmp_path / "final",
        model="openai/gpt-4o-mini",
        max_usd=0.05,
        seed=42,
        tracer=tracer,
    )
    assert backend.calls == 1
    assert summary["stopped_reason"] == "budget"
    assert summary["n"] == 1
    assert summary["total_cost_usd"] == 0.04
    quality = (tmp_path / "final" / "quality.csv").read_text(encoding="utf-8")
    cost = (tmp_path / "final" / "cost.csv").read_text(encoding="utf-8")
    assert "openai/gpt-4o-mini" in quality
    assert "openrouter_usage.cost" in cost
    assert "budget" in cost
    assert tracer.events and tracer.events[0]["model"] == "openai/gpt-4o-mini"
    assert tracer.flushed is True
    assert (
        json.loads((tmp_path / "final" / "openai-gpt-4o-mini.jsonl").read_text().splitlines()[0])[
            "cost_source"
        ]
        == "openrouter_usage.cost"
    )


def test_zero_budget_sends_nothing(tmp_path: Path):
    root = materialize_fixture(tmp_path / "fix")
    rows = load_jsonl(root / "test.jsonl", 2)
    backend = _Meter([0.01])
    with pytest.raises(SystemExit):
        run_baseline(
            rows=rows,
            database_dir=root / "database",
            backend=backend,
            out_dir=tmp_path / "final",
            model="openai/gpt-4o-mini",
            max_usd=0,
            seed=1,
            tracer=NullTracer(),
        )
    assert backend.calls == 0


def test_kaggle_config_caps_steps():
    config = load_train_config(Path("configs/train_qlora_kaggle.yaml"))
    assert config.method == "qlora"
    assert config.max_steps == 900
    assert config.num_train_epochs == 2
    assert config.max_train_examples is None
    assert config.train_jsonl == "data/spider/train.jsonl"


def test_drive_form_and_sqlite_layout(tmp_path: Path):
    html = """
    <form id="download-form" action="https://drive.usercontent.google.com/download">
      <input type="hidden" name="confirm" value="t">
      <input type="hidden" name="uuid" value="abc-123">
    </form>
    """
    assert parse_drive_confirm(html) == ("t", "abc-123")
    extract = tmp_path / "zip"
    folder = extract / "spider" / "database" / "school"
    folder.mkdir(parents=True)
    db = folder / "school.sqlite"
    db.write_bytes(b"SQLite format 3\x00" + b"rest")
    (folder / "._school.sqlite").write_bytes(b"\x00\x05Mac OS X")
    archive_path = tmp_path / "spider_data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(db, "spider/database/school/school.sqlite")
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(unpacked)
    placed = place_sqlite_tree(unpacked, tmp_path / "database")
    assert placed == 1
    assert (tmp_path / "database" / "school" / "school.sqlite").is_file()
    assert not (tmp_path / "database" / "school" / "._school.sqlite").exists()
