import ast
import json
from pathlib import Path

import yaml

from localllm.eval.error_analysis import classify
from localllm.eval.regression import assert_no_regression
from localllm.eval.render_tables import render_block
from localllm.quant.awq import SCHEME, awq_recipe, dry_run
from localllm.serve.vllm_cmd import build_vllm_command
from localllm.train.config import load_train_config
from localllm.train.formatting import example_messages, mask_prompt_labels
from localllm.train.sft import dry_run as train_dry_run


def test_lora_and_qlora_configs_load():
    lora = load_train_config(Path("configs/train_lora.yaml"))
    assert lora.method == "lora"
    assert lora.lora_r == 16
    assert lora.lora_alpha == 32
    assert lora.bf16 is True
    assert lora.load_in_4bit is False
    assert lora.assistant_only_loss is True
    qlora = load_train_config(Path("configs/train_qlora.yaml"))
    assert qlora.method == "qlora" and qlora.load_in_4bit is True
    t4 = load_train_config(Path("configs/train_qlora_t4.yaml"))
    assert t4.fp16 is True and t4.bf16 is False
    assert t4.max_steps == 50


def test_assistant_tokens_are_the_only_trained_span():
    row = {
        "schema": "Table students\n  id INTEGER primary key",
        "question": "How many students are there?",
        "gold_sql": "SELECT COUNT(*) FROM students",
    }
    messages = example_messages(row)
    assert messages[-1]["role"] == "assistant"
    labels = mask_prompt_labels([1, 2, 3, 4, 5], prompt_token_len=3)
    assert labels == [-100, -100, -100, 4, 5]


def test_train_dry_run_writes_manifest(tmp_path, monkeypatch):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        Path("configs/train_qlora_t4.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded["output_dir"] = str(tmp_path / "out")
    loaded["data"]["train_jsonl"] = str(Path("tests/fixtures/prompts.jsonl"))
    config_path.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    config = load_train_config(config_path)
    path = train_dry_run(config, Path("tests/fixtures/prompts.jsonl"), max_steps=50)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["train_rows"] == 2
    assert manifest["assistant_only_loss"] is True
    assert manifest["git_commit"]
    assert manifest["params"]["max_steps"] == 50


def test_awq_recipe_and_vllm_command():
    recipe = awq_recipe()
    assert recipe[1]["scheme"] == SCHEME
    assert recipe[1]["ignore"] == ["lm_head"]
    payload = dry_run(
        Path("tests/fixtures/prompts.jsonl"), 4, "Qwen/Qwen2.5-1.5B-Instruct", Path("outputs/awq")
    )
    assert payload["prompts_loaded"] == 2
    command = build_vllm_command(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization="compressed-tensors",
        lora_modules="spider-sql=/weights/adapter",
    )
    assert command[:3] == ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"]
    assert "--enable-lora" in command
    assert "--enable-prefix-caching" in command
    assert "compressed-tensors" in command


def test_regression_threshold():
    drop = assert_no_regression(0.93, 0.95, 0.03)
    assert abs(drop - 0.02) < 1e-9
    try:
        assert_no_regression(0.90, 0.95, 0.03)
    except SystemExit as exc:
        assert "dropped" in str(exc)
    else:
        raise AssertionError("expected a loud failure")


def test_error_labels_and_pending_tables():
    assert (
        classify("SELECT a FROM t JOIN u ON t.id = u.id", "SELECT a FROM t", None) == "wrong join"
    )
    block = render_block(Path("results/does-not-exist"))
    assert "pending" in block
    assert "GPT-4o-mini" in block


def test_deploy_files_have_no_secrets():
    root = Path(".")
    banned = ("sk-or-v1-", "sk-lf-", "AKIA")
    for path in list(root.glob("deploy/**/*.py")) + list(root.glob("src/**/*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text
        ast.parse(text)
    modal = (root / "deploy" / "modal_vllm.py").read_text(encoding="utf-8")
    assert "min_containers=0" in modal
    assert "scaledown_window" in modal
    assert "Secret.from_name" in modal
    assert "Qwen/Qwen2.5-1.5B-Instruct" in modal


def test_monitoring_artifacts():
    dashboard = json.loads(
        Path("deploy/monitoring/grafana/dashboards/localllm.json").read_text(encoding="utf-8")
    )
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert "Gateway p95 latency" in titles
    assert "KV cache usage" in titles
    alerts = Path("deploy/monitoring/alerts.yml").read_text(encoding="utf-8")
    assert "> 0.02" in alerts
    assert "HighP95Latency" in alerts
    gguf = Path("src/localllm/quant/gguf.sh").read_text(encoding="utf-8")
    assert "Q4_K_M" in gguf and "Q8_0" in gguf
