import json
from pathlib import Path

import yaml

from localllm.eval.render_tables import END, START, render_block


def test_readme_matches_measured_openrouter_files():
    block = render_block(Path("results/final"))
    assert "| Qwen2.5-1.5B base | pending | pending | pending | zero-shot |" in block
    assert "| + QLoRA | pending | pending | pending |  |" in block
    assert "| GPT-4o-mini (OpenRouter) | 0.762 | 0.064 | 0.68% |" in block
    assert "| GPT-5-mini (OpenRouter) | 0.756 | 0.059 | 0.68% |" in block
    assert "reasoning minimal" in block
    assert "p50/p95 1103/1898 ms" in block
    assert "p50/p95 1182/2015 ms" in block
    assert "$0.059 (measured 10 Nov 2026, OpenRouter usage.cost, total $0.061, n=1034)" in block
    assert (
        "$0.172 (measured 10 Nov 2026, OpenRouter usage.cost, reasoning minimal, total $0.178, n=1034)"
        in block
    )
    assert "Training GPU-hours | pending" in block
    assert "Self-hosted cost per 1k queries | pending" in block
    assert "list price" not in block
    assert "Execution accuracy is the main metric" in block
    text = Path("README.md").read_text(encoding="utf-8")
    rendered = text.split(START, 1)[1].split(END, 1)[0].strip()
    assert rendered == block


def test_missing_final_dir_stays_pending(tmp_path: Path):
    block = render_block(tmp_path)
    assert "GPT-5-mini (OpenRouter) | pending" in block
    assert "0.762" not in block
    assert "list price" not in block


def test_deploy_manifests_parse():
    roots = [Path("deploy"), Path(".github/workflows")]
    seen = 0
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix in {".yml", ".yaml"}:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                assert loaded, path
                seen += 1
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                seen += 1
    assert seen >= 8
