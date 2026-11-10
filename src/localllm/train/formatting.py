"""Chat messages and completion-only labels. No torch import."""

from __future__ import annotations

import json
from pathlib import Path

from localllm.data.prompts import build_messages

IGNORE_INDEX = -100


def example_messages(row: dict) -> list[dict[str, str]]:
    return build_messages(row["schema"], row["question"], row["gold_sql"])


def load_sft_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({"messages": example_messages(row)})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def mask_prompt_labels(input_ids: list[int], prompt_token_len: int) -> list[int]:
    """Set prompt tokens to -100 so the loss is on the assistant SQL only."""
    if prompt_token_len < 0 or prompt_token_len > len(input_ids):
        raise ValueError("prompt_token_len is outside the token sequence")
    labels = list(input_ids)
    for index in range(prompt_token_len):
        labels[index] = IGNORE_INDEX
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("every token was masked; the assistant span is missing")
    return labels
