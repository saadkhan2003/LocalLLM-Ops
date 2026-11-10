"""Stream Spider questions and write train, val, and test jsonl files.

The Hugging Face ``validation`` split is the test set. It is never mixed into
training. Five percent of the HF ``train`` split is held out as ``val``.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

from localllm.data.schema import (
    resolve_sqlite,
    schema_from_schema_row,
    schema_from_sqlite,
    schema_from_tables_json,
    truncate_schema,
)

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "xlangai/spider"
HF_CONFIG = "spider"
SCHEMA_MIRROR_URL = (
    "https://huggingface.co/datasets/richardr1126/spider-schema/resolve/main/"
    "spider_schema_rows_v2.json"
)
# Official Spider release linked from https://yale-lily.github.io/spider on 10 Nov 2026.
SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
VAL_RATIO = 0.05
EXPECTED_TRAIN = 6650
EXPECTED_VAL = 350
EXPECTED_TEST = 1034


def split_train_val(
    rows: list[dict], val_ratio: float = VAL_RATIO, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Hold out ``val_ratio`` of rows. 7,000 rows and 0.05 yield 6,650 / 350."""
    ordered = sorted(rows, key=lambda row: (row["db_id"], row["question"], row["gold_sql"]))
    random.Random(seed).shuffle(ordered)
    n_val = int(round(len(ordered) * val_ratio))
    if len(ordered) > 1:
        n_val = min(n_val, len(ordered) - 1)
    return ordered[n_val:], ordered[:n_val]


def iter_hf_split(
    split: str,
    limit: int | None = None,
    page_size: int = 100,
    opener=None,
) -> Iterator[dict]:
    """Stream rows from the Hugging Face datasets server. No dataset download cache."""
    opener = opener or urllib.request.urlopen
    offset = 0
    produced = 0
    while True:
        if limit is not None and produced >= limit:
            return
        length = page_size if limit is None else min(page_size, limit - produced)
        query = (
            f"{HF_ROWS_URL}?dataset={HF_DATASET}&config={HF_CONFIG}"
            f"&split={split}&offset={offset}&length={length}"
        )
        with opener(query, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows") or []
        if not rows:
            return
        for item in rows:
            yield item["row"]
            produced += 1
            if limit is not None and produced >= limit:
                return
        offset += len(rows)
        if len(rows) < length:
            return


def load_schema_map(path: Path) -> dict[str, str]:
    """Load either official ``tables.json`` or the schema-row mirror."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    if isinstance(data, list) and data and "table_names_original" in data[0]:
        for entry in data:
            mapping[entry["db_id"]] = schema_from_tables_json(entry)
        return mapping
    if isinstance(data, list):
        for entry in data:
            mapping[entry["db_id"]] = schema_from_schema_row(entry)
        return mapping
    raise ValueError(f"unrecognized schema file: {path}")


def download_schema_mirror(dest: Path, opener=None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = opener or urllib.request.urlopen
    with opener(SCHEMA_MIRROR_URL, timeout=60) as response:
        dest.write_bytes(response.read())
    return dest


def example_from_row(
    row: dict,
    schema_map: dict[str, str],
    database_dir: Path | None,
    max_schema_chars: int,
) -> dict[str, str]:
    db_id = row["db_id"]
    schema = ""
    if database_dir is not None:
        sqlite_path = resolve_sqlite(database_dir, db_id)
        if sqlite_path is not None:
            schema = schema_from_sqlite(sqlite_path)
    if not schema:
        schema = schema_map.get(db_id, "")
    schema, _truncated = truncate_schema(schema, max_schema_chars) if schema else ("", False)
    return {
        "db_id": db_id,
        "schema": schema,
        "question": row["question"],
        "gold_sql": row["query"],
    }


def missing_databases(rows: Iterable[dict], database_dir: Path) -> list[str]:
    missing = []
    for db_id in sorted({row["db_id"] for row in rows}):
        if resolve_sqlite(database_dir, db_id) is None:
            missing.append(db_id)
    return missing


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def prepare(
    out_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 42,
    database_dir: Path | None = None,
    schema_path: Path | None = None,
    max_schema_chars: int = 6000,
    require_sqlite: bool = False,
    fetch_split=None,
    schema_map: dict[str, str] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch_split = fetch_split or iter_hf_split
    if schema_map is None:
        if schema_path is None:
            schema_path = out_dir / "spider_schema.json"
            if not schema_path.exists():
                download_schema_mirror(schema_path)
        schema_map = load_schema_map(schema_path)

    train_rows = [
        example_from_row(row, schema_map, database_dir, max_schema_chars)
        for row in fetch_split("train", limit)
    ]
    test_rows = [
        example_from_row(row, schema_map, database_dir, max_schema_chars)
        for row in fetch_split("validation", limit)
    ]
    train_split, val_split = split_train_val(train_rows, seed=seed)
    counts = {
        "train": write_jsonl(out_dir / "train.jsonl", train_split),
        "val": write_jsonl(out_dir / "val.jsonl", val_split),
        "test": write_jsonl(out_dir / "test.jsonl", test_rows),
    }
    db_dir = database_dir or (out_dir / "database")
    missing = missing_databases(train_rows + test_rows, db_dir) if db_dir.exists() else sorted(
        {row["db_id"] for row in train_rows + test_rows}
    )
    manifest = {
        "counts": counts,
        "limit": limit,
        "seed": seed,
        "val_ratio": VAL_RATIO,
        "expected_full_counts": {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },
        "sqlite_missing": missing,
        "schema_source": str(schema_path) if schema_path else "injected",
        "official_sqlite_drive_id": SPIDER_DRIVE_ID,
        "note": (
            "HF validation is the test split and is not used for training. "
            "SQLite files are not in the HF dataset. Download the official "
            "Spider zip (Google Drive id above) for execution accuracy."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"train={counts['train']} val={counts['val']} test={counts['test']} "
        f"sqlite_missing={len(missing)}"
    )
    if require_sqlite and missing:
        raise SystemExit(
            f"{len(missing)} databases have no SQLite file under {db_dir}. "
            "See RUNBOOK.md for the official Spider download."
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider jsonl splits.")
    parser.add_argument("--out", type=Path, default=Path("data/spider"))
    parser.add_argument("--limit", type=int, default=None, help="Stream at most this many rows per split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--database-dir", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None, help="tables.json or schema mirror.")
    parser.add_argument("--max-schema-chars", type=int, default=6000)
    parser.add_argument("--require-sqlite", action="store_true")
    args = parser.parse_args()
    prepare(
        args.out,
        limit=args.limit,
        seed=args.seed,
        database_dir=args.database_dir,
        schema_path=args.schema,
        max_schema_chars=args.max_schema_chars,
        require_sqlite=args.require_sqlite,
    )


if __name__ == "__main__":
    main()
