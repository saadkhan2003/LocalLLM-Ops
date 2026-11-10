import json
from pathlib import Path

from localllm.data.prompts import (
    build_messages,
    prompt_texts_from_jsonl,
    render_prompt,
    token_length_report,
    whitespace_tokenize,
)
from localllm.data.schema import schema_from_sqlite, truncate_schema

SCHEMA = "Table students\n  id INTEGER primary key\n  name TEXT"
QUESTION = "How many students are there?"
SQL = "SELECT COUNT(*) FROM students"
EXPECTED = (
    "system: You write SQLite SQL. Return only SQL.\n"
    "user: Schema:\n"
    "Table students\n"
    "  id INTEGER primary key\n"
    "  name TEXT\n"
    "\n"
    "Question: How many students are there?\n"
    "assistant: SELECT COUNT(*) FROM students"
)


def test_prompt_exact_string():
    assert render_prompt(SCHEMA, QUESTION, SQL) == EXPECTED
    messages = build_messages(SCHEMA, QUESTION, SQL)
    assert messages[0] == {"role": "system", "content": "You write SQLite SQL. Return only SQL."}
    assert messages[2]["content"] == SQL


def test_schema_from_sqlite_includes_keys(tmp_path: Path):
    from localllm.data.fixtures import materialize_fixture

    root = materialize_fixture(tmp_path)
    schema = schema_from_sqlite(root / "database" / "school" / "school.sqlite")
    assert "Table classes" in schema
    assert "id INTEGER primary key" in schema
    assert "students.class_id = classes.id" in schema


def test_truncate_keeps_tables_columns_and_keys():
    blocks = []
    for index in range(30):
        blocks.append(f"Table t{index}\n  id INTEGER primary key\n  name TEXT")
    blocks.append("Foreign keys:\n  t1.id = t0.id")
    schema = "\n".join(blocks)
    truncated, did = truncate_schema(schema, 180)
    assert did
    assert len(truncated) <= 180
    assert "Table t0" in truncated
    assert "-- schema truncated" in truncated


def test_fixture_prompt_p95_under_1024(tmp_path: Path):
    from localllm.data.fixtures import materialize_fixture

    root = materialize_fixture(tmp_path)
    texts = prompt_texts_from_jsonl(root / "test.jsonl")
    report = token_length_report(texts, whitespace_tokenize)
    assert report["n"] == 20
    assert report["p95"] <= 1024
    assert report["max"] <= 1024


def test_prompt_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"schema": SCHEMA, "question": QUESTION, "gold_sql": SQL}) + "\n",
        encoding="utf-8",
    )
    assert prompt_texts_from_jsonl(path) == [EXPECTED]
