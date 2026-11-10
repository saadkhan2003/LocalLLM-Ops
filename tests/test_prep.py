from localllm.data.prep_spider import (
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    EXPECTED_VAL,
    example_from_row,
    missing_databases,
    prepare,
    split_train_val,
)
from localllm.data.schema import schema_from_schema_row, schema_from_tables_json


def _rows(n, split_name):
    return [
        {
            "db_id": "school" if i % 2 == 0 else "library",
            "question": f"{split_name} question {i}",
            "query": "SELECT 1",
            "gold_sql": "SELECT 1",
        }
        for i in range(n)
    ]


def test_full_split_counts():
    train, val = split_train_val(_rows(7000, "train"))
    assert len(train) == EXPECTED_TRAIN
    assert len(val) == EXPECTED_VAL
    assert EXPECTED_TEST == 1034


def test_prepare_writes_fields_and_checks_sqlite(tmp_path):
    from localllm.data.fixtures import materialize_fixture

    fixture = materialize_fixture(tmp_path / "fix")
    schema_map = {
        "school": "Table students\n  id INTEGER primary key",
        "library": "Table books\n  id INTEGER primary key",
    }

    def fetch(split, limit):
        count = 20 if limit is None else limit
        return _rows(count, split)

    manifest = prepare(
        tmp_path / "out",
        limit=20,
        database_dir=fixture / "database",
        schema_map=schema_map,
        fetch_split=fetch,
        require_sqlite=True,
    )
    assert manifest["counts"]["train"] == 19
    assert manifest["counts"]["val"] == 1
    assert manifest["counts"]["test"] == 20
    assert manifest["sqlite_missing"] == []
    line = (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    import json

    row = json.loads(line)
    assert set(row) == {"db_id", "schema", "question", "gold_sql"}
    assert "Table classes" in row["schema"] or "Table authors" in row["schema"]


def test_missing_database_detected(tmp_path):
    rows = [{"db_id": "missing", "question": "q", "gold_sql": "SELECT 1"}]
    assert missing_databases(rows, tmp_path) == ["missing"]


def test_tables_json_and_schema_row():
    tables = schema_from_tables_json(
        {
            "db_id": "school",
            "table_names_original": ["students"],
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
            "column_types": ["text", "number", "text"],
            "primary_keys": [1],
            "foreign_keys": [],
        }
    )
    assert "id number primary key" in tables
    mirror = schema_from_schema_row(
        {
            "db_id": "school",
            "Schema (values (type))": "students : id (number) , name (text)",
            "Primary Keys": "students : id",
            "Foreign Keys": "",
        }
    )
    assert "Table students" in mirror
    assert "id number primary key" in mirror


def test_sqlite_schema_preferred_over_map(tmp_path):
    from localllm.data.fixtures import materialize_fixture

    fixture = materialize_fixture(tmp_path)
    row = example_from_row(
        {"db_id": "school", "question": "q", "query": "SELECT 1"},
        {"school": "Table fake\n  id INTEGER"},
        fixture / "database",
        6000,
    )
    assert "Table classes" in row["schema"]
