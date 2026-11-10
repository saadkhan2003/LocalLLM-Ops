import sqlite3

import pytest

from localllm.data.fixtures import materialize_fixture
from localllm.eval.execution import (
    exact_match,
    execution_match,
    remove_distinct,
    result_eq,
    strip_sql,
)


def test_strip_fence_and_exact_match():
    assert strip_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert exact_match("SELECT  1", "select 1")
    assert not exact_match("SELECT 2", "SELECT 1")


def test_distinct_removed_outside_strings_only():
    assert "distinct" not in remove_distinct("SELECT DISTINCT name FROM t").lower()
    kept = remove_distinct("SELECT name FROM t WHERE name = 'distinct'")
    assert "'distinct'" in kept


def test_bag_semantics_ignore_row_and_column_order():
    gold = [("Ali", 17), ("Noor", 19)]
    pred = [(19, "Noor"), (17, "Ali")]
    assert result_eq(gold, pred, order_matters=False)
    assert not result_eq(gold, pred, order_matters=True)


def test_order_by_is_sensitive(tmp_path):
    root = materialize_fixture(tmp_path)
    db = root / "database" / "library" / "library.sqlite"
    gold = "SELECT title FROM books ORDER BY year ASC"
    same = "SELECT title FROM books ORDER BY year"
    flipped = "SELECT title FROM books ORDER BY year DESC"
    assert execution_match(db, same, gold)[0]
    assert not execution_match(db, flipped, gold)[0]


def test_execution_match_on_fixture(tmp_path):
    root = materialize_fixture(tmp_path)
    db = root / "database" / "school" / "school.sqlite"
    ok, err = execution_match(db, "SELECT count(*) FROM students", "SELECT COUNT(*) FROM students")
    assert ok and err is None
    bad, reason = execution_match(db, "SELECT * FROM students", "SELECT age FROM students")
    assert not bad and reason == "result_mismatch"


def test_test_suite_requires_every_sqlite_file(tmp_path):
    folder = tmp_path / "school"
    folder.mkdir()
    for name, age in (("one.sqlite", 10), ("two.sqlite", 30)):
        conn = sqlite3.connect(folder / name)
        conn.execute("CREATE TABLE students (age INTEGER)")
        conn.execute("INSERT INTO students (age) VALUES (?)", (age,))
        conn.commit()
        conn.close()
    gold = "SELECT COUNT(*) FROM students WHERE age > 18"
    passes_only_the_older_file = "SELECT COUNT(*) FROM students"
    match, reason = execution_match(folder / "one.sqlite", passes_only_the_older_file, gold)
    assert not match and reason == "result_mismatch"


def test_invalid_sql_and_timeout(tmp_path):
    root = materialize_fixture(tmp_path)
    db = root / "database" / "school" / "school.sqlite"
    match, reason = execution_match(db, "DROP TABLE students", "SELECT COUNT(*) FROM students")
    assert not match
    assert reason.startswith("execution_error") or reason.startswith("gold") or "error" in reason
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT x FROM c"
    match, reason = execution_match(db, slow, "SELECT 1", timeout_s=0.2)
    assert not match
    assert reason == "timeout"


def test_timeout_raises_inside_limit():
    # Guard against a timeout implementation that never returns.
    import time

    from localllm.eval.execution import SqlTimeout, execute_sql

    root_db = "/tmp/localllm-timeout.sqlite"
    conn = sqlite3.connect(root_db)
    conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
    conn.commit()
    conn.close()
    started = time.perf_counter()
    with pytest.raises(SqlTimeout):
        execute_sql(
            root_db,
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT x FROM c",
            timeout_s=0.3,
        )
    assert time.perf_counter() - started < 5
