"""A 20-example Spider-shaped fixture with real SQLite files.

The questions and gold SQL are original. They exercise the same execution
metric as Spider. They are not a slice of the Spider dataset.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from localllm.data.schema import schema_from_sqlite

SCHOOL_ROWS = [
    (
        "How many students are there?",
        "SELECT COUNT(*) FROM students",
        "easy",
    ),
    (
        "How many classes are there?",
        "SELECT COUNT(*) FROM classes",
        "easy",
    ),
    (
        "How many students are older than 18?",
        "SELECT COUNT(*) FROM students WHERE age > 18",
        "easy",
    ),
    (
        "Which students are older than 18?",
        "SELECT name FROM students WHERE age > 18",
        "easy",
    ),
    (
        "What are the names of the students?",
        "SELECT name FROM students",
        "easy",
    ),
    (
        "What are the names of the classes?",
        "SELECT name FROM classes",
        "easy",
    ),
    (
        "Who teaches each class?",
        "SELECT teacher FROM classes",
        "easy",
    ),
    (
        "List student ages.",
        "SELECT age FROM students",
        "easy",
    ),
    (
        "What is the average student age?",
        "SELECT AVG(age) FROM students",
        "medium",
    ),
    (
        "How many students are in the History class?",
        "SELECT COUNT(*) FROM students AS s JOIN classes AS c "
        "ON s.class_id = c.id WHERE c.name = 'History'",
        "medium",
    ),
    (
        "What are the names of students in the Math class?",
        "SELECT s.name FROM students AS s JOIN classes AS c "
        "ON s.class_id = c.id WHERE c.name = 'Math'",
        "medium",
    ),
    (
        "How many students are younger than 18?",
        "SELECT COUNT(*) FROM students WHERE age < 18",
        "easy",
    ),
]

LIBRARY_ROWS = [
    (
        "How many books are there?",
        "SELECT COUNT(*) FROM books",
        "easy",
    ),
    (
        "How many authors are there?",
        "SELECT COUNT(*) FROM authors",
        "easy",
    ),
    (
        "What are the names of the authors?",
        "SELECT name FROM authors",
        "easy",
    ),
    (
        "What are the titles of the books?",
        "SELECT title FROM books",
        "easy",
    ),
    (
        "Which books were published after 1970?",
        "SELECT title FROM books WHERE year > 1970",
        "medium",
    ),
    (
        "How many books did Le Guin write?",
        "SELECT COUNT(*) FROM books AS b JOIN authors AS a "
        "ON b.author_id = a.id WHERE a.name = 'Le Guin'",
        "hard",
    ),
    (
        "Show the country of each author.",
        "SELECT country FROM authors",
        "easy",
    ),
    (
        "What is the title of the oldest book?",
        "SELECT title FROM books ORDER BY year ASC LIMIT 1",
        "hard",
    ),
]


def fixture_examples() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for question, sql, difficulty in SCHOOL_ROWS:
        rows.append(
            {
                "db_id": "school",
                "question": question,
                "gold_sql": sql,
                "difficulty": difficulty,
            }
        )
    for question, sql, difficulty in LIBRARY_ROWS:
        rows.append(
            {
                "db_id": "library",
                "question": question,
                "gold_sql": sql,
                "difficulty": difficulty,
            }
        )
    return rows


def materialize_fixture(root: Path) -> Path:
    """Create SQLite files and jsonl splits. Safe to call more than once."""
    root = Path(root)
    database = root / "database"
    _write_school(database / "school" / "school.sqlite")
    _write_library(database / "library" / "library.sqlite")
    examples = []
    for row in fixture_examples():
        db_path = database / row["db_id"] / f"{row['db_id']}.sqlite"
        item = {
            "db_id": row["db_id"],
            "schema": schema_from_sqlite(db_path),
            "question": row["question"],
            "gold_sql": row["gold_sql"],
            "difficulty": row["difficulty"],
        }
        examples.append(item)
    _write_jsonl(root / "test.jsonl", examples)
    # 19 / 1 matches the 5% holdout on 20 rows (round(20 * 0.05) == 1).
    _write_jsonl(root / "train.jsonl", examples[1:])
    _write_jsonl(root / "val.jsonl", examples[:1])
    manifest = {
        "train": 19,
        "val": 1,
        "test": 20,
        "note": "Built-in fixture, not the Spider dataset.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_school(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE classes (
          id INTEGER PRIMARY KEY,
          name TEXT,
          teacher TEXT
        );
        CREATE TABLE students (
          id INTEGER PRIMARY KEY,
          name TEXT,
          age INTEGER,
          class_id INTEGER,
          FOREIGN KEY (class_id) REFERENCES classes(id)
        );
        INSERT INTO classes (id, name, teacher) VALUES
          (1, 'History', 'Ada'),
          (2, 'Math', 'Grace');
        INSERT INTO students (id, name, age, class_id) VALUES
          (1, 'Ali', 17, 1),
          (2, 'Noor', 19, 1),
          (3, 'Omar', 20, 2),
          (4, 'Hana', 16, 2);
        """
    )
    conn.commit()
    conn.close()


def _write_library(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE authors (
          id INTEGER PRIMARY KEY,
          name TEXT,
          country TEXT
        );
        CREATE TABLE books (
          id INTEGER PRIMARY KEY,
          title TEXT,
          year INTEGER,
          author_id INTEGER,
          FOREIGN KEY (author_id) REFERENCES authors(id)
        );
        INSERT INTO authors (id, name, country) VALUES
          (1, 'Le Guin', 'US'),
          (2, 'Calvino', 'IT');
        INSERT INTO books (id, title, year, author_id) VALUES
          (1, 'The Dispossessed', 1974, 1),
          (2, 'The Left Hand of Darkness', 1969, 1),
          (3, 'Invisible Cities', 1972, 2);
        """
    )
    conn.commit()
    conn.close()
