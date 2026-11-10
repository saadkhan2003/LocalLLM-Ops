"""Deterministic keyword SQL generator for CPU smoke tests.

This is not a neural network. It lets the harness, gateway, and Locust file
run with no GPU and no API key. README results for Qwen stay empty until a
real run.
"""

from __future__ import annotations

import re

from localllm.data.schema import parse_schema_tables


class TinySqlModel:
    def generate(self, schema: str, question: str) -> str:
        tables = parse_schema_tables(schema)
        if not tables:
            return "SELECT 1"
        mentioned = _mentioned_tables(question, list(tables))
        table = mentioned[0] if mentioned else next(iter(tables))
        columns = {name.lower(): name for name, _kind in tables[table]}
        older = _older_than(question)
        text = question.lower()
        if "how many" in text and older is not None and "age" in columns:
            return f"SELECT COUNT(*) FROM {table} WHERE {columns['age']} > {older}"
        if "how many" in text:
            return f"SELECT COUNT(*) FROM {table}"
        if older is not None and "age" in columns:
            projected = columns.get("name", "*")
            target = projected if projected == "*" else projected
            return f"SELECT {target} FROM {table} WHERE {columns['age']} > {older}"
        if _asks_names(text) and "name" in columns:
            return f"SELECT {columns['name']} FROM {table}"
        if "title" in text and "title" in columns:
            return f"SELECT {columns['title']} FROM {table}"
        if "teach" in text and "teacher" in columns:
            return f"SELECT {columns['teacher']} FROM {table}"
        return f"SELECT * FROM {table}"


def _mentioned_tables(question: str, tables: list[str]) -> list[str]:
    text = question.lower()
    hits: list[tuple[int, str]] = []
    for table in tables:
        forms = {table.lower()}
        if table.lower().endswith("s") and len(table) > 1:
            forms.add(table.lower()[:-1])
        for form in forms:
            match = re.search(rf"\b{re.escape(form)}\b", text)
            if match:
                hits.append((match.start(), table))
                break
    hits.sort()
    seen: list[str] = []
    for _pos, table in hits:
        if table not in seen:
            seen.append(table)
    return seen


def _older_than(question: str) -> int | None:
    match = re.search(r"older than (\d+)", question.lower())
    if not match:
        return None
    return int(match.group(1))


def _asks_names(text: str) -> bool:
    return "names of" in text or "what are the names" in text or "name of each" in text
