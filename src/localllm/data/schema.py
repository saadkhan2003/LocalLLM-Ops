"""Serialize a SQLite database or a Spider schema record into a stable text schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

TRUNCATION_MARK = "-- schema truncated"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def schema_from_sqlite(db_path: Path | str) -> str:
    """Tables, columns, types, primary keys, and foreign keys. No sample rows."""
    path = Path(db_path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY rowid"
            )
        ]
        lines: list[str] = []
        foreign_keys: list[str] = []
        for table in tables:
            lines.append(f"Table {table}")
            columns = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            for _cid, name, col_type, _notnull, _default, pk in columns:
                kind = (col_type or "TEXT").strip() or "TEXT"
                suffix = " primary key" if pk else ""
                lines.append(f"  {name} {kind}{suffix}")
            for fk in conn.execute(f"PRAGMA foreign_key_list({quote_ident(table)})"):
                # id, seq, ref_table, from_col, to_col, on_update, on_delete, match
                foreign_keys.append(f"  {table}.{fk[3]} = {fk[2]}.{fk[4]}")
        if foreign_keys:
            lines.append("Foreign keys:")
            lines.extend(foreign_keys)
        return "\n".join(lines)
    finally:
        conn.close()


def schema_from_tables_json(entry: dict) -> str:
    """Official Spider ``tables.json`` object for one database."""
    tables: list[str] = list(entry["table_names_original"])
    columns = list(entry["column_names_original"])
    types = list(entry["column_types"])
    primary = set(entry.get("primary_keys") or [])
    lines: list[str] = []
    for index, table in enumerate(tables):
        lines.append(f"Table {table}")
        for col_index, (table_index, name) in enumerate(columns):
            if table_index != index or name == "*":
                continue
            kind = types[col_index] if col_index < len(types) else "text"
            suffix = " primary key" if col_index in primary else ""
            lines.append(f"  {name} {kind}{suffix}")
    foreign_keys = entry.get("foreign_keys") or []
    if foreign_keys:
        lines.append("Foreign keys:")
        for left, right in foreign_keys:
            left_table, left_col = _column_ref(columns, tables, left)
            right_table, right_col = _column_ref(columns, tables, right)
            lines.append(f"  {left_table}.{left_col} = {right_table}.{right_col}")
    return "\n".join(lines)


def schema_from_schema_row(row: dict) -> str:
    """Row from the public ``richardr1126/spider-schema`` JSON file.

    The official Spider zip's ``tables.json`` is preferred when you have it.
    This mirror is only a schema cache so prompts can be built before the
    SQLite files are downloaded.
    """
    raw = row.get("Schema (values (type))") or row.get("schema") or ""
    primary_raw = row.get("Primary Keys") or ""
    foreign_raw = row.get("Foreign Keys") or ""
    primary = _pipe_map(primary_raw)
    lines: list[str] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        table, cols = part.split(":", 1)
        table = table.strip()
        lines.append(f"Table {table}")
        pk_cols = {c.strip() for c in primary.get(table, [])}
        for col in cols.split(","):
            col = col.strip()
            if not col:
                continue
            name, _, kind = col.partition("(")
            name = name.strip()
            kind = kind.rstrip(")").strip() or "text"
            suffix = " primary key" if name in pk_cols else ""
            lines.append(f"  {name} {kind}{suffix}")
    foreign_lines = _foreign_lines(foreign_raw)
    if foreign_lines:
        lines.append("Foreign keys:")
        lines.extend(foreign_lines)
    return "\n".join(lines)


def truncate_schema(schema: str, max_chars: int) -> tuple[str, bool]:
    """Drop whole tables from the end until the text fits.

    Foreign keys that mention a dropped table are dropped with it. The marker
    is included in the returned text, which stays within ``max_chars``.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(schema) <= max_chars:
        return schema, False
    tables, foreign = _split_schema(schema)
    while tables and len(_join_schema(tables, foreign, truncated=True)) > max_chars:
        dropped = tables.pop()
        dropped_name = dropped[0].split(" ", 1)[1]
        foreign = [
            line
            for line in foreign
            if not line.strip().startswith(f"{dropped_name}.")
            and f"= {dropped_name}." not in line
        ]
    text = _join_schema(tables, foreign, truncated=True)
    if len(text) > max_chars:
        # Last resort: one table, columns trimmed from the bottom.
        if not tables:
            mark = TRUNCATION_MARK
            return mark[:max_chars], True
        header = tables[0][:1]
        body = tables[0][1:]
        while body and len(_join_schema([header + body], [], truncated=True)) > max_chars:
            body.pop()
        text = _join_schema([header + body], [], truncated=True)
    return text, True


def parse_schema_tables(schema: str) -> dict[str, list[tuple[str, str]]]:
    """Map table name to a list of (column, type) pairs. Primary-key flag is ignored."""
    tables: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for line in schema.splitlines():
        if line.startswith("Table "):
            current = line.split(" ", 1)[1].strip()
            tables[current] = []
            continue
        if line.startswith("Foreign keys:") or line.startswith(TRUNCATION_MARK):
            current = None
            continue
        if current is None or not line.startswith("  "):
            continue
        if "=" in line and "." in line:
            continue
        body = line.strip()
        if body.endswith(" primary key"):
            body = body[: -len(" primary key")].rstrip()
        parts = body.split()
        if len(parts) < 2:
            continue
        tables[current].append((parts[0], " ".join(parts[1:])))
    return tables


def resolve_sqlite(database_dir: Path | str, db_id: str) -> Path | None:
    root = Path(database_dir)
    candidates = [
        root / db_id / f"{db_id}.sqlite",
        root / db_id / f"{db_id}.sqlite3",
        root / f"{db_id}.sqlite",
        root / db_id / "database.sqlite",
    ]
    for path in candidates:
        if path.is_file():
            return path
    folder = root / db_id
    if folder.is_dir():
        found = sorted(folder.glob("*.sqlite"))
        if found:
            return found[0]
    return None


def _column_ref(columns: list, tables: list[str], index: int) -> tuple[str, str]:
    table_index, name = columns[index]
    return tables[table_index], name


def _pipe_map(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for part in text.split("|"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        table, cols = part.split(":", 1)
        found[table.strip()] = [c.strip() for c in cols.split(",") if c.strip()]
    return found


def _foreign_lines(text: str) -> list[str]:
    lines: list[str] = []
    for part in text.split("|"):
        part = part.strip()
        if " equals " not in part:
            continue
        left, right = part.split(" equals ", 1)
        if ":" not in left or ":" not in right:
            continue
        left_table, left_col = (piece.strip() for piece in left.split(":", 1))
        right_table, right_col = (piece.strip() for piece in right.split(":", 1))
        lines.append(f"  {left_table}.{left_col} = {right_table}.{right_col}")
    return lines


def _split_schema(schema: str) -> tuple[list[list[str]], list[str]]:
    tables: list[list[str]] = []
    foreign: list[str] = []
    current: list[str] | None = None
    in_fk = False
    for line in schema.splitlines():
        if line.startswith("Foreign keys:"):
            in_fk = True
            current = None
            continue
        if in_fk:
            if line.strip():
                foreign.append(line)
            continue
        if line.startswith("Table "):
            current = [line]
            tables.append(current)
            continue
        if current is not None:
            current.append(line)
    return tables, foreign


def _join_schema(tables: list[list[str]], foreign: list[str], *, truncated: bool) -> str:
    lines: list[str] = []
    for block in tables:
        lines.extend(block)
    if foreign:
        lines.append("Foreign keys:")
        lines.extend(foreign)
    if truncated:
        lines.append(TRUNCATION_MARK)
    return "\n".join(lines)
