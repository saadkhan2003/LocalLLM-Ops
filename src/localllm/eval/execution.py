"""Execution accuracy in the style of ``taoyds/test-suite-sql-eval``.

A prediction is correct when it executes on every ``*.sqlite`` file in the
database directory and the result matches the gold query under bag semantics.
Column order may differ. Row order matters only when the gold SQL contains
``ORDER BY``. ``DISTINCT`` is stripped by default, matching the official CLI
(``--keep_distinct`` is off unless requested). Value plugging is not
implemented; the official CLI leaves ``--plug_value`` off as well.

String exact match is whitespace-normalized equality. It is not the official
clause-level Spider exact match, which needs the Spider SQL parser.
"""

from __future__ import annotations

import itertools
import random
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0


class SqlTimeout(Exception):
    pass


def postprocess_sql(query: str) -> str:
    query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020", query, flags=re.IGNORECASE)


def remove_distinct(sql: str) -> str:
    """Drop ``DISTINCT`` tokens that are not inside quotes."""
    out: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(sql):
        char = sql[i]
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            i += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            out.append(char)
            i += 1
            continue
        if _is_distinct_at(sql, i):
            i += len("distinct")
            continue
        out.append(char)
        i += 1
    return "".join(out)


def strip_sql(text: str) -> str:
    """Remove a code fence and keep the first SQL statement."""
    stripped = text.strip()
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if ";" in stripped:
        stripped = stripped.split(";", 1)[0]
    return stripped.strip()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", strip_sql(sql)).strip().lower().rstrip(";")


def exact_match(predicted: str, gold: str) -> bool:
    return normalize_sql(predicted) == normalize_sql(gold)


def order_matters(gold_sql: str) -> bool:
    return "order by" in gold_sql.lower()


def execute_sql(db_path: Path | str, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[tuple]:
    """Run a read-only query. Raises ``SqlTimeout`` or ``sqlite3.Error``."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda raw: raw.decode(errors="ignore") if isinstance(raw, bytes) else raw
    started = time.monotonic()

    def handler() -> int:
        if time.monotonic() - started > timeout_s:
            return 1
        return 0

    conn.set_progress_handler(handler, 10_000)
    try:
        try:
            conn.execute("PRAGMA query_only = ON")
        except sqlite3.Error:
            pass
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [tuple(row) for row in rows]
    except sqlite3.OperationalError as exc:
        if "interrupt" in str(exc).lower():
            raise SqlTimeout(sql) from exc
        raise
    finally:
        conn.close()


def _usable_sqlite(path: Path) -> bool:
    if path.name.startswith("._"):
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(16).startswith(b"SQLite format 3")
    except OSError:
        return False


def sqlite_files(db_path: Path | str) -> list[Path]:
    path = Path(db_path)
    folder = path.parent if path.is_file() else path
    files = sorted(
        item
        for item in folder.iterdir()
        if item.suffix == ".sqlite" and item.is_file() and _usable_sqlite(item)
    )
    if not files and path.is_file():
        return [path]
    return files


def execution_match(
    db_path: Path | str,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    keep_distinct: bool = False,
) -> tuple[bool, str | None]:
    """Return ``(match, error)``. ``error`` is set when the prediction fails."""
    predicted = postprocess_sql(strip_sql(predicted_sql))
    gold = postprocess_sql(strip_sql(gold_sql))
    if not keep_distinct:
        predicted = remove_distinct(predicted)
        gold = remove_distinct(gold)
    ordered = order_matters(gold)
    files = sqlite_files(db_path)
    if not files:
        return False, "missing_database"
    last_error: str | None = None
    for database in files:
        try:
            gold_rows = execute_sql(database, gold, timeout_s=timeout_s)
        except SqlTimeout:
            return False, "gold_timeout"
        except sqlite3.Error as exc:
            return False, f"gold_error: {exc}"
        try:
            pred_rows = execute_sql(database, predicted, timeout_s=timeout_s)
        except SqlTimeout:
            return False, "timeout"
        except sqlite3.Error as exc:
            return False, f"execution_error: {exc}"
        if not result_eq(gold_rows, pred_rows, order_matters=ordered):
            last_error = "result_mismatch"
            return False, last_error
    return True, last_error


def result_eq(result1: list[tuple], result2: list[tuple], order_matters: bool) -> bool:
    """Bag-semantics denotation check, including column permutations.

    The permutation search is exhaustive up to 6 columns and deterministic.
    The official evaluator samples constraints randomly above 3 columns; this
    project does not, so repeated evals do not flicker.
    """
    if len(result1) == 0 and len(result2) == 0:
        return True
    if len(result1) != len(result2):
        return False
    if not result1 or not result2:
        return False
    if len(result1[0]) != len(result2[0]):
        return False
    if not _quick_reject(result1, result2, order_matters):
        return False
    num_cols = len(result1[0])
    for perm in _column_permutations(result1, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            permuted = result2
        else:
            permuted = [tuple(row[index] for index in perm) for row in result2]
        if order_matters:
            if result1 == permuted:
                return True
        elif set(result1) == set(permuted) and _multiset_eq(result1, permuted):
            return True
    return False


def _is_distinct_at(sql: str, index: int) -> bool:
    word = "distinct"
    if sql[index : index + len(word)].lower() != word:
        return False
    before = sql[index - 1] if index else " "
    after_index = index + len(word)
    after = sql[after_index] if after_index < len(sql) else " "
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _unorder_row(row: tuple) -> tuple:
    return tuple(sorted(row, key=lambda value: (str(type(value)), str(value))))


def _quick_reject(result1: list[tuple], result2: list[tuple], order_matters: bool) -> bool:
    left = [_unorder_row(row) for row in result1]
    right = [_unorder_row(row) for row in result2]
    if order_matters:
        return left == right
    return set(left) == set(right)


def _multiset_eq(left: list[tuple], right: list[tuple]) -> bool:
    if len(left) != len(right):
        return False
    counts: dict[tuple, int] = defaultdict(int)
    for item in left:
        counts[item] += 1
    for item in right:
        if item not in counts:
            return False
        counts[item] -= 1
        if counts[item] < 0:
            return False
    return True


def _column_permutations(result1: list[tuple], result2: list[tuple]):
    num_cols = len(result1[0])
    if num_cols > 6:
        return _constrained_permutations(result1, result2)
    return itertools.permutations(range(num_cols))


def _constrained_permutations(result1: list[tuple], result2: list[tuple]):
    """Deterministic stand-in for the official sampled constraint search."""
    num_cols = len(result1[0])
    columns = [{row[index] for row in result1} for index in range(num_cols)]
    allowed: list[set[int]] = [set(range(num_cols)) for _ in range(num_cols)]
    sample = result2[:20]
    # Shuffle with a fixed seed so the constraint order does not depend on dict order,
    # then apply every sampled row. No rows are dropped.
    rng = random.Random(0)
    order = list(range(len(sample)))
    rng.shuffle(order)
    for index in order:
        row = sample[index]
        for src in range(num_cols):
            allowed[src] = {
                dest for dest in allowed[src] if dest < len(row) and row[dest] in columns[src]
            }
    return itertools.product(*[sorted(choices) for choices in allowed])
