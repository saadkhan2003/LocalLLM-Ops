"""Allow a single read-only SELECT. Reject everything else before it can run."""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

_FORBIDDEN_NAMES = (
    "Drop",
    "Delete",
    "Update",
    "Insert",
    "Alter",
    "Command",
    "Create",
    "Attach",
    "Detach",
)
FORBIDDEN = tuple(getattr(exp, name) for name in _FORBIDDEN_NAMES if hasattr(exp, name))


class SqlRejected(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_select_only(sql: str) -> str:
    """Return the stripped SQL when it is one SELECT-family statement.

    WITH, UNION, INTERSECT, and EXCEPT are allowed. DDL, DML, PRAGMA, and
    multiple statements are not. A string literal that mentions DROP is
    allowed, because the parser sees one statement.
    """
    cleaned = _without_fence(sql).strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        raise SqlRejected("empty SQL")
    try:
        statements = sqlglot.parse(cleaned, read="sqlite")
    except sqlglot.errors.SqlglotError as exc:
        raise SqlRejected(f"parse error: {exc}") from exc
    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1:
        raise SqlRejected("multiple statements")
    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SqlRejected(f"only SELECT is allowed, got {type(statement).__name__}")
    for node in statement.walk():
        if isinstance(node, FORBIDDEN):
            raise SqlRejected(f"forbidden {type(node).__name__}")
    return cleaned


def _without_fence(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped
