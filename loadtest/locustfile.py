"""Locust load test for the gateway. Defaults to the local tiny backend.

Streaming is on: each task reads the SSE body and records time to first byte.
Set LOCUST_BACKEND, GATEWAY_API_KEY, and --host. See RUNBOOK.md for the GPU run.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

from locust import HttpUser, between, events, task

from localllm.data.fixtures import fixture_examples, materialize_fixture
from localllm.data.schema import schema_from_sqlite

_ROOT = materialize_fixture(Path("data/fixtures"))
_SCHEMAS: dict[str, str] = {}
PROMPTS = []
for _row in fixture_examples():
    if _row["db_id"] not in _SCHEMAS:
        db_path = _ROOT / "database" / _row["db_id"] / f"{_row['db_id']}.sqlite"
        _SCHEMAS[_row["db_id"]] = schema_from_sqlite(db_path)
    PROMPTS.append(_row)


class SqlUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        self.backend = os.environ.get("LOCUST_BACKEND", "tiny")
        key = os.environ.get("GATEWAY_API_KEY", "")
        self.headers = {"content-type": "application/json"}
        if key:
            self.headers["authorization"] = f"Bearer {key}"

    @task
    def ask(self) -> None:
        row = random.choice(PROMPTS)
        payload = {
            "question": row["question"],
            "schema": _SCHEMAS[row["db_id"]],
            "backend": self.backend,
            "execute": False,
        }
        started = time.perf_counter()
        first_byte = None
        with self.client.post(
            "/v1/sql/stream",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True,
            name="/v1/sql/stream",
        ) as response:
            try:
                for chunk in response.iter_content(chunk_size=64):
                    if chunk and first_byte is None:
                        first_byte = time.perf_counter()
            except Exception as exc:
                response.failure(str(exc))
                return
            if response.status_code >= 400:
                response.failure(f"status {response.status_code}")
                return
            response.success()
        # Separate row so TTFT is visible. Do not add it into requests/s;
        # read the POST row for throughput. One chunk from the tiny backend
        # makes TTFT almost equal to the full response.
        if first_byte is not None:
            events.request.fire(
                request_type="TTFT",
                name="/v1/sql/stream",
                response_time=(first_byte - started) * 1000,
                response_length=0,
                exception=None,
                context={},
            )
