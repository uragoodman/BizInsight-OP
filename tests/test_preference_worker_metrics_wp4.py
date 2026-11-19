from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from app.memory.preference_worker import start_metrics_server
from app.observability import record_event


def test_worker_metrics_server_exposes_prometheus_and_rejects_other_paths():
    record_event("memory.preference_dead_letter", {"memory.operation": "test"})
    server = start_metrics_server(0, host="127.0.0.1")
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/metrics", timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/plain")
            payload = response.read().decode("utf-8")
            assert "bizinsight_memory_preference_dead_letter_total" in payload
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://{host}:{port}/", timeout=2)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
