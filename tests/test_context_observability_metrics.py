from app.observability import tracing


def test_context_measurements_are_exported_without_prometheus(monkeypatch):
    monkeypatch.setattr(tracing, "PrometheusCounter", None)
    monkeypatch.setattr(tracing, "PrometheusHistogram", None)
    monkeypatch.setattr(tracing, "generate_latest", None)
    monkeypatch.setattr(tracing, "_MEASUREMENTS", {})
    monkeypatch.setattr(tracing, "_METRICS", {})
    monkeypatch.setattr(tracing, "_PROM_MEASUREMENTS", {})

    tracing.record_measurement("context_bundle_tokens", 128, labels={"agent": "triage"})
    tracing.record_measurement("context_bundle_tokens", 256, labels={"agent": "triage"})
    tracing.record_measurement("context_resolver_decisions", labels={"outcome": "continue"})

    snapshot = tracing.snapshot_metrics()
    assert snapshot['context_bundle_tokens{agent=triage}_count'] == 2
    assert snapshot["context_resolver_decisions{outcome=continue}"] == 1

    payload = tracing.prometheus_payload().decode("utf-8")
    assert 'bizinsight_context_bundle_tokens_count{agent="triage"} 2' in payload
    assert 'bizinsight_context_bundle_tokens_sum{agent="triage"} 384.0' in payload
    assert 'bizinsight_context_resolver_decisions_total{outcome="continue"} 1.0' in payload


def test_context_measurements_reject_unbounded_metric_labels():
    original = tracing._MEASUREMENTS
    tracing._MEASUREMENTS = {}
    try:
        tracing.record_measurement("context_bundle_tokens", 10, labels={"agent": "raw-user-id"})
        snapshot = tracing.snapshot_metrics()
        assert "raw-user-id" not in repr(snapshot)
        assert snapshot["context_bundle_tokens{agent=invalid}_count"] == 1
    finally:
        tracing._MEASUREMENTS = original
