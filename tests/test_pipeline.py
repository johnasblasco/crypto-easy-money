import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cryptopredict.data import generate_synthetic
from cryptopredict.predict import load_bundle, predict_latest
from cryptopredict.train import run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    df = generate_synthetic(1500)
    metrics = run(df, symbol="BTCUSDT", interval="1h", source="synthetic", out_dir=out, verbose=False)
    return out, metrics


def test_training_writes_artifacts(trained):
    out, metrics = trained
    for name in ("model.joblib", "metrics.json", "test_predictions.csv", "report.md"):
        assert (out / name).exists()
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["best_model"] == metrics["best_model"]
    models = [r for r in saved["results"] if not r["baseline"]]
    baselines = [r for r in saved["results"] if r["baseline"]]
    assert len(models) >= 3 and len(baselines) == 2
    for r in models:
        assert 0 <= r["accuracy"] <= 1
        assert 0 <= r["roc_auc"] <= 1
        assert "cv_accuracy" in r


def test_test_set_comes_after_training_data(trained):
    out, metrics = trained
    preds = pd.read_csv(out / "test_predictions.csv", parse_dates=["timestamp"])
    assert len(preds) == metrics["rows"]["test"]
    assert preds["timestamp"].is_monotonic_increasing
    # train rows + horizon gap + test rows = all rows
    rows = metrics["rows"]
    assert rows["train"] + metrics["horizon"] + rows["test"] == rows["total"]


def test_predict_latest(trained):
    out, _ = trained
    bundle = load_bundle(out / "model.joblib")
    result = predict_latest(generate_synthetic(300), bundle)
    assert result["direction"] in {"UP", "DOWN"}
    assert 0 <= result["prob_up"] <= 1
    assert 0.5 <= result["confidence"] <= 1
    assert len(result["indicators"]) == 6


def test_dashboard_api(trained, monkeypatch):
    from app import server

    out, _ = trained
    monkeypatch.setattr(server, "MODEL_PATH", out / "model.joblib")
    server._cache.update(at=0.0, payload=None, model_mtime=None)
    client = TestClient(server.app)

    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["candles"]) == server.CANDLES
    assert body["prediction"]["direction"] in {"UP", "DOWN"}
    assert body["history"], "expected out-of-sample predictions on the chart"
    candle_times = {c["time"] for c in body["candles"]}
    assert all(h["time"] in candle_times for h in body["history"])

    assert client.get("/").status_code == 200


def test_dashboard_without_model(tmp_path, monkeypatch):
    from app import server

    monkeypatch.setattr(server, "MODEL_PATH", tmp_path / "missing.joblib")
    server._cache.update(at=0.0, payload=None, model_mtime=None)
    resp = TestClient(server.app).get("/api/dashboard")
    assert resp.status_code == 503
    assert "train" in resp.json()["detail"]
