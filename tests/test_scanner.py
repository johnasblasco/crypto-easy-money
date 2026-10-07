import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cryptopredict import scanner
from cryptopredict.experiments import run_grid


@pytest.fixture(scope="module")
def exp_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("experiments")
    run_grid(["AAAUSDT", "BBBUSDT"], ["1h"], [1, 4], source="synthetic", limit=1200, out_dir=out, verbose=False)
    return out


def _mark_validated(exp_dir, symbol):
    path = exp_dir / "summary.csv"
    summary = pd.read_csv(path)
    original = summary.copy()
    summary.loc[summary["symbol"] == symbol, "verdict"] = scanner.VALIDATED
    summary.to_csv(path, index=False)
    return lambda: original.to_csv(path, index=False)


def test_hold_time_formatting():
    assert scanner.format_duration(scanner.hold_time("1h", 1)) == "1 hour"
    assert scanner.format_duration(scanner.hold_time("4h", 3)) == "12 hours"
    assert scanner.format_duration(scanner.hold_time("1d", 3)) == "3 days"
    assert scanner.format_duration(scanner.hold_time("1d", 1)) == "1 day"
    assert scanner.format_duration(scanner.hold_time("4h", 9)) == "36 hours"
    assert scanner.format_duration(scanner.hold_time("15m", 1)) == "15 min"


@pytest.mark.parametrize(
    "direction, actionable, verdict, status",
    [
        ("UP", True, "POSSIBLE EDGE", "BUY"),
        ("DOWN", True, "POSSIBLE EDGE", "AVOID"),
        ("UP", False, "POSSIBLE EDGE", "WAIT"),
        ("UP", True, "NO EDGE", "NOT VALIDATED"),
        ("UP", True, "UNPROVEN", "NOT VALIDATED"),
    ],
)
def test_only_validated_models_can_signal(direction, actionable, verdict, status):
    assert scanner.classify({"direction": direction, "actionable": actionable}, verdict) == status


def test_scan_lists_every_configuration(exp_dir):
    rows = scanner.scan(exp_dir, source="synthetic")
    assert len(rows) == 4
    assert {(r["symbol"], r["horizon"]) for r in rows} == {
        ("AAAUSDT", 1), ("AAAUSDT", 4), ("BBBUSDT", 1), ("BBBUSDT", 4)
    }
    for r in rows:
        assert r["status"] in scanner.STATUS_ORDER
        assert pd.Timestamp(r["exit_time"]) - pd.Timestamp(r["signal_time"]) == scanner.hold_time("1h", r["horizon"])
    statuses = [scanner.STATUS_ORDER[r["status"]] for r in rows]
    assert statuses == sorted(statuses)


def test_scan_uses_corrected_verdict(exp_dir):
    restore = _mark_validated(exp_dir, "AAAUSDT")
    try:
        rows = scanner.scan(exp_dir, source="synthetic")
    finally:
        restore()
    for r in rows:
        if r["symbol"] == "AAAUSDT":
            assert r["status"] in {"BUY", "AVOID", "WAIT"}
        else:
            assert r["status"] == "NOT VALIDATED"


def test_alerts_are_sent_once_per_signal(tmp_path, monkeypatch):
    row = {"status": "BUY", "symbol": "AAAUSDT", "interval": "1h", "horizon": 1,
           "signal_time": "2026-01-01T00:00:00+00:00"}
    sent_file = tmp_path / "sent.json"
    assert len(scanner.new_buy_signals([row], sent_file)) == 1
    assert scanner.new_buy_signals([row], sent_file) == []
    later = {**row, "signal_time": "2026-01-01T01:00:00+00:00"}
    assert len(scanner.new_buy_signals([row, later], sent_file)) == 1


def test_send_alert_uses_configured_channels(monkeypatch):
    posts = []

    class Resp:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(scanner.requests, "post", lambda url, json, timeout: posts.append((url, json)) or Resp())
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert scanner.send_alert("hi") == []

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    assert scanner.send_alert("hi") == ["telegram", "discord"]
    assert posts[0][1] == {"chat_id": "42", "text": "hi"}
    assert posts[1] == ("https://discord.example/webhook", {"content": "hi"})


def test_scanner_and_per_token_dashboard_api(exp_dir, monkeypatch):
    from app import server

    monkeypatch.setattr(server, "EXPERIMENTS_DIR", exp_dir)
    monkeypatch.setenv("DATA_SOURCE", "synthetic")
    server._cache.clear()
    client = TestClient(server.app)

    body = client.get("/api/scanner").json()
    assert len(body["rows"]) == 4
    config = body["rows"][0]["config"]

    dash = client.get(f"/api/dashboard?config={config}")
    assert dash.status_code == 200
    assert dash.json()["prediction"]["symbol"] == body["rows"][0]["symbol"]

    assert client.get("/api/dashboard?config=../../etc").status_code == 400
    assert client.get("/api/dashboard?config=ZZZUSDT_1h_h1").status_code == 404
