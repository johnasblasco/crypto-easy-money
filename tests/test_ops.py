"""Operational safety: secrets never reach logs, recorded data survives a stop."""
import pandas as pd
import requests

from cryptopredict import scanner
from quant import record_book


def test_failed_alert_never_prints_the_token(monkeypatch, capsys, tmp_path):
    secret = "123456:SECRET-BOT-TOKEN"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def boom(url, **kw):
        resp = requests.Response()
        resp.status_code = 401
        raise requests.HTTPError(f"401 Client Error: Unauthorized for url: {url}", response=resp)

    monkeypatch.setattr(scanner.requests, "post", boom)
    row = {"symbol": "BTCUSDT", "status": "BUY", "interval": "1h", "horizon": 1, "signal_time": "t"}
    monkeypatch.setattr(scanner, "scan", lambda *a, **k: [row])
    monkeypatch.setattr(scanner, "new_buy_signals", lambda rows, *a, **k: rows)
    monkeypatch.setattr(scanner, "alert_text", lambda r: "hello")
    monkeypatch.setattr(scanner, "format_table", lambda rows: "", raising=False)

    class Stop(Exception):
        pass

    def stop(_):
        raise Stop                            # end the --watch loop after one cycle

    monkeypatch.setattr(scanner.time, "sleep", stop)
    try:
        scanner.main(["--dir", str(tmp_path), "--watch", "15"])
    except Stop:
        pass
    out = capsys.readouterr()
    assert secret not in out.out + out.err
    assert "HTTP 401" in out.out


def test_recorder_flushes_on_interrupt_and_writes_atomically(monkeypatch, tmp_path):
    monkeypatch.setattr(record_book, "BOOK_DIR", tmp_path)
    calls = {"n": 0}

    def snap(symbol, session):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt          # what `docker compose stop` (SIGINT) delivers
        return {"ts": pd.Timestamp("2026-10-07 12:00", tz="UTC") + pd.Timedelta(seconds=calls["n"]), "mid": 1.0}

    monkeypatch.setattr(record_book, "snapshot", snap)
    record_book.run(["BTCUSDT"], every=0.0)
    files = list((tmp_path / "BTCUSDT").glob("*"))
    assert [f.suffix for f in files] == [".parquet"]          # no leftover .tmp
    assert len(pd.read_parquet(files[0])) == 3
