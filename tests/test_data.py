import pandas as pd

from cryptopredict.data import COLUMNS, generate_synthetic, load_csv


def test_synthetic_shape_and_ohlc_consistency():
    df = generate_synthetic(500)
    assert list(df.columns) == COLUMNS
    assert len(df) == 500
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] > 0).all()


def test_shorter_synthetic_is_tail_of_longer():
    long = generate_synthetic(2000)
    short = generate_synthetic(300)
    pd.testing.assert_frame_equal(short, long.iloc[-300:].reset_index(drop=True))


def test_load_csv_accepts_epoch_ms(tmp_path):
    df = generate_synthetic(50)
    raw = df.copy()
    raw["timestamp"] = raw["timestamp"].astype("int64") // 10**6
    raw.columns = [c.upper() for c in raw.columns]  # header case shouldn't matter
    path = tmp_path / "candles.csv"
    raw.iloc[::-1].to_csv(path, index=False)  # unsorted on purpose

    loaded = load_csv(path)
    assert loaded["timestamp"].is_monotonic_increasing
    pd.testing.assert_series_equal(loaded["close"], df["close"])
    assert (loaded["timestamp"].dt.tz is not None)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_binance_pages_and_drops_open_candle(monkeypatch):
    from cryptopredict import data

    hour = 3_600_000
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) // hour * hour
    # 2500 hourly candles; the newest one opened this hour and is still forming.
    opens = [now_ms - hour * i for i in range(2500)][::-1]
    klines = [[t, "1", "2", "0.5", "1.5", "10", t + hour - 1] for t in opens]
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        end = params.get("endTime", float("inf"))
        batch = [k for k in klines if k[0] <= end][-params["limit"]:]
        return _FakeResponse(batch)

    monkeypatch.setattr(data.requests, "get", fake_get)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)

    df = data.fetch_binance("btcusdt", "1h", limit=2500)
    assert len(calls) == 3  # 1000 + 1000 + 500
    assert calls[0]["symbol"] == "BTCUSDT"
    assert len(df) == 2499  # still-open candle removed
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique
    assert df["close"].dtype == float


def test_fetch_binance_falls_back_to_next_host(monkeypatch):
    from cryptopredict import data

    hosts = []

    def fake_get(url, params, timeout):
        hosts.append(url)
        if "binance.vision" in url:
            raise data.requests.ConnectionError("blocked")
        return _FakeResponse([[0, "1", "1", "1", "1", "1", 1]])

    monkeypatch.setattr(data.requests, "get", fake_get)
    df = data.fetch_binance("BTCUSDT", "1d", limit=10)
    assert len(df) == 1
    assert "api.binance.com" in hosts[-1]
