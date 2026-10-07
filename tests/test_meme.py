"""Sub-cent meme coins: readable prices and the tick-size cost that comes with them."""
import pytest

from cryptopredict import train
from cryptopredict.data import generate_synthetic
from cryptopredict.scanner import format_price


@pytest.mark.parametrize("price, text", [
    (61234.5, "61,234.50"),
    (2.3456, "2.3456"),
    (0.0010118, "0.0010118"),
    (4.07e-06, "0.0000040700"),      # PEPE: 4 decimals would print 0.0000
])
def test_format_price_keeps_significant_digits(price, text):
    assert format_price(price) == text


def test_binance_training_charges_half_a_tick(tmp_path, monkeypatch):
    df = generate_synthetic(1500, start_price=4e-6)          # a PEPE-sized price
    monkeypatch.setattr(train, "tick_size", lambda symbol: 1e-8)
    m = train.run(df, symbol="PEPEUSDT", interval="1h", source="binance", out_dir=tmp_path, verbose=False)
    expected = 0.5 * 1e-8 / float(df["close"].median())
    assert m["spread"] == pytest.approx(expected)
    assert m["spread"] > 0.0005                               # >5 bp per side: it matters
    assert m["cost_per_trade"] == pytest.approx(m["fee"] + expected)


def test_non_binance_sources_have_no_tick_cost(tmp_path):
    m = train.run(generate_synthetic(1500), symbol="AAAUSDT", interval="1h", source="synthetic",
                  out_dir=tmp_path, verbose=False)
    assert m["spread"] == 0.0 and m["cost_per_trade"] == m["fee"]
