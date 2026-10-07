"""Scan many tokens and timeframes for a current BUY signal, and send alerts.

Usage:
    python -m cryptopredict.experiments            # train/validate the models first
    python -m cryptopredict.scanner                # one scan, printed as a table
    python -m cryptopredict.scanner --watch 15     # re-scan every 15 minutes and send alerts

It uses the models saved by ``cryptopredict.experiments`` (one per token,
interval and horizon). A configuration may only produce a BUY signal if its
model passed the edge check on unseen data (verdict "POSSIBLE EDGE", after the
multiple-comparison correction). Every other configuration is shown as
"NOT VALIDATED": its prediction is listed for information, never as a signal.

Signals are long-only (spot): BUY means "the validated model expects the price
to rise over the hold time"; AVOID means it expects a fall. Hold time is the
model's horizon, e.g. horizon 3 on 4h candles = hold 12 hours.

Alerts: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, and/or DISCORD_WEBHOOK_URL.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .data import INTERVAL_MS, load_data
from .experiments import EXPERIMENTS_DIR
from .predict import load_bundle, predict_latest

VALIDATED = "POSSIBLE EDGE"
STATUS_ORDER = {"BUY": 0, "AVOID": 1, "WAIT": 2, "NOT VALIDATED": 3}
SENT_FILE = EXPERIMENTS_DIR / ".alerts_sent.json"


def hold_time(interval: str, horizon: int) -> pd.Timedelta:
    return pd.Timedelta(milliseconds=INTERVAL_MS[interval] * horizon)


def format_duration(td: pd.Timedelta) -> str:
    hours = td.total_seconds() / 3600
    if hours < 1:
        return f"{round(hours * 60)} min"
    if hours >= 24 and hours % 24 == 0:
        days = int(hours // 24)
        return "1 day" if days == 1 else f"{days} days"
    return "1 hour" if hours == 1 else f"{hours:g} hours"


def load_configs(exp_dir: Path = EXPERIMENTS_DIR) -> list[dict]:
    """Every trained configuration plus its (multiple-comparison corrected) verdict."""
    adjusted = {}
    summary_path = exp_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        for r in summary.itertuples():
            adjusted[(r.symbol, r.interval, int(r.horizon))] = r.verdict

    configs = []
    for model_path in sorted(exp_dir.glob("*/model.joblib")):
        metrics_path = model_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        key = (metrics["symbol"], metrics["interval"], int(metrics["horizon"]))
        configs.append({
            "path": model_path,
            "symbol": key[0],
            "interval": key[1],
            "horizon": key[2],
            "verdict": adjusted.get(key, metrics["edge"]["verdict"]),
            "test_strategy_return": metrics["edge"]["strategy_return"],
            "test_buy_and_hold_return": metrics["edge"]["buy_and_hold_return"],
            "trained_at": metrics["trained_at"],
        })
    return configs


def classify(prediction: dict, verdict: str) -> str:
    if verdict != VALIDATED:
        return "NOT VALIDATED"
    if not prediction["actionable"]:
        return "WAIT"
    return "BUY" if prediction["direction"] == "UP" else "AVOID"


def scan(exp_dir: Path = EXPERIMENTS_DIR, source: str | None = None, candles: int = 300) -> list[dict]:
    """Predict the next move for every configuration, best opportunities first."""
    configs = load_configs(exp_dir)
    data_cache: dict = {}
    rows = []
    for cfg in configs:
        bundle = load_bundle(cfg["path"])
        src = source or bundle["source"]
        key = (src, cfg["symbol"], cfg["interval"])
        try:
            if key not in data_cache:
                data_cache[key] = load_data(src, cfg["symbol"], cfg["interval"], candles, save=False)
            pred = predict_latest(data_cache[key], bundle)
        except Exception as exc:  # one token failing shouldn't stop the scan
            rows.append({**_base_row(cfg), "status": "ERROR", "error": str(exc)})
            continue

        hold = hold_time(cfg["interval"], cfg["horizon"])
        # The last closed candle closes one interval after it opens.
        signal_time = pd.Timestamp(pred["as_of"]) + pd.Timedelta(milliseconds=INTERVAL_MS[cfg["interval"]])
        rows.append({
            **_base_row(cfg),
            "status": classify(pred, cfg["verdict"]),
            "direction": pred["direction"],
            "confidence": pred["confidence"],
            "threshold": pred["threshold"],
            "price": pred["price"],
            "hold": format_duration(hold),
            "signal_time": signal_time.isoformat(),
            "exit_time": (signal_time + hold).isoformat(),
        })
    rows.sort(key=lambda r: (STATUS_ORDER.get(r["status"], 9), -(r.get("confidence") or 0)))
    return rows


def _base_row(cfg: dict) -> dict:
    return {
        "symbol": cfg["symbol"],
        "interval": cfg["interval"],
        "horizon": cfg["horizon"],
        "verdict": cfg["verdict"],
        "test_strategy_return": cfg["test_strategy_return"],
        "test_buy_and_hold_return": cfg["test_buy_and_hold_return"],
        "trained_at": cfg["trained_at"],
    }


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "No trained models found. Run `python -m cryptopredict.experiments` first."
    lines = [
        f"{'Status':<14} {'Token':<10} {'Candle':<6} {'Hold':<10} {'Call':<5} {'Conf':>6} {'Price':>14}  Model check",
    ]
    for r in rows:
        if r["status"] == "ERROR":
            lines.append(f"{'ERROR':<14} {r['symbol']:<10} {r['interval']:<6} {r['error']}")
            continue
        lines.append(
            f"{r['status']:<14} {r['symbol']:<10} {r['interval']:<6} {r['hold']:<10} {r['direction']:<5} "
            f"{r['confidence']:>6.1%} {format_price(r['price']):>14}  {r['verdict']}"
        )
    buys = [r for r in rows if r["status"] == "BUY"]
    validated = {(r["symbol"], r["interval"], r["horizon"]) for r in rows if r["verdict"] == VALIDATED}
    lines.append("")
    if not validated:
        lines.append(
            "No token/timeframe passed the edge check, so there are no signals to act on. "
            "Predictions above are for information only."
        )
    elif not buys:
        lines.append("Validated models exist, but none of them has a confident BUY call right now.")
    return "\n".join(lines)


def alert_text(row: dict) -> str:
    exit_at = pd.Timestamp(row["exit_time"]).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"BUY {row['symbol']} @ {format_price(row['price'])}\n"
        f"Hold {row['hold']} (until {exit_at}), {row['interval']} candles\n"
        f"Confidence {row['confidence']:.1%} (threshold {row['threshold']:.0%})\n"
        f"Back-test on unseen data: strategy {row['test_strategy_return']:+.1%} "
        f"vs buy & hold {row['test_buy_and_hold_return']:+.1%}\n"
        f"Not financial advice. Past results don't guarantee future ones."
    )


def format_price(p: float) -> str:
    """About 5 significant digits, so sub-cent meme coins (PEPE ~ $0.000004) stay readable."""
    if not math.isfinite(p) or p <= 0:
        return str(p)
    digits = 2 if p >= 1000 else 4 if p >= 1 else min(12, max(4, math.ceil(-math.log10(p)) + 4))
    return f"{p:,.{digits}f}"


def send_alert(text: str, timeout: float = 10.0) -> list[str]:
    """Send ``text`` to every configured channel; returns the channels used."""
    sent = []
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text},
            timeout=timeout,
        ).raise_for_status()
        sent.append("telegram")
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook:
        requests.post(webhook, json={"content": text}, timeout=timeout).raise_for_status()
        sent.append("discord")
    return sent


def new_buy_signals(rows: list[dict], sent_file: Path = SENT_FILE) -> list[dict]:
    """BUY rows not alerted before (one alert per token/timeframe/candle)."""
    try:
        sent = set(json.loads(sent_file.read_text()))
    except (FileNotFoundError, ValueError):
        sent = set()
    fresh = []
    for r in rows:
        if r["status"] != "BUY":
            continue
        key = f"{r['symbol']}|{r['interval']}|{r['horizon']}|{r['signal_time']}"
        if key not in sent:
            fresh.append(r)
            sent.add(key)
    sent_file.parent.mkdir(parents=True, exist_ok=True)
    sent_file.write_text(json.dumps(sorted(sent)[-1000:]))
    return fresh


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=EXPERIMENTS_DIR, help="experiments folder with trained models")
    p.add_argument("--source", choices=["binance", "csv", "synthetic"], help="default: same as training")
    p.add_argument("--watch", type=float, metavar="MINUTES", help="re-scan every MINUTES and send alerts")
    args = p.parse_args(argv)

    while True:
        rows = scan(args.dir, args.source)
        print(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"))
        print(format_table(rows) + "\n")
        if args.watch is None:
            break
        for row in new_buy_signals(rows):
            try:
                channels = send_alert(alert_text(row))
                print(f"Alert sent for {row['symbol']} {row['interval']} via {', '.join(channels) or 'nothing (no channel configured)'}")
            except requests.RequestException as exc:
                # str(exc) contains the request URL, i.e. the bot token or webhook secret.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                print(f"Could not send alert for {row['symbol']}: {type(exc).__name__}"
                      f"{f' (HTTP {status})' if status else ''}")
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
