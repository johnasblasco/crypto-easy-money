"""Autopilot: pretend-money accounts that copy the app's signals automatically, plus the trend guide.

Usage:
    python -m cryptopredict.autopilot              # one update, printed
    python -m cryptopredict.autopilot --watch 15   # update every 15 minutes (Docker runs this for you)
    python -m cryptopredict.autopilot --reset      # start again with fresh pretend money

Three pretend accounts start with the same money (AUTOPILOT_START_CASH, default 1000 USDT):

* Copy the model calls: whenever a token model from ``cryptopredict.experiments`` that passed
  the edge check makes a confident UP call (confidence at or above its own threshold), buy
  AUTOPILOT_TRADE_PCT % of the starting money (default 10%), hold for the model's hold time,
  then sell. Models that failed the edge check are not copied (they lost to fees in the live
  test); AUTOPILOT_COPY_ALL=1 copies them anyway, to watch them lose with pretend money.
* Follow the trend guide: split the money equally over the guide coins and keep in each the
  share the daily trend rule suggests, rebalanced once a day after 00:00 UTC.
* Buy & hold: buy the guide coins equally on the first day and never trade. The yardstick.

Every pretend buy and sell pays the fee (and half a price tick for the model trades, as in
their back-tests) at a Binance price fetched in the same update; nothing is bought, sold or
valued at an old price. No real money and no exchange account are used. All orders go through
``PaperBroker``; connecting a real exchange later means swapping in a broker that sends the same
orders, once the "before real money" checks pass.

After a pause (PC or Docker off), the next update replays the missed time from Binance's price
history as if it had been running, up to 14 days. If the market data can't be loaded, it trades
nothing and retries on the next update.

Alerts use the scanner's settings (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, DISCORD_WEBHOOK_URL):
trend changes, a daily report (after AUTOPILOT_REPORT_HOUR UTC, default 1) and, with
AUTOPILOT_TRADE_ALERTS=1, every pretend trade. Coins: AUTOPILOT_COINS (default: the 16 main coins).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from . import trend_guide as TG
from .data import BINANCE_HOSTS, DATA_DIR, INTERVAL_MS, load_data
from .experiments import EXPERIMENTS_DIR
from .scanner import VALIDATED, format_duration, format_price, hold_time, load_configs, scan, send_alert

STATE_PATH = DATA_DIR / "autopilot" / "state.json"
TREND_FEE = 0.001 + 0.0003        # spot taker fee + 3 bp spread floor, as in the trend research
EQUITY_EVERY = pd.Timedelta(hours=1)
STALE_AFTER = pd.Timedelta(minutes=45)   # no update for this long -> the dashboard says it is off
CATCH_UP_AFTER = pd.Timedelta(minutes=40)   # updates run every 15 minutes: a longer gap means it was off
CATCH_UP_MAX = pd.Timedelta(days=14)        # missed time older than this is not replayed
GO_LIVE_DAYS, GO_LIVE_TRADES = 90, 30
MAX_HISTORY = 3000


def _setting(name: str, default, cast, ok, notes: list):
    raw = (os.environ.get(name) or "").strip().replace("%", "").replace(",", "")
    if not raw:
        return default
    try:
        value = cast(raw)
        if ok(value):
            return value
    except ValueError:
        pass
    notes.append(f"ignored {name}={os.environ.get(name)!r} (not a valid value); using {default}")
    return default


def settings() -> dict:
    from quant.data import UNIVERSE

    notes: list = []
    coins = [s.strip().upper() for s in (os.environ.get("AUTOPILOT_COINS") or "").split(",") if s.strip()]
    return {"start_cash": _setting("AUTOPILOT_START_CASH", 1000.0, float, lambda v: v > 0, notes),
            "trade_share": _setting("AUTOPILOT_TRADE_PCT", 10.0, float, lambda v: 0 < v <= 100, notes) / 100,
            "report_hour": _setting("AUTOPILOT_REPORT_HOUR", 1, int, lambda v: 0 <= v <= 23, notes),
            "trade_alerts": (os.environ.get("AUTOPILOT_TRADE_ALERTS") or "").strip().lower() in ("1", "true", "yes", "on"),
            "copy_all": (os.environ.get("AUTOPILOT_COPY_ALL") or "").strip().lower() in ("1", "true", "yes", "on"),
            "coins": coins or list(UNIVERSE), "notes": notes}


# ------------------------------------------------------------------ orders

class PaperBroker:
    """Fills every order at the given price and charges ``fee`` per side. Nothing leaves this computer."""

    name = "paper"

    def buy(self, account: dict, usdt: float, price: float, fee: float) -> float:
        usdt = min(usdt, account["cash"])
        account["cash"] -= usdt
        return usdt * (1 - fee) / price

    def sell(self, account: dict, qty: float, price: float, fee: float) -> float:
        usdt = qty * price * (1 - fee)
        account["cash"] += usdt
        return usdt


# ------------------------------------------------------------------ state

def new_state(now: pd.Timestamp, cfg: dict) -> dict:
    def account(name, **extra):
        return {"name": name, "start_cash": cfg["start_cash"], "cash": cfg["start_cash"], "trades": [],
                "equity": [], **extra}

    return {"version": 1, "started": now.isoformat(), "coins": cfg["coins"], "broker": PaperBroker.name,
            "accounts": {"models": account("Copy the model calls", positions=[],
                                           stats=new_stats()),
                         "trend": account("Follow the trend guide", holdings={}, rebalanced_for=None),
                         "hold": account("Just buy and hold", holdings={})},
            "seen_signals": [], "guide": None, "guide_day": None, "prices": {}, "last_report": None,
            "last_run": None, "synced_until": None, "uncovered_s": 0.0, "log": []}


def new_stats() -> dict:
    return {"count": 0, "validated": 0, "pnl": 0.0, "wins": 0, "fees": 0.0, "win_pnl": 0.0, "loss_pnl": 0.0}


def _backup(path: Path) -> Path:
    return path.with_name(path.name + ".bak")


def load_state(path: Path = STATE_PATH, repair: bool = False) -> dict | None:
    """The saved accounts; falls back to the previous save if the file is damaged (e.g. a power cut)."""
    for candidate in (path, _backup(path)):
        try:
            return json.loads(candidate.read_text())
        except FileNotFoundError:
            continue
        except ValueError:
            if repair:                     # keep the damaged file for inspection, out of the way
                candidate.rename(candidate.with_name(f"{candidate.name}.corrupt-{int(time.time())}"))
                print(f"{candidate} was damaged; moved it aside and used the previous save", flush=True)
    return None


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(state, default=str))
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        shutil.copyfile(path, _backup(path))
    os.replace(tmp, path)


def account_value(acct: dict, prices: dict) -> float:
    value = acct["cash"]
    for p in acct.get("positions", []):
        value += p["qty"] * prices.get(p["symbol"], p["entry_price"])
    for sym, qty in acct.get("holdings", {}).items():
        value += qty * prices.get(sym, 0.0)
    return value


def account_symbols(acct: dict) -> set:
    return set(acct.get("holdings", {})) | {p["symbol"] for p in acct.get("positions", [])}


def trade_fees(t: dict) -> float:
    """Fees of a finished trade in USDT: on the money put in, and on the sale."""
    if "fees" in t:
        return t["fees"]
    return t["invested"] * t["fee"] + t["qty"] * t["exit_price"] * t["fee"]


def record_trade(acct: dict, trade: dict) -> None:
    """Keep the last MAX_HISTORY trades, and running totals over all of them for the checks."""
    stats = acct.setdefault("stats", new_stats())
    for key, value in new_stats().items():         # states saved before a total existed
        stats.setdefault(key, value)
    stats["count"] += 1
    stats["validated"] += trade.get("verdict") == VALIDATED
    stats["pnl"] += trade["pnl"]
    stats["fees"] += trade_fees(trade)
    if trade["pnl"] > 0:
        stats["wins"] += 1
        stats["win_pnl"] += trade["pnl"]
    else:
        stats["loss_pnl"] += trade["pnl"]
    acct["trades"] = (acct["trades"] + [trade])[-MAX_HISTORY:]


def close_trade(acct: dict, p: dict, price: float, closed: str, broker, replayed: bool = False) -> dict:
    got = broker.sell(acct, p["qty"], price, p["fee"])
    trade = {**p, "closed": closed, "exit_price": price, "proceeds": got, "pnl": got - p["invested"],
             "pnl_pct": got / p["invested"] - 1, "price_move": price / p["entry_price"] - 1,
             "fees": p["invested"] * p["fee"] + p["qty"] * price * p["fee"]}
    if replayed:
        trade["replayed"] = True
    record_trade(acct, trade)
    return trade


# ------------------------------------------------------------------ market data

def latest_prices(symbols, timeout: float = 10.0) -> dict:
    """Latest Binance spot prices, one request for all symbols (per symbol if the batch fails).

    Returns only the prices it got; a symbol missing from the result means "no price now".
    """
    symbols = sorted(set(symbols))
    if not symbols:
        return {}
    for host in BINANCE_HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/ticker/price",
                             params={"symbols": json.dumps(symbols, separators=(",", ":"))}, timeout=timeout)
            r.raise_for_status()
            return {d["symbol"]: float(d["price"]) for d in r.json()}
        except Exception:
            continue
    out = {}
    for sym in symbols:                      # e.g. one unknown symbol makes the batch fail
        try:
            r = requests.get(f"{BINANCE_HOSTS[0]}/api/v3/ticker/price", params={"symbol": sym}, timeout=timeout)
            r.raise_for_status()
            out[sym] = float(r.json()["price"])
        except Exception:
            continue
    return out


def daily_bars(symbol: str) -> pd.DataFrame:
    from quant.live import fetch_daily

    return fetch_daily(symbol)


def model_costs(exp_dir: Path = EXPERIMENTS_DIR) -> dict:
    """Per-side trading cost of every token model trained on Binance data, by config name."""
    out = {}
    for path in exp_dir.glob("*/metrics.json"):
        try:
            m = json.loads(path.read_text())
        except ValueError:
            continue
        if m.get("source") == "binance":
            out[path.parent.name] = float(m.get("cost_per_trade") or m.get("fee") or 0.001)
    return out


# ------------------------------------------------------------------ the three accounts

def config_name(row: dict) -> str:
    return f"{row['symbol']}_{row['interval']}_h{row['horizon']}"


def step_models(state: dict, rows: list[dict], prices: dict, costs: dict, now: pd.Timestamp,
                share: float, broker=PaperBroker(), copy_all: bool = False) -> list[dict]:
    """Sell finished pretend trades, then copy every new confident UP call. Returns the events.

    ``prices`` must hold only prices fetched in this update: a finished trade without one waits
    for the next update, and a buy without one fills at the signal candle's close.
    """
    acct = state["accounts"]["models"]
    events = []
    still = []
    for p in acct["positions"]:
        price = prices.get(p["symbol"])
        if pd.Timestamp(p["exit_time"]) > now or price is None:
            still.append(p)
            continue
        events.append({"kind": "sell", "trade": close_trade(acct, p, price, now.isoformat(), broker)})
    acct["positions"] = still
    open_configs = {p["config"] for p in still}
    seen = list(state["seen_signals"])
    seen_set = set(seen)
    for r in rows:
        if r.get("status") == "ERROR" or r.get("direction") != "UP" or r.get("confidence", 0) < r.get("threshold", 1):
            continue
        if not copy_all and r.get("verdict") != VALIDATED:
            continue                       # failed the edge check: its calls lose to fees
        cfg = config_name(r)
        key = f"{cfg}|{r['signal_time']}"
        signal_age = now - pd.Timestamp(r["signal_time"])
        if (key in seen_set or cfg in open_configs or cfg not in costs
                or signal_age > pd.Timedelta(milliseconds=INTERVAL_MS[r["interval"]])):
            continue                       # already copied, still holding, not Binance data, or stale
        seen.append(key)
        seen_set.add(key)
        invest = min(acct["start_cash"] * share, acct["cash"])
        if invest < 1:
            events.append({"kind": "skip", "text": f"no pretend cash left for {r['symbol']} {r['interval']}"})
            continue
        price = prices.get(r["symbol"], r["price"])
        qty = broker.buy(acct, invest, price, costs[cfg])
        pos = {"config": cfg, "symbol": r["symbol"], "interval": r["interval"], "hold": r["hold"],
               "opened": now.isoformat(), "entry_price": price, "qty": qty, "invested": invest, "fee": costs[cfg],
               "exit_time": max(pd.Timestamp(r["exit_time"]), now).isoformat(), "confidence": r["confidence"],
               "verdict": r["verdict"]}
        acct["positions"].append(pos)
        open_configs.add(cfg)
        events.append({"kind": "buy", "position": pos})
    state["seen_signals"] = seen[-MAX_HISTORY:]
    return events


def step_trend(state: dict, prices: dict, now: pd.Timestamp, broker=PaperBroker(),
               guide: list | None = None, day: str | None = None) -> list[dict] | None:
    """Once per new daily guide: hold equity x exposure / number of coins in each coin.

    Returns None (and stays due) if a coin has no price or no guide data: valuing it at 0 would
    make the account sell the coins it should keep.
    """
    acct = state["accounts"]["trend"]
    guide = guide if guide is not None else state["guide"]
    day = day or state["guide_day"]
    if not guide or acct["rebalanced_for"] == day:
        return []
    coins = state["coins"]
    if any(s not in prices for s in set(coins) | set(acct["holdings"])) or any(c.get("trend") == "ERROR" for c in guide):
        return None
    equity = account_value(acct, prices)
    targets = {c["symbol"]: equity * max(c.get("exposure") or 0.0, 0.0) / len(coins) for c in guide}
    events = []
    orders = [(targets.get(sym, 0.0) - acct["holdings"].get(sym, 0.0) * prices[sym], sym, prices[sym]) for sym in coins]
    for diff, sym, price in sorted(orders):          # sells (negative) first, to free cash for the buys
        if abs(diff) < 1:
            continue
        if diff < 0:
            qty = min(acct["holdings"].get(sym, 0.0), -diff / price)
            got = broker.sell(acct, qty, price, TREND_FEE)
            acct["holdings"][sym] = acct["holdings"].get(sym, 0.0) - qty
            events.append({"time": now.isoformat(), "symbol": sym, "side": "sell", "usdt": got, "price": price})
        else:
            spend = min(diff, acct["cash"])
            if spend < 1:
                continue
            acct["holdings"][sym] = acct["holdings"].get(sym, 0.0) + broker.buy(acct, spend, price, TREND_FEE)
            events.append({"time": now.isoformat(), "symbol": sym, "side": "buy", "usdt": spend, "price": price})
    acct["holdings"] = {s: q for s, q in acct["holdings"].items() if q > 1e-12}
    acct["trades"] = (acct["trades"] + events)[-MAX_HISTORY:]
    acct["rebalanced_for"] = day
    return events


def step_hold(state: dict, prices: dict, broker=PaperBroker()) -> None:
    """Buy every guide coin equally once (when all of them have a price), then never trade."""
    acct = state["accounts"]["hold"]
    if acct["holdings"] or any(s not in prices for s in state["coins"]):
        return
    each = acct["cash"] / len(state["coins"])
    for sym in state["coins"]:
        acct["holdings"][sym] = broker.buy(acct, each, prices[sym], TREND_FEE)


# ------------------------------------------------------------------ catching up after a pause

def replay_calls(configs: list[dict], since: pd.Timestamp, until: pd.Timestamp, costs: dict,
                 load=None, reach: dict | None = None, copy_all: bool = False) -> tuple[list[dict], dict, list[str]]:
    """What the live scan would have seen while the autopilot was off.

    Returns every confident UP call each Binance-trained model made on candles that closed in
    (since, until] (signal at the candle's close, entry at that close, the same rule as live),
    the candle closes by (symbol, interval) to price the sells at their exit times (back to
    ``reach[(symbol, interval)]`` for trades opened before the pause), and the models whose
    data could not be loaded.
    """
    from .features import add_features
    from .predict import load_bundle

    load = load or (lambda sym, interval, n: load_data("binance", sym, interval, n, save=False))
    reach = reach or {}
    calls, closes, candles, failed = [], {}, {}, []
    for cfg in configs:
        name = config_name(cfg)
        key = (cfg["symbol"], cfg["interval"])
        copied = copy_all or cfg["verdict"] == VALIDATED
        if name not in costs or not (copied or key in reach):
            continue                      # not copied, and no open trade of it to price
        step = pd.Timedelta(milliseconds=INTERVAL_MS[cfg["interval"]])
        hold = hold_time(cfg["interval"], cfg["horizon"])
        try:
            if key not in candles:
                start = min(since, reach.get(key, since))
                candles[key] = load(cfg["symbol"], cfg["interval"], int((until - start) / step) + cfg["horizon"] + 400)
                df = candles[key]
                closes[key] = pd.Series(df["close"].to_numpy(float), index=pd.DatetimeIndex(df["timestamp"]) + step)
            bundle = cfg.get("bundle") or load_bundle(cfg["path"])
            feats = add_features(candles[key]).dropna(subset=bundle["features"])
        except Exception:                 # no data now: the whole catch-up retries on the next update
            failed.append(name)
            continue
        close_at = feats["timestamp"] + step
        window = feats[(close_at > since) & (close_at <= until)]
        if window.empty or not copied:
            continue
        prob_up = bundle["model"].predict_proba(window[bundle["features"]].astype(float))[:, 1]
        need = max(0.5, bundle.get("threshold", 0.5))      # an UP call at or above the model's own threshold
        for ts, close, p in zip(window["timestamp"], window["close"], prob_up):
            if p < need:
                continue
            signal = pd.Timestamp(ts) + step
            calls.append({"config": name, "symbol": cfg["symbol"], "interval": cfg["interval"],
                          "hold": format_duration(hold), "signal_time": signal.isoformat(),
                          "exit_time": (signal + hold).isoformat(), "price": float(close), "confidence": float(p),
                          "verdict": cfg["verdict"], "fee": costs[name]})
    return calls, closes, failed


def catch_up_models(state: dict, calls: list[dict], closes: dict, until: pd.Timestamp, share: float,
                    broker=PaperBroker()) -> list[dict]:
    """Replay the missed time in order: sell each trade at the candle close at its exit time, copy each call."""
    acct = state["accounts"]["models"]
    seen = list(state["seen_signals"])
    seen_set = set(seen)
    events = []

    def exit_price(p):
        s = closes.get((p["symbol"], p["interval"]))
        s = s[s.index <= pd.Timestamp(p["exit_time"])] if s is not None else None
        return float(s.iloc[-1]) if s is not None and len(s) else None

    def sell_due(t):
        keep = []
        for p in sorted(acct["positions"], key=lambda p: p["exit_time"]):
            price = exit_price(p) if pd.Timestamp(p["exit_time"]) <= t else None
            if price is None:
                keep.append(p)
                continue
            events.append({"kind": "sell", "trade": close_trade(acct, p, price, p["exit_time"], broker, replayed=True)})
        acct["positions"] = keep

    # Same priority as the live scanner when cash is short: validated models first, then the most confident.
    for c in sorted(calls, key=lambda c: (c["signal_time"], c["verdict"] != VALIDATED, -c["confidence"])):
        sell_due(pd.Timestamp(c["signal_time"]))            # sells free their cash before the next buy
        key = f"{c['config']}|{c['signal_time']}"
        if key in seen_set or c["config"] in {p["config"] for p in acct["positions"]}:
            continue
        seen.append(key)
        seen_set.add(key)
        invest = min(acct["start_cash"] * share, acct["cash"])
        if invest < 1:
            events.append({"kind": "skip", "text": f"no pretend cash left for {c['symbol']} {c['interval']}"})
            continue
        qty = broker.buy(acct, invest, c["price"], c["fee"])
        pos = {"config": c["config"], "symbol": c["symbol"], "interval": c["interval"], "hold": c["hold"],
               "opened": c["signal_time"], "entry_price": c["price"], "qty": qty, "invested": invest, "fee": c["fee"],
               "exit_time": c["exit_time"], "confidence": c["confidence"], "verdict": c["verdict"], "replayed": True}
        acct["positions"].append(pos)
        events.append({"kind": "buy", "position": pos})
    sell_due(until)
    state["seen_signals"] = seen[-MAX_HISTORY:]
    return events


def catch_up_trend(state: dict, daily, today: pd.Timestamp, broker=PaperBroker()) -> tuple[int, bool]:
    """Rebalance the trend account for every daily decision it missed, at that day's close.

    Stops at the first day with missing data and reports (days done, finished?); the rest is
    retried on the next update.
    """
    acct = state["accounts"]["trend"]
    if not acct.get("rebalanced_for"):
        return 0, True
    days = pd.date_range(pd.Timestamp(acct["rebalanced_for"]) + pd.Timedelta("1D"), today - pd.Timedelta("1D"),
                         freq="1D")[-CATCH_UP_MAX.days:]
    for i, day in enumerate(days):
        guide = TG.guide(state["coins"], daily, day)
        prices = {c["symbol"]: c["price"] for c in guide if c.get("price") is not None}
        if step_trend(state, prices, day, broker, guide=guide, day=day.isoformat()) is None:
            return i, False
    return len(days), True


def catch_up_text(gap: pd.Timedelta, events: list[dict], trend_days: int) -> str:
    hours = gap.total_seconds() / 3600
    off = f"{hours:.0f} hours" if hours < 48 else f"{hours / 24:.0f} days"
    sells = [e["trade"] for e in events if e["kind"] == "sell"]
    buys = sum(e["kind"] == "buy" for e in events)
    lines = [f"⏸ Autopilot was off for {off}. It caught up as if it had been running:"]
    lines.append(f"• model calls: {buys} pretend buys, {len(sells)} sells"
                 + (f" ({sum(t['pnl'] for t in sells):+,.2f} USDT)" if sells else ""))
    lines.append(f"• trend guide: rebalanced for {trend_days} missed day(s)")
    if gap > CATCH_UP_MAX:
        lines.append(f"Only the last {CATCH_UP_MAX.days} days were replayed; the rest doesn't count toward the 90 days.")
    return "\n".join(lines)


# ------------------------------------------------------------------ reports and checks

def go_live_checks(state: dict, now: pd.Timestamp) -> list[dict]:
    """What has to be true before copying the model calls with real money makes sense."""
    days = int((now - pd.Timestamp(state["started"])).total_seconds() - state.get("uncovered_s", 0.0)) // 86400
    prices = state["prices"]
    models, hold = state["accounts"]["models"], state["accounts"]["hold"]
    stats = models.get("stats") or {"count": len(models["trades"]), "pnl": sum(t["pnl"] for t in models["trades"]),
                                    "validated": sum(t.get("verdict") == VALIDATED for t in models["trades"])}
    r_models = account_value(models, prices) / models["start_cash"] - 1
    r_hold = account_value(hold, prices) / hold["start_cash"] - 1
    return [
        {"label": f"At least {GO_LIVE_DAYS} days of pretend trading", "ok": days >= GO_LIVE_DAYS,
         "detail": f"{max(days, 0)} days so far"},
        {"label": f"At least {GO_LIVE_TRADES} finished pretend trades", "ok": stats["count"] >= GO_LIVE_TRADES,
         "detail": f"{stats['count']} so far"},
        {"label": "Copying the model calls made money and beat buy & hold, after fees",
         "ok": stats["count"] > 0 and stats["pnl"] > 0 and r_models > max(r_hold, 0.0),
         "detail": f"{r_models:+.2%} vs {r_hold:+.2%} for buy & hold; finished trades {stats['pnl']:+,.2f} USDT"},
        {"label": "The copied models passed the edge check on unseen data",
         "ok": stats["count"] > 0 and stats["validated"] == stats["count"],
         "detail": f"{stats['validated']} of {stats['count']} trades came from validated models"},
    ]


def summary(state: dict, now: pd.Timestamp) -> dict:
    prices = state["prices"]
    accounts = []
    for key, acct in state["accounts"].items():
        value = account_value(acct, prices)
        accounts.append({"key": key, "name": acct["name"], "value": value, "start_cash": acct["start_cash"],
                         "return": value / acct["start_cash"] - 1, "cash": acct["cash"], "equity": acct["equity"]})
    models = state["accounts"]["models"]
    open_trades = []
    for p in models["positions"]:
        price = prices.get(p["symbol"], p["entry_price"])
        open_trades.append({**p, "price": prices.get(p["symbol"]),
                            "pnl_pct": price * (1 - p["fee"]) * p["qty"] / p["invested"] - 1,
                            "price_move": price / p["entry_price"] - 1,
                            "fees_pct": -(1 - (1 - p["fee"]) ** 2)})   # buy fee paid + sell fee to come
    last_run = pd.Timestamp(state["last_run"]) if state.get("last_run") else None
    return {"running": last_run is not None and now - last_run < STALE_AFTER, "started": state["started"],
            "last_run": state.get("last_run"), "broker": state.get("broker", "paper"), "accounts": accounts,
            "guide": state.get("guide") or [], "guide_day": state.get("guide_day"),
            "open_trades": open_trades, "recent_trades": models["trades"][-15:][::-1],
            "trend_trades": state["accounts"]["trend"]["trades"][-10:][::-1],
            "checks": go_live_checks(state, now), "log": state.get("log", [])[-10:],
            "why": why(models), "copy": state.get("copy")}


def why(models: dict) -> dict:
    """Where the copied trades' money went: wins vs losses, fees vs price moves, per model."""
    trades = models["trades"]
    stats = {**new_stats(), "count": len(trades), "pnl": sum(t["pnl"] for t in trades),
             "wins": sum(t["pnl"] > 0 for t in trades), "fees": sum(trade_fees(t) for t in trades),
             "win_pnl": sum(t["pnl"] for t in trades if t["pnl"] > 0),
             "loss_pnl": sum(t["pnl"] for t in trades if t["pnl"] <= 0)}
    if models.get("stats") and all(k in models["stats"] for k in new_stats()):
        stats = models["stats"]           # totals over every trade, not just the kept history
    losses = stats["count"] - stats["wins"]
    by_model: dict = {}
    for t in trades:
        m = by_model.setdefault(t["config"], {"config": t["config"], "verdict": t.get("verdict"), "trades": 0,
                                              "wins": 0, "pnl": 0.0, "fees": 0.0})
        m["trades"] += 1
        m["wins"] += t["pnl"] > 0
        m["pnl"] += t["pnl"]
        m["fees"] += trade_fees(t)
    return {"count": stats["count"], "wins": stats["wins"],
            "win_rate": stats["wins"] / stats["count"] if stats["count"] else None,
            "avg_win": stats["win_pnl"] / stats["wins"] if stats["wins"] else None,
            "avg_loss": stats["loss_pnl"] / losses if losses else None,
            "pnl": stats["pnl"], "fees": stats["fees"], "pnl_before_fees": stats["pnl"] + stats["fees"],
            "by_model": sorted(by_model.values(), key=lambda m: m["pnl"])}


def report_text(state: dict, now: pd.Timestamp) -> str:
    s = summary(state, now)
    since = (now - pd.Timestamp(state["started"])).days
    lines = [f"📊 Autopilot report, day {since + 1} (pretend money, not real)"]
    for a in s["accounts"]:
        lines.append(f"• {a['name']}: {a['value']:,.2f} USDT ({a['return']:+.1%})")
    day_ago = (now - pd.Timedelta("1D")).isoformat()
    done = [t for t in state["accounts"]["models"]["trades"] if t["closed"] >= day_ago]
    if done:
        wins = sum(t["pnl"] > 0 for t in done)
        lines.append(f"Model trades finished in the last day: {len(done)} ({wins} won), "
                     f"{sum(t['pnl'] for t in done):+,.2f} USDT")
    up = [f"{TG.coin_name(c['symbol'])} {c['exposure']:.0%}" for c in s["guide"] if (c.get("exposure") or 0) > 0]
    unknown = [TG.coin_name(c["symbol"]) for c in s["guide"] if c.get("exposure") is None and c["trend"] != "UNTESTED"]
    if up:
        lines.append("Trend guide, keep in coin: " + ", ".join(up))
    elif s["guide"] and not unknown:
        lines.append("Trend guide: all trends are down, hold USDT.")
    if unknown:
        lines.append("Trend guide has no data yet for: " + ", ".join(unknown))
    lines.append("Not financial advice.")
    return "\n".join(lines)


def trade_text(event: dict) -> str:
    if event["kind"] == "buy":
        p = event["position"]
        return (f"🟢 Pretend BUY {p['symbol']} @ {format_price(p['entry_price'])} for {p['invested']:,.2f} USDT, "
                f"hold {p['hold']} ({p['interval']} model, {p['confidence']:.0%} confident, {p['verdict']})")
    t = event["trade"]
    return (f"{'✅' if t['pnl'] > 0 else '🔴'} Pretend SELL {t['symbol']} @ {format_price(t['exit_price'])}: "
            f"{t['pnl']:+,.2f} USDT ({t['pnl_pct']:+.2%})")


# ------------------------------------------------------------------ one update

def run_once(now: pd.Timestamp | None = None, path: Path = STATE_PATH, scan_fn=scan, daily_fn=daily_bars,
             prices_fn=latest_prices, costs_fn=model_costs, notify=send_alert, cfg: dict | None = None,
             replay_fn=None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    cfg = cfg or settings()
    state = load_state(path, repair=True) or new_state(now, cfg)
    log = list(cfg.get("notes", []))
    messages = []
    today = now.floor("D").isoformat()
    # The catch-up cursor only moves when a whole update finished with all the data it needed,
    # so a pause is replayed even if the first update after it is cut short or offline.
    synced = state.get("synced_until") or state.get("last_run")
    synced = pd.Timestamp(synced) if synced else None
    gap = now - synced if synced is not None else pd.Timedelta(0)
    catching_up = gap > CATCH_UP_AFTER
    days_cache: dict = {}

    def daily(sym):
        if sym not in days_cache:
            days_cache[sym] = daily_fn(sym)
        return days_cache[sym]

    trend_days, trend_done = catch_up_trend(state, daily, now.floor("D")) if catching_up else (0, True)
    if state["guide_day"] != today:                  # the daily rule decides once per UTC day
        new_guide = TG.guide(state["coins"], daily, now)
        if not any(c["trend"] == "ERROR" for c in new_guide):
            messages += TG.changes(state["guide"], new_guide)
            state["guide"] = new_guide
            state["guide_day"] = today
        else:                                        # keep the last complete guide (and its alerts) until then
            log.append("some coins had no daily data; the trend guide retries on the next update")

    fresh = prices_fn(set(state["coins"]) | {p["symbol"] for p in state["accounts"]["models"]["positions"]})
    state["prices"].update(fresh)
    step_hold(state, fresh)
    if state["guide_day"] == today and trend_done:
        if step_trend(state, fresh, now) is None:
            log.append("no price for some coins; the trend account rebalances on the next update")
    state["last_run"] = now.isoformat()
    save_state(state, path)        # the dashboard fills in now; checking every token model below can take minutes

    try:
        rows = scan_fn()
    except Exception as exc:
        rows = []
        log.append(f"scan failed: {exc}")
    positions = state["accounts"]["models"]["positions"]
    now_prices = prices_fn({r["symbol"] for r in rows if r.get("direction") == "UP"} | {p["symbol"] for p in positions})
    state["prices"].update(now_prices)
    costs = costs_fn()
    configs = {config_name(r): r.get("verdict") for r in rows if r.get("status") != "ERROR"}
    state["copy"] = {"all": cfg.get("copy_all", False), "models": len(configs),
                     "validated": sum(v == VALIDATED for v in configs.values())}
    state["accounts"]["models"]["name"] = "Copy every model call" if cfg.get("copy_all") else "Copy validated model calls"
    events = []
    models_done = True
    if catching_up:
        since = max(synced, now - CATCH_UP_MAX)
        reach = {}
        for p in positions:
            key = (p["symbol"], p["interval"])
            reach[key] = min(reach.get(key, now), pd.Timestamp(p["exit_time"]))
        try:
            if replay_fn is None:
                missed, closes, failed = replay_calls(load_configs(EXPERIMENTS_DIR), since, now, costs, reach=reach,
                                                      copy_all=cfg.get("copy_all", False))
            else:
                missed, closes, failed = replay_fn(since, now, costs)
        except Exception as exc:
            missed, closes, failed = [], {}, [f"replay error: {exc}"]
        models_done = not failed
        if models_done:
            events = catch_up_models(state, missed, closes, now, cfg["trade_share"])
            unpriced = {p["symbol"] for p in state["accounts"]["models"]["positions"]} - set(now_prices)
            if unpriced:
                got = prices_fn(unpriced)
                now_prices.update(got)
                state["prices"].update(got)
    if models_done:
        events += step_models(state, rows, now_prices, costs, now, cfg["trade_share"], copy_all=cfg.get("copy_all", False))
    else:                         # trading now would skip the missed time for good: wait for the data
        log.append(f"no market data for {len(failed)} model(s) ({', '.join(failed[:3])}); "
                   "the pretend trades wait and the catch-up retries on the next update")
    log += [e["text"] for e in events if e["kind"] == "skip"]
    caught_up = catching_up and models_done and trend_done
    if caught_up:
        messages.append(catch_up_text(gap, events, trend_days))
        log.append(f"caught up {gap}: {len(events)} replayed trades, {trend_days} trend day(s)")
        if gap > CATCH_UP_MAX:
            state["uncovered_s"] = state.get("uncovered_s", 0.0) + (gap - CATCH_UP_MAX).total_seconds()
    elif catching_up and not trend_done:
        log.append("no daily data for the trend catch-up; it retries on the next update")

    priced = set(fresh) | set(now_prices)
    for acct in state["accounts"].values():
        last = pd.Timestamp(acct["equity"][-1][0]) if acct["equity"] else None
        if account_symbols(acct) <= priced and (last is None or now - last >= EQUITY_EVERY or events):
            acct["equity"] = (acct["equity"] + [[now.isoformat(), round(account_value(acct, state["prices"]), 4)]])[-MAX_HISTORY:]

    if cfg["trade_alerts"]:                          # replayed trades are summed up in the catch-up message
        messages += [trade_text(e) for e in events if e["kind"] in ("buy", "sell")
                     and not (e.get("position") or e.get("trade") or {}).get("replayed")]
    if now.hour >= cfg["report_hour"] and state["last_report"] != today:
        messages.append(report_text(state, now))
        state["last_report"] = today
    for text in messages:
        try:
            notify(text)
        except requests.RequestException as exc:
            # str(exc) contains the request URL, i.e. the bot token or webhook secret.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            log.append(f"could not send an alert: {type(exc).__name__}{f' (HTTP {status})' if status else ''}")

    state["log"] = (state["log"] + [f"{now:%Y-%m-%d %H:%M} {line}" for line in log])[-50:]
    state["last_run"] = now.isoformat()
    if not catching_up or caught_up:
        state["synced_until"] = now.isoformat()
    save_state(state, path)
    return state


def status(path: Path = STATE_PATH, now: pd.Timestamp | None = None) -> dict:
    state = load_state(path)
    if state is None:
        return {"running": False, "started": None}
    return summary(state, now or pd.Timestamp.now(tz="UTC"))


def format_status(s: dict) -> str:
    if not s.get("started"):
        return "Autopilot has not run yet."
    lines = [f"Autopilot (pretend money) since {s['started'][:10]}, last update {s['last_run'][:16]} UTC"]
    for a in s["accounts"]:
        lines.append(f"  {a['name']:<24} {a['value']:>12,.2f} USDT  {a['return']:+.2%}")
    lines.append(f"  open pretend trades: {len(s['open_trades'])}")
    lines.append("Trend guide (keep in coin): " + ", ".join(
        f"{TG.coin_name(c['symbol'])} {c['exposure']:.0%}" if c.get("exposure") is not None
        else f"{TG.coin_name(c['symbol'])} ?" for c in s["guide"]))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch", type=float, metavar="MINUTES", help="update every MINUTES")
    p.add_argument("--reset", action="store_true", help="delete the pretend accounts and start again")
    args = p.parse_args(argv)
    if args.reset:
        for f in (STATE_PATH, _backup(STATE_PATH)):
            f.unlink(missing_ok=True)
        print("Pretend accounts deleted; the next update starts fresh.")
    while True:
        try:
            run_once()
            print(format_status(status()) + "\n", flush=True)
        except Exception as exc:          # keep the loop alive through network trouble
            print(f"update failed: {type(exc).__name__}: {exc}", flush=True)
        if args.watch is None:
            break
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
