"""Autopilot: pretend-money accounts that copy the app's signals automatically, plus the trend guide.

Usage:
    python -m cryptopredict.autopilot              # one update, printed
    python -m cryptopredict.autopilot --watch 15   # update every 15 minutes (Docker runs this for you)
    python -m cryptopredict.autopilot --reset      # start again with fresh pretend money

Three pretend accounts start with the same money (AUTOPILOT_START_CASH, default 1000 USDT):

* Copy the model calls: whenever a token model from ``cryptopredict.experiments`` makes a
  confident UP call (confidence at or above its own threshold), buy AUTOPILOT_TRADE_PCT % of
  the starting money (default 10%), hold for the model's hold time, then sell. Validated or
  not: the point is to see, live, whether the calls make money after fees.
* Follow the trend guide: split the money equally over the guide coins and keep in each the
  share the daily trend rule suggests, rebalanced once a day after 00:00 UTC.
* Buy & hold: buy the guide coins equally on the first day and never trade. The yardstick.

Every pretend buy and sell pays the fee (and half a price tick for the model trades, as in
their back-tests) at the latest Binance price. No real money and no exchange account are used.
All orders go through ``PaperBroker``; connecting a real exchange later means swapping in a
broker that sends the same orders, once the "before real money" checks pass.

Alerts use the scanner's settings (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, DISCORD_WEBHOOK_URL):
trend changes, a daily report (after AUTOPILOT_REPORT_HOUR UTC, default 1) and, with
AUTOPILOT_TRADE_ALERTS=1, every pretend trade. Coins: AUTOPILOT_COINS (default: the 16 main coins).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from . import trend_guide as TG
from .data import BINANCE_HOSTS, DATA_DIR, INTERVAL_MS
from .experiments import EXPERIMENTS_DIR
from .scanner import VALIDATED, format_price, scan, send_alert

STATE_PATH = DATA_DIR / "autopilot" / "state.json"
TREND_FEE = 0.001 + 0.0003        # spot taker fee + 3 bp spread floor, as in the trend research
EQUITY_EVERY = pd.Timedelta(hours=1)
STALE_AFTER = pd.Timedelta(minutes=45)   # no update for this long -> the dashboard says it is off
GO_LIVE_DAYS, GO_LIVE_TRADES = 90, 30
MAX_HISTORY = 3000


def settings() -> dict:
    from quant.data import UNIVERSE

    coins = [s.strip().upper() for s in (os.environ.get("AUTOPILOT_COINS") or "").split(",") if s.strip()]
    return {"start_cash": float(os.environ.get("AUTOPILOT_START_CASH") or 1000),
            "trade_share": float(os.environ.get("AUTOPILOT_TRADE_PCT") or 10) / 100,
            "report_hour": int(os.environ.get("AUTOPILOT_REPORT_HOUR") or 1),
            "trade_alerts": os.environ.get("AUTOPILOT_TRADE_ALERTS") == "1",
            "coins": coins or list(UNIVERSE)}


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
            "accounts": {"models": account("Copy the model calls", positions=[]),
                         "trend": account("Follow the trend guide", holdings={}, rebalanced_for=None),
                         "hold": account("Just buy and hold", holdings={})},
            "seen_signals": [], "guide": None, "guide_day": None, "prices": {}, "last_report": None,
            "last_run": None, "log": []}


def load_state(path: Path = STATE_PATH) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, default=str))
    os.replace(tmp, path)


def account_value(acct: dict, prices: dict) -> float:
    value = acct["cash"]
    for p in acct.get("positions", []):
        value += p["qty"] * prices.get(p["symbol"], p["entry_price"])
    for sym, qty in acct.get("holdings", {}).items():
        value += qty * prices.get(sym, 0.0)
    return value


# ------------------------------------------------------------------ market data

def latest_prices(symbols, timeout: float = 10.0) -> dict:
    """Latest Binance spot prices, one request for all symbols (per symbol if the batch fails)."""
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
                share: float, broker=PaperBroker()) -> list[dict]:
    """Sell finished pretend trades, then copy every new confident UP call. Returns the events."""
    acct = state["accounts"]["models"]
    events = []
    still = []
    for p in acct["positions"]:
        price = prices.get(p["symbol"])
        if pd.Timestamp(p["exit_time"]) > now or price is None:
            still.append(p)
            continue
        got = broker.sell(acct, p["qty"], price, p["fee"])
        trade = {**p, "closed": now.isoformat(), "exit_price": price, "proceeds": got,
                 "pnl": got - p["invested"], "pnl_pct": got / p["invested"] - 1}
        acct["trades"].append(trade)
        events.append({"kind": "sell", "trade": trade})
    acct["positions"] = still
    open_configs = {p["config"] for p in still}
    seen = list(state["seen_signals"])
    for r in rows:
        if r.get("status") == "ERROR" or r.get("direction") != "UP" or r.get("confidence", 0) < r.get("threshold", 1):
            continue
        cfg = config_name(r)
        key = f"{cfg}|{r['signal_time']}"
        signal_age = now - pd.Timestamp(r["signal_time"])
        if (key in seen or cfg in open_configs or cfg not in costs
                or signal_age > pd.Timedelta(milliseconds=INTERVAL_MS[r["interval"]])):
            continue                       # already copied, still holding, not Binance data, or stale
        seen.append(key)
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


def step_trend(state: dict, prices: dict, now: pd.Timestamp, broker=PaperBroker()) -> list[dict]:
    """Once per new daily guide: hold equity x exposure / number of coins in each coin."""
    acct = state["accounts"]["trend"]
    guide = state["guide"]
    if not guide or acct["rebalanced_for"] == state["guide_day"]:
        return []
    coins = state["coins"]
    equity = account_value(acct, prices)
    targets = {c["symbol"]: equity * max(c.get("exposure") or 0.0, 0.0) / len(coins) for c in guide}
    events = []
    orders = []
    for sym in coins:
        price = prices.get(sym)
        if price is None:
            continue
        have = acct["holdings"].get(sym, 0.0) * price
        orders.append((targets.get(sym, 0.0) - have, sym, price))
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
    acct["rebalanced_for"] = state["guide_day"]
    return events


def step_hold(state: dict, prices: dict, broker=PaperBroker()) -> None:
    """Buy every guide coin equally once, then never trade."""
    acct = state["accounts"]["hold"]
    if acct["holdings"]:
        return
    priced = [s for s in state["coins"] if s in prices]
    if not priced:
        return
    each = acct["cash"] / len(priced)
    for sym in priced:
        acct["holdings"][sym] = broker.buy(acct, each, prices[sym], TREND_FEE)


# ------------------------------------------------------------------ reports and checks

def go_live_checks(state: dict, now: pd.Timestamp) -> list[dict]:
    """What has to be true before copying the model calls with real money makes sense."""
    days = (now - pd.Timestamp(state["started"])).days
    prices = state["prices"]
    models, hold = state["accounts"]["models"], state["accounts"]["hold"]
    trades = models["trades"]
    r_models = account_value(models, prices) / models["start_cash"] - 1
    r_hold = account_value(hold, prices) / hold["start_cash"] - 1
    validated = sum(t.get("verdict") == VALIDATED for t in trades)
    return [
        {"label": f"At least {GO_LIVE_DAYS} days of pretend trading", "ok": days >= GO_LIVE_DAYS,
         "detail": f"{days} days so far"},
        {"label": f"At least {GO_LIVE_TRADES} finished pretend trades", "ok": len(trades) >= GO_LIVE_TRADES,
         "detail": f"{len(trades)} so far"},
        {"label": "Copying the model calls beat buy & hold, after fees", "ok": bool(trades) and r_models > r_hold,
         "detail": f"{r_models:+.2%} vs {r_hold:+.2%}"},
        {"label": "The copied models passed the edge check on unseen data", "ok": bool(trades) and validated == len(trades),
         "detail": f"{validated} of {len(trades)} trades came from validated models"},
    ]


def summary(state: dict, now: pd.Timestamp) -> dict:
    prices = state["prices"]
    accounts = []
    for key, acct in state["accounts"].items():
        value = account_value(acct, prices)
        accounts.append({"key": key, "name": acct["name"], "value": value, "start_cash": acct["start_cash"],
                         "return": value / acct["start_cash"] - 1, "cash": acct["cash"], "equity": acct["equity"]})
    models = state["accounts"]["models"]
    open_trades = [{**p, "price": prices.get(p["symbol"]),
                    "pnl_pct": (prices.get(p["symbol"], p["entry_price"]) * (1 - p["fee"]) * p["qty"]) / p["invested"] - 1}
                   for p in models["positions"]]
    last_run = pd.Timestamp(state["last_run"]) if state.get("last_run") else None
    return {"running": last_run is not None and now - last_run < STALE_AFTER, "started": state["started"],
            "last_run": state.get("last_run"), "broker": state.get("broker", "paper"), "accounts": accounts,
            "guide": state.get("guide") or [], "guide_day": state.get("guide_day"),
            "open_trades": open_trades, "recent_trades": models["trades"][-15:][::-1],
            "trend_trades": state["accounts"]["trend"]["trades"][-10:][::-1],
            "checks": go_live_checks(state, now), "log": state.get("log", [])[-10:]}


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
    lines.append("Trend guide, keep in coin: " + (", ".join(up) if up else "nothing (all trends down: hold USDT)"))
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
             prices_fn=latest_prices, costs_fn=model_costs, notify=send_alert, cfg: dict | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    cfg = cfg or settings()
    state = load_state(path) or new_state(now, cfg)
    log = []

    try:
        rows = scan_fn()
    except Exception as exc:
        rows = []
        log.append(f"scan failed: {exc}")

    messages = []
    today = now.floor("D").isoformat()
    if state["guide_day"] != today:                  # the daily rule decides once per UTC day
        new_guide = TG.guide(state["coins"], daily_fn, now)
        messages += TG.changes(state["guide"], new_guide)
        state["guide"] = new_guide
        if not any(c["trend"] == "ERROR" for c in new_guide):
            state["guide_day"] = today
        else:
            log.append("some coins had no daily data; the guide retries on the next update")

    wanted = set(state["coins"]) | {p["symbol"] for p in state["accounts"]["models"]["positions"]}
    wanted |= {r["symbol"] for r in rows if r.get("direction") == "UP"}
    state["prices"].update(prices_fn(wanted))

    events = step_models(state, rows, state["prices"], costs_fn(), now, cfg["trade_share"])
    step_hold(state, state["prices"])
    if state["guide_day"] == today:
        step_trend(state, state["prices"], now)
    log += [e["text"] for e in events if e["kind"] == "skip"]

    for acct in state["accounts"].values():
        last = pd.Timestamp(acct["equity"][-1][0]) if acct["equity"] else None
        if last is None or now - last >= EQUITY_EVERY or events:
            acct["equity"] = (acct["equity"] + [[now.isoformat(), round(account_value(acct, state["prices"]), 4)]])[-MAX_HISTORY:]

    if cfg["trade_alerts"]:
        messages += [trade_text(e) for e in events if e["kind"] in ("buy", "sell")]
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
    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()
        print("Pretend accounts deleted; the next update starts fresh.")
    while True:
        try:
            run_once()
        except Exception as exc:          # keep the loop alive through network trouble
            print(f"update failed: {type(exc).__name__}: {exc}", flush=True)
        print(format_status(status()) + "\n", flush=True)
        if args.watch is None:
            break
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
