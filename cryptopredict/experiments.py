"""Run the full evaluation over many coins, intervals and horizons.

Usage:
    python -m cryptopredict.experiments                        # 9 major tokens x 1h/4h/1d
    python -m cryptopredict.experiments --symbols BTCUSDT ETHUSDT --intervals 4h --horizons 1 3

Each configuration is trained and evaluated exactly like ``cryptopredict.train``
and saved under ``models/experiments/<SYMBOL>_<interval>_h<horizon>/``. The
summary table goes to ``models/experiments/summary.md`` and ``summary.csv``.

Testing many configurations means some will look good by chance, so the
summary corrects the p-values for multiple comparisons (Holm's method).
"""
from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_data
from .train import DEFAULT_FEE, MODELS_DIR, SIGNIFICANCE, run

EXPERIMENTS_DIR = MODELS_DIR / "experiments"
# Large, liquid Binance spot pairs. Thin markets have wider spreads than the fee model assumes.
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
]


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def summarize(metrics: dict) -> dict:
    best = next(r for r in metrics["results"] if r["model"] == metrics["best_model"])
    edge = metrics["edge"]
    return {
        "symbol": metrics["symbol"],
        "interval": metrics["interval"],
        "horizon": metrics["horizon"],
        "test_rows": metrics["rows"]["test"],
        "best_model": metrics["best_model"],
        "accuracy": best["accuracy"],
        "baseline_accuracy": metrics["best_baseline_accuracy"],
        "roc_auc": best["roc_auc"],
        "p_vs_baseline": best["p_vs_baseline"],
        "threshold": edge["threshold"],
        "strategy_return": edge["strategy_return"],
        "buy_and_hold_return": edge["buy_and_hold_return"],
        "verdict": edge["verdict"],
    }


def _run_pair(symbol, interval, horizons, source, limit, fee, out_dir) -> tuple[list[dict], list[str]]:
    """Train every horizon for one token/interval. Returns (summary rows, log lines)."""
    rows, logs = [], []
    try:
        df = load_data(source, symbol, interval, limit)
    except Exception as exc:
        return rows, [f"!! {symbol} {interval}: could not load data ({exc})"]
    for horizon in horizons:
        name = f"{symbol}_{interval}_h{horizon}"
        try:
            metrics = run(
                df, symbol=symbol, interval=interval, source=source, horizon=horizon,
                fee=fee, out_dir=out_dir / name, verbose=False,
            )
        except ValueError as exc:
            logs.append(f"!! {name}: {exc}")
            continue
        row = summarize(metrics)
        logs.append(
            f"== {name} ({len(df)} candles)\n"
            f"   {row['verdict']}: accuracy {row['accuracy']:.1%} vs baseline "
            f"{row['baseline_accuracy']:.1%}, p={row['p_vs_baseline']:.3f}, "
            f"strategy {row['strategy_return']:+.1%} vs hold {row['buy_and_hold_return']:+.1%}"
        )
        rows.append(row)
    return rows, logs


def run_grid(
    symbols,
    intervals,
    horizons,
    source: str = "binance",
    limit: int = 5000,
    fee: float = DEFAULT_FEE,
    out_dir: Path = EXPERIMENTS_DIR,
    verbose: bool = True,
    jobs: int = 1,
) -> pd.DataFrame:
    log = print if verbose else (lambda *a, **k: None)
    pairs = [(sym, iv) for sym in symbols for iv in intervals]
    args = [(sym, iv, list(horizons), source, limit, fee, out_dir) for sym, iv in pairs]
    rows = []
    if jobs > 1:
        # "spawn", not fork: forking a parent that already runs OpenMP/BLAS threads
        # (LightGBM, numpy) can deadlock the children.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_run_pair, *a) for a in args]
            for fut in as_completed(futures):
                pair_rows, logs = fut.result()
                rows += pair_rows
                for line in logs:
                    log(line, flush=True)
    else:
        for a in args:
            pair_rows, logs = _run_pair(*a)
            rows += pair_rows
            for line in logs:
                log(line, flush=True)
    # Stable order regardless of which worker finished first.
    order = {pair: i for i, pair in enumerate(pairs)}
    rows.sort(key=lambda r: (order[(r["symbol"], r["interval"])], r["horizon"]))

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["p_adjusted"] = holm(summary["p_vs_baseline"].fillna(1.0))
    # After correcting for the number of configurations tested, an edge only
    # counts if it is still significant.
    summary.loc[
        (summary["verdict"] == "POSSIBLE EDGE") & (summary["p_adjusted"] >= SIGNIFICANCE), "verdict"
    ] = "UNPROVEN"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(format_summary(summary, fee))
    return summary


def format_summary(summary: pd.DataFrame, fee: float) -> str:
    pct = lambda v: f"{v * 100:+.1f}%"  # noqa: E731
    lines = [
        "# Experiment summary",
        "",
        f"{len(summary)} configurations. Strategy = long-only at the confidence threshold chosen on "
        f"training data, {fee * 100:.2f}% fee per trade. p adj = Holm-corrected for "
        f"{len(summary)} comparisons.",
        "",
        "| Coin | Interval | Horizon | Model | Accuracy | Baseline | p | p adj | Threshold | Strategy | Buy & hold | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.symbol} | {r.interval} | {r.horizon} | {r.best_model} | {r.accuracy:.1%} | "
            f"{r.baseline_accuracy:.1%} | {r.p_vs_baseline:.3f} | {r.p_adjusted:.3f} | {r.threshold:.0%} | "
            f"{pct(r.strategy_return)} | {pct(r.buy_and_hold_return)} | **{r.verdict}** |"
        )
    edges = summary[summary["verdict"] == "POSSIBLE EDGE"]
    lines += ["", "## Bottom line", ""]
    if edges.empty:
        lines.append(
            "No configuration shows an edge that is both statistically significant (after correction) "
            "and profitable after fees. Treat the predictions as an academic result, not a trading signal."
        )
    else:
        names = ", ".join(f"{r.symbol} {r.interval} h{r.horizon}" for r in edges.itertuples())
        lines.append(
            f"Possible edge in: {names}. Before risking money, paper-trade these forward on new data "
            "for several weeks and check the edge holds."
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--intervals", nargs="+", default=["1h", "4h", "1d"])
    p.add_argument("--horizons", nargs="+", type=int, default=[1])
    p.add_argument("--source", choices=["binance", "csv", "synthetic"], default="binance")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--fee", type=float, default=DEFAULT_FEE)
    p.add_argument("--out", type=Path, default=EXPERIMENTS_DIR)
    p.add_argument("--jobs", type=int, default=1, help="token/interval pairs to train in parallel")
    args = p.parse_args(argv)

    summary = run_grid(
        args.symbols, args.intervals, args.horizons, args.source, args.limit, args.fee, args.out,
        jobs=args.jobs,
    )
    if summary.empty:
        print("No experiments completed.")
        return
    print("\n" + (args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
