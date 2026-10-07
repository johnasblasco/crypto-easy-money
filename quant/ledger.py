"""Trial ledger: every evaluated configuration is recorded, good or bad.

Selection bias is the main way research fools itself: try 200 ideas, report
the best. The ledger keeps the denominator honest. ``deflated_sharpe`` and
``pbo`` read every trial on the same (horizon, cost) grid, so the bar a
winner must clear rises with every idea tried.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .data import ROOT

LEDGER_DIR = ROOT / "data" / "ledger"


def config_id(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def record(config: dict, report: dict, portfolio_net: pd.Series, ledger_dir: Path = LEDGER_DIR) -> str:
    """Append a trial. ``portfolio_net`` = per-decision-time net returns (zeros when flat)."""
    ledger_dir.mkdir(parents=True, exist_ok=True)
    cid = config_id(config)
    (ledger_dir / "returns").mkdir(exist_ok=True)
    portfolio_net.rename("net").to_frame().to_parquet(ledger_dir / "returns" / f"{cid}.parquet")
    row = {"id": cid, "config": config, "report": report}
    with open(ledger_dir / "trials.jsonl", "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")
    return cid


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


def trials(ledger_dir: Path = LEDGER_DIR) -> list[dict]:
    path = ledger_dir / "trials.jsonl"
    if not path.exists():
        return []
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[row["id"]] = row  # latest record per config wins
    return list(out.values())


def returns_matrix(ids: list[str], ledger_dir: Path = LEDGER_DIR) -> pd.DataFrame:
    cols = {}
    for cid in ids:
        path = ledger_dir / "returns" / f"{cid}.parquet"
        if path.exists():
            cols[cid] = pd.read_parquet(path)["net"]
    return pd.DataFrame(cols).fillna(0.0)


def _horizon(cfg: dict):
    return cfg.get("h", cfg.get("horizon"))


def selection_stats(match: dict, candidate: str, ledger_dir: Path = LEDGER_DIR) -> dict:
    """DSR of ``candidate`` against every recorded trial matching ``match``, and their PBO.

    ``match`` may contain ``h`` (horizon, matched whether the trial stored it as
    ``h`` or ``horizon``); every other key must match exactly. Pass an empty
    dict to deflate against ALL trials ever run (most conservative).
    """
    def ok(cfg):
        for k, v in match.items():
            if k == "h":
                if _horizon(cfg) != v:
                    return False
            elif cfg.get(k) != v:
                return False
        return True

    rows = [t for t in trials(ledger_dir) if ok(t["config"])]
    ids = [t["id"] for t in rows]
    if candidate not in ids:
        raise KeyError(candidate)
    R = returns_matrix(ids, ledger_dir)
    srs = np.array([M.sharpe(R[c].to_numpy()) for c in R.columns])
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    return {
        "n_trials": len(ids),
        "candidate_sharpe_per_period": float(M.sharpe(R[candidate].to_numpy())),
        "expected_max_sharpe_from_luck": M.expected_max_sharpe(len(ids), var_sr),
        "deflated_sharpe": M.deflated_sharpe(R[candidate].to_numpy(), max(len(ids), 1), var_sr),
        "pbo": M.pbo_cscv(R.to_numpy(), n_blocks=10) if R.shape[1] >= 2 else float("nan"),
    }
