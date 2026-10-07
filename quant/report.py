"""Collate saved study results into markdown tables (numbers straight from data/results)."""
from __future__ import annotations

import json
import sys

import numpy as np

from .experiment import RESULTS_DIR


def load(name: str):
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(v, fmt="{:.2f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def events_table(rows, cost="perp_taker", group="majors"):
    rows = [r for r in rows if r["cost"] == cost and r["group"] == group and r.get("n", 0)]
    out = ["| Hypothesis | Params | h (min) | Events | Days | Gross bp | Placebo bp | Net bp | p (net>0) | Holm p | Years net>0 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['hypothesis']} | {r['params']} | {r['h']} | {r['n']} | {r['days']} | {f(r['gross_bps'], '{:.1f}')} | "
                   f"{f(r.get('placebo_gross_bps'), '{:.1f}')} | {f(r['net_bps'], '{:.1f}')} | {f(r.get('p_net'), '{:.3f}')} | "
                   f"{f(r.get('p_holm'), '{:.3f}')} | {r.get('positive_years', '—')} |")
    return "\n".join(out)


def conversion_table(rows):
    out = ["| Set | h | Cost | Decision | AUC | Skilled folds | Trading folds | Trades | Net EV bp | p | Sharpe | Max DD |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['set']} | {r['h']} | {r['cost']} | {r['decision']} | {f(r['auc'], '{:.4f}')} | {r['skilled_folds']} | "
                   f"{r['trading_folds']}/{r['folds']} | {r['active_rows']} | {f(r['ev_bps'], '{:.1f}')} | {f(r['ev_pvalue'], '{:.3f}')} | "
                   f"{f(r['sharpe_ann'])} | {f(r['max_drawdown'], '{:.0%}')} |")
    return "\n".join(out)


def main():
    sections = []
    for name in sys.argv[1:] or ["intraday_conversion", "event_hypotheses", "trend_variants", "volatility_forecast",
                                  "sequence_models", "flow_regressions", "calibration_regimes", "daily_panel_h1",
                                  "daily_panel_h3", "trend_robustness", "xsection_momentum", "ablation"]:
        data = load(name)
        if data is None:
            sections.append(f"## {name}\n\n(not available)")
            continue
        if name == "event_hypotheses":
            sections.append(f"## {name} (majors, perp taker)\n\n" + events_table(data) +
                            f"\n\n## {name} (alts, perp taker)\n\n" + events_table(data, group="alts"))
        elif name == "intraday_conversion":
            sections.append(f"## {name}\n\n" + conversion_table(data))
        else:
            sections.append(f"## {name}\n\n```json\n{json.dumps(data, indent=1, default=str)[:6000]}\n```")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
