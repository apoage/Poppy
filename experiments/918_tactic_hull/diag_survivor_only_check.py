"""POST-HOC robustness/diagnostic (NOT part of the pre-registered SPEC.md gate) — re-scores
`reversal_21d_xs` on the SURVIVOR-ONLY panel (include_dead=False, 363 assets, drops the 35
dead/delisted names) to check whether STEP-0's large reversal Sharpe is a dead-coin artifact.
Finding: NO -- survivor-only gross 5.67 / net21 3.98 (dead-inclusive: 6.16 / 4.45), essentially
the same magnitude, so the effect lives in the broadly-illiquid tail (many still-listed
small-cap/memecoin names), not specifically in delisted ones. net80 (the 916 verified realistic
taker tier) is -0.78 -- see README.md "Robustness checks" for the full readout and its
consequence for how to read the STEP-0 verdict.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import loader, tactics, scorer
from trader import backtest as bt

panel = loader.load_panel(include_dead=False)
day_ts, close, ids, syms = panel["day_ts"], panel["close"], panel["ids"], panel["syms"]
with np.errstate(divide="ignore", invalid="ignore"):
    logc = np.log(np.where(close > 0, close, np.nan))
print("n_assets (survivor-only)", logc.shape[1])

pos_arms, pres, warm, diag = tactics.build_all(logc, syms)
d = scorer.score_arm(logc, pos_arms["reversal_21d_xs"], pres, day_ts, warm, syms, H=1,
                     start_ts=bt._to_epoch("2024-01-01"), end_ts=bt._to_epoch("2026-07-01"),
                     n_decisions_monthly=30, n_boot=500)
print("survivor-only reversal_21d_xs: gross_log", d["gross_log_sharpe"], "net21", d["net_log"]["legacy21"],
     "net80", d["net_log"]["base_taker80"], "n", d["n_daily_decisions"])

# also: split reversal book pnl attribution by "how many days old is this asset's LISTING"
# quick proxy: was the asset ever in the include_dead=True-only extra set (i.e. dead/delisted)?
