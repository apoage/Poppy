"""POST-HOC robustness/diagnostic (NOT part of the pre-registered SPEC.md gate) — run after
the STEP-0 result showed an unexpectedly large `reversal_21d_xs` Sharpe, to check whether it is
concentrated in a handful of extreme decisions/assets (a red flag for a data artifact) or
broadly diversified (consistent with a genuine, if likely untradeable-at-modeled-cost,
small-cap/illiquid daily cross-sectional reversal effect — see README.md "Robustness checks").
Finding: NOT concentrated — ~100-asset terciles every day, largest single-day |gross| 5.4%,
per-asset weights 0.39-0.68% (no outlier position size). See README.md for the full readout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import loader, tactics
from trader import backtest as bt

panel = loader.load_panel()
day_ts, close, ids, syms = panel["day_ts"], panel["close"], panel["ids"], panel["syms"]
with np.errstate(divide="ignore", invalid="ignore"):
    logc = np.log(np.where(close > 0, close, np.nan))

pos_arms, pres, warm, diag = tactics.build_all(logc, syms)
pos = pos_arms["reversal_21d_xs"]

r_lo = int(np.searchsorted(day_ts, bt._to_epoch("2024-01-01")))
r_hi = int(np.searchsorted(day_ts, bt._to_epoch("2026-07-01")))
n_present_window = pres[r_lo:r_hi].sum(axis=1)
print("n_present in window: min/mean/max", n_present_window.min(), n_present_window.mean(), n_present_window.max())

H = 1
res = bt.regime_series(logc, pos, pres, r_lo, r_hi, H, warm, xp=np)
gross, turn, bh = res
print("gross stats: mean", gross.mean(), "std", gross.std(), "min", gross.min(), "max", gross.max())
order = np.argsort(-np.abs(gross))
print("top 15 abs-gross decisions (row_in_window, gross, day):")
t_grid = np.arange(max(warm, r_lo), min(r_hi - H, logc.shape[0]-1-H) + 1, H)
for i in order[:15]:
    row = t_grid[i]
    import datetime
    day = datetime.datetime.utcfromtimestamp(int(day_ts[row])).date()
    print(i, row, day, "gross=", gross[i])

# which assets are driving the biggest single-day contributions?
biggest_idx = order[0]
row = t_grid[biggest_idx]
contrib = pos[row] * (logc[row+H] - logc[row])
contrib = np.where(np.isfinite(contrib), contrib, 0.0)
top_assets = np.argsort(-np.abs(contrib))[:10]
print("Biggest single day contributors:", [(syms[j], pos[row,j], contrib[j]) for j in top_assets])

# distribution of position sizes
nz = pos[t_grid] != 0
print("nonzero position magnitudes: min/mean/max", np.abs(pos[t_grid][nz]).min(), np.abs(pos[t_grid][nz]).mean(), np.abs(pos[t_grid][nz]).max())
