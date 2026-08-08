"""Rung-1 PREFLIGHT: does the training env actually load all 64 frame slots?

Why this exists. mcts_trader/training/setup.py:429 pads a short price matrix with
    pad_indices = [i % n_loaded for i in range(64)]
which DUPLICATES columns and silently re-maps slot -> asset. The encoder is positionally
universe-locked, so that is an 882-class identity scramble that no error message announces.
If prices.shape[1] < 64 for the rung-1 window, the run must NOT launch.

CPU only, read-only DB, no GPU.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import sys

sys.path.insert(0, "/home/lukas/Trader")
from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

PERIODS = os.environ.get("RUNG1_PERIODS", "/home/lukas/Trader/experiments/821_planner_daily/period_821_train.json")
FRAME = "/home/lukas/Trader/runs/832_e5/frame_provenance_imprint.json"

per = json.load(open(PERIODS))["periods"][0]
frame = json.load(open(FRAME))["frame_symbols"]
assets = per["assets"]

print("period      :", per["name"])
print("window      :", per["start_date"], "->", per["end_date"])
print("n assets    :", len(assets))
print("order-exact vs imprint frame:", assets == frame)

env = VectorBTTradingEnv(
    db_path="/home/lukas/Trader/trading_data.db",
    n_assets=64,
    start_date=per["start_date"],
    end_date=per["end_date"],
    specific_symbols=assets,
    initial_cash=10000.0,
    max_steps=100,
)
p = env.prices
print()
print("prices.shape:", p.shape)
print("n symbols loaded:", len(env.symbols))
print("PAD WOULD FIRE (shape[1] < 64):", p.shape[1] < 64)
print("symbols order-exact vs periods assets:", list(env.symbols) == assets)
missing = [s for s in assets if s not in set(env.symbols)]
print("missing symbols:", missing)
import numpy as np
print("any zero prices:", bool((p <= 0).any()), "| n zero cells:", int((p <= 0).sum()))
print("n non-finite:", int((~np.isfinite(p)).sum()))
