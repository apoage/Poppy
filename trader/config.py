#!/usr/bin/env python3
"""TASK_820 — frozen configuration for the trend CTA.

All tunables live here in three frozen dataclasses + a paths block. Nothing in the
strategy/risk/backtest layers re-decides these constants; they read them from here.

The StrategyConfig defaults ARE the 819 Phase-3 frozen cell (broad / equal-weight /
single-30d-sign trend, H=1d daily rebalance, gross <= 1). The RiskConfig overlay defaults
to OFF so the backtest stays bit-comparable to the 819 research numbers (gate 1a); paper
trading flips it ON.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as _np

# --- swappable array backend (operator override 2026-07-05) ------------------
# Math kernels are written against a module-level `xp` handle so heavy batch work can
# run on jax.numpy (GPU). numpy float64 is the DEFAULT and the determinism reference for
# parity/selftest; jax is opt-in via set_backend("jax") / CLI --backend jax.
_BACKEND = {"xp": _np, "name": "numpy"}


def set_backend(name: str):
    """Select the array backend. 'numpy' (default, float64 determinism reference) or
    'jax' (jax.numpy; enables float64 so results match the numpy reference). Returns xp."""
    if name == "numpy":
        _BACKEND["xp"], _BACKEND["name"] = _np, "numpy"
    elif name == "jax":
        import jax  # noqa: WPS433 (opt-in heavy import)
        jax.config.update("jax_enable_x64", True)  # match the numpy float64 reference
        import jax.numpy as jnp
        _BACKEND["xp"], _BACKEND["name"] = jnp, "jax"
    else:
        raise ValueError(f"unknown backend {name!r} (expected 'numpy' or 'jax')")
    return _BACKEND["xp"]


def get_backend():
    """Return the active array module (numpy by default)."""
    return _BACKEND["xp"]


def backend_name() -> str:
    return _BACKEND["name"]


def to_numpy(a):
    """Materialize any backend array to a numpy float64 array (host). Scoring/bootstrap/IO
    run on numpy for RNG determinism, so kernels hand results back through this boundary."""
    return _np.asarray(a, dtype=_np.float64)

# --- repo-relative paths -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(REPO_ROOT / "trading_data.db")
NPZ_CACHE = str(REPO_ROOT / "runs" / "819_single_asset" / "_close_broad_daily.npz")
PAPER_DIR = REPO_ROOT / "runs" / "820_paper"
PAPER_STATE = PAPER_DIR / "state.json"
PAPER_LEDGER = PAPER_DIR / "ledger.csv"

# --- global constants (819/816-frozen) ---------------------------------------
K = 1440                         # bars per daily timeframe (1-min bars; 819 daily trend)
BAR_SEC = 60                     # 1-minute source bars
SEC_PER_DAY = 86400
MINUTES_PER_YEAR = 525600.0      # 365 * 24 * 60 (used by the annualized Sharpe)
VOL_FLOOR = 1e-6                 # trailing vol below this -> asset treated inactive
VOL_WINDOW = 30                  # trailing realized-vol window (days) for present-mask

# Regime windows (819). The 2022 BEAR is the judge; FULL is the deployable span.
REGIMES = [
    ("BULL21", "2021-01-01", "2021-11-10"),
    ("BEAR22", "2021-11-10", "2022-12-31"),
    ("REC23",  "2023-01-01", "2023-12-31"),
    ("BULL24", "2024-01-01", "2024-12-31"),
    ("CHOP25", "2025-01-01", "2025-09-01"),
    ("FULL",   "2021-01-01", "2025-09-01"),
]
REG_NAMES = [r[0] for r in REGIMES]

# Stablecoins / fiat tokens excluded from the point-in-time USD-crypto universe.
STABLE_FIAT = frozenset({
    "USDT", "USDC", "DAI", "TUSD", "USDD", "PYUSD", "GUSD", "USDP", "FRAX", "LUSD",
    "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "USD", "USDG", "RLUSD", "EURT", "USDS",
})


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen 819 Phase-3 cell. Signal = sign(logC_t - logC_{t-30d}); equal weight over
    present assets; gross <= 1; daily (H=1) rebalance at close. invvol / tanh / lookback
    ensemble are kept AVAILABLE (strategy.py supports them) but are NON-DEFAULT — 819
    lever attribution rejected inverse-vol (-0.06) and the lookback ensemble (-0.16)."""
    lookbacks: Tuple[int, ...] = (30,)     # single 30d sign trend (819 best)
    sizing: str = "equal"                  # equal | invvol (invvol available, non-default)
    signal: str = "sign"                   # sign | tanh
    vol_window: int = VOL_WINDOW           # trailing realized-vol window (days)
    vol_floor: float = VOL_FLOOR
    H_rebal: int = 1                       # rebalance every 1 day (daily close)
    gross_cap: float = 1.0                 # $1 long/short book (gross exposure <= 1)
    costs_bp: Tuple[int, ...] = (0, 3, 8, 21)   # bp/side, one-sided on turnover
    gate_costs: Tuple[int, ...] = (8, 21)  # gate at net@8 (perps) or net@21 (taker)
    n_boot: int = 2000                     # block-bootstrap resamples
    seed: int = 819


@dataclass(frozen=True)
class RiskConfig:
    """Execution-level risk overlay (the 'with risk management' fork of TASK_819).
    NOT signal shaping — it caps, throttles, bands, and staleness-guards the frozen
    signal's positions. Defaults OFF so the backtest reproduces 819 exactly; paper ON."""
    enabled: bool = False                  # OFF for backtest parity, ON for paper
    per_asset_cap: float = 0.05            # |pos_i| <= 5% of book
    dd_halve: float = -0.20                # drawdown from HWM at which gross -> half
    dd_flat: float = -0.35                 # drawdown at which flat + halt flag
    rebalance_band: float = 0.002          # skip |Delta pos_i| < 0.2% of book
    staleness_days: int = 2                # newest bar older than this -> no new positions


@dataclass(frozen=True)
class PaperConfig:
    """Paper-trading cost model + panel sizing."""
    cost_bp_per_side: float = 21.0         # Kraken spot taker16 + slip5 (bp/side)
    history_days: int = 400                # trailing daily bars pulled per step (>> warm)


# Convenience singletons (the frozen canon).
STRATEGY = StrategyConfig()
RISK = RiskConfig()
PAPER = PaperConfig()
