#!/usr/bin/env python3
"""TASK_820 — production trend CTA package.

Ships the real-but-marginal diversified crypto-OHLC trend edge from TASK_819 (fork branch
ACCEPT) as a proper CTA with an execution-level risk overlay, through the 4-gate deployment
ladder. Frozen strategy: sign(logC_t - logC_{t-30d}) trend, equal weight over present assets,
gross <= 1, daily rebalance. See tasks/TASK_820_SHIP_TREND_CTA.md.

Layout: config (frozen dataclasses + backend), data (daily panels + ohlcv_daily), strategy
(819 signal/mask/weights), risk (overlay 1-4), backtest (accounting + parity), paper (daily
loop), selftest (ST1-6), cli (`python -m trader ...`).
"""

from .config import (StrategyConfig, RiskConfig, PaperConfig, STRATEGY, RISK, PAPER,
                     get_backend, set_backend, backend_name)

__all__ = ["StrategyConfig", "RiskConfig", "PaperConfig", "STRATEGY", "RISK", "PAPER",
           "get_backend", "set_backend", "backend_name"]
__version__ = "0.1.0"
