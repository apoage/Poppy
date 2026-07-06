"""TASK_739 ε calibration.

Loads one (or N) Foxglove period(s) of real market data, computes
20-step forward log-return cumulants per (t, asset), and reports
the empirical trade density at a sweep of ε values. Recommends
an ε for the spec's 5–15% target trade density.

Usage:
    python tools/calibrate_739_epsilon.py
    python tools/calibrate_739_epsilon.py --n-periods 3 --lookahead 20
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from mcts_trader.data.history import HistoricalMarket


def compute_trade_density(returns: np.ndarray, lookahead: int, epsilons: list[float]) -> list[dict]:
    T, n_assets = returns.shape
    T_valid = T - lookahead + 1
    if T_valid <= 0:
        return []
    kernel_cumsum = np.cumsum(returns, axis=0)
    r_fwd = np.empty((T_valid, n_assets), dtype=np.float64)
    r_fwd[0] = kernel_cumsum[lookahead - 1]
    r_fwd[1:] = kernel_cumsum[lookahead:] - kernel_cumsum[:-lookahead]

    abs_fwd = np.abs(r_fwd)
    std = float(r_fwd.std())

    rows = []
    for eps in epsilons:
        trade_frac = float((abs_fwd > eps).mean())
        buy_frac = float((r_fwd > eps).mean())
        sell_frac = float((r_fwd < -eps).mean())
        rows.append({
            'epsilon': eps,
            'trade_density': trade_frac,
            'buy_frac': buy_frac,
            'sell_frac': sell_frac,
            'hold_frac': 1.0 - trade_frac,
        })
    return rows, std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--periods-file', default='config/foxglove_periods.json')
    p.add_argument('--db-path', default='trading_data.db')
    p.add_argument('--n-periods', type=int, default=3,
                   help='Number of periods to aggregate over (default: 3)')
    p.add_argument('--n-assets', type=int, default=64)
    p.add_argument('--lookahead', type=int, default=20)
    p.add_argument('--target-density', type=float, default=0.10,
                   help='Target trade density for ε recommendation (default: 0.10 = 10%%)')
    args = p.parse_args()

    epsilons = [1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7.5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2]

    with open(args.periods_file) as f:
        periods_cfg = json.load(f)
    # periods_file is a nested object {group: [runs...]}. Flatten to list of runs.
    runs = []
    if isinstance(periods_cfg, dict):
        for group_runs in periods_cfg.values():
            if isinstance(group_runs, list):
                runs.extend(group_runs)
    elif isinstance(periods_cfg, list):
        runs = periods_cfg
    runs = runs[:args.n_periods]

    print(f"Calibrating TASK_739 ε across {len(runs)} period(s), lookahead={args.lookahead}")
    print()

    all_rows_by_eps: dict[float, list[dict]] = {eps: [] for eps in epsilons}
    stds = []

    for i, run in enumerate(runs, 1):
        label = f"{run['start_date']}→{run['end_date']}"
        print(f"[{i}/{len(runs)}] Loading {label}...")
        try:
            hm = HistoricalMarket(
                db_path=args.db_path,
                n_assets=args.n_assets,
                specific_symbols=run.get('assets'),
                start_date=run['start_date'],
                end_date=run['end_date'],
                use_ram_cache=True,
                skip_validation=True,
            )
            hm.close_db()
        except Exception as e:
            print(f"  skipped: {e}")
            continue

        close = hm.cache_data[:, :, 3].astype(np.float64)
        close = np.maximum(close, 1e-12)
        returns = np.log(close[1:] / close[:-1])
        returns = np.clip(returns, -1.0, 1.0)

        rows, std = compute_trade_density(returns, args.lookahead, epsilons)
        stds.append(std)
        for row in rows:
            all_rows_by_eps[row['epsilon']].append(row)
        print(f"  T={returns.shape[0]}, n_assets={returns.shape[1]}, std(r_fwd)={std:.5f}")

    if not stds:
        print("No periods loaded — aborting.")
        return

    print()
    print("Aggregate across periods")
    print("=" * 80)
    print(f"{'ε':>10}  {'trade_density':>14}  {'BUY':>8}  {'HOLD':>8}  {'SELL':>8}")
    recommended = None
    for eps in epsilons:
        rows = all_rows_by_eps[eps]
        if not rows:
            continue
        td = float(np.mean([r['trade_density'] for r in rows]))
        bf = float(np.mean([r['buy_frac'] for r in rows]))
        hf = float(np.mean([r['hold_frac'] for r in rows]))
        sf = float(np.mean([r['sell_frac'] for r in rows]))
        mark = ''
        if recommended is None and td <= args.target_density:
            recommended = eps
            mark = '   ← recommended'
        print(f"{eps:>10.1e}  {td*100:>13.1f}%  {bf*100:>7.1f}%  {hf*100:>7.1f}%  {sf*100:>7.1f}%{mark}")

    print()
    print(f"Mean std(r_fwd) across periods: {np.mean(stds):.5f}")
    print(f"Normal-approx ε for trade density {args.target_density*100:.0f}%: "
          f"{np.mean(stds) * 1.645:.5f}  (1-tail 90% of std)")
    if recommended is not None:
        print(f"Empirical ε recommendation: {recommended:.2e} "
              f"(first ε at-or-below {args.target_density*100:.0f}% trade density)")
    else:
        print("No ε in the sweep hit the target density from above — widen the sweep.")


if __name__ == '__main__':
    main()
