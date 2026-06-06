"""asset_select.py — stratified, non-overlapping BUILD/HOLDOUT asset split for the candle-language
cross-asset transfer test (TASK_796 scale-up).

The A8 transfer claim is only as good as the assets it is tested on. The v1/v2 baseline used 9
build / 5 holdout (hand-picked). To make "a vocabulary fit on BUILD transfers to HELD-OUT assets"
a real claim rather than a narrow-regime artifact, we pick a large, PRICE-STRATIFIED, NON-OVERLAPPING
split spanning the whole eligible universe: both sets cover sub-cent to thousands-of-dollars assets,
so transfer cannot be explained by "both sets happen to be the same kind of asset".

Selection:
  - eligible = assets with >= min_bars bars (enough for a 20k-window sample).
  - sort eligible by AVG(close) (a cheap price-regime proxy).
  - take `n_build + n_holdout` evenly-spaced-by-price assets across the full range, then assign them
    to BUILD / HOLDOUT in an interleaved, stratified way so each set independently spans the range.
  - deterministic given (n_build, n_holdout, min_bars, seed).

CPU-only, one GROUP BY scan. Emits JSON {build, holdout, ...} so the orchestrator computes the split
ONCE and passes explicit asset lists to every member (no per-member re-scan).
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np

from . import config as C


def asset_stats(db: str, min_bars: int):
    """Return [(asset_id, n_bars, avg_close), ...] for assets with >= min_bars bars and a positive
    average close. One GROUP BY scan of the ohlcv table."""
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT asset_id, COUNT(*) AS n, AVG(close) AS avg_close FROM ohlcv GROUP BY asset_id"
        ).fetchall()
    finally:
        con.close()
    out = []
    for a, n, p in rows:
        if n is None or p is None:
            continue
        if int(n) >= int(min_bars) and float(p) > 0.0:
            out.append((int(a), int(n), float(p)))
    return out


def select_split(db: str, n_build: int, n_holdout: int, min_bars: int = 20000, seed: int = 42) -> dict:
    """Stratified, non-overlapping build/holdout split spanning the price range. Deterministic."""
    stats = asset_stats(db, min_bars)
    total = int(n_build) + int(n_holdout)
    if len(stats) < total:
        raise ValueError(
            f"asset_select: only {len(stats)} eligible assets (>= {min_bars} bars); "
            f"need {total} ({n_build} build + {n_holdout} holdout). Lower --min-bars or the counts."
        )
    # sort by price proxy, take `total` evenly-spaced-by-price assets across the full range.
    stats.sort(key=lambda t: t[2])
    ids = [t[0] for t in stats]
    prices = [t[2] for t in stats]
    pick_idx = np.unique(np.linspace(0, len(ids) - 1, total).round().astype(int))
    # backfill if uniqueness collapsed some indices (dense low-price end): add nearest unused.
    if pick_idx.size < total:
        used = set(int(i) for i in pick_idx.tolist())
        j = 0
        extra = []
        while pick_idx.size + len(extra) < total and j < len(ids):
            if j not in used:
                extra.append(j); used.add(j)
            j += 1
        pick_idx = np.sort(np.concatenate([pick_idx, np.array(extra, dtype=int)]))
    pick_idx = pick_idx[:total]

    # interleave the price-sorted picks into build/holdout so BOTH sets span the full range:
    # assign the i-th sorted pick to BUILD when the running build-quota advances, else HOLDOUT.
    build, holdout = [], []
    for i in range(total):
        to_build = (i * n_build) // total != ((i + 1) * n_build) // total
        (build if to_build else holdout).append(int(ids[pick_idx[i]]))

    # guardrails: exact sizes, no overlap.
    assert len(set(build) & set(holdout)) == 0, "asset_select: build/holdout overlap"
    # (the quota rule yields exactly n_build/n_holdout for integer i; verify.)
    if len(build) != n_build or len(holdout) != n_holdout:
        # rare rounding drift: rebalance by moving from the larger to the smaller, price-spread-preserving.
        pool = build if len(build) > n_build else holdout
        other = holdout if pool is build else build
        while len(build) != n_build or len(holdout) != n_holdout:
            if len(build) > n_build:
                holdout.append(build.pop(len(build) // 2))
            elif len(holdout) > n_holdout:
                build.append(holdout.pop(len(holdout) // 2))
            else:
                break

    bset = set(build)
    bp = [p for a, _, p in stats if a in bset]
    hp = [p for a, _, p in stats if a in set(holdout)]
    return dict(
        build=sorted(build), holdout=sorted(holdout),
        n_build=len(build), n_holdout=len(holdout),
        min_bars=int(min_bars), seed=int(seed),
        n_eligible=len(stats),
        build_price_range=[float(min(bp)), float(max(bp))] if bp else None,
        holdout_price_range=[float(min(hp)), float(max(hp))] if hp else None,
        full_price_range=[float(prices[0]), float(prices[-1])],
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Stratified build/holdout asset split for candle-lang A8.")
    p.add_argument("--db", type=str, default=None, help="DB path (default config.db_path()).")
    p.add_argument("--n-build", type=int, default=150)
    p.add_argument("--n-holdout", type=int, default=150)
    p.add_argument("--min-bars", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="write JSON here (else stdout).")
    args = p.parse_args(argv)
    db = args.db if args.db else C.db_path()
    split = select_split(db, args.n_build, args.n_holdout, args.min_bars, args.seed)
    js = json.dumps(split, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js + "\n")
        print(f"[asset_select] wrote {args.out}: {split['n_build']} build / {split['n_holdout']} holdout "
              f"from {split['n_eligible']} eligible; build price {split['build_price_range']}, "
              f"holdout price {split['holdout_price_range']}")
    else:
        print(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
