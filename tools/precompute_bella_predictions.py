"""TASK_740 Part A: Offline pre-compute of Bella-generated log-returns per
training window.

Bella (diff_teacher market generator) is unconditional + regime-conditional —
it has no native mechanism to condition on a specific 60-step OHLCV prefix. We
therefore approximate a *window-specific* Bella forecast by:

  1. Estimating the regime (crash / rally / sideways / volatile / calm) of the
     current window from simple statistics of its log-returns.
  2. Asking Bella to roll out K unconditional samples under that regime.
  3. Averaging the forward per-asset log-returns across the K samples to form
     a single target per horizon.

This matches the task spec's "roll out K samples, average across K" without
inventing a conditioning mechanism Bella doesn't have. Outputs live in the
same log-return space as AuxBellaHead / AuxPriceHead so a single MSE loss is
meaningful.

Output layout (one NPZ per period):
  preds      : (n_windows, n_horizons, n_assets) float32 log-returns
  horizons   : (n_horizons,) int32
  generated_at       : ISO timestamp string
  bella_checkpoint_sha: sha256 of the checkpoint file (first 16 hex chars)

Manifest at config/bella_preds_manifest.json maps period_tag → shard path.

Usage:
  # Smoke-test one window:
  python tools/precompute_bella_predictions.py --dry-run

  # Full pre-compute (multi-hour, writes ~8 GB into data/bella_preds/):
  python tools/precompute_bella_predictions.py \
      --bella-checkpoint checkpoints/diff_teacher/generator_best \
      --periods-file config/foxglove_periods.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('precompute_bella')


def _rotation_hash(period_assets: List[str]) -> str:
    """TASK_745: Stable 8-char hash of a rotation's asset universe (order-
    independent, since the trainer's per-rotation column order is determined by
    DB-ID-asc SQL row order — not the JSON list order — so hashing the SORTED
    symbol list is the right invariant).
    """
    payload = ",".join(sorted(period_assets or []))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _load_bella_canonical_assets(db_path: str, n_assets: int = 64) -> List[str]:
    """TASK_745: Reconstruct Bella's training-time asset list, IN BELLA'S
    OUTPUT-COLUMN ORDER (count-desc, delisted dropped, top-N kept). This must
    match diff_teacher/data_loader.py::MarketDataLoader._load_assets exactly.
    """
    import sqlite3
    DELISTED = {"EOS/USD", "LUNA/USD", "FTT/USD", "SRM/USD"}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
            FROM assets a
            JOIN ohlcv o ON a.id = o.asset_id
            WHERE a.is_active = 1
            GROUP BY a.id
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (n_assets + 20,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    symbols: List[str] = []
    for _aid, sym, _cnt in rows:
        if sym in DELISTED:
            continue
        symbols.append(sym)
        if len(symbols) >= n_assets:
            break
    return symbols


def _regime_index_from_returns(returns_window: np.ndarray) -> int:
    """Classify a (ws, n_assets) log-return window into one of Bella's 5 regimes.
    Convention: 0=crash, 1=rally, 2=sideways, 3=volatile, 4=calm.
    Thresholds chosen to spread across regimes without fine-tuning — final
    regime conditioning is a soft signal, not a critical correctness gate.
    """
    mean_ret = float(np.mean(returns_window))
    std_ret = float(np.std(returns_window))
    if std_ret > 0.015 and mean_ret < -0.001:
        return 0  # crash
    if std_ret > 0.015 and mean_ret > +0.001:
        return 1  # rally
    if std_ret > 0.015:
        return 3  # volatile
    if std_ret < 0.003:
        return 4  # calm
    return 2      # sideways


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_bella(checkpoint_path: str):
    from mcts_trader.training.setup import setup_bella_generator
    generator, cfg = setup_bella_generator(checkpoint_path)
    logger.info(f"Bella loaded: {cfg.architecture}, {cfg.hidden_dim}d, "
                f"{cfg.num_layers}L, seq_length={cfg.seq_length}")
    return generator, cfg


def _window_forward_log_returns(generator, cfg, ws: int, horizons: Tuple[int, ...],
                                k_samples: int,
                                regime_idx: int, seed: int) -> np.ndarray:
    """Generate k_samples Bella rollouts under the given regime and return
    the per-asset *cumulative* log-return at each horizon, averaged across K.
    Shape: (n_horizons, n_assets_bella). TASK_745: returns Bella-canonical
    asset order; caller is responsible for remapping to the trainer's axis.
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(seed)
    regime_arr = jnp.full((k_samples,), regime_idx, dtype=jnp.int32)
    generated = generator.generate_on_device(
        n_samples=k_samples,
        seed=seed,
        regime=regime_arr,
    )
    # Reshape (K, seq, assets*features) → (K, seq, n_assets, 5)
    seq_len = cfg.seq_length
    n_assets_bella = cfg.n_assets
    n_features = cfg.n_features
    generated = generated.reshape(k_samples, seq_len, n_assets_bella, n_features)

    # Channel 3 = close. Bella emits z-scored per-step returns at channel 3;
    # see mcts_trader/training/setup.py::generate_bella_prices_on_device.
    volatility_scale = 0.015
    close_returns = np.array(generated[:, :, :, 3]) * volatility_scale
    jax.block_until_ready(generated)

    # Forward segment starts at index `ws` (position where a newly-observed
    # 60-step window would end).
    forward = close_returns[:, ws:ws + max(horizons), :]  # (K, Hmax, n_assets_bella)
    cum = np.cumsum(forward, axis=1)                       # log-return = Σ per-step returns
    picks = np.array([h - 1 for h in horizons], dtype=np.int32)
    cum_picked = cum[:, picks, :]                          # (K, n_horizons, n_assets_bella)
    mean_across_k = np.mean(cum_picked, axis=0)            # (n_horizons, n_assets_bella)
    # TASK_745: no cyclic pad/truncate here — return Bella-canonical order;
    # the per-period remap in _process_period aligns columns to the trainer.
    return mean_across_k.astype(np.float32)


# TASK_754: trajectory-shape descriptor schema.
SHAPE_SCHEMA_VERSION = 1  # see compute_shape_descriptor() invariants below
SHAPE_DESCRIPTOR_DIM = 14
# Component layout (indices into the 14-vector):
#   0-3 : (mean, std, skew, excess_kurt) over flatten(K, n_assets)  — cross moments
#   4-7 : (mean, std, skew, excess_kurt) over preds[0, :]            — horizon-1 only
#   8-11: (mean, std, skew, excess_kurt) over preds[-1, :]           — last horizon
#   12  : s1               (top singular value of preds: (K, n_assets))
#   13  : s1 / sum(s_i)    (top-mode relative weight)
# Component-group labels (matches default loss weights in train_phase2.py and §4.1 of the spec):
#   cross  → (mean=w0, std=w1, skew=w2, kurt=w3)
#   h1     → (mean=w0, std=w1, skew=w2, kurt=w3)
#   hlast  → (mean=w0, std=w1, skew=w2, kurt=w3)
#   svd    → (s1=w4, s1_ratio=w5)
SHAPE_COMPONENT_GROUPS = ('cross', 'h1', 'hlast', 'svd')


def compute_shape_descriptor(preds_window: np.ndarray) -> np.ndarray:
    """TASK_754: collapse a Bella prediction window into a 14-float
    trajectory-shape descriptor.

    Args:
        preds_window: (K, n_assets) array of Bella-predicted log-returns at
            horizons K and assets in any order. NaN cells are treated as
            absent — see "finite-subset" rule below.

    Returns:
        (14,) float32 descriptor — permutation-invariant under asset
        permutation (moments are symmetric sums; singular value spectrum is
        invariant under column permutation).

    Invariants (schema_version=1):
        - Skew via scipy.stats.skew(bias=False) — sample-size-corrected.
        - Kurtosis via scipy.stats.kurtosis(fisher=True, bias=False) — i.e.
          *excess* kurt, normal → 0, sample-size-corrected.
        - sigma clamp at 1e-6 (~10× smallest expected per-step log-return std).
        - NaN cells dropped before moment/SVD computation (TASK_745 cyclic-pad
          may leave ~half the n_assets=64 columns as NaN per rotation).
        - SVD denominator (sum_i s_i) clamped at 1e-12 to avoid div-by-zero.
        - When the finite subset is empty or degenerate (<2 elements), the
          descriptor row is filled with zeros (loss masks out under
          jnp.isfinite anyway, and zero is permutation-invariant).
    """
    from scipy import stats as _sps

    arr = np.asarray(preds_window, dtype=np.float64)
    assert arr.ndim == 2, f"expected (K, n_assets), got {arr.shape}"
    K = arr.shape[0]

    SIGMA_FLOOR = 1e-6
    SVD_FLOOR = 1e-12

    def _moments(x: np.ndarray) -> Tuple[float, float, float, float]:
        finite = x[np.isfinite(x)]
        if finite.size < 2:
            return (0.0, 0.0, 0.0, 0.0)
        m = float(np.mean(finite))
        s = float(np.std(finite, ddof=0))
        s_clamped = max(s, SIGMA_FLOOR)
        # scipy expects size >= 8 for stable bias-corrected skew/kurt, but
        # for log-returns at horizons 1/5/10/20 with ~33 finite assets per
        # row we routinely sit at 33 ≥ 8 — fine.
        try:
            sk = float(_sps.skew(finite, bias=False))
            ku = float(_sps.kurtosis(finite, fisher=True, bias=False))
        except Exception:
            sk = 0.0
            ku = 0.0
        # Replace any inf/nan that scipy may return on degenerate inputs.
        if not np.isfinite(sk):
            sk = 0.0
        if not np.isfinite(ku):
            ku = 0.0
        return (m, s_clamped, sk, ku)

    # 1) cross moments over (K, n_assets) flattened.
    cross = _moments(arr.reshape(-1))
    # 2) horizon-1 moments
    h1 = _moments(arr[0, :])
    # 3) last-horizon moments
    hlast = _moments(arr[K - 1, :])
    # 4) SVD spectrum: restrict to columns where ALL horizons are finite (so
    #    the matrix is rectangular without NaN). For typical TASK_745 rotation
    #    this drops ~31/64 cyclic-pad columns + EUR/CAD/GBP crosses.
    finite_cols = np.all(np.isfinite(arr), axis=0)
    if finite_cols.sum() >= 1 and K >= 1:
        M = arr[:, finite_cols].astype(np.float64)
        try:
            # full_matrices=False keeps the spectrum even when K != n_finite.
            s_vals = np.linalg.svd(M, compute_uv=False)
        except np.linalg.LinAlgError:
            s_vals = np.zeros(min(M.shape), dtype=np.float64)
        s_sum = float(np.sum(s_vals))
        s1 = float(s_vals[0]) if s_vals.size > 0 else 0.0
        s1_ratio = s1 / max(s_sum, SVD_FLOOR)
    else:
        s1 = 0.0
        s1_ratio = 0.0

    desc = np.array((
        cross[0], cross[1], cross[2], cross[3],
        h1[0], h1[1], h1[2], h1[3],
        hlast[0], hlast[1], hlast[2], hlast[3],
        s1, s1_ratio,
    ), dtype=np.float32)
    assert desc.shape == (SHAPE_DESCRIPTOR_DIM,)
    # Final guard: replace any non-finite with 0.0 (the head trains toward 0,
    # which is a no-op for a zero-init head, and the trainer's NaN mask
    # wouldn't fire because the row is all-finite by construction).
    return np.where(np.isfinite(desc), desc, 0.0).astype(np.float32)


def _period_tag(period: dict) -> str:
    """TASK_745: include rotation hash so 270 unique rotations get unique
    shards. Legacy '<start>__<end>' tags are intentionally NOT recognized by
    the consumer — they were column-mismatched and must be re-precomputed.
    """
    return (f"{period['start_date']}__{period['end_date']}"
            f"__{_rotation_hash(period.get('assets', []))}")


def _process_period(period: dict, generator, cfg, ws: int, horizons: Tuple[int, ...],
                    k_samples: int, n_assets_target: int, out_dir: Path,
                    bella_sha: str, db_path: str, stride: int = 1,
                    bella_canonical: List[str] | None = None) -> dict | None:
    from mcts_trader.data.history import HistoricalMarket

    tag = _period_tag(period)
    out_path = out_dir / f"bella_preds_{tag}.npz"
    if out_path.exists():
        logger.info(f"[{tag}] shard exists — skip")
        return {'path': str(out_path.relative_to(REPO_ROOT)) if out_path.is_relative_to(REPO_ROOT) else str(out_path),
                'k_samples': k_samples,
                'horizons': list(horizons),
                'stride': int(stride),
                'rotation_hash': _rotation_hash(period.get('assets', [])),
                'bella_checkpoint_sha': bella_sha}

    try:
        hm = HistoricalMarket(
            db_path=db_path,
            n_assets=n_assets_target,
            specific_symbols=period.get('assets'),
            start_date=period['start_date'],
            end_date=period['end_date'],
            use_ram_cache=True,
            skip_validation=True,
        )
        hm.close_db()
    except Exception as e:
        logger.warning(f"[{tag}] failed to load: {e}")
        return None

    # TASK_745: derive the trainer's column→symbol mapping for THIS rotation,
    # then build a remap from trainer column i → Bella canonical index j.
    # `hm.assets` is the dict from _load_specific_assets (SQL-row order, which
    # is what the trainer also gets). After cyclic-pad to n_assets_target,
    # trainer column i = hm_loaded_symbols[i % n_loaded].
    hm_loaded = [hm.assets[aid] for aid in hm.asset_ids]
    if not hm_loaded:
        logger.warning(f"[{tag}] HistoricalMarket loaded 0 assets — skip")
        return None
    if bella_canonical is None:
        bella_canonical = _load_bella_canonical_assets(db_path, n_assets=cfg.n_assets)
    sym_to_bella_col = {s: i for i, s in enumerate(bella_canonical)}
    remap = np.full((n_assets_target,), -1, dtype=np.int32)
    n_loaded = len(hm_loaded)
    n_unknown = 0
    for i in range(n_assets_target):
        sym = hm_loaded[i % n_loaded]
        j = sym_to_bella_col.get(sym, -1)
        if j < 0:
            n_unknown += 1
        remap[i] = j
    if n_unknown:
        logger.info(f"[{tag}] {n_unknown}/{n_assets_target} trainer columns not in "
                    f"Bella canonical universe — will be NaN-masked")

    close = hm.cache_data[:, :, 3]                                    # (T, n_assets_real)
    returns = np.log(np.maximum(close[1:], 1e-8) / np.maximum(close[:-1], 1e-8))
    T_ret = returns.shape[0]
    n_windows = max(T_ret - ws, 0)
    if n_windows <= 0:
        logger.warning(f"[{tag}] period too short ({T_ret} returns ≤ {ws}) — skip")
        return None

    stride = max(1, int(stride))
    n_compute = (n_windows + stride - 1) // stride  # ceil
    logger.info(f"[{tag}] T_ret={T_ret:,} n_windows={n_windows:,} K={k_samples} "
                f"stride={stride} (computing {n_compute:,} windows, "
                f"{n_windows - n_compute:,} NaN-masked)")
    # Option B: full-shape array; positions not on stride grid are NaN so the
    # trainer's existing NaN mask filters them out without consumer changes.
    if stride > 1:
        preds = np.full((n_windows, len(horizons), n_assets_target), np.nan,
                        dtype=np.float32)
    else:
        preds = np.zeros((n_windows, len(horizons), n_assets_target),
                         dtype=np.float32)
    # Seed salt derived from the tag so reruns are reproducible per-period.
    salt = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], 'big')

    t0 = time.time()
    computed = 0
    # Precompute boolean mask + safe gather index for the remap, so each
    # window's gather is a single fancy-index op.
    valid_mask = remap >= 0                              # (n_assets_target,)
    safe_idx = np.where(valid_mask, remap, 0).astype(np.int64)
    for w in range(0, n_windows, stride):
        window_ret = returns[w:w + ws]
        regime_idx = _regime_index_from_returns(window_ret)
        bella_out = _window_forward_log_returns(
            generator, cfg, ws=ws, horizons=horizons,
            k_samples=k_samples,
            regime_idx=regime_idx, seed=(salt + w) & 0x7FFFFFFF,
        )                                                # (n_horizons, n_assets_bella)
        # TASK_745: remap Bella columns → trainer columns; NaN where unknown.
        gathered = bella_out[:, safe_idx]                # (n_horizons, n_assets_target)
        preds[w] = np.where(valid_mask[None, :], gathered, np.nan)
        computed += 1
        if computed % 200 == 0:
            rate = computed / (time.time() - t0)
            logger.info(f"[{tag}] {computed}/{n_compute} computed windows "
                        f"({rate:.1f} w/s)")

    np.savez_compressed(
        out_path,
        preds=preds,
        horizons=np.array(horizons, dtype=np.int32),
        generated_at=np.array(_dt.datetime.utcnow().isoformat()),
        bella_checkpoint_sha=np.array(bella_sha),
        # TASK_745: persist the remap + canonical universe so consumers can
        # audit column semantics without re-deriving from the DB.
        rotation_hash=np.array(_rotation_hash(period.get('assets', []))),
        trainer_asset_order=np.array(hm_loaded),
        bella_canonical_assets=np.array(bella_canonical),
        remap=remap,
    )
    logger.info(f"[{tag}] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) "
                f"in {time.time() - t0:.1f}s")
    return {'path': str(out_path.relative_to(REPO_ROOT)) if out_path.is_relative_to(REPO_ROOT) else str(out_path),
            'k_samples': k_samples,
            'horizons': list(horizons),
            'stride': int(stride),
            'rotation_hash': _rotation_hash(period.get('assets', [])),
            'n_remap_unknown': int(n_unknown),
            'bella_checkpoint_sha': bella_sha,
            'n_windows': int(n_windows),
            'n_computed': int(computed)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bella-checkpoint', type=str,
                    default='checkpoints/diff_teacher/generator_best/generator',
                    help='Bella generator checkpoint directory (must contain '
                         'the Orbax-format checkpoint itself)')
    ap.add_argument('--periods-file', type=str,
                    default='config/foxglove_periods.json')
    ap.add_argument('--output-dir', type=str, default='data/bella_preds')
    ap.add_argument('--manifest', type=str,
                    default='config/bella_preds_manifest.json')
    ap.add_argument('--horizons', type=str, default='1,5,10,20',
                    help='Comma-separated list of forward horizons in steps.')
    ap.add_argument('--k-samples', type=int, default=4,
                    help='Number of Bella samples to average per window.')
    ap.add_argument('--window-size', type=int, default=60,
                    help='Observed window length (must match agent ws).')
    ap.add_argument('--n-assets', type=int, default=64)
    ap.add_argument('--db-path', type=str, default='trading_data.db')
    ap.add_argument('--limit-periods', type=int, default=0,
                    help='Process only the first N periods (0=all). Debug use.')
    ap.add_argument('--stride', type=int, default=1,
                    help='Compute Bella prediction every Nth window; intermediate '
                         'positions in the shard are filled with NaN. Default 1 '
                         '(dense, byte-identical to legacy). The consumer masks '
                         'NaN samples (train_phase2.py NaN filter), so shards '
                         'remain full-shape and indexing is unchanged.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Load Bella, compute one window, print shape + preview, exit.')
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    horizons = tuple(int(x) for x in args.horizons.split(','))
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint and compute sha fingerprint for reproducibility.
    ckpt_path = Path(args.bella_checkpoint).resolve()
    if ckpt_path.is_dir():
        candidates = [p for p in ckpt_path.rglob('*') if p.is_file()]
        if not candidates:
            raise SystemExit(f"Bella checkpoint dir is empty: {ckpt_path}")
        fp_file = sorted(candidates, key=lambda p: -p.stat().st_size)[0]
    elif ckpt_path.is_file():
        fp_file = ckpt_path
    else:
        raise SystemExit(f"Bella checkpoint not found: {ckpt_path}")
    bella_sha = _sha16(fp_file)
    logger.info(f"Bella sha16={bella_sha} (fp file: {fp_file.name})")

    generator, cfg = _load_bella(str(ckpt_path))

    # TASK_745: derive Bella's canonical asset list once (same SQL/filter as
    # diff_teacher/data_loader.py::_load_assets) so every period reuses it.
    bella_canonical = _load_bella_canonical_assets(args.db_path, n_assets=cfg.n_assets)
    logger.info(f"Bella canonical universe: {len(bella_canonical)} symbols "
                f"(top by row count from {args.db_path})")

    with open(args.periods_file) as f:
        periods = json.load(f).get('periods', [])
    if args.limit_periods > 0:
        periods = periods[:args.limit_periods]

    if args.dry_run:
        from mcts_trader.data.history import HistoricalMarket
        p = periods[0]
        logger.info(f"[dry-run] period {_period_tag(p)}")
        try:
            hm = HistoricalMarket(
                db_path=args.db_path, n_assets=args.n_assets,
                specific_symbols=p.get('assets'),
                start_date=p['start_date'], end_date=p['end_date'],
                use_ram_cache=True, skip_validation=True,
            )
            hm.close_db()
        except Exception as e:
            raise SystemExit(f"[dry-run] failed to load period: {e}")
        close = hm.cache_data[:, :, 3]
        returns = np.log(np.maximum(close[1:], 1e-8) / np.maximum(close[:-1], 1e-8))
        window_ret = returns[:args.window_size]
        regime_idx = _regime_index_from_returns(window_ret)
        logger.info(f"[dry-run] regime={regime_idx}")
        t0 = time.time()
        preds = _window_forward_log_returns(
            generator, cfg, ws=args.window_size, horizons=horizons,
            k_samples=args.k_samples,
            regime_idx=regime_idx, seed=12345,
        )
        logger.info(f"[dry-run] preds shape = {preds.shape} (Bella-canonical order)")
        logger.info(f"[dry-run] horizons     = {horizons}")
        logger.info(f"[dry-run] preds[:, :3] = {preds[:, :3].tolist()}")
        logger.info(f"[dry-run] wall = {time.time() - t0:.1f}s — NOT writing files")
        return

    free_gb = shutil.disk_usage(out_dir).free / 1e9
    if free_gb < 8.0:
        logger.warning(f"Only {free_gb:.1f} GB free under {out_dir} — target ≥ 8 GB.")

    stride = max(1, int(args.stride))
    manifest = {
        'bella_checkpoint': str(ckpt_path.relative_to(REPO_ROOT)) if ckpt_path.is_relative_to(REPO_ROOT) else str(ckpt_path),
        'bella_checkpoint_sha': bella_sha,
        'window_size': args.window_size,
        'horizons': list(horizons),
        'k_samples': args.k_samples,
        'stride': stride,
        'generated_at': _dt.datetime.utcnow().isoformat(),
        'shards': {},
    }
    if Path(args.manifest).exists():
        try:
            old = json.load(open(args.manifest))
            manifest['shards'].update(old.get('shards', {}))
        except Exception:
            pass

    for i, period in enumerate(periods):
        tag = _period_tag(period)
        logger.info(f"[{i + 1}/{len(periods)}] {tag}")
        entry = _process_period(period, generator, cfg, args.window_size, horizons,
                                 args.k_samples, args.n_assets, out_dir, bella_sha,
                                 args.db_path, stride=stride,
                                 bella_canonical=bella_canonical)
        if entry is not None:
            manifest['shards'][tag] = entry
            with open(args.manifest, 'w') as f:
                json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {args.manifest} with "
                f"{len(manifest['shards'])}/{len(periods)} shards.")


if __name__ == '__main__':
    main()
