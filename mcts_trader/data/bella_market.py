"""TASK_753: Bella as data augmentation — synthetic OHLCV market.

A drop-in replacement for `HistoricalMarket` whose `cache_data` is sourced
from Bella (diff_teacher generator) rather than the database. Mirrors
HistoricalMarket's public surface (`cache_data`, `assets`, `asset_ids`,
`valid_timestamps`, `close_db`, `symbol_to_id`, `id_to_symbol`, `n_assets`,
`min_time`, `max_time`) so the rest of `SegmentPool._make_slot` is
byte-identical between real and Bella slot paths.

Architecture notes (from runs/000_smoke/bella_augment_smoke at task time):
- Bella checkpoint at `checkpoints/diff_teacher/generator_best/generator`
  emits `seq_length=400` per `generate_on_device` call (conv generator,
  768d / 12L / 10-step OT-CFM ODE integration). NOT 1440 as the TASK_753
  spec §4.4 estimated.
- `MarketGenerator.generate(seq_length=1440)` is BROKEN in the current
  checkpoint (conv padding shape mismatch). We therefore chain N
  `generate_on_device` calls and concatenate to reach the target
  `seq_length`. Documented as the "C1 seam tolerance" in TASK_753_REVIEW.md.
- Per TASK_745 the per-rotation column remap is mandatory. For
  augmentation (not distillation), NaN-masked columns are
  cyclic-pad-filled from neighbouring valid columns because the encoder
  cannot consume NaN — this differs from TASK_745 distill which masks
  NaNs at loss-time only.
- OHLCV reconstruction is delegated to
  `mcts_trader.training.setup.generate_bella_prices_on_device` (lifts
  the 90-line OHLCV-constraint logic; do NOT reimplement).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_bella_canonical_assets(db_path: str, n_assets: int = 64) -> List[str]:
    """TASK_745: reconstruct Bella's training-time asset list IN BELLA'S
    OUTPUT-COLUMN ORDER (count-desc, delisted dropped, top-N kept). Lifted
    verbatim from `tools/precompute_bella_predictions.py:70-102`. Must match
    `diff_teacher/data_loader.py::MarketDataLoader._load_assets` exactly.
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


def _rotation_hash(period_assets: List[str]) -> str:
    """TASK_745: stable 8-char hash of a rotation's asset universe (order-
    independent). Lifted from `tools/precompute_bella_predictions.py:60-67`.
    """
    payload = ",".join(sorted(period_assets or []))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _build_remap(
    period_assets: List[str],
    bella_canonical: List[str],
    n_assets_target: int,
) -> tuple[np.ndarray, int]:
    """TASK_745: derive trainer-column → Bella-column remap for one rotation.

    Trainer column i (after cyclic-pad to n_assets_target) corresponds to
    `period_assets[i % n_loaded]`. Bella column j corresponds to
    `bella_canonical[j]`. Returns a `(n_assets_target,)` int32 remap where
    entry i is the Bella column j for that trainer asset, or -1 if the
    trainer asset is not in Bella's universe.
    """
    if not period_assets:
        # Defensive: zero-length period spec → no remap available.
        return np.full((n_assets_target,), -1, dtype=np.int32), n_assets_target

    sym_to_bella_col: Dict[str, int] = {s: i for i, s in enumerate(bella_canonical)}
    remap = np.full((n_assets_target,), -1, dtype=np.int32)
    n_loaded = len(period_assets)
    n_unknown = 0
    for i in range(n_assets_target):
        sym = period_assets[i % n_loaded]
        j = sym_to_bella_col.get(sym, -1)
        if j < 0:
            n_unknown += 1
        remap[i] = j
    return remap, n_unknown


def _cyclic_pad_fill(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill NaN columns in a `(T, A, 5)` OHLCV tensor by cyclic-pad from
    neighbouring valid columns. The encoder cannot consume NaN; for
    augmentation we need a complete tensor (vs distillation, which can mask
    NaN at loss-time).

    `valid_mask`: `(A,)` bool, True where the column is finite.
    """
    if valid_mask.all():
        return arr
    if not valid_mask.any():
        # Degenerate case — no valid columns. Fall back to a constant 100.0
        # close + tight OHLC (synthetic flat market). Volume = 1.
        T, A, _ = arr.shape
        out = np.empty_like(arr)
        out[..., :4] = 100.0
        out[..., 4] = 1.0
        return out

    valid_idx = np.flatnonzero(valid_mask)
    n_valid = valid_idx.size
    out = arr.copy()
    n_assets = arr.shape[1]
    cyclic_idx = valid_idx[np.arange(n_assets) % n_valid]
    # Only overwrite NaN columns to preserve any per-column fidelity in valid columns.
    fill_mask = ~valid_mask
    out[:, fill_mask, :] = arr[:, cyclic_idx[fill_mask], :]
    return out


class BellaMarket:
    """Bella-generated synthetic market with a `HistoricalMarket`-shaped
    interface. Drop-in replacement under `slot['hm']`.

    Construction calls Bella's `generate_on_device` N times (chained, to
    reach the requested `seq_length`), reconstructs OHLCV via
    `generate_bella_prices_on_device`, applies per-rotation column remap,
    and cyclic-pad-fills any NaN columns. The result is a `(seq_length, A,
    5)` float32 OHLCV tensor with all-finite prices.

    Attributes (mirrors HistoricalMarket public surface):
        cache_data: (T, A, 5) float32 OHLCV
        n_assets: int
        assets: dict[int, str]   id → symbol (synthetic IDs 0..A-1)
        asset_ids: list[int]     [0, 1, ..., A-1]
        symbol_to_id: dict[str, int]
        id_to_symbol: dict[int, str]
        valid_timestamps: list[int]   [0, 1, ..., T-1]
        min_time: 0
        max_time: T - 1

    Augmentation-only fields (informational):
        regime_idx: int          regime used at generation time (0..4)
        n_unknown_columns: int   how many columns were NaN-filled
    """

    def __init__(
        self,
        generator,
        cfg,
        seq_length: int,
        n_assets: int,
        regime: int,
        seed: int,
        volatility_scale: float = 0.015,
        asset_policy: str = "remap",
        period_assets: Optional[List[str]] = None,
        bella_canonical: Optional[List[str]] = None,
        db_path: str = "trading_data.db",
    ):
        """
        Args:
            generator: MarketGenerator instance (from setup_bella_generator).
            cfg: GeneratorConfig instance (config of `generator`).
            seq_length: Desired temporal length T. If > cfg.seq_length,
                multiple Bella calls are chained and concatenated.
            n_assets: Trainer-axis asset count (cyclic-pad target).
            regime: Bella regime index 0..4 (crash, rally, sideways,
                volatile, calm) or 5 for unconditional.
            seed: PRNG seed for reproducibility.
            volatility_scale: z-score → log-return scale for Bella's raw
                output (matches `generate_bella_prices_on_device`'s
                hard-coded 0.015).
            asset_policy: 'remap' (per-rotation TASK_745 column remap;
                default), 'canonical' (use Bella's universe directly,
                ignore rotation), 'random' (random asset permutation).
            period_assets: Rotation's asset symbol list (mandatory when
                asset_policy='remap').
            bella_canonical: Bella's canonical asset list (optional;
                derived from `db_path` if not provided).
            db_path: SQLite DB for canonical asset list derivation.
        """
        import jax
        import jax.numpy as jnp
        from mcts_trader.training.setup import generate_bella_prices_on_device

        if asset_policy not in ("remap", "canonical", "random"):
            raise ValueError(
                f"BellaMarket: asset_policy={asset_policy!r} unsupported; "
                f"expected one of remap/canonical/random."
            )

        self.regime_idx = int(regime)
        self.n_assets = int(n_assets)
        self.asset_policy = asset_policy

        # --- 1) Generate Bella OHLCV in n_chunks of cfg.seq_length each.
        native_T = int(cfg.seq_length)
        target_T = int(seq_length) if seq_length > 0 else native_T
        n_chunks = max(1, (target_T + native_T - 1) // native_T)  # ceil
        bella_n_assets = int(cfg.n_assets)

        chunks = []
        for ci in range(n_chunks):
            # Per-chunk seed derived from base seed + chunk index. Each chunk
            # is an independent Bella rollout; concatenated chunks have a
            # seam at every native_T boundary (C1 seam tolerance — the
            # encoder sees a small discontinuity in price level; the oracle
            # path solver tolerates this because each lookahead segment fits
            # within one chunk for typical lookaheads ≤ 20).
            key = jax.random.PRNGKey((int(seed) * 1_000_003 + ci) & 0x7FFFFFFF)
            # generate_bella_prices_on_device uses regime randomly per
            # n_envs; for a deterministic-regime augmentation we call the
            # generator directly with the requested regime, then run the
            # same OHLCV reconstruction inline (matches setup.py:660-700).
            regime_arr = jnp.full((1,), self.regime_idx, dtype=jnp.int32)
            generated = generator.generate_on_device(
                n_samples=1,
                seed=int((int(seed) * 1_000_003 + ci) & 0x7FFFFFFF),
                regime=regime_arr,
            )
            # Reshape (1, native_T, A*F) → (1, native_T, A, F)
            generated = generated.reshape(1, native_T, bella_n_assets, int(cfg.n_features))

            # OHLCV reconstruction — mirrors setup.py:660-700 with
            # `volatility_scale` parameterised.
            close_raw = generated[:, :, :, 3]
            close_returns = close_raw * volatility_scale
            close_log = jnp.cumsum(close_returns, axis=1)
            close_prices = 100.0 * jnp.exp(close_log)

            open_prices = jnp.concatenate(
                [close_prices[:, :1], close_prices[:, :-1]], axis=1
            )
            high_raw = jnp.abs(generated[:, :, :, 1]) * volatility_scale
            high_prices = close_prices * (1.0 + high_raw)
            low_raw = jnp.abs(generated[:, :, :, 2]) * volatility_scale
            low_prices = close_prices * (1.0 - low_raw)
            vol_raw = generated[:, :, :, 4]
            volume = jnp.maximum(jnp.exp(vol_raw * 0.5 + 10.0), 1.0)

            ohlcv = jnp.stack(
                [open_prices, high_prices, low_prices, close_prices, volume],
                axis=-1,
            )
            # Positivity guard on OHLC.
            ohlcv = ohlcv.at[:, :, :, :4].set(jnp.maximum(ohlcv[:, :, :, :4], 1.0))
            jax.block_until_ready(ohlcv)
            chunks.append(np.array(ohlcv[0]))  # (native_T, bella_n_assets, 5)
            # Free underlying GPU buffer.
            del generated, ohlcv

        # --- 2) Concatenate chunks then truncate to target_T.
        if n_chunks == 1:
            bella_ohlcv = chunks[0]
        else:
            bella_ohlcv = np.concatenate(chunks, axis=0)
        bella_ohlcv = bella_ohlcv[:target_T]  # (target_T, bella_n_assets, 5)

        # --- 3) Column remap per asset_policy.
        if asset_policy == "remap":
            if not period_assets:
                logger.warning(
                    "BellaMarket: asset_policy='remap' but period_assets is "
                    "empty — falling back to 'canonical'."
                )
                remapped = self._take_canonical(bella_ohlcv, self.n_assets)
                self.n_unknown_columns = max(0, self.n_assets - bella_n_assets)
            else:
                if bella_canonical is None:
                    bella_canonical = _load_bella_canonical_assets(
                        db_path, n_assets=bella_n_assets
                    )
                remap, n_unknown = _build_remap(
                    period_assets, bella_canonical, self.n_assets
                )
                # Safe gather: -1 entries map to column 0 then NaN-fill.
                safe_idx = np.where(remap >= 0, remap, 0).astype(np.int64)
                gathered = bella_ohlcv[:, safe_idx, :]  # (T, A, 5)
                valid_mask = remap >= 0
                if not valid_mask.all():
                    gathered[:, ~valid_mask, :] = np.nan
                # Cyclic-pad fill NaN columns (augmentation needs all-finite).
                remapped = _cyclic_pad_fill(gathered, valid_mask)
                self.n_unknown_columns = int(n_unknown)
        elif asset_policy == "canonical":
            remapped = self._take_canonical(bella_ohlcv, self.n_assets)
            self.n_unknown_columns = max(0, self.n_assets - bella_n_assets)
        else:  # 'random'
            rng = np.random.default_rng(int(seed))
            perm = rng.permutation(bella_n_assets)
            taken = bella_ohlcv[:, perm, :]
            remapped = self._take_canonical(taken, self.n_assets)
            self.n_unknown_columns = max(0, self.n_assets - bella_n_assets)

        # --- 4) Final guards.
        if not np.isfinite(remapped).all():
            # Should not happen post-fill; this is a defensive paranoia.
            logger.warning(
                "BellaMarket: NaN survived cyclic-pad fill — replacing with 100.0 "
                "(close-band default)."
            )
            remapped = np.where(np.isfinite(remapped), remapped, 100.0)
        assert remapped.shape == (target_T, self.n_assets, 5), (
            f"BellaMarket cache_data shape {remapped.shape} != expected "
            f"({target_T}, {self.n_assets}, 5)"
        )
        # Positivity assertion on OHLC.
        assert (remapped[..., :4] > 0).all(), "BellaMarket: non-positive prices found"

        self.cache_data = remapped.astype(np.float32)

        # --- 5) HistoricalMarket-compatible metadata.
        # Synthetic IDs 0..A-1. Symbols are tagged so debugging sees they're
        # synthetic, and so that downstream code looking up `symbol_to_id`
        # doesn't accidentally pick up a real-market remap.
        if asset_policy == "remap" and period_assets:
            # Carry through the rotation's symbol order (cyclic-padded);
            # this preserves "trainer column i = rotation_assets[i % n_loaded]".
            n_loaded = len(period_assets)
            symbols = [period_assets[i % n_loaded] for i in range(self.n_assets)]
        else:
            # 'canonical' / 'random' / empty rotation: synthesised tags.
            tag = "BELLA" if asset_policy != "random" else "BELLA_RND"
            symbols = [f"{tag}_{i:02d}" for i in range(self.n_assets)]
        self.asset_ids = list(range(self.n_assets))
        self.assets = {i: symbols[i] for i in self.asset_ids}
        self.id_to_symbol = self.assets
        self.symbol_to_id = {v: k for k, v in self.assets.items()}
        self.valid_timestamps = list(range(target_T))
        self.min_time = 0
        self.max_time = target_T - 1
        # `id_to_col` / `ts_to_idx` are inferred from contiguous IDs.
        self.id_to_col = {i: i for i in self.asset_ids}
        self.ts_to_idx = {ts: ts for ts in self.valid_timestamps}

    @staticmethod
    def _take_canonical(arr: np.ndarray, n_target: int) -> np.ndarray:
        """Take/cyclic-pad columns of arr to n_target on axis=1."""
        A = arr.shape[1]
        if A == n_target:
            return arr
        if A > n_target:
            return arr[:, :n_target, :]
        # A < n_target — cyclic-pad.
        repeats = (n_target + A - 1) // A
        return np.tile(arr, (1, repeats, 1))[:, :n_target, :]

    def close_db(self):
        """No-op (no DB connection)."""
        return None
