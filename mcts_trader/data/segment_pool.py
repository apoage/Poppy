"""
mcts_trader/data/segment_pool.py — SegmentPool + state-channel helpers.

Extracted from train_easy.py (TASK_772.5, 2026-05-23). Holds:
- `SegmentPool` class — pool of active historical segments with linear
  cursors, the canonical RAM-resident data source for both train_easy.py
  (Phase 1 imprint) and train_phase2.py (Stage A/B via load_segment_pool).
- `build_state_channels` — JIT-friendly lax.scan over (target, returns) to
  produce 6 per-step state channels (position, PnL, hold, heat fast/slow,
  capacity); used both inside SegmentPool._make_slot and downstream
  generators (OnTheFlyGenerator in train_easy.py).
- `corrupt_oracle_selections` — TASK_691 entropy injection on oracle path.
- `compute_target_position` — derive {-1, 0, +1} positions from selections
  and returns; shared by SegmentPool oracle precompute and OnTheFlyGenerator.

train_easy.py re-imports all four symbols from this module so existing
callers (including train_phase2.py's `from train_easy import SegmentPool`)
keep working without changes.
"""

import json
import logging
import sqlite3
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# TASK_772.5 (2026-05-23): cross-module dependencies SegmentPool used to get
# transitively via train_easy.py's module-level imports. Replicated here so
# the class is self-contained (no circular dependency on train_easy).
from mcts_trader.data.history import HistoricalMarket
from mcts_trader.jax_agent.path_solver import (
    greedy_lookahead_path,
    optimal_path_to_policy,
    dense_sign_oracle,
    dense_sign_labels_to_policy,
)
from mcts_trader.training.setup import setup_bella_generator

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=['n_assets', 'k', 'window_size'])
def build_state_channels(
    target_positions,   # (T, n_assets) from oracle {-1, 0, +1}
    returns,            # (T, n_assets) log returns
    n_assets=64,
    k=4,
    window_size=64,
    decay_fast=0.80,
    decay_slow=0.99,
):
    """Build 6 per-step state channels via lax.scan.

    Returns:
        state_channels: (T, 6, n_assets) — ready to tile per window
    """
    def step(carry, inputs):
        prev_pos, pnl, steps, h_fast, h_slow = carry
        pos, ret = inputs

        # Detect trades (position changed)
        traded = (pos != prev_pos).astype(jnp.float32)
        # Flips (long→short or reverse) count as double trade
        flipped = (prev_pos != 0) & (pos != 0) & (pos != prev_pos)
        trade_strength = traded + flipped.astype(jnp.float32)

        # Unrealized PnL: signed relative to position direction
        held = (pos == prev_pos) & (prev_pos != 0)
        signed_ret = ret * prev_pos  # long: +ret, short: -ret
        new_pnl = jnp.where(held, pnl + signed_ret, 0.0)
        new_pnl = jnp.clip(new_pnl, -1.0, 1.0)

        # Hold duration: normalized by window_size
        new_steps = jnp.where(traded > 0.5, 0.0, steps + 1.0)

        # Multi-scale heat
        new_h_fast = h_fast * decay_fast + trade_strength
        new_h_slow = h_slow * decay_slow + trade_strength

        # Global: capacity usage
        n_active = jnp.sum(jnp.abs(pos) > 0.5)
        capacity = n_active / k

        out = jnp.stack([
            pos,                                          # ch27: position
            new_pnl,                                      # ch28: unrealized PnL
            jnp.clip(new_steps / window_size, 0.0, 1.0), # ch29: hold duration
            jnp.clip(new_h_fast, 0.0, 1.0),              # ch30: heat fast
            jnp.clip(new_h_slow, 0.0, 1.0),              # ch31: heat slow
            jnp.full(n_assets, capacity),                 # ch32: capacity (broadcast)
        ])  # (6, n_assets)

        return (pos, new_pnl, new_steps, new_h_fast, new_h_slow), out

    init = tuple(jnp.zeros(n_assets) for _ in range(5))
    _, channels = jax.lax.scan(step, init, (target_positions, returns))
    return channels  # (T, 6, n_assets)


def corrupt_oracle_selections(selections, key, p_drop=0.15, q_phantom=0.05):
    """TASK_691: Inject entropy into oracle path before state channel computation.

    Trade drops: oracle selected → force unselected (simulates missed entries)
    Phantom trades: oracle didn't select → force selected (simulates false entries)

    All downstream state channels remain internally consistent because they're
    computed FROM the noisy path. Targets stay clean (uncorrupted oracle actions).
    """
    k1, k2 = jax.random.split(key)
    drop_mask = jax.random.bernoulli(k1, p_drop, selections.shape)
    noisy = jnp.where(drop_mask & (selections > 0.5), 0.0, selections)
    phantom_mask = jax.random.bernoulli(k2, q_phantom, selections.shape)
    noisy = jnp.where(phantom_mask & (selections < 0.5), 1.0, noisy)
    return noisy


def compute_target_position(selections, returns):
    """Derive oracle target position from selections and returns.

    Args:
        selections: (T, n_assets) binary mask from greedy_lookahead_path
        returns: (T, n_assets) log returns

    Returns:
        target_position: (T, n_assets) values in {-1, 0, +1}
    """
    return jnp.where(
        selections > 0.5,
        jnp.where(returns > 0, 1.0, jnp.where(returns < 0, -1.0, 0.0)),
        0.0
    )


class SegmentPool:
    """Pool of active historical segments with linear cursors.

    Maintains N_SLOTS segments in RAM from different date ranges (regime diversity).
    Each segment has a linear cursor that walks sequentially through timestamps.
    When a segment is exhausted, it's replaced with the next run from the same
    date range (same regime, different assets), then a new date range when the
    group is fully exhausted.

    This ensures:
    - Full data coverage (one pass through all 70M timestamps)
    - Linear coherency (sequential walk, not random sampling)
    - Regime diversity (slots from different date ranges)
    - Bounded RAM (~1.9GB for 6 slots × ~316MB each)
    """

    def __init__(self, period_specs, db_path, n_assets=64, n_slots=6, window_length=589,
                 max_runs_per_group=0, min_assets=32, oracle_config=None):
        """
        Args:
            period_specs: List of dicts with start_date, end_date, assets
            db_path: SQLite database path
            n_assets: Assets per segment (for cyclic padding)
            n_slots: Number of concurrent segments in RAM
            window_length: Episode length for get_window (stride = window_length // 2)
            max_runs_per_group: TASK_712: Cap runs per date range (0=unlimited).
                Reduces asset diversity but speeds up training proportionally.
            min_assets: TASK_712: Skip periods with fewer assets than this.
                Tiny periods (e.g. 2 assets padded to 64) are ~97% duplicate data.
            oracle_config: TASK_714: Dict with oracle params for full-period precomputation.
                Keys: oracle_k, oracle_lookahead, transaction_cost, min_alpha_threshold
        """
        self.db_path = db_path
        self.n_assets = n_assets
        self.n_slots = n_slots
        self.window_length = window_length
        self._stride = max(window_length // 2, 1)
        # TASK_714: Oracle config for full-period precomputation
        self._oracle_config = oracle_config or {}

        # Group specs by date range
        self.groups = {}  # (start_date, end_date) → [spec, spec, ...]
        self.group_keys = []  # Ordered list of (start_date, end_date)
        for spec in period_specs:
            key = (spec['start_date'], spec['end_date'])
            if key not in self.groups:
                self.groups[key] = []
                self.group_keys.append(key)
            self.groups[key].append(spec)

        # TASK_712: Filter tiny periods (too few assets → mostly cyclic-pad duplicates)
        if min_assets > 0:
            tiny_groups = []
            for key in self.group_keys:
                # Check first run's asset count
                first_spec = self.groups[key][0]
                n_spec_assets = len(first_spec.get('assets') or [])
                if 0 < n_spec_assets < min_assets:
                    tiny_groups.append(key)
            for key in tiny_groups:
                n_skipped = len(self.groups[key])
                logger.info(f"[POOL] Skipping {key[0][-10:]}→{key[1][-10:]}: "
                            f"{len(self.groups[key][0].get('assets', []))} assets < {min_assets} min "
                            f"({n_skipped} runs)")
                del self.groups[key]
                self.group_keys.remove(key)

        # TASK_712: Cap runs per group for faster training
        n_total_before = sum(len(v) for v in self.groups.values())
        if max_runs_per_group > 0:
            for key in self.groups:
                self.groups[key] = self.groups[key][:max_runs_per_group]

        n_groups = len(self.group_keys)
        n_total = sum(len(v) for v in self.groups.values())
        if max_runs_per_group > 0 and n_total < n_total_before:
            logger.info(f"[POOL] Capped to {max_runs_per_group} runs/group: "
                        f"{n_total_before}→{n_total} segments")
        logger.info(f"[POOL] {n_total} segments across {n_groups} date ranges, {n_slots} slots")

        # Per-group tracking: which run index we're on
        self._group_run_idx = {k: 0 for k in self.group_keys}
        # TASK_712: Track completed passes per group (for logging)
        self._passes_completed = {k: 0 for k in self.group_keys}
        # Queue of groups not yet assigned to a slot
        self._available_groups = list(self.group_keys)

        # Initialize slots
        self.slots = []  # List of slot dicts
        self._round_robin_idx = 0
        self.rng = np.random.default_rng(42)
        # TASK_753 step 0: defensive init of _exhausted_groups. Previously
        # referenced at .add(group_key) below without ever being defined in
        # __init__ — crashed the first time a slot failed to load. Untriggered
        # historically because foxglove periods load cleanly, but exposed by
        # any path that introduces a new failure mode at slot construction.
        self._exhausted_groups = set()

        # TASK_753: Bella-as-data augmentation knobs from oracle_config.
        # When bella_augment is True, slot init coin-flips between
        # _make_slot (historical) and _make_bella_slot (synthetic). The
        # ratio of Bella slots in the pool is tracked via _slot_origins
        # (True = Bella). _rotate_slot preserves the per-slot origin
        # with a fresh coin flip so the ratio holds across the run.
        self.bella_augment = bool(self._oracle_config.get('bella_augment', False))
        self.bella_augment_ratio = float(self._oracle_config.get('bella_augment_ratio', 0.0))
        self.bella_augment_mode = str(self._oracle_config.get('bella_augment_mode', 'append'))
        if self.bella_augment and self.bella_augment_mode != 'append':
            raise NotImplementedError(
                f"TASK_753 v1: --bella-augment-mode '{self.bella_augment_mode}' "
                f"is reserved for v2 (TASK_753b). Use 'append'."
            )
        self._slot_origins = []  # list[bool], True per Bella-sourced slot
        self._bella_generator = None
        self._bella_cfg = None
        self._bella_canonical_assets = None
        self._bella_slot_counter = 0

        # TASK_801b-ii (Fork A hot-path wiring): token front-end. OFF (default)
        # ⇒ self._token_tokenizer stays None, _finalize_slot never tokenizes,
        # slot dicts carry no 'token_ids' ⇒ canon bit-identical, zero CPU (C1).
        # ON ⇒ lazy-load the frozen vocab ONCE here; every slot (init + rotation)
        # is tokenized ONCE in _finalize_slot into a (T, n_assets) int32 RAM grid.
        self._use_token_frontend = bool(self._oracle_config.get('use_token_frontend', False))
        self._token_language = str(self._oracle_config.get('token_frontend_language', 'L1_bpe'))
        self._token_tokenizer = None
        self._token_cfg = None
        if self._use_token_frontend:
            self._init_token_frontend()

        n_init = min(n_slots, n_groups)
        for i in range(n_init):
            group_key = self._available_groups.pop(0)
            slot, is_bella = self._make_slot_with_origin(group_key)
            if slot is not None:
                self.slots.append(slot)
                self._slot_origins.append(is_bella)
            else:
                self._exhausted_groups.add(group_key)

        if not self.slots:
            raise ValueError("No valid segments could be loaded!")

        logger.info(f"[POOL] Initialized {len(self.slots)} slots from {n_init} groups")
        if self.bella_augment:
            n_bella = sum(self._slot_origins)
            logger.info(
                f"[POOL:BELLA-AUG] Bella-augment ON "
                f"(ratio={self.bella_augment_ratio}, mode={self.bella_augment_mode}, "
                f"regime_mode={self._oracle_config.get('bella_augment_regime_mode', 'uniform')}, "
                f"asset_policy={self._oracle_config.get('bella_augment_asset_policy', 'remap')}) — "
                f"{n_bella}/{len(self.slots)} initial slots are Bella-sourced"
            )

        # Stats
        self._segments_loaded = len(self.slots)
        self._groups_started = n_init

    def _init_token_frontend(self):
        """TASK_801b-ii: lazy-load the frozen L1 vocab ONCE for the whole pool.

        The vocab path comes from token_frontend_vocab_file (the pinned 799 codebook at real train
        time, server-only). The tokenizer + its fit-time CandleConfig (window/halflife/sigma_floor)
        are stashed; every slot reuses them. ZERO loading happens when use_token_frontend is OFF.
        """
        from tools.candle_lang.token_hotpath import load_tokenizer
        vocab_file = str(self._oracle_config.get('token_frontend_vocab_file', '') or '')
        if not vocab_file:
            raise ValueError(
                "TASK_801b-ii: use_token_frontend=True but token_frontend_vocab_file is empty. "
                "Set it to a frozen L1 codebook JSON (e.g. runs/799_grammar/L1_RESULT_vocab.json)."
            )
        tok = load_tokenizer(vocab_file)
        self._token_tokenizer = tok
        self._token_cfg = tok.cfg
        logger.info(
            f"[POOL:TOKEN] Token front-end ON: vocab={vocab_file} "
            f"codebook_size={tok.codebook_size} language={self._token_language} "
            f"(window={tok.cfg.window} halflife={tok.cfg.ewma_halflife})"
        )

    def _make_slot_with_origin(self, group_key):
        """TASK_753: coin-flip wrapper around _make_slot / _make_bella_slot.

        Returns (slot, is_bella) where is_bella is True iff the slot was
        produced by _make_bella_slot. Mode is always 'append' in v1 — see
        the NotImplementedError in __init__ for reserved modes.
        """
        if self.bella_augment and self.bella_augment_ratio > 0.0:
            coin = float(self.rng.random())
            if coin < self.bella_augment_ratio:
                slot = self._make_bella_slot(group_key)
                if slot is not None:
                    return slot, True
                # Bella construction failed — fall through to historical.
                logger.warning(
                    f"[POOL:BELLA-AUG] Bella slot for {group_key} failed; "
                    f"falling back to historical."
                )
        slot = self._make_slot(group_key)
        return slot, False

    def _make_slot(self, group_key):
        """Create a slot from the next available run in a group."""
        from mcts_trader.data.history import HistoricalMarket
        runs = self.groups[group_key]

        while self._group_run_idx[group_key] < len(runs):
            run_idx = self._group_run_idx[group_key]
            spec = runs[run_idx]
            label = f"{spec['start_date']}→{spec['end_date']} run {run_idx+1}/{len(runs)}"
            try:
                hm = HistoricalMarket(
                    db_path=self.db_path,
                    n_assets=self.n_assets,
                    specific_symbols=spec.get('assets'),
                    start_date=spec['start_date'],
                    end_date=spec['end_date'],
                    use_ram_cache=True,
                    skip_validation=True,
                )
                hm.close_db()
            except Exception as e:
                logger.warning(f"[POOL] Failed to load {label}: {e}")
                self._group_run_idx[group_key] = run_idx + 1
                continue

            # Cyclic-pad assets if fewer than n_assets
            if hm.cache_data.shape[1] < self.n_assets:
                n_have = hm.cache_data.shape[1]
                repeats = (self.n_assets + n_have - 1) // n_have
                hm.cache_data = np.tile(hm.cache_data, (1, repeats, 1))[:, :self.n_assets, :]

            slot = self._finalize_slot(
                hm, label, group_key, run_idx,
                period_assets=spec.get('assets'),
                source='historical',
            )
            return slot

        return None  # Group exhausted

    def _finalize_slot(self, hm, label, group_key, run_idx,
                       period_assets=None, source='historical',
                       bella_meta=None):
        """TASK_753 step 3 refactor: precompute pipeline shared between
        historical and Bella slots. Takes an `hm`-shaped object with a
        populated `cache_data`, runs the oracle + state channels + policy
        targets + confidence + optional Bella-distill/shape shard loads,
        and returns the slot dict.

        Args:
            hm: HistoricalMarket or BellaMarket with `cache_data` populated.
            label: human-readable slot label for logging.
            group_key: SegmentPool group key.
            run_idx: SegmentPool run index within the group.
            period_assets: rotation's asset symbol list (used for Bella
                shard tag construction in distill/shape paths). For Bella
                slots, pass the rotation's assets — keeps the column
                semantics aligned across real/Bella feeds.
            source: 'historical' or 'bella' — annotation only; downstream
                code does not branch on this.
            bella_meta: optional dict with augmentation metadata
                (regime_idx, n_unknown_columns, seed). Stashed on the slot
                under 'bella_meta' for logging.
        """
        import time as _time
        n_timestamps = len(hm.valid_timestamps)

        # TASK_714: Precompute oracle on full period
        t_oracle = _time.time()
        oracle_sel, oracle_cum_pnl, returns = self._precompute_oracle(hm, label)
        t_oracle = _time.time() - t_oracle

        # TASK_715: Precompute state channels, policy targets, confidence
        T_ret = len(returns)
        oc = self._oracle_config
        oracle_sel_jax = jnp.array(oracle_sel)
        returns_jax = jnp.array(returns)

        # State channels from oracle trajectory
        target_pos = compute_target_position(oracle_sel_jax, returns_jax)
        state_ch_full = build_state_channels(
            target_pos, returns_jax,
            n_assets=self.n_assets,
            k=oc.get('oracle_k', 4),
            window_size=self.window_length,  # approximate ws
        )
        jax.block_until_ready(state_ch_full)
        state_ch_np = np.array(state_ch_full).astype(np.float16)  # (T_ret, 6, n_assets)

        # Policy targets (state-aware)
        prev_sel = jnp.zeros((T_ret, self.n_assets))
        prev_sel = prev_sel.at[1:].set(
            jnp.where(oracle_sel_jax[:-1] > 0.5,
                      jnp.where(returns_jax[:-1] > 0, 1.0, -1.0), 0.0)
        )
        # TASK_739: select BC target oracle. "topk_path" (default) keeps the
        # pre-739 state-aware top-k policy. "dense_sign" replaces the BC
        # target with per-asset sign-of-forward-return labels (every asset,
        # every step). oracle_sel / oracle_cum_pnl are unchanged — still
        # top-k for value / per-step reward / n-step returns.
        oracle_mode = oc.get('oracle_mode', 'topk_path')
        if oracle_mode == 'dense_sign':
            # Match the max_len formula used by greedy_lookahead_path in
            # _precompute_oracle so JIT compiles once (pad to ≥ 262144,
            # rounded up to a multiple of 1024).
            ds_max_len = max(262144, ((T_ret + 1023) // 1024) * 1024)
            dense_labels = dense_sign_oracle(
                returns_jax,
                lookahead=oc.get('oracle_lookahead', 20),
                epsilon=oc.get('sign_epsilon', 1e-4),
                max_len=ds_max_len,
            )
            policy_targets_full = dense_sign_labels_to_policy(
                dense_labels, n_actions_per_asset=5,
            )
        elif oracle_mode == 'hybrid':
            raise NotImplementedError(
                "oracle_mode='hybrid' is reserved — TODO(TASK_739 follow-up). "
                "Expected semantics: top-k drives value signal, dense-sign drives BC."
            )
        else:
            policy_targets_full = optimal_path_to_policy(
                oracle_sel_jax, returns_jax,
                prev_selections=prev_sel,
                n_actions_per_asset=5, temperature=1.0,
            )
        jax.block_until_ready(policy_targets_full)
        policy_np = np.array(policy_targets_full).astype(np.float16)  # (T_ret, n_assets*5)

        # Confidence (max softmax per asset)
        conf_full = np.max(
            policy_np.reshape(T_ret, self.n_assets, 5), axis=-1
        ).astype(np.float16)  # (T_ret, n_assets)

        mb_hm = hm.cache_data.nbytes / 1024**2
        mb_precomp = (oracle_sel.nbytes + oracle_cum_pnl.nbytes + returns.nbytes +
                      state_ch_np.nbytes + policy_np.nbytes + conf_full.nbytes) / 1024**2
        # TASK_753 C3 signal: cumulative oracle PnL per slot. Used by the
        # selector to compute Bella-vs-historical oracle PnL ratios when
        # augmentation is enabled.
        oracle_cum_pnl_final = float(oracle_cum_pnl[-1]) if len(oracle_cum_pnl) else 0.0
        logger.info(f"[POOL] Loaded {label}: {n_timestamps:,} ts, "
                    f"{mb_hm:.0f}+{mb_precomp:.0f} MB (data+precomp) "
                    f"src={source} oracleCumPnL={oracle_cum_pnl_final:.4f}")

        # TASK_733: cheat-channel visibility — fraction of channel-32 values
        # in {−1, 0, +1} for this segment. Uses the same log-return stream
        # the encoder ingests. Logs only when the diagnostic is enabled.
        if oc.get('use_cheat_channel', False):
            la = int(oc.get('cheat_lookahead', 20))
            eps = float(oc.get('cheat_epsilon', 1e-4))
            if T_ret > la:
                # Forward cum log-return: r_fwd[t] = sum(returns[t+1:t+1+la]).
                # np.cumsum then diff: cs[t+la] - cs[t] → (T_ret-la,) aligned to t.
                cs = np.cumsum(np.asarray(returns), axis=0)
                r_fwd = cs[la:] - cs[:-la]  # (T_ret-la, n_assets)
                frac_pos = float(np.mean(r_fwd > eps))
                frac_neg = float(np.mean(r_fwd < -eps))
                frac_hold = 1.0 - frac_pos - frac_neg
                logger.info(
                    f"[POOL:CHEAT] {label}: la={la}, eps={eps:.1e} — "
                    f"+1={frac_pos:.1%}  -1={frac_neg:.1%}  hold={frac_hold:.1%}  "
                    f"(non-HOLD={frac_pos + frac_neg:.1%})"
                )

        max_cursor = max(n_timestamps - self.window_length, 0)

        # TASK_740: Load precomputed Bella predictions for this period if
        # the distillation head is enabled and the manifest lists a shard.
        # Shape: (T_ret, n_horizons, n_assets) of log-returns. Missing
        # shards fall back to None and the trainer masks those samples.
        # TASK_753 note: Bella-sourced slots get bella_preds_full=None
        # unconditionally (no shard would match their synthetic context).
        bella_preds_full = None
        if source == 'historical' and oc.get('use_bella_distill_head', False):
            # TASK_745: tag must include the rotation hash so the consumer
            # only picks up shards whose column semantics actually match
            # this rotation's trainer-axis.
            bella_preds_full = self._load_bella_shard(group_key, T_ret,
                                                     oc.get('bella_pred_horizons', (1, 5, 10, 20)),
                                                     manifest_path=oc.get('bella_preds_manifest',
                                                                           'config/bella_preds_manifest.json'),
                                                     period_assets=period_assets)

        # TASK_754: Load precomputed Bella trajectory-shape descriptors for
        # this period (additive to TASK_740). Shape: (T_ret, descriptor_dim)
        # float32. Missing shards fall back to None and the trainer masks
        # those samples sample-wise.
        bella_shape_full = None
        if source == 'historical' and oc.get('use_bella_shape_head', False):
            bella_shape_full = self._load_bella_shape_shard(
                group_key, T_ret,
                descriptor_dim=oc.get('bella_shape_descriptor_dim', 14),
                manifest_path=oc.get('bella_shape_manifest',
                                     'config/bella_shape_manifest.json'),
                period_assets=period_assets,
            )

        slot = {
            'hm': hm,
            'cursor': 0,
            'max_cursor': max_cursor,
            'group_key': group_key,
            'run_idx': run_idx,
            'label': label,
            # TASK_714: Precomputed oracle
            'oracle_selections': oracle_sel,   # (T-1, n_assets)
            'oracle_cum_pnl': oracle_cum_pnl,  # (T-1,)
            'returns': returns,                 # (T-1, n_assets)
            # TASK_715: Precomputed state/targets/confidence (float16)
            'state_channels': state_ch_np,     # (T-1, 6, n_assets) fp16
            'policy_targets': policy_np,        # (T-1, n_assets*5) fp16
            'confidence': conf_full,            # (T-1, n_assets) fp16
            # TASK_740: Precomputed Bella distillation targets (fp32 or None)
            'bella_preds_full': bella_preds_full,
            # TASK_754: Precomputed Bella trajectory-shape descriptors (fp32 or None)
            'bella_shape_full': bella_shape_full,
            # TASK_753 augmentation metadata (annotation only).
            'source': source,                   # 'historical' | 'bella'
            'oracle_cum_pnl_final': oracle_cum_pnl_final,
            'oracle_wall_s': float(t_oracle),
        }
        if bella_meta is not None:
            slot['bella_meta'] = bella_meta

        # TASK_801b-ii (Fork A hot-path wiring): tokenize this slot ONCE into a
        # (T, n_assets) int32 token grid on the slot's price grid. OFF ⇒ this
        # block never runs ⇒ slot carries no 'token_ids' ⇒ canon bit-identical.
        # ON ⇒ annotate-once-cache-to-RAM (C4); the batch loop only SLICES it.
        # The grid is POST serving-lag (join_timeline_to_grid applies it); the
        # batch slice must NOT re-apply lag.
        if self._use_token_frontend and self._token_tokenizer is not None:
            import time as _time
            from tools.candle_lang.token_hotpath import tokenize_grid
            t_tok = _time.time()
            token_grid = tokenize_grid(
                hm, source, self._token_tokenizer, self._token_cfg,
                db_path=self.db_path, language=self._token_language,
            )
            t_tok = _time.time() - t_tok
            slot['token_ids'] = token_grid     # (T, n_assets) int32 — lag-applied
            n_valid = int(np.count_nonzero(token_grid >= 0))
            logger.info(
                f"[POOL:TOKEN] Tokenized {label}: grid={token_grid.shape} "
                f"valid={n_valid}/{token_grid.size} ({100.0*n_valid/max(token_grid.size,1):.1f}%) "
                f"src={source} in {t_tok:.2f}s"
            )
        return slot

    def _ensure_bella_generator(self):
        """TASK_753: lazy-load the Bella generator on first Bella slot
        construction so non-augment runs pay zero load cost.
        """
        if getattr(self, '_bella_generator', None) is not None:
            return self._bella_generator, self._bella_cfg
        oc = self._oracle_config
        ckpt = oc.get('bella_augment_checkpoint',
                      'checkpoints/diff_teacher/generator_best/generator')
        from mcts_trader.training.setup import setup_bella_generator
        import time as _time
        t0 = _time.time()
        gen, cfg = setup_bella_generator(ckpt)
        load_wall = _time.time() - t0
        logger.info(f"[POOL:BELLA] Generator loaded from {ckpt} "
                    f"(arch={cfg.architecture}, hidden_dim={cfg.hidden_dim}, "
                    f"num_layers={cfg.num_layers}, seq_length={cfg.seq_length}, "
                    f"n_steps={cfg.n_steps}) in {load_wall:.1f}s")
        # TASK_753 C2: log actual architecture facts on first load.
        self._bella_generator = gen
        self._bella_cfg = cfg
        # Lazy-derive Bella's canonical asset list once for all subsequent
        # Bella slots in this pool (matches the TASK_745 reuse pattern in
        # precompute_bella_predictions.py).
        from mcts_trader.data.bella_market import _load_bella_canonical_assets
        self._bella_canonical_assets = _load_bella_canonical_assets(
            self.db_path, n_assets=cfg.n_assets,
        )
        logger.info(f"[POOL:BELLA] Canonical universe: "
                    f"{len(self._bella_canonical_assets)} symbols")
        return gen, cfg

    def _pick_bella_regime(self, group_key, run_idx):
        """TASK_753: choose Bella regime per --bella-augment-regime-mode.

        Returns int in 0..4 (Bella's five regimes) or 5 for unconditional.
        """
        oc = self._oracle_config
        mode = oc.get('bella_augment_regime_mode', 'uniform')
        if mode == 'unconditional':
            return 5
        if mode == 'uniform':
            # Use the pool's rng (seeded with 42 in __init__) for reproducibility.
            return int(self.rng.integers(0, 5))
        if mode == 'match':
            # 'match' would require the rotation's first-window stats —
            # implement when we have a running pool with returns access.
            # For v1 we treat 'match' identically to 'uniform' to keep the
            # slot factory deterministic w.r.t. group_key (it would be
            # confusing for two adjacent Bella slots from the same rotation
            # to get different regimes via 'match').
            return int(self.rng.integers(0, 5))
        raise ValueError(f"Unknown bella_augment_regime_mode={mode!r}")

    def _make_bella_slot(self, group_key, regime_idx=None, seed=None):
        """TASK_753: construct a synthetic Bella-sourced slot.

        Mirrors _make_slot but with a BellaMarket replacing HistoricalMarket.
        Same precompute path (oracle / state channels / policy targets /
        confidence) — AsyncBatchPipeline does not branch on slot source.
        bella_preds_full and bella_shape_full are forced to None (the
        TASK_740/754 distillation paths shouldn't see synthetic context).

        Args:
            group_key: SegmentPool group key (carried through to the slot
                dict for _rotate_slot bookkeeping; the rotation's
                period_assets list is used for the column remap).
            regime_idx: optional Bella regime override (0..5). When None,
                resolved via _pick_bella_regime.
            seed: optional PRNG seed (default: deterministic per
                group_key + Bella slot index for reproducibility).
        """
        import time as _time
        from mcts_trader.data.bella_market import BellaMarket
        oc = self._oracle_config

        # Need rotation's asset list for the column remap (asset_policy='remap').
        # Use the FIRST run's assets for the group as the canonical rotation —
        # consistent with the trainer's view of "one rotation = one asset set".
        runs = self.groups.get(group_key, [])
        if not runs:
            logger.warning(f"[POOL:BELLA] group {group_key} has no runs — cannot construct Bella slot")
            return None
        first_spec = runs[0]
        period_assets = first_spec.get('assets', [])

        # Resolve regime + seed.
        if regime_idx is None:
            regime_idx = self._pick_bella_regime(group_key, run_idx=0)
        if seed is None:
            self._bella_slot_counter = getattr(self, '_bella_slot_counter', 0) + 1
            seed = (hash(group_key) ^ self._bella_slot_counter * 1_000_003) & 0x7FFFFFFF

        # Lazy-load the generator (one-shot per pool lifetime).
        gen, cfg = self._ensure_bella_generator()

        # Determine target seq_length. 0 → native; >0 → chained chunks.
        target_seq = int(oc.get('bella_augment_seq_length', 0) or cfg.seq_length)
        # Must be at least window_length so the pool can actually use the slot.
        if target_seq < self.window_length:
            logger.warning(f"[POOL:BELLA] bella_augment_seq_length={target_seq} < "
                           f"window_length={self.window_length}; bumping to window_length.")
            target_seq = int(self.window_length)

        t_build = _time.time()
        bm = BellaMarket(
            generator=gen,
            cfg=cfg,
            seq_length=target_seq,
            n_assets=self.n_assets,
            regime=int(regime_idx),
            seed=int(seed),
            volatility_scale=float(oc.get('bella_augment_volatility_scale', 0.015)),
            asset_policy=str(oc.get('bella_augment_asset_policy', 'remap')),
            period_assets=period_assets,
            bella_canonical=getattr(self, '_bella_canonical_assets', None),
            db_path=self.db_path,
        )
        bella_build_wall = _time.time() - t_build

        label = (f"BELLA {group_key[0][-10:]}→{group_key[1][-10:]} "
                 f"reg={regime_idx} seed={seed % 1_000_000:06d}")
        bella_meta = {
            'regime_idx': int(regime_idx),
            'seed': int(seed),
            'n_unknown_columns': int(getattr(bm, 'n_unknown_columns', 0)),
            'seq_length': int(target_seq),
            'build_wall_s': float(bella_build_wall),
        }
        self._bella_regime_counts = getattr(self, '_bella_regime_counts', None)
        if self._bella_regime_counts is None:
            import collections as _coll
            self._bella_regime_counts = _coll.Counter()
        self._bella_regime_counts[int(regime_idx)] += 1
        logger.info(
            f"[POOL:BELLA] {label}: build={bella_build_wall:.2f}s "
            f"unknown_cols={bm.n_unknown_columns}/{self.n_assets} "
            f"seq_length={target_seq}"
        )

        slot = self._finalize_slot(
            bm, label, group_key, run_idx=0,
            period_assets=period_assets, source='bella',
            bella_meta=bella_meta,
        )
        return slot

    def _precompute_oracle(self, hm, label):
        """TASK_714: Compute oracle on full period. Returns numpy arrays."""
        close = hm.cache_data[:, :, 3]  # (T, n_assets)
        returns = np.log(np.maximum(close[1:], 1e-8) / np.maximum(close[:-1], 1e-8))
        T_ret = returns.shape[0]

        oc = self._oracle_config
        # Use fixed max_len to minimize JIT recompilations (round up to 262144 = 2^18)
        max_len = max(262144, ((T_ret + 1023) // 1024) * 1024)

        logger.info(f"[POOL] Computing oracle for {label}: {T_ret:,} steps (max_len={max_len})")
        t0 = time.time()
        oracle_result = greedy_lookahead_path(
            jnp.array(returns),
            k=oc.get('oracle_k', 4),
            lookahead=oc.get('oracle_lookahead', 20),
            transaction_cost=oc.get('transaction_cost', 0.001),
            min_alpha_threshold=oc.get('min_alpha_threshold', 0.0005),
            max_len=max_len,
        )
        # Block until ready, then transfer to numpy (RAM)
        jax.block_until_ready(oracle_result.selections)
        oracle_sel = np.array(oracle_result.selections[:T_ret])
        oracle_cum_pnl = np.array(oracle_result.cumulative_pnl[:T_ret])
        logger.info(f"[POOL] Oracle computed in {time.time() - t0:.1f}s "
                    f"(trades={int(np.sum(np.diff(oracle_sel, axis=0) != 0) / 2)})")

        return oracle_sel, oracle_cum_pnl, returns.astype(np.float32)

    def _load_bella_shard(self, group_key, T_ret, horizons, manifest_path,
                          period_assets=None):
        """TASK_740: Load a Bella predictions shard for this group.

        Shard format (NPZ):
          preds     — (n_windows, n_horizons, n_assets) log-returns, float32
          horizons  — (n_horizons,) int32
          generated_at, bella_checkpoint_sha — metadata

        Returns an array of shape (T_ret, n_horizons, n_assets) aligned with
        decision_indices, or None if no shard is registered / loadable. The
        trainer masks samples whose rows contain NaN, so pad-right with NaN
        when n_windows < T_ret and truncate when n_windows > T_ret.
        """
        import json
        manifest_p = Path(manifest_path)
        if not manifest_p.exists():
            logger.warning(f"[POOL] Bella manifest missing at {manifest_p} — "
                           f"distillation loss will be fully masked out.")
            return None
        try:
            with open(manifest_p) as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"[POOL] Failed to read Bella manifest: {e}")
            return None

        # TASK_745: rotation-aware tag. Legacy '<start>__<end>' shards are
        # ignored intentionally — they were column-mismatched and must be
        # re-precomputed under the new (Option A) remap.
        import hashlib as _hashlib
        rot_hash = _hashlib.sha256(
            ",".join(sorted(period_assets or [])).encode()
        ).hexdigest()[:8]
        period_tag = f"{group_key[0]}__{group_key[1]}__{rot_hash}"
        entry = manifest.get('shards', {}).get(period_tag)
        if entry is None:
            return None

        shard_path = Path(entry['path'])
        if not shard_path.is_absolute():
            # Shard paths in the manifest are relative to the repo root where
            # precompute_bella_predictions.py was run.
            shard_path = Path.cwd() / shard_path
        if not shard_path.exists():
            logger.warning(f"[POOL] Bella shard missing: {shard_path}")
            return None

        try:
            data = np.load(shard_path)
            preds = data['preds']
            shard_horizons = tuple(int(h) for h in data['horizons'])
        except Exception as e:
            logger.warning(f"[POOL] Bella shard {shard_path} load failed: {e}")
            return None

        if shard_horizons != tuple(horizons):
            logger.warning(f"[POOL] Bella horizons mismatch in {shard_path}: "
                           f"shard={shard_horizons} cfg={horizons}")
            return None

        n_windows = preds.shape[0]
        n_horizons = len(horizons)
        out = np.full((T_ret, n_horizons, self.n_assets), np.nan, dtype=np.float32)
        take = min(n_windows, T_ret)
        out[:take] = preds[:take].astype(np.float32)
        return out

    def _load_bella_shape_shard(self, group_key, T_ret, descriptor_dim,
                                manifest_path, period_assets=None):
        """TASK_754: Load a Bella trajectory-shape descriptor shard for this group.

        Shard format (NPZ):
          descriptor      — (n_windows, descriptor_dim) float32
          schema_version  — int (must equal 1)
          generated_at, bella_checkpoint_sha, rotation_hash — metadata

        Returns an array of shape (T_ret, descriptor_dim) aligned with
        decision_indices, or None if no shard is registered / loadable. The
        trainer masks samples whose rows contain NaN (mirrors TASK_740 path).
        """
        import json
        manifest_p = Path(manifest_path)
        if not manifest_p.exists():
            logger.warning(f"[POOL] Bella shape manifest missing at {manifest_p} — "
                           f"shape-distill loss will be fully masked out.")
            return None
        try:
            with open(manifest_p) as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"[POOL] Failed to read Bella shape manifest: {e}")
            return None

        # TASK_745: rotation-aware tag — same hash construction as the legacy
        # bella_preds loader, so a single rotation gets the same tag in both
        # manifests.
        import hashlib as _hashlib
        rot_hash = _hashlib.sha256(
            ",".join(sorted(period_assets or [])).encode()
        ).hexdigest()[:8]
        period_tag = f"{group_key[0]}__{group_key[1]}__{rot_hash}"
        entry = manifest.get('shards', {}).get(period_tag)
        if entry is None:
            return None

        shard_path = Path(entry['path'])
        if not shard_path.is_absolute():
            shard_path = Path.cwd() / shard_path
        if not shard_path.exists():
            logger.warning(f"[POOL] Bella shape shard missing: {shard_path}")
            return None

        try:
            data = np.load(shard_path)
            desc = data['descriptor']
            shard_schema = int(data['schema_version']) if 'schema_version' in data.files else -1
        except Exception as e:
            logger.warning(f"[POOL] Bella shape shard {shard_path} load failed: {e}")
            return None

        if shard_schema != 1:
            logger.warning(f"[POOL] Bella shape shard {shard_path} schema_version="
                           f"{shard_schema}, expected 1 — refusing to use.")
            return None
        if desc.shape[-1] != int(descriptor_dim):
            logger.warning(f"[POOL] Bella shape shard {shard_path} descriptor_dim="
                           f"{desc.shape[-1]}, expected {descriptor_dim} — refusing to use.")
            return None

        n_windows = desc.shape[0]
        out = np.full((T_ret, int(descriptor_dim)), np.nan, dtype=np.float32)
        take = min(n_windows, T_ret)
        out[:take] = desc[:take].astype(np.float32)
        return out

    def get_episode_range(self, episode_length, window_size):
        """TASK_714: Get next contiguous episode range from round-robin slot.

        Returns:
            slot: slot dict (or None if all exhausted)
            cursor_start: starting timestep for this episode
            n_steps: number of valid training windows in this episode
        """
        if not self.slots:
            return None, 0, 0

        slot_idx = self._round_robin_idx % len(self.slots)
        self._round_robin_idx += 1
        slot = self.slots[slot_idx]

        cursor = slot['cursor']
        T_ret = len(slot['returns'])
        # Last valid decision index: T_ret - 1
        # Decision index for cursor c with window_size ws: c + ws - 1
        # For episode of ep_len steps starting at c: last decision at c + ep_len - 1 + ws - 1
        # Need: c + ep_len - 1 + ws - 1 < T_ret → c < T_ret - ep_len - ws + 2
        max_start = T_ret - episode_length - window_size + 2
        if max_start <= 0:
            max_start = 1  # Segment too short, use what we can

        # Clamp episode length if near end of segment
        avail = max(T_ret - cursor - window_size + 1, 0)
        n_steps = min(episode_length, avail)

        if n_steps <= 0:
            # Segment exhausted — rotate
            self._rotate_slot(slot_idx)
            # Retry with new slot
            slot_idx = (self._round_robin_idx - 1) % len(self.slots) if self.slots else 0
            if not self.slots:
                return None, 0, 0
            slot = self.slots[slot_idx]
            cursor = slot['cursor']
            avail = max(len(slot['returns']) - cursor - window_size + 1, 0)
            n_steps = min(episode_length, avail)
            if n_steps <= 0:
                return None, 0, 0

        cursor_start = cursor
        # Advance cursor by n_steps (stride=1 within episode, stride=episode_length between episodes)
        slot['cursor'] += n_steps

        return slot, cursor_start, n_steps

    def get_window(self):
        """Get next sequential window from round-robin slot.

        Returns:
            prices: (window_length, n_assets, 5) OHLCV, or None if all exhausted
            slot_label: str describing current slot (for logging)
        """
        if not self.slots:
            return None, "exhausted"

        # Round-robin slot selection
        slot_idx = self._round_robin_idx % len(self.slots)
        self._round_robin_idx += 1
        slot = self.slots[slot_idx]

        # Get sequential window at cursor position
        hm = slot['hm']
        start = slot['cursor']
        prices = hm.cache_data[start:start + self.window_length]

        # Advance cursor
        slot['cursor'] += self._stride

        # Check if segment exhausted
        if slot['cursor'] > slot['max_cursor']:
            logger.info(f"[POOL] Segment exhausted: {slot['label']} "
                        f"(walked {slot['cursor']}/{slot['max_cursor']} with stride {self._stride})")
            self._rotate_slot(slot_idx)

        # Cyclic-pad if fewer assets than needed
        if prices.shape[1] < self.n_assets:
            n_have = prices.shape[1]
            repeats = (self.n_assets + n_have - 1) // n_have
            prices = np.tile(prices, (1, repeats, 1))[:, :self.n_assets, :]

        return prices, slot['label']

    def _rotate_slot(self, slot_idx):
        """Replace exhausted slot with next segment. Cycles indefinitely.

        TASK_753: under Bella-augment, the replacement slot's source is
        chosen by a fresh coin flip via _make_slot_with_origin. This keeps
        the rolling Bella ratio stable across the run; without the fresh
        coin flip, once a Bella slot was made it would stay Bella forever.
        """
        old_slot = self.slots[slot_idx]
        group_key = old_slot['group_key']
        runs = self.groups[group_key]
        next_run = old_slot['run_idx'] + 1

        if next_run < len(runs):
            # Next run in group — different assets, must reload
            old_slot['hm'] = None
            self._group_run_idx[group_key] = next_run
            new_slot, is_bella = self._make_slot_with_origin(group_key)
            if new_slot is not None:
                self.slots[slot_idx] = new_slot
                if slot_idx < len(self._slot_origins):
                    self._slot_origins[slot_idx] = is_bella
                self._segments_loaded += 1
                return

        # Group fully exhausted — wrap around to run 0
        self._passes_completed[group_key] = self._passes_completed.get(group_key, 0) + 1
        passes = self._passes_completed[group_key]
        self._group_run_idx[group_key] = 0

        if len(runs) == 1:
            # Single-run group: just reset cursor, no disk I/O
            # TASK_753: under Bella-augment, a single-run group's reset can
            # still optionally re-coin-flip into a Bella slot. We preserve
            # the historical "cursor reset" path when augment is OFF;
            # otherwise, fresh coin flip.
            if self.bella_augment and self.bella_augment_ratio > 0.0:
                old_slot['hm'] = None
                new_slot, is_bella = self._make_slot_with_origin(group_key)
                if new_slot is not None:
                    self.slots[slot_idx] = new_slot
                    if slot_idx < len(self._slot_origins):
                        self._slot_origins[slot_idx] = is_bella
                    self._segments_loaded += 1
                    if passes % 50 == 0:
                        logger.info(f"[POOL] {new_slot['label']} pass {passes} — "
                                    f"single-run group resampled (Bella={is_bella})")
                    return
                # Fallback: keep the old slot, reset cursor.
            old_slot['cursor'] = 0
            if passes % 50 == 0:  # Log sparingly for fast-cycling small segments
                logger.info(f"[POOL] {old_slot['label']} pass {passes} — cursor reset")
            return

        # Multi-run group: load run 0 (different assets from current run)
        old_slot['hm'] = None
        new_slot, is_bella = self._make_slot_with_origin(group_key)
        if new_slot is not None:
            self.slots[slot_idx] = new_slot
            if slot_idx < len(self._slot_origins):
                self._slot_origins[slot_idx] = is_bella
            self._segments_loaded += 1
            logger.info(f"[POOL] Group {group_key[0][-10:]}→{group_key[1][-10:]} "
                        f"pass {passes}, cycling to run 1/{len(runs)} (Bella={is_bella})")
            return

        # Should never happen — group had at least 1 valid run on init
        logger.error(f"[POOL] Group {group_key} failed to reload on cycle — removing slot")
        self.slots.pop(slot_idx)
        if slot_idx < len(self._slot_origins):
            self._slot_origins.pop(slot_idx)
        if self._round_robin_idx > 0:
            self._round_robin_idx -= 1

    @property
    def is_exhausted(self):
        """Always False — pool cycles indefinitely."""
        return False

    def sample_positions(self, n_samples, window_size, jitter_interval=50):
        """TASK_715: Get n_samples (slot_idx, cursor) positions, round-robin across slots.

        Sequential cursor within each slot (stride=1). Temporal jitter every
        jitter_interval steps to prevent position memorization.
        """
        positions = []
        if not self.slots:
            return positions

        n_slots = len(self.slots)
        per_slot = n_samples // n_slots + 1

        for i in range(n_slots):
            slot_idx = (self._round_robin_idx + i) % n_slots
            slot = self.slots[slot_idx]
            T_ret = len(slot['returns'])

            for _ in range(per_slot):
                if len(positions) >= n_samples:
                    break
                cursor = slot['cursor']
                # Bounds check: decision_idx = cursor + ws - 1 must be < T_ret
                if cursor + window_size - 1 >= T_ret:
                    self._rotate_slot(slot_idx)
                    if slot_idx >= len(self.slots):
                        break
                    slot = self.slots[slot_idx]
                    T_ret = len(slot['returns'])
                    continue

                positions.append((slot_idx, cursor))
                slot['cursor'] += 1

                # Temporal jitter
                slot['_step_count'] = slot.get('_step_count', 0) + 1
                if slot['_step_count'] % jitter_interval == 0:
                    slot['cursor'] = min(slot['cursor'] + self.rng.integers(0, 6),
                                         T_ret - window_size)

        self._round_robin_idx += n_slots
        return positions[:n_samples]

    def status(self):
        """Return a status string for logging."""
        if not self.slots:
            return "EXHAUSTED"
        parts = []
        for i, s in enumerate(self.slots):
            pct = (s['cursor'] / max(s['max_cursor'], 1)) * 100
            parts.append(f"S{i}:{s['group_key'][0][-5:]} {pct:.0f}%")
        return " | ".join(parts)
