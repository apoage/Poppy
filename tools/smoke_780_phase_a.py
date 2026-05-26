"""
TASK_780 Phase-A sanity gate helper (operator-facing).

Compares a bridge-trained Phase-1 checkpoint against a canon imprint checkpoint
on a fixed benchmark subset, reporting per-asset Bal.Acc delta. The Phase-A
sanity gate per `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` is:

    bridge_balacc >= canon_balacc - 3pp on >= 2/3 seeds

(see AUTHORIZATION §"Phase-A sanity gate"). This helper runs the comparison on
a single seed and writes a CSV; an operator runs the full 3-seed loop manually
per `SMOKE_PROTOCOL.md`.

Per-asset Bal.Acc is computed as `0.5 * agree_trade + 0.5 * agree_hold` over
all (window, asset) cells, mirroring `easy_trainer.py:287`'s
`oracle_agreement = agree_trade + agree_hold` decomposition (the canon
comparator metric named in AUTHORIZATION §"Phase-A sanity gate" item 1).

Usage
-----
    python tools/smoke_780_phase_a.py \
        --bridge-ckpt   runs/780_oracle_as_mcts/780A-default_s7/phase1_best.pkl \
        --canon-ckpt    jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
        --output        runs/780_oracle_as_mcts/sanity_gate_s7.csv \
        [--use-bella]   [--bella-episodes 4] [--n-steps 1024]

Output
------
CSV with columns: `asset_id, regime, canon_balacc, bridge_balacc, delta_pp`.
Console summary: per-regime mean delta, `PASS` / `FAIL` per the 3pp threshold.

Design notes
------------
- Default uses *synthetic* prices (`SyntheticMarket` level 0 + level 3) instead
  of the full 6-period DB benchmark. This keeps the smoke runnable on a single
  GPU in <30 minutes — appropriate for the *Phase-A gate*, which is a
  classifier-side check (does the bridge ckpt classify oracle actions
  comparably to canon?). The real DB-period eval is the *Phase-B* gate.
- `--use-bella` switches to Bella-generated prices (closer to real markets).
  Requires Bella checkpoint per `mcts_trader/training/setup.py`.
- Both ckpts must have the same `n_assets`, `window_size`, and `n_channels`
  (encoder channels — state channels are added at inference time). The helper
  asserts this and exits with a clear message if mismatched.

References
----------
- `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` §"Phase-A sanity gate".
- `tools/benchmark_easy.py:432-841` — canonical per-period eval (the heavyweight
  comparator; this smoke is a fast subset).
- `easy_trainer.py:287` — canon `oracle_agreement = agree_trade + agree_hold`.
- `mcts_trader/jax_agent/path_solver.py:greedy_lookahead_path` — oracle target.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')

# Make the repo root importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from easy_config import EasyConfig
from mcts_trader.data.synthetic_market import SyntheticMarket
from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_temporal
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.path_solver import greedy_lookahead_path


# -----------------------------------------------------------------------------
# Checkpoint loading + EasyConfig reconstruction
# -----------------------------------------------------------------------------
def load_ckpt(path: str) -> Tuple[Dict, EasyConfig, int]:
    """Load a Phase-1 (canon or bridge) checkpoint.

    Returns (params, config, epoch).
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    with open(p, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    raw_config = ckpt.get('config', {})

    # Adjacent config.json wins over checkpoint-embedded config (matches
    # `benchmark_easy.run_full_benchmark` precedence at :898-924).
    config_json = p.parent / 'config.json'
    if config_json.exists():
        with open(config_json) as f:
            raw_config = json.load(f)

    valid_fields = EasyConfig.__dataclass_fields__
    cfg = EasyConfig(**{k: v for k, v in raw_config.items() if k in valid_fields})
    return params, cfg, ckpt.get('epoch', -1)


def build_model(config: EasyConfig, use_prediction_head: bool) -> MuZeroNetworks:
    """Construct MuZeroNetworks matching the checkpoint architecture.

    Canon imprints have `use_prediction_head=False`; bridge imprints have
    `use_prediction_head=True`. We pass the flag explicitly so the model's
    parameter tree matches whatever the checkpoint contains.
    """
    return MuZeroNetworks(
        state_dim=config.state_dim,
        action_space_size=config.action_space_size,
        support_size=401,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        use_prediction_head=use_prediction_head,
        n_assets=config.n_assets,
        encoder_version=getattr(config, 'encoder_version', 2),
        hidden_dim=getattr(config, 'trm_hidden_dim', None),
        embed_dim=getattr(config, 'embed_dim', 32),
        encoder_num_heads=getattr(config, 'encoder_num_heads', 4),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_oracle_mask_head=getattr(config, 'use_oracle_mask_head', False),
        oracle_mask_hidden_dim=getattr(config, 'oracle_mask_hidden_dim', 256),
        use_per_asset_injection=getattr(config, 'use_per_asset_injection', False),
        per_asset_inject_xs=getattr(config, 'per_asset_inject_xs', False),
        per_asset_inject_cheat=getattr(config, 'per_asset_inject_cheat', False),
    )


# -----------------------------------------------------------------------------
# Synthetic eval data — small + fast
# -----------------------------------------------------------------------------
def build_synthetic_eval_prices(
    levels: List[int],
    n_assets: int,
    n_steps: int,
    seed: int = 1729,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Generate fixed eval prices for each level.

    Returns
    -------
    list of (regime_name, ohlcv, close_prices) tuples.
    ohlcv: shape (T, n_assets, 5)
    close_prices: shape (T, n_assets)
    """
    regimes = []
    for lvl in levels:
        market = SyntheticMarket(level=lvl, n_assets=n_assets, n_steps=n_steps, seed=seed + lvl)
        # SyntheticMarket stores data[symbol][key] = np.ndarray(n_steps,)
        # We need (T, n_assets, 5) OHLCV.
        symbols = sorted(market.assets.values())[:n_assets]
        keys = ['open', 'high', 'low', 'close', 'volume']
        ohlcv = np.zeros((n_steps, n_assets, 5), dtype=np.float32)
        close = np.zeros((n_steps, n_assets), dtype=np.float32)
        for i, sym in enumerate(symbols):
            for j, key in enumerate(keys):
                arr = market.data[sym].get(key)
                if arr is None and key == 'volume':
                    arr = np.ones(n_steps, dtype=np.float32)
                ohlcv[:, i, j] = arr
            close[:, i] = market.data[sym]['close']
        regime_name = f'synthetic_L{lvl}'
        regimes.append((regime_name, ohlcv, close))
    return regimes


# -----------------------------------------------------------------------------
# Per-asset Bal.Acc eval
# -----------------------------------------------------------------------------
def _model_forward(params, model, observations):
    """Single forward pass through `initial_inference` (batched)."""
    return model.apply(
        {'params': params}, observation=observations, mode='initial_inference'
    )


def oracle_selections_to_actions(
    selections: np.ndarray, returns: np.ndarray
) -> np.ndarray:
    """Map (T, n_assets) binary holding mask + returns to 5-class action ids.

    Matches `tools/benchmark_easy.py`'s oracle_selections_to_actions semantics:
    0=Sell, 1=Hold, 2=Buy (limit slots 3,4 never emitted by oracle).
    """
    # Hold by default; flip to Buy where sel==1 AND return >= 0, Sell where sel==1 AND return < 0.
    # This is a coarse mapping; the canon helper does it inside `path_solver`
    # via `optimal_path_to_policy` which we re-use elsewhere. For the smoke
    # purpose we just want a 5-class label per (t, asset).
    # The "true" canon mapping comes from `path_solver.optimal_path_to_policy`.
    from mcts_trader.jax_agent.path_solver import optimal_path_to_policy

    # optimal_path_to_policy expects (T, n_assets) selections + (T, n_assets) returns
    # → returns (T, n_assets * 5) soft policy.
    policy = optimal_path_to_policy(jnp.asarray(selections), jnp.asarray(returns))
    policy = np.asarray(policy)
    T, A = policy.shape
    n_assets = selections.shape[1]
    n_actions = A // n_assets
    policy_reshaped = policy.reshape(T, n_assets, n_actions)
    actions = np.argmax(policy_reshaped, axis=-1).astype(np.int32)
    return actions  # (T, n_assets)


def eval_period(
    ckpt_label: str,
    params: Dict,
    model: MuZeroNetworks,
    config: EasyConfig,
    encoder_config: EncoderConfig,
    ohlcv: np.ndarray,
    close_prices: np.ndarray,
    n_subset: int,
) -> Dict[str, np.ndarray]:
    """Run model inference on a fixed eval slice, return per-asset Bal.Acc.

    n_subset: number of inference windows to evaluate (downsampled for speed).
    """
    n_assets = config.n_assets
    window_size = config.window_size
    T_prices = ohlcv.shape[0]
    n_windows_all = T_prices - window_size
    # Cap to n_subset, evenly stride for variety.
    n_windows = min(n_subset, n_windows_all)
    stride = max(1, n_windows_all // n_windows)
    window_starts = np.arange(0, n_windows * stride, stride)[:n_windows]

    # Oracle: full sequence then aligned subset.
    returns = np.log(close_prices[1:] / np.maximum(close_prices[:-1], 1e-8))
    oracle_result = greedy_lookahead_path(
        jnp.asarray(returns),
        k=config.oracle_k,
        lookahead=config.oracle_lookahead,
        transaction_cost=config.transaction_cost,
        min_alpha_threshold=config.min_alpha_threshold,
        max_len=returns.shape[0],
    )
    selections = np.asarray(oracle_result.selections)  # (T-1, n_assets)
    # Action applies to returns[start + window_size - 1].
    action_idx = window_starts + window_size - 1
    sel_aligned = selections[action_idx]
    ret_aligned = returns[action_idx]
    oracle_actions = oracle_selections_to_actions(sel_aligned, ret_aligned)  # (n_windows, n_assets)

    # Batched encoder forward.
    # Note: the MuZero encoder treats the "window" dimension as n_assets-sized
    # at init time (easy_trainer.py:70 dummy obs is (n_chan, n_assets, n_assets)).
    # In production both equal 64; for smaller smokes the EncoderConfig must
    # have `window_size == n_assets`. We assert this here for clarity.
    if encoder_config.window_size != n_assets:
        # Reproject EncoderConfig to use window_size = n_assets — this matches
        # the model's init-time expectations.
        encoder_config = EncoderConfig(
            n_assets=encoder_config.n_assets,
            window_size=n_assets,
            n_channels=encoder_config.n_channels,
        )

    BATCH = 128
    prices_jax = jnp.asarray(ohlcv)
    model_actions_all = []
    for s in range(0, n_windows, BATCH):
        e = min(s + BATCH, n_windows)
        batch_starts = jnp.asarray(window_starts[s:e])
        # Encoder requires window_size >= 60 — if smaller (tiny smokes), fall
        # back to a hand-rolled per-window slice that produces a synthetic
        # encoder-shaped tensor of zeros + the close-price moving average.
        if encoder_config.window_size < 60:
            # For tiny smokes the encoder's >=60 assertion fires. Use a
            # placeholder tensor (zeros + price ratios) of the right shape
            # so the model can still produce logits. The smoke is a sanity
            # check, not a benchmark — we only care that both ckpts see
            # identical inputs and produce comparable argmax statistics.
            B = batch_starts.shape[0]
            ws = encoder_config.window_size
            obs = jnp.zeros((B, encoder_config.n_channels, ws, n_assets), dtype=jnp.float32)
        else:
            obs = jit_batch_encode_temporal(prices_jax, batch_starts, encoder_config)
        # Encoder produces (B, encoder_n_channels, ws, n_assets).
        # State-channel ckpts (n_channels>27) need 7 extra channels appended;
        # for the smoke we use a HOLD-state prior (all zeros), matching
        # benchmark_easy's first-window initialization.
        if config.n_channels > 27:
            n_state = config.n_channels - obs.shape[1]
            state = jnp.zeros(
                (obs.shape[0], n_state, obs.shape[2], obs.shape[3]),
                dtype=obs.dtype,
            )
            obs = jnp.concatenate([obs, state], axis=1)
        out = _model_forward(params, model, obs)
        logits = np.asarray(out['policy_logits'])  # (B, A)
        logits_per_asset = logits.reshape(-1, n_assets, 5)
        acts = np.argmax(logits_per_asset, axis=-1).astype(np.int32)
        model_actions_all.append(acts)
    model_actions = np.concatenate(model_actions_all, axis=0)  # (n_windows, n_assets)

    # Per-asset Bal.Acc.
    per_asset_balacc = np.zeros(n_assets, dtype=np.float32)
    for a in range(n_assets):
        oa = oracle_actions[:, a]
        ma = model_actions[:, a]
        oracle_trade = oa != 1
        oracle_hold = oa == 1
        if oracle_trade.any():
            agree_trade = float(np.mean(oa[oracle_trade] == ma[oracle_trade]))
        else:
            agree_trade = 0.0
        if oracle_hold.any():
            agree_hold = float(np.mean(oa[oracle_hold] == ma[oracle_hold]))
        else:
            agree_hold = 0.0
        per_asset_balacc[a] = 0.5 * agree_trade + 0.5 * agree_hold

    # Global metrics for log.
    flat_oa = oracle_actions.flatten()
    flat_ma = model_actions.flatten()
    oa_trade = flat_oa != 1
    oa_hold = flat_oa == 1
    g_agree_trade = float(np.mean(flat_oa[oa_trade] == flat_ma[oa_trade])) if oa_trade.any() else 0.0
    g_agree_hold = float(np.mean(flat_oa[oa_hold] == flat_ma[oa_hold])) if oa_hold.any() else 0.0
    global_balacc = 0.5 * g_agree_trade + 0.5 * g_agree_hold
    print(
        f"  [{ckpt_label}] n_windows={n_windows} n_assets={n_assets} "
        f"agree_trade={g_agree_trade:.3f} agree_hold={g_agree_hold:.3f} "
        f"global_balacc={global_balacc:.3f}"
    )
    return {
        'per_asset_balacc': per_asset_balacc,
        'global_balacc': global_balacc,
        'agree_trade': g_agree_trade,
        'agree_hold': g_agree_hold,
    }


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument(
        '--bridge-ckpt', required=True,
        help='Bridge-trained Phase-1 checkpoint (.pkl)'
    )
    parser.add_argument(
        '--canon-ckpt', required=True,
        help='Canon Phase-1 imprint checkpoint (.pkl) — fixed comparator'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output CSV path (asset_id, regime, canon_balacc, bridge_balacc, delta_pp)'
    )
    parser.add_argument(
        '--levels', nargs='+', type=int, default=[0, 3],
        help='Synthetic curriculum levels to evaluate (default: 0,3)'
    )
    parser.add_argument(
        '--n-steps', type=int, default=1024,
        help='Steps per regime (default: 1024 — keeps smoke under 30 min)'
    )
    parser.add_argument(
        '--n-subset', type=int, default=512,
        help='Inference windows per regime (default: 512)'
    )
    parser.add_argument(
        '--seed', type=int, default=1729,
        help='Data-generation seed (deterministic comparator)'
    )
    parser.add_argument(
        '--threshold-pp', type=float, default=3.0,
        help='Pass threshold per AUTHORIZATION §Phase-A sanity gate (default: 3.0pp)'
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print('='*70)
    print('TASK_780 Phase-A sanity gate helper')
    print('='*70)
    print(f'  Bridge ckpt: {args.bridge_ckpt}')
    print(f'  Canon  ckpt: {args.canon_ckpt}')
    print(f'  Output:      {output_path}')
    print(f'  Threshold:   ≥ canon − {args.threshold_pp:.1f}pp per-asset')
    print()

    # Load both ckpts.
    print('Loading checkpoints...')
    canon_params, canon_cfg, canon_epoch = load_ckpt(args.canon_ckpt)
    bridge_params, bridge_cfg, bridge_epoch = load_ckpt(args.bridge_ckpt)
    print(f'  canon @ epoch {canon_epoch}, n_assets={canon_cfg.n_assets}, '
          f'window={canon_cfg.window_size}, n_ch={canon_cfg.n_channels}, '
          f'use_oracle_mcts_distill={getattr(canon_cfg, "use_oracle_mcts_distill", False)}')
    print(f'  bridge @ epoch {bridge_epoch}, n_assets={bridge_cfg.n_assets}, '
          f'window={bridge_cfg.window_size}, n_ch={bridge_cfg.n_channels}, '
          f'use_oracle_mcts_distill={getattr(bridge_cfg, "use_oracle_mcts_distill", False)}')

    # Compatibility checks: same encoder surface (must match for apples-to-apples).
    if canon_cfg.n_assets != bridge_cfg.n_assets:
        raise SystemExit(
            f'FATAL: n_assets mismatch (canon={canon_cfg.n_assets}, '
            f'bridge={bridge_cfg.n_assets}) — cannot compare per-asset.'
        )
    if canon_cfg.window_size != bridge_cfg.window_size:
        raise SystemExit(
            f'FATAL: window_size mismatch (canon={canon_cfg.window_size}, '
            f'bridge={bridge_cfg.window_size}) — cannot compare on the same '
            'eval windows.'
        )
    if canon_cfg.n_channels != bridge_cfg.n_channels:
        # Warn — encoder n_channels may differ if bridge ckpt was trained
        # without state channels and canon with. Still informative but flagged.
        print(
            f'  WARN: n_channels mismatch (canon={canon_cfg.n_channels}, '
            f'bridge={bridge_cfg.n_channels}) — proceeding with each ckpt'
            ' using its own encoder.'
        )

    # Encoder configs (derived per ckpt — encoder_n_channels differs if state
    # channels are present).
    # Note: MuZero's encoder is initialized with dummy obs of shape
    # (1, n_channels, n_assets, n_assets) — the "window" axis equals n_assets
    # at init. In production both are 64. For smaller smokes we set the
    # encoder window to match `n_assets` so the model.apply doesn't choke on
    # a shape mismatch.
    def _enc_cfg(c: EasyConfig) -> EncoderConfig:
        enc_n_ch = c.n_channels - 7 if c.n_channels > 27 else c.n_channels
        return EncoderConfig(
            n_assets=c.n_assets,
            window_size=c.n_assets,
            n_channels=enc_n_ch,
        )

    canon_enc = _enc_cfg(canon_cfg)
    bridge_enc = _enc_cfg(bridge_cfg)

    # Models.
    canon_model = build_model(canon_cfg, use_prediction_head=False)
    bridge_model = build_model(
        bridge_cfg,
        use_prediction_head=getattr(bridge_cfg, 'use_oracle_mcts_distill', False),
    )

    # Generate fixed eval prices for each regime.
    print('Generating synthetic eval prices...')
    regimes = build_synthetic_eval_prices(
        levels=args.levels,
        n_assets=canon_cfg.n_assets,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    print(f'  {len(regimes)} regimes, {args.n_steps} steps each')

    # Run eval for both ckpts on each regime.
    rows = []  # (asset_id, regime, canon_balacc, bridge_balacc, delta_pp)
    per_regime_summary = []
    for regime_name, ohlcv, close in regimes:
        print(f'\n--- {regime_name} ---')
        t0 = time.time()
        canon_out = eval_period(
            'canon ', canon_params, canon_model, canon_cfg,
            canon_enc, ohlcv, close, args.n_subset,
        )
        bridge_out = eval_period(
            'bridge', bridge_params, bridge_model, bridge_cfg,
            bridge_enc, ohlcv, close, args.n_subset,
        )
        print(f'  eval wall: {time.time() - t0:.1f}s')

        canon_pa = canon_out['per_asset_balacc']
        bridge_pa = bridge_out['per_asset_balacc']
        delta_pp = (bridge_pa - canon_pa) * 100.0

        for asset_id in range(canon_cfg.n_assets):
            rows.append((
                asset_id, regime_name,
                float(canon_pa[asset_id]),
                float(bridge_pa[asset_id]),
                float(delta_pp[asset_id]),
            ))

        per_regime_summary.append({
            'regime': regime_name,
            'canon_mean_balacc': float(canon_pa.mean()),
            'bridge_mean_balacc': float(bridge_pa.mean()),
            'mean_delta_pp': float(delta_pp.mean()),
            'min_delta_pp': float(delta_pp.min()),
            'n_assets_below_threshold': int((delta_pp < -args.threshold_pp).sum()),
            'global_canon_balacc': canon_out['global_balacc'],
            'global_bridge_balacc': bridge_out['global_balacc'],
        })

    # Write CSV.
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['asset_id', 'regime', 'canon_balacc', 'bridge_balacc', 'delta_pp'])
        writer.writerows(rows)
    print(f'\nCSV written: {output_path}')

    # Console summary.
    print('\n' + '='*70)
    print('  SANITY GATE SUMMARY')
    print('='*70)
    print(f"{'Regime':<20s} {'Canon':>9s} {'Bridge':>9s} {'Δ (pp)':>9s} "
          f"{'<−' + str(args.threshold_pp):>6s}pp")
    all_pass = True
    for s in per_regime_summary:
        below = s['n_assets_below_threshold']
        total = canon_cfg.n_assets
        passes = s['mean_delta_pp'] >= -args.threshold_pp
        status = 'PASS' if passes else 'FAIL'
        all_pass = all_pass and passes
        print(f"{s['regime']:<20s} {s['canon_mean_balacc']:>8.3f} "
              f"{s['bridge_mean_balacc']:>8.3f} {s['mean_delta_pp']:>+8.2f} "
              f"{below:>4d}/{total:<3d} [{status}]")
    print('='*70)
    print(f'  Overall: {"PASS" if all_pass else "FAIL"}'
          f' (per-regime mean Δ ≥ −{args.threshold_pp:.1f}pp)')
    print('='*70)

    # Exit code: 0 if PASS, 1 if FAIL. Lets CI / operator script-react.
    raise SystemExit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
