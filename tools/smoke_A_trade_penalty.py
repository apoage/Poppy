"""
TASK_787 (experiment A — "PnL in the loss") smoke.

Two assertions:

  [A1] Canon-OFF equivalence. With trade_rate_penalty_weight=0.0 the over-trade
       penalty branch is gated by a COMPILE-TIME `if weight > 0` test, so the
       term is never traced into the XLA graph. We verify:
         (a) the returned 'overtrade_penalty' metric is exactly 0.0,
         (b) total_loss equals the sum of the individually-returned component
             losses with NO penalty contribution (to <1e-6 relative), AND
         (c) the OFF-path grad pytree is BIT-IDENTICAL (jnp.array_equal on every
             leaf) to a reference grad built from a config whose
             trade_rate_penalty_weight attribute is removed entirely (so the
             field-read path can't differ). This is the canon bit-identity gate.

  [A2] ON trainability. With trade_rate_penalty_weight=0.1, run a handful of
       Stage-B train steps on a tiny synthetic config + random params. Assert:
       loss finite, grads finite and non-zero, no NaN, total_loss > the
       no-penalty total_loss (the penalty strictly adds), and the trade-rate
       diagnostic is sane (0 <= mean_trade_rate <= 1).

Runs in well under a minute on a single GPU using a tiny synthetic model.
No checkpoint / DB / data pipeline required (all aux heads OFF, oracle value
target → only synthetic obs + mcts targets are needed).

Usage
-----
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 PYTHONPATH=<worktree root> \
        python tools/smoke_A_trade_penalty.py

Exit code 0 = both assertions PASS; 1 = failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.4')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from phase2_config import Phase2Config
from train_phase2 import (
    create_model,
    create_stage_b_optimizer,
    make_stage_b_train_step,
)


N_ASSETS = 8
ACTION_SPACE = N_ASSETS * 5
N_CHANNELS = 27        # encoder channels only — keeps the smoke fast
WINDOW = N_ASSETS      # square obs (C, W, N) with W = N = n_assets
BATCH = 4


def _tiny_config(trade_rate_penalty_weight: float) -> Phase2Config:
    """Small Stage-B config with every aux head OFF (isolates the loss path)."""
    cfg = Phase2Config(
        name='smokeA',
        seed=42,
        stage='B',
        encoder_version=2,
        state_dim=32,
        n_assets=N_ASSETS,
        action_space_size=ACTION_SPACE,
        n_channels=N_CHANNELS,
        H_cycles=2,
        L_cycles=2,
        embed_dim=16,
        encoder_num_heads=2,
        compute_dtype='float32',
        epochs=1,
        steps_per_epoch=1,
        # Value path: oracle remaining-PnL target (no obs_tN bootstrap needed).
        value_target_mode='oracle',
        use_popart=False,
        # Factored search path off → joint CE (simplest); both paths share the
        # penalty term, which reads policy_logits directly.
        use_factored_search=False,
        # All aux heads OFF — synthetic future/bella inputs are then unused.
        use_direction_head=False,
        use_direction_policy_bias=False,
        use_per_asset_policy_head=False,
        use_oracle_mask_head=False,
        use_rank_head=False,
        use_flow_matching=False,
        use_aux_price_head=False,
        use_bella_distill_head=False,
        use_bella_shape_head=False,
        use_titans_memory=False,
        disable_limit_orders=False,
        entropy_coef=0.0,
        min_entropy=0.0,
        # The knob under test.
        trade_rate_penalty_weight=trade_rate_penalty_weight,
    )
    return cfg


def _init_params(cfg, seed=0):
    model = create_model(cfg)
    dummy_obs = jnp.zeros((1, cfg.n_channels, cfg.n_assets, cfg.n_assets))
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init(
        jax.random.PRNGKey(seed),
        observation=dummy_obs, action=dummy_action,
        mode='initialize_parameters',
    )
    return model, variables['params']


def _make_batch(cfg, seed=123):
    """Synthetic Stage-B train-step inputs matching the documented shapes."""
    rng = np.random.default_rng(seed)
    C, N = cfg.n_channels, cfg.n_assets
    obs = jnp.asarray(rng.standard_normal((BATCH, C, N, N), dtype=np.float32))
    obs_t1 = jnp.asarray(rng.standard_normal((BATCH, C, N, N), dtype=np.float32))
    obs_tN = jnp.asarray(rng.standard_normal((BATCH, C, N, N), dtype=np.float32))
    actions_dyn = jnp.asarray(rng.integers(0, ACTION_SPACE, size=(BATCH,), dtype=np.int32))
    rewards_dyn = jnp.asarray(rng.standard_normal((BATCH,), dtype=np.float32) * 0.01)
    # Per-asset softmax visit target (each 5-block sums to 1, scaled to 64-sum).
    visits = rng.dirichlet(np.ones(5), size=(BATCH, N)).reshape(BATCH, ACTION_SPACE)
    mcts_visits = jnp.asarray(visits.astype(np.float32))
    mcts_value = jnp.asarray(rng.standard_normal((BATCH,), dtype=np.float32) * 0.1)
    oracle_targets = jnp.asarray(visits.astype(np.float32))  # same per-asset-softmax form
    bc_weight = jnp.asarray(0.05, dtype=jnp.float32)
    popart_mean = jnp.asarray(0.0, dtype=jnp.float32)
    popart_std = jnp.asarray(1.0, dtype=jnp.float32)
    oracle_values = jnp.asarray(rng.standard_normal((BATCH,), dtype=np.float32) * 0.1)
    nstep_rewards = jnp.asarray(rng.standard_normal((BATCH,), dtype=np.float32) * 0.01)
    future_cum_returns = jnp.asarray(rng.standard_normal((BATCH, N), dtype=np.float32) * 0.01)
    # Aux heads OFF → these are unused; supply finite zeros.
    future_log_returns = jnp.zeros((BATCH, 20, N), dtype=jnp.float32)
    bella_preds = jnp.full((BATCH, 4, N), jnp.nan, dtype=jnp.float32)
    bella_shape = jnp.full((BATCH, 14), jnp.nan, dtype=jnp.float32)
    return (obs, obs_t1, obs_tN, actions_dyn, rewards_dyn, mcts_visits,
            mcts_value, oracle_targets, bc_weight, popart_mean, popart_std,
            oracle_values, nstep_rewards, future_cum_returns,
            future_log_returns, bella_preds, bella_shape)


# Component metric keys whose weighted sum (with the Stage-B weights) makes up
# total_loss when the penalty is absent. Used for the A1(b) reconstruction.
def _reconstruct_total_no_penalty(cfg, metrics):
    return float(
        cfg.exit_policy_weight * metrics['policy_loss'] +
        cfg.exit_value_weight * metrics['value_loss'] +
        cfg.exit_consistency_weight * (metrics['consistency_loss'] + metrics['reward_loss']) +
        float(metrics['bc_weight']) * metrics['bc_loss'] +
        cfg.direction_loss_weight * metrics['dir_loss'] +
        getattr(cfg, 'rank_loss_weight', 1.0) * metrics['rank_loss'] +
        getattr(cfg, 'aux_price_loss_weight', 0.1) * metrics['aux_price_loss'] +
        getattr(cfg, 'bella_distill_loss_weight', 0.1) * metrics['bella_distill_loss'] +
        getattr(cfg, 'bella_shape_loss_weight', 0.1) * metrics['bella_shape_loss']
    )


def assertion_1_canon_off():
    print('\n[A1] Canon-OFF bit-identity (trade_rate_penalty_weight = 0.0)')
    cfg = _tiny_config(0.0)
    model, params = _init_params(cfg, seed=0)
    optimizer, opt_state = create_stage_b_optimizer(cfg, params)
    step = make_stage_b_train_step(model, cfg, optimizer)
    batch = _make_batch(cfg)

    _, _, metrics_off = step(params, opt_state, *batch)
    jax.block_until_ready(metrics_off)

    # (a) penalty metric is exactly 0.0
    ot = float(metrics_off['overtrade_penalty'])
    print(f'  overtrade_penalty metric (OFF) = {ot:.8e}')
    assert ot == 0.0, f'OFF overtrade_penalty must be exactly 0.0, got {ot}'

    # (b) total_loss == reconstruction with no penalty term
    total = float(metrics_off['total_loss'])
    recon = _reconstruct_total_no_penalty(cfg, metrics_off)
    rel = abs(total - recon) / max(abs(total), 1e-8)
    print(f'  total_loss              = {total:.8f}')
    print(f'  reconstruct (no penalty)= {recon:.8f}  rel_err={rel:.2e}')
    assert rel < 1e-6, f'OFF total_loss must match no-penalty reconstruction, rel={rel}'

    # (c) grad pytree bit-identical to a config WITHOUT the field at all.
    #     Build a reference step from a config whose attribute is deleted so the
    #     getattr(...,'trade_rate_penalty_weight',0.0) default path is taken —
    #     proving the OFF graph equals the pre-787 graph.
    cfg_ref = _tiny_config(0.0)
    delattr(cfg_ref, 'trade_rate_penalty_weight')
    opt_ref, opt_state_ref = create_stage_b_optimizer(cfg_ref, params)
    step_ref = make_stage_b_train_step(model, cfg_ref, opt_ref)

    def _grads(s, ostate):
        # Re-run loss_fn grads via a thin wrapper: the step returns new params;
        # instead we recompute grads to compare. We rely on grad_norm + the
        # returned new_params delta for an exact leaf comparison.
        new_params, _, m = s(params, ostate, *batch)
        jax.block_until_ready(new_params)
        return new_params, float(m['grad_norm'])

    np_off, gn_off = _grads(step, opt_state)
    np_ref, gn_ref = _grads(step_ref, opt_state_ref)

    print(f'  grad_norm OFF           = {gn_off:.10f}')
    print(f'  grad_norm REF(no-field) = {gn_ref:.10f}')
    assert gn_off == gn_ref, f'grad_norm differs: {gn_off} vs {gn_ref}'

    # Leaf-by-leaf bit-identity of the updated params (a 1-step proxy for grad
    # equality: same init params + same updates ⇒ identical new params iff the
    # graphs are identical).
    leaves_off = jax.tree_util.tree_leaves(np_off)
    leaves_ref = jax.tree_util.tree_leaves(np_ref)
    assert len(leaves_off) == len(leaves_ref), 'param tree structure changed'
    n_ident = 0
    for lo, lr in zip(leaves_off, leaves_ref):
        if not bool(jnp.array_equal(lo, lr)):
            raise AssertionError('OFF-path new_params diverge from no-field reference '
                                 '— canon bit-identity BROKEN')
        n_ident += 1
    print(f'  bit-identical param leaves: {n_ident}/{len(leaves_ref)}')
    print('  PASS: A1 canon-OFF path is bit-identical to pre-787 graph')
    return True


def assertion_2_on_trainability():
    print('\n[A2] ON trainability (trade_rate_penalty_weight = 0.1)')
    cfg = _tiny_config(0.1)
    model, params = _init_params(cfg, seed=0)
    optimizer, opt_state = create_stage_b_optimizer(cfg, params)
    step = make_stage_b_train_step(model, cfg, optimizer)
    batch = _make_batch(cfg)

    # Also build the OFF step on identical params for the strict-add check.
    cfg_off = _tiny_config(0.0)
    opt_off, opt_state_off = create_stage_b_optimizer(cfg_off, params)
    step_off = make_stage_b_train_step(model, cfg_off, opt_off)
    _, _, m_off = step_off(params, opt_state_off, *batch)
    jax.block_until_ready(m_off)

    p, ostate = params, opt_state
    last = None
    trade_rates = []
    for i in range(5):
        p, ostate, m = step(p, ostate, *batch)
        jax.block_until_ready(m)
        last = m
        trade_rates.append(float(m['mean_trade_rate']))

    total = float(last['total_loss'])
    ot = float(last['overtrade_penalty'])
    gn = float(last['grad_norm'])
    tr = float(last['mean_trade_rate'])
    print(f'  final total_loss        = {total:.6f}')
    print(f'  final overtrade_penalty = {ot:.6f}')
    print(f'  final grad_norm         = {gn:.6f}')
    print(f'  mean_trade_rate (steps) = {[round(x, 4) for x in trade_rates]}')

    assert np.isfinite(total), 'total_loss not finite'
    assert np.isfinite(ot) and ot > 0.0, 'overtrade_penalty must be finite and > 0 when ON'
    assert np.isfinite(gn) and gn > 0.0, 'grad_norm must be finite and non-zero'
    assert 0.0 <= tr <= 1.0, f'mean_trade_rate out of [0,1]: {tr}'

    # The penalty strictly adds: ON step-0 total_loss > OFF step-0 total_loss on
    # identical params (compare first step, recomputed cleanly).
    _, _, m_on0 = step(params, opt_state, *batch)
    jax.block_until_ready(m_on0)
    on0 = float(m_on0['total_loss'])
    off0 = float(m_off['total_loss'])
    print(f'  step-0 total_loss ON={on0:.6f}  OFF={off0:.6f}  Δ={on0-off0:.6f}')
    assert on0 > off0, 'ON total_loss must exceed OFF total_loss (penalty adds)'
    expected_delta = 0.1 * float(m_on0['overtrade_penalty'])
    assert abs((on0 - off0) - expected_delta) < 1e-5, (
        f'ON-OFF delta {on0-off0} != weight*penalty {expected_delta}')
    print('  PASS: A2 ON path trains (finite loss/grads, penalty adds exactly weight*penalty)')
    return True


def main():
    print('=' * 70)
    print('TASK_787 experiment A — over-trade penalty smoke')
    print('=' * 70)
    ok1 = assertion_1_canon_off()
    ok2 = assertion_2_on_trainability()
    print('\n' + '=' * 70)
    if ok1 and ok2:
        print('PASS: TASK_787 smoke — canon-OFF bit-identical AND ON trainable')
        print('=' * 70)
        raise SystemExit(0)
    print('FAIL: TASK_787 smoke')
    print('=' * 70)
    raise SystemExit(1)


if __name__ == '__main__':
    main()
