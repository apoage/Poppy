# TASK_713: Trade Quality Feedback

**Status**: DONE
**Priority**: 1 (benchmark trade agreement 12-33% vs training 75% — model can't pick the right assets)
**Depends on**: TASK_712 (all regimes active)

---

## Problem

The model learns to imitate oracle trades at 75% accuracy in-distribution, but drops to 12-33% on held-out benchmark periods. Pure imitation gives no feedback on trade QUALITY — only whether the trade matches the oracle. The model doesn't learn that trading asset 7 (which earns +2%) is better than trading asset 12 (which earns -0.5%).

Cross-asset signals are subtle but powerful (relative strength, correlation breaks, momentum divergence). The model sees all 64 assets in the observation but doesn't get enough signal to distinguish high-value from low-value trade opportunities.

## Solution: Return-Weighted Policy Loss

Weight the policy cross-entropy by the magnitude of the oracle action's return. High-value trades (oracle picked asset 7, it earned +2%) get more gradient than low-value trades (asset 3, earned +0.1%). Holds stay at weight 1.0.

### Mechanism

In `easy_trainer.py` loss function, after computing per-asset cross-entropy:

```python
# Current: equal-weight trades via adaptive balancing
per_asset_ce = optax.softmax_cross_entropy(policy_reshaped, targets_reshaped)

# NEW: Return-weighted trades — focus learning on high-value trade signals
# oracle_returns: (batch, 64) — per-asset return at this timestep
# trade_mask: (batch, 64) — 1.0 where oracle trades, 0.0 where holds
abs_returns = jnp.abs(oracle_returns)  # magnitude of return
# Normalize: top-k returns get weight ~k, holds get weight 1.0
return_weight = 1.0 + trade_mask * abs_returns / jnp.maximum(jnp.mean(abs_returns), 1e-6)
per_asset_ce = per_asset_ce * return_weight
```

This is additive to TASK_149's adaptive trade/hold balancing. TASK_149 ensures trades and holds get equal gradient mass. TASK_713 ensures WITHIN trades, high-value ones dominate.

### Data Requirement

Need per-asset returns in the training batch. Currently `batch['returns']` is scalar remaining PnL. Need to also pass per-asset returns at the decision timestep. This requires:

1. In `_generate_batch_episodes`: store `log_returns[start_idx:end_idx]` as `ep_data['per_asset_returns']`
2. In `generate_batch`: include `per_asset_returns` in batch
3. In `_train_step_impl`: use it for return weighting

### Expected Impact

- High-value trades (the ones that matter for PnL) get 5-10x more gradient
- Model learns "when cross-asset signal X appears, trade asset Y specifically"
- Low-value trades (near-zero return) get less gradient — reduces noise
- Should improve benchmark trade agreement from 12-33% → higher

### Risks

- Reward hacking: model might learn to only trade extreme moves (ignoring subtle signals)
- Mitigation: cap return_weight at max 10x to prevent single outliers dominating

---

## Changes

### easy_trainer.py
- `_train_step_impl`: Add return-weighted policy loss after TASK_149 adaptive weighting
- New batch key: `per_asset_returns`

### train_easy.py
- `_generate_batch_episodes`: Store per-asset returns in episode data
- `generate_batch`: Pass through to batch dict

### No changes needed
- Oracle, encoder, state channels — all unchanged
- Checkpoints — fully compatible (just a loss change)
