# TASK_726: Class-Balanced BC Loss — Fix 93.75% HOLD Dominance

**Status**: READY
**Priority**: CRITICAL — root cause of bull market blindness
**Depends on**: TASK_724 (factored search, DONE), encoder audit (DONE)
**Blocks**: Balanced accuracy gate (75%), bull alpha improvement

---

## Problem Statement

The encoder direction audit (`prompt-lib/out_encoder_audit/findings.md`) proved:
1. Encoder preserves direction (cosine 0.38 bull/bear in latent)
2. Policy head outputs near-uniform (~20% per action) regardless of direction
3. Root cause: **oracle class imbalance** — k=4 of 64 assets selected = 93.75% HOLD by construction

The BC loss in Phase 2 is plain cross-entropy:
```python
# train_phase2.py line 798-800:
policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
oracle_reshaped = oracle_targets.reshape(-1, n_assets, n_actions_per_asset)
bc_loss = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean()
```

This treats every (sample, asset) pair equally. With 93.75% HOLD, the gradient is 94% "predict HOLD" and 6% "predict BUY/SELL." The model converges on the marginal (94% HOLD) and never learns the per-asset conditional (WHICH assets to BUY/SELL).

**Phase 1 already solved this** (`easy_trainer.py:170-202`) with adaptive trade weighting, return-weighted boost, and warmup ramping. **Phase 2 dropped all of it** when the training loop was rebuilt. This is the bug.

---

## Solution: Port Phase 1 Class Balancing into Phase 2 BC Loss

### What Phase 1 Does (easy_trainer.py:170-202)

```python
# 1. Continuous trade strength from oracle targets
hold_prob = targets_reshaped[:, :, 1]  # (B, 64) — HOLD probability per asset
trade_strength = 1.0 - hold_prob       # 0.0 = pure hold, 1.0 = pure trade

# 2. Adaptive weight: equalize gradient between trades and holds
n_trades = jnp.maximum(jnp.sum(trade_strength), 1.0)
n_holds = jnp.maximum(jnp.sum(hold_prob), 1.0)
target_trade_weight = jnp.minimum(n_holds / n_trades, 64.0)  # Up to 64× boost

# 3. Per-asset weights: holds=1.0, trades=adaptive_weight
loss_weights = 1.0 + trade_strength * (target_trade_weight - 1.0)  # (B, 64)

# 4. Return-weighted boost (TASK_713): high-return trades get extra gradient
abs_ret = jnp.abs(per_asset_returns)
return_boost = 1.0 + trade_strength * jnp.minimum(abs_ret / mean_ret, 10.0)
loss_weights = loss_weights * return_boost

# 5. Weighted loss
policy_loss = jnp.sum(per_asset_ce * loss_weights) / jnp.sum(loss_weights)
```

### What Phase 2 Needs (the fix)

**File**: `train_phase2.py`, inside `make_stage_b_train_step()`, replace the BC loss block (around line 796-800).

#### Change 1: Class-Balanced BC Loss

Replace:
```python
# === Oracle BC regularizer (decaying, per-asset CE matching Phase 1) ===
policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
oracle_reshaped = oracle_targets.reshape(-1, n_assets, n_actions_per_asset)
bc_loss = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean()
```

With:
```python
# === Oracle BC regularizer — class-balanced (TASK_726) ===
# Oracle targets are 93.75% HOLD (k=4/64). Without rebalancing, the model
# learns the HOLD marginal and ignores the per-asset BUY/SELL conditional.
# Port Phase 1's adaptive trade weighting (easy_trainer.py:170-202).
policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
oracle_reshaped = oracle_targets.reshape(-1, n_assets, n_actions_per_asset)

# Per-asset CE: (B, 64)
per_asset_ce = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped)

# Trade strength from oracle targets (continuous, 0=hold, 1=trade)
hold_prob = oracle_reshaped[:, :, 1]  # HOLD probability per asset (B, 64)
trade_strength = 1.0 - hold_prob      # (B, 64)

# Adaptive weight: equalize gradient mass between trades and holds
n_trades = jnp.maximum(jnp.sum(trade_strength), 1.0)
n_holds = jnp.maximum(jnp.sum(hold_prob), 1.0)
trade_weight = jnp.minimum(n_holds / n_trades, 64.0)  # Cap at 64× (full k=4/64 balance)

# Per-asset loss weights: holds=1.0, trades=trade_weight
bc_weights = 1.0 + trade_strength * (trade_weight - 1.0)  # (B, 64)

# Weighted mean (normalized to preserve gradient scale)
bc_loss = jnp.sum(per_asset_ce * bc_weights) / jnp.sum(bc_weights)
```

This change:
- Gives trade decisions (BUY/SELL) up to 64× more weight than HOLD decisions
- Self-adapting: `n_holds / n_trades` is computed per-batch, tracks the actual imbalance
- HOLD still gets gradient (weight=1.0) — model doesn't forget to hold when appropriate
- Preserves gradient magnitude via `/jnp.sum(bc_weights)` normalization

#### Change 2: Also Balance the Factored Policy Loss (Optional but Recommended)

The factored search targets also have a HOLD bias — when Q-values are flat, the softmax puts ~20% per action with a slight HOLD edge. The per-asset CE policy loss inherits this bias.

Same rebalancing can be applied to the policy loss. In the `if use_factored:` branch (around line 735-740):

```python
if use_factored:
    pol_r = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
    tgt_r = mcts_visits.reshape(-1, n_assets, n_actions_per_asset)
    per_asset_ce_policy = optax.softmax_cross_entropy(pol_r, tgt_r)

    # Rebalance: weight non-HOLD targets higher
    hold_prob_policy = tgt_r[:, :, 1]
    trade_strength_policy = 1.0 - hold_prob_policy
    n_t = jnp.maximum(jnp.sum(trade_strength_policy), 1.0)
    n_h = jnp.maximum(jnp.sum(hold_prob_policy), 1.0)
    tw = jnp.minimum(n_h / n_t, 64.0)
    policy_weights = 1.0 + trade_strength_policy * (tw - 1.0)
    policy_loss = jnp.sum(per_asset_ce_policy * policy_weights) / jnp.sum(policy_weights)
else:
    policy_loss = optax.softmax_cross_entropy(policy_logits, mcts_visits).mean()
```

#### Change 3: Return-Weighted Boost (Optional, Phase 2 of the Fix)

Phase 1 also has return-weighted boosting (TASK_713) — high-return trades get up to 10× extra weight. This naturally emphasizes bull-market BUY signals (which have high returns).

This requires `per_asset_returns` in the training batch, which Phase 2's `AsyncBatchPipeline` doesn't currently provide. Adding it requires:

**File**: `train_phase2.py`, `AsyncBatchPipeline._prepare_numpy()`, around line 468:

After computing `step_returns = slot['returns'][decision_indices]`:
```python
# Per-asset absolute returns for return-weighted BC (TASK_726)
all_per_asset_returns.append(np.abs(step_returns).astype(np.float32))
```

And in the returned dict:
```python
'per_asset_returns': np.concatenate(all_per_asset_returns)[:batch_size],
```

Then in `_encode_jax`, pass through to the batch:
```python
'per_asset_returns': jnp.array(np_batch['per_asset_returns']),
```

Then in `make_stage_b_train_step`, add to the loss:
```python
# After computing bc_weights:
if per_asset_returns is not None:
    abs_ret = jnp.abs(per_asset_returns)  # (B, 64)
    mean_ret = jnp.maximum(jnp.mean(abs_ret), 1e-6)
    return_boost = 1.0 + trade_strength * jnp.minimum(abs_ret / mean_ret, 10.0)
    bc_weights = bc_weights * return_boost
```

**Defer this to Phase 2 of the fix.** Change 1 alone should have significant impact.

---

## Config Changes

**File**: `phase2_config.py`

```python
# === Class-Balanced BC (TASK_726) ===
bc_class_balanced: bool = True     # Adaptive trade/hold weight balancing
bc_max_trade_weight: float = 64.0  # Cap on trade weight boost (64 = full k/n balance)
bc_return_weighted: bool = False   # Phase 2: also weight by per-asset return magnitude
```

**File**: `train_phase2.py`, `parse_args()`

```python
parser.add_argument('--no-bc-balance', action='store_true',
                    help='Disable class-balanced BC (use plain mean)')
parser.add_argument('--bc-max-trade-weight', type=float, default=None,
                    help='Max trade weight boost (default: 64.0)')
```

Override:
```python
if args.no_bc_balance:
    config.bc_class_balanced = False
if args.bc_max_trade_weight is not None:
    config.bc_max_trade_weight = args.bc_max_trade_weight
```

Guard in `make_stage_b_train_step`:
```python
# Capture in closure:
use_bc_balance = config.bc_class_balanced
bc_max_tw = config.bc_max_trade_weight

# In loss_fn:
if use_bc_balance:
    # ... (class-balanced BC from Change 1)
else:
    bc_loss = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean()
```

---

## Training Command

```bash
mkdir -p jax_ckpt_easy_phase2_726 && nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl \
  --stage B --epochs 300 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.3 \
  --no-reset-value-head \
  --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 \
  --bc-weight-min 0.05 \
  --output-dir jax_ckpt_easy_phase2_726 \
  > jax_ckpt_easy_phase2_726/stage_b_nohup.log 2>&1 &
```

Note: `--bc-weight-min 0.05` slightly higher than before (0.03) since BC is now the primary directional signal and class-balanced. The balanced BC at 0.05 provides more directional gradient than unbalanced BC at 0.15.

---

## Expected Impact

With class balancing, the effective gradient becomes:
- Before: 94% HOLD signal, 6% BUY/SELL signal → model learns HOLD marginal
- After: 50% HOLD signal, 50% BUY/SELL signal (by gradient mass) → model learns per-asset conditional

The 1.2% BUY/SELL swing between bull and bear regimes is now amplified to ~50% of the BC gradient. The model should start differentiating its response to oracle BUY vs oracle SELL within 50-100 epochs.

**Conservative estimate**: Bull alpha improves from -24% to -10% to -15% (model starts buying some assets in uptrends)
**Optimistic estimate**: Bull alpha improves to -5% to 0% (model reproduces oracle's bull BUY behavior)

---

## Diagnostic Check

After training, re-run the benchmark with action distribution diagnostics:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_easy.py --checkpoint jax_ckpt_easy_phase2_726/phase2_stageB_e200.pkl
```

Look for:
```
When oracle BUYs → model: BUY > 10% (was 2.9% before)
When oracle SELLs → model: SELL > 10% (was 2.5% before)
```

If the model's response to BUY vs SELL diverges, the class balancing worked. If still identical, the problem is deeper (TRM reasoning can't extract the conditional from the latent — try unfreezing encoder or auxiliary direction head).

---

## Why This Should Work

Phase 1 had this same class balancing and achieved policy_loss ~1.4 with meaningful BUY/SELL discrimination. Phase 2 dropped it and the model regressed to 94% HOLD. The mechanism is proven — just needs to be ported.

The encoder audit confirmed the latent HAS the directional signal (cosine 0.38). The policy head HAS the capacity (5 outputs per asset). The training just isn't providing enough gradient on the minority class for the policy to learn the conditional mapping latent → BUY/SELL.

---

## Files Modified

| File | Changes |
|------|---------|
| `train_phase2.py` | Replace BC loss with class-balanced version (Change 1), optionally balance policy loss (Change 2) |
| `phase2_config.py` | Add `bc_class_balanced`, `bc_max_trade_weight`, `bc_return_weighted` |
| `train_phase2.py` parse_args | Add `--no-bc-balance`, `--bc-max-trade-weight` CLI args |

**No architecture changes. No checkpoint incompatibility. No new files.**
