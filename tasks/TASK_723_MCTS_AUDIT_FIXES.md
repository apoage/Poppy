# TASK_723: MCTS Audit Fixes — Reward, Discount, Exploration, Data

**Status**: IMPLEMENTED (Fixes 1,2,4,5,6 — Fix 3,7 deferred per plan)
**Priority**: CRITICAL — fixes compound bugs that make MCTS search degenerate
**Based on**: Three independent deep analyses (Explore agent + 2× Opus agents, ~300k tokens)
**File scope**: `mctx_search.py`, `train_phase2.py`, `phase2_config.py`, `tools/benchmark_easy.py`

---

## Problem Summary

MCTS tree search is **functionally broken** due to three compounding bugs:
1. Rewards are invisible (0.04% of Q-value)
2. Discount mismatch between MCTS and value training
3. Top-16 action filter only explores HOLD actions

Combined effect: MCTS produces uninformative visit distributions → policy trains on noise → regresses from E93 (1.71, BC-driven) to E999 (1.93, MCTS-driven). **MCTS is actively hurting the policy.**

---

## Fix 1: Remove /100.0 Reward Descaling in MCTS [CRITICAL, 2 min]

**File**: `mcts_trader/jax_agent/mctx_search.py`, line 110

**Bug**: Reward head outputs values in [-10, 10] (trained on `REWARD_SCALE=100` targets). After `_categorical_to_scalar` recovers the scalar in [-10, 10], `/100.0` crushes it to ~0.001. Meanwhile value ≈ 2.5 (PopArt denormalized). So `Q = R + γ*V ≈ 0.001 + 0.997*2.5 ≈ 2.49`. Reward is **0.04% of Q**.

The reward difference between BUY and HOLD on a +0.1% asset is ~0.00001 — completely invisible. MCTS tree search is blind to rewards and degenerates to single-step value lookahead.

**Why /100.0 exists**: TASK_034 comment says "descale to match actual environment reward scale." But MCTS uses `qtransform_by_parent_and_siblings` which normalizes Q-values per-node — absolute scale doesn't matter, only relative ordering. The /100.0 destroys relative reward ordering by making ALL rewards negligible vs value.

**Fix**:
```python
# BEFORE (line 110):
reward_scalar = self._categorical_to_scalar(output['reward']) / 100.0

# AFTER:
reward_scalar = self._categorical_to_scalar(output['reward'])
```

**Note**: `_categorical_to_scalar` also applies PopArt `value_scale`/`value_shift` (lines 138-140). This is wrong for rewards (rewards are NOT PopArt-normalized). But the PopArt params are set per-epoch in `train_stage_b` on the search object — they apply to ALL categorical-to-scalar calls. Two sub-fixes needed:

**Sub-fix 1a**: Remove `/100.0` (line 110)

**Sub-fix 1b**: Don't apply PopArt scale/shift to rewards. Create a separate method:
```python
def _reward_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
    """Convert reward categorical to scalar. No PopArt — rewards aren't normalized."""
    from . import get_value_support
    support = get_value_support()
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * support, axis=-1)
```

Then line 110 becomes:
```python
reward_scalar = self._reward_to_scalar(output['reward'])
```

---

## Fix 2: Match MCTS Discount to Value Training Discount [HIGH, 1 min]

**File**: `phase2_config.py`, line 38

**Bug**: MCTS uses `mcts_discount=0.997` but value head is trained with `td_discount=0.95`. At MCTS depth d:
```
Q = Σ_{k=0}^{d-1} (0.997^k × R_k) + 0.997^d × V(leaf)
```
But `V(leaf)` was trained assuming γ=0.95 for future discounting. The tree applies 0.997^d to a value already calibrated for 0.95 → systematic overvaluation of future states.

**Fix**: Either in config default or via CLI override:
```python
# phase2_config.py — change default:
mcts_discount: float = 0.95  # Must match td_discount

# Or via CLI:
--mcts-sims 32  # (add --mcts-discount 0.95 CLI arg if needed)
```

Since there's no `--mcts-discount` CLI arg yet, also add one in `parse_args()`:
```python
parser.add_argument('--mcts-discount', type=float, default=None,
                    help='MCTS tree search discount (should match td_discount)')
```
And in the override section:
```python
if args.mcts_discount is not None:
    config.mcts_discount = args.mcts_discount
```

---

## Fix 3: Increase max_considered_actions [HIGH, 1 min]

**File**: `phase2_config.py`, line 35

**Bug**: With 320 actions and `max_considered_actions=16`, Gumbel MuZero only evaluates the 16 highest-prior actions. Since the policy is HOLD-biased (97% trade density = 3.1%), these 16 are all HOLD variants for different assets. BUY and SELL actions are **never explored**.

Feedback loop: HOLD prior → MCTS explores HOLD → visits = HOLD → policy trains HOLD → more HOLD prior.

**Fix**:
```python
# phase2_config.py:
max_considered_actions: int = 64  # Was 16 — one per asset minimum
```

With 64 considered, at least some BUY/SELL actions per asset can be explored. Cost: MCTS slightly slower but still within budget.

**Important**: With `max_considered_actions=64` and `mcts_sims=32`, each action gets only 0.5 simulations on average. Gumbel MuZero handles this (designed for max_considered > sims), but confidence will be low. **Also increase `mcts_sims` to 64** to maintain ~1 sim per action. Cost: MCTS ~38s→~76s/epoch, but the search is now meaningful rather than wasted compute.

Compromise if training time is a constraint: `mcts_sims=48, max_considered_actions=48`.

**Alternative**: If 64 is too slow, use 32 (ensures at least half the assets get non-HOLD consideration).

---

## Fix 4: Increase BC Weight Floor [MEDIUM, 1 min]

**File**: `phase2_config.py`, lines 44-45

**Bug**: BC decays from 0.05 to 0.005 over first 12 epochs, then stays at 0.005 for 988 epochs. The oracle signal (which is correct) has 200x less influence than MCTS (which is uninformative).

Evidence: Policy was BEST at E93 (1.71) when BC was still ~0.02. After BC decayed, MCTS took over → policy regressed to 1.93.

**Fix**:
```python
bc_weight_min: float = 0.03   # Was 0.005 — keep oracle signal meaningful
bc_decay_steps: int = 20000   # Was 5000 — slower decay (50 epochs)
```

**Rationale**: Until MCTS is fixed (Fixes 1-3), BC provides the only correct policy signal. After MCTS is fixed, BC should still anchor the policy to prevent drift. The oracle knows exactly which assets to trade — MCTS should improve on the oracle, not replace it with noise.

---

## Fix 5: Increase Value Loss Weight [MEDIUM, 1 min]

**File**: `phase2_config.py`, line 42

**Bug**: `exit_value_weight=0.5` means value's loss-weighted contribution is ~12.5% of total loss (0.5×1.09 / 4.4 total). By weight alone it's 25% (0.5 / 2.0 total weights). Either way, value is underweighted relative to its importance — MCTS depends on good value to guide search.

**Fix**:
```python
exit_value_weight: float = 1.0  # Was 0.5 — match policy weight
```

---

## Fix 6: Add Slot Rotation to AsyncBatchPipeline [HIGH, 30 min]

**File**: `train_phase2.py`, epoch loop in `train_stage_b` (around line 1095)

**Bug**: Random sampling doesn't trigger `_rotate_slot()`. Same 17 segments for 1000 epochs. 140 available, 17 used (12%).

**Fix**: Add periodic rotation between epochs (main thread, safe from background thread):

```python
# In train_stage_b epoch loop, before the step loop:
if epoch > 0 and epoch % 20 == 0:
    # Rotate one slot — round-robin across regimes
    slot_to_rotate = (epoch // 20) % len(pool.slots)
    pool._rotate_slot(slot_to_rotate)
    logger.info(f"  Rotated slot {slot_to_rotate}")
```

**Thread safety**: The background thread holds a reference to the slot dict and accesses `slot['hm'].cache_data`. If `_rotate_slot` sets `old_slot['hm'] = None` mid-read, the thread crashes with `AttributeError: 'NoneType' has no attribute 'cache_data'`. Queue-draining doesn't prevent this — the thread may be mid-`_prepare_numpy`.

**Correct approach — stop/restart the pipeline:**
```python
if epoch > 0 and epoch % 20 == 0:
    slot_to_rotate = (epoch // 20) % len(pool.slots)
    # Stop background thread (waits for current _prepare_numpy to finish)
    pipeline.stop()
    # Rotate safely — no concurrent readers
    pool._rotate_slot(slot_to_rotate)
    logger.info(f"  Rotated slot {slot_to_rotate} → {pool.slots[slot_to_rotate].get('label', '?')}")
    # Restart pipeline with fresh thread
    pipeline = AsyncBatchPipeline(pool, config, prefetch=2, batch_size=batch_size)
```

Cost: ~2-3 seconds per rotation (thread join + restart). Happens every 20 epochs (~1000s of training). Negligible (0.2% overhead).

---

## Fix 7: Fix Benchmark Value MSE [LOW, 15 min]

**File**: `tools/benchmark_easy.py`, around line 689

**Bug**: Benchmark computes `value_mse = (model_values / value_scale - oracle_remaining) ** 2` but:
1. `model_values` is PopArt-normalized (model trained with PopArt) — not denormalized
2. `oracle_remaining` is full-segment remaining PnL — training uses 400-step horizon

The MSE=302 is meaningless — comparing wrong-scale numbers against wrong-horizon targets.

**Fix**: Load PopArt state from checkpoint and denormalize:
```python
# In benchmark, after loading checkpoint:
popart_state = ckpt.get('popart_state', None)
if popart_state:
    pa_mean = popart_state['mean']
    pa_std = np.sqrt(popart_state['var'] + 1e-8)
else:
    pa_mean, pa_std = 0.0, 1.0

# When computing value MSE:
model_values_raw = (model_values * pa_std + pa_mean) / config.value_scale
oracle_remaining_400 = oracle_cum_pnl[min(t+400, T-1)] - oracle_cum_pnl[t]  # 400-step horizon
value_mse = np.mean((model_values_raw - oracle_remaining_400) ** 2)
```

---

## Implementation Order

```
Fix 1  (reward descaling + PopArt separation)    → 10 min, CRITICAL
Fix 2  (discount match)                          → 2 min, config change
Fix 3  (max_considered_actions 16→64)            → 1 min, config change
Fix 4  (bc_weight_min 0.005→0.03)               → 1 min, config change
Fix 5  (value_weight 0.5→1.0)                   → 1 min, config change
Fix 6  (slot rotation)                           → 30 min, code change
Fix 7  (benchmark MSE)                           → 15 min, diagnostic only
```

Fixes 1-5 can ship together. Fix 6 is independent. Fix 7 is diagnostic (doesn't affect training).

---

## Expected Outcome

With Fixes 1-6 applied:
- MCTS can actually see reward differences between actions → meaningful tree search
- MCTS explores BUY/SELL actions (not just HOLD) → trades more in bull markets
- BC anchors policy to oracle while MCTS improves → no regression from E93
- Diverse data → better generalization to benchmark periods
- Value properly weighted → better MCTS Q-value estimates

**Conservative estimate**: 4/6 positive alpha (sideways flips from -0.34% to positive, plus potential bull improvement from actually trading)

**Optimistic estimate**: 5/6 positive alpha (bull improves from -22% to single digits due to MCTS actually exploring BUY actions)

---

## Future Consideration: Factored Per-Asset MCTS

All three analyses agreed the 320-action joint MCTS is fundamentally mismatched to the factored 64×5 action space. Even with max_considered=64, MCTS treats "BUY asset 7" and "SELL asset 23" as competing alternatives when they're independent decisions.

The correct long-term fix is factored per-asset search (64 independent 5-action MCTS, one per asset). But Fixes 1-6 should be attempted first — they may be sufficient for the 4/6 gate.

If not, file as TASK_724.
