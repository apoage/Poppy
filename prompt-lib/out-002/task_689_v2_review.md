# TASK_689 v2 Review — Agent State Channels

**Date**: 2026-02-21
**Task Version**: v2 (revised after 3 independent reviews)
**Review Focus**: Remaining issues, verification of fixes, new concerns

---

## Executive Summary

**Overall Assessment**: The v2 revision addresses all 3 HIGH priority issues from the previous review:

✅ **CRIT-1 FIXED**: Position direction now derived from actions, not returns
✅ **CRIT-2 FIXED**: Scheduled sampling implemented with 3-phase approach
✅ **CRIT-3 FIXED**: Tiled state (current only) instead of evolving history

**New concerns**: 1 HIGH, 1 MEDIUM, and several implementation details that need clarification.

**Recommendation**: Task is now **READY to proceed with implementation**, but address the HIGH issue first (scheduled sampling cost).

---

## Remaining Issues

### CRIT-4: Scheduled Sampling Cost is Underestimated

**Description**:
The task claims "One forward pass per timestep × 400 steps = 400 forward passes per episode" and suggests "only run for 20% of episodes" to mitigate. This is still significant:

- Baseline: 1 forward pass per batch step (training)
- Phase 2: +0.2 × 400 = **80 extra forward passes per episode** for 20% of episodes
- Per epoch (200 steps, batch=256): 200 × 256 × 0.2 = 10,240 episodes → **~820k extra forward passes**

**Impact**: 2-3× training slowdown in Phase 2, not "only for 20%" as implied.

**Evidence**:
- Line 225: "The expensive part: running model forward per-episode to get its own actions"
- Line 265: "Cost: One forward pass per timestep × 400 steps = 400 forward passes per episode"
- Line 267: "Optimization: Only run model-forward for a fraction of episodes. Even 20% model-state episodes is enough"

**Proposed fix**:
Implement **gradient-free scheduled sampling** that doesn't require full forward passes:

```python
# Option A: Cached model predictions (recommended)
# During training, cache model's last predictions per episode
# On next step, use cached predictions for state (no forward pass)
# Only recompute when RNG says "use model state"

# Option B: Predictions from previous epoch
# At epoch N, use predictions from epoch N-1's checkpoint
# This shifts scheduled sampling to be lagged, but no runtime cost

# Option C: Stochastic model-forward (compromise)
# Only run full forward pass for 5% of episodes, not 20%
# The rest use oracle with added noise to simulate "model state"
```

**Rationale for Option A**:
```python
# In training loop
episode_predictions = []  # Accumulate during episode

for step in range(episode_length):
    obs = get_observation(step)
    state = get_state_channels(step)

    if should_use_model_state(epoch):
        # Use previous prediction (no forward pass!)
        action = episode_predictions[step - 1]  # Cached
    else:
        action = model(obs)  # Normal training forward pass
        episode_predictions.append(action)

    loss = compute_loss(action, oracle_action[step])
```

This gives us "model-like" state without the 400× forward pass cost.

**Severity**: HIGH — Unmitigated, this will slow training 2-3×

---

### MED-3: State Corruption May Break Training Signal

**Description**:
Lines 179-203 propose "state corruption" with:
- 10% position bit-flip
- Gaussian PnL noise (0.2 std)
- 5% zero dropout of ALL state channels

While the intent is denoising robustness, this may backfire:

1. **Position bit-flip**: Flipping long→short changes everything. If model learns "heat says hot, position says long" but position is actually short, it gets contradictory signals.
2. **Zero dropout**: 5% of samples see ZERO state channels. This is too aggressive — model learns to ignore state entirely for these samples.
3. **Corruption rate is cumulative**: 10% + 5% = 15% of training samples have corrupted state.

**Evidence**:
- Lines 179-203: `corrupt_state()` with flip_mask, pnl_noise, zero_mask
- Line 205: "This trains a denoising component: model learns 'state is a useful hint, not a command.'"

**Proposed fix**:
Reduce corruption and make it more targeted:

```python
def corrupt_state(state_channels, key, corruption_level=0.05):  # Was 0.2
    """Conservative state corruption — 5% of samples."""
    k1, k2 = jax.random.split(key, 2)

    # 1. Position jitter: 2% of assets get small random noise, not full flip
    jitter_mask = jax.random.bernoulli(k1, 0.02, state_channels[0].shape)
    jitter = jax.random.uniform(k1, state_channels[0].shape, -0.1, 0.1)
    state_channels = state_channels.at[0].set(
        jnp.where(jitter_mask, state_channels[0] + jitter, state_channels[0])
    )

    # 2. PnL noise: Reduced magnitude
    pnl_noise = jax.random.normal(k2, state_channels[1].shape) * 0.05  # Was 0.2
    state_channels = state_channels.at[1].add(pnl_noise)

    # REMOVE: Zero dropout — too aggressive, 5% is too high
    # If we want zero dropout, use 1% not 5%

    return state_channels
```

**Severity**: MEDIUM — May hurt training, but can be tuned later

---

### MED-4: Confidence Channel Depends on Oracle Behavior

**Description**:
The confidence channel uses `max(oracle_soft_policy[asset, :])` during training. But oracle's soft targets come from **multipass with sharpening temperature=0.1** (TASK_677). This creates two issues:

1. **Oracle confidence ≠ model confidence**: Oracle confidence reflects "how consistent was the oracle across 10 noisy passes?" — this is NOT the same as "how confident should the model be?"
2. **Sharpening skews confidence**: Temperature=0.1 makes oracle targets very sharp (95% on correct action). Model will rarely see low-confidence oracle examples.

At inference, `confidence = max(model_softmax[asset, :])` — this is a different distribution entirely.

**Evidence**:
- Line 55: `confidence: [0.2, 1.0] — Max softmax probability per asset (oracle or model)`
- Line 71: "Training: `confidence = max(oracle_soft_policy[asset, :])` per asset"
- Line 72: "Inference: `confidence = max(model_softmax[asset, :])` per asset"
- Line 75: "Self-correcting: uncertain model predictions → low confidence → model learns to hold when uncertain"
- easy_config.py line 75: `oracle_sharpen_temperature: float = 0.1` (from TASK_677)

**Proposed fix**:
Option A: Drop confidence channel from state
- Rationale: 7 channels is already complex. Confidence adds little beyond what position/PnL/heat already signal.

Option B: Match oracle confidence to model confidence distribution
```python
# During oracle generation, track "consistency" across 10 passes
# Confidence = fraction of passes agreeing on top action
# This is already computed for oracle_sharpening!

# Use consistency directly (not sharpened softmax)
confidence = oracle_consistency  # From multipass aggregation
# This ranges 0.2-1.0 naturally (no artificial bounds)
```

Option C: Synthetic oracle confidence during training
```python
# During Phase 1 (oracle state), simulate "oracle confidence" as:
confidence = jnp.ones_like(oracle_actions) * 0.8  # Consistent high confidence
# During Phase 2/3 (model state), use actual model confidence:
confidence = model_softmax.max(axis=-1)
```

**Severity**: MEDIUM — Confidence channel may not help, could hurt

---

### LOW-2: Implementation Phase C (Confidence Channel) Should Be Optional

**Description**:
Phase C adds confidence channel, but it's optional given MED-4 concerns. Should be a separate experiment, not part of core implementation.

**Proposed fix**:
Phase C → **Optional Experiment** after Phase A and B are validated.

**Severity**: LOW — Nice to separate, but doesn't block implementation

---

## Verification Checklist (Source Code Reading Required)

Before proceeding, verify these by reading actual source code:

- [ ] Read `mcts_trader/jax_agent/path_solver.py` lines 150-250:
  - Verify `optimal_path_to_policy()` returns action indices (0-4), not soft probabilities
  - Confirm action encoding matches lines 99-102 (Buy=2, Sell=0, Hold=1, LimitSell=3, LimitBuy=4)

- [ ] Read `mcts_trader/jax_agent/muzero_model.py` lines 1-100:
  - Verify `n_channels` parameter flows through to encoder
  - Check if encoder projection layer is `nn.Dense(n_channels, embed_dim)` or similar

- [ ] Read `mcts_trader/env/jax_encoder.py` lines 40-80:
  - Verify `n_channels` can be parameterized (currently hardcoded to 27)
  - Check observation shape handling for concatenation

- [ ] Read `train_easy.py` lines 100-200:
  - Verify oracle policy computation flow
  - Check where state channels should be inserted

---

## Design Verification (Conceptual, No Code Reading)

### ✅ Position from Actions (CRIT-1 FIX VERIFIED)
Lines 98-102 correctly derive position from action type:
- `action == 2 | action == 4` → Buy/LimitBuy → +1 (long)
- `action == 0 | action == 3` → Sell/LimitSell → -1 (short)
- Otherwise → keep previous position

**Status**: ✅ CORRECT — matches oracle action encoding

---

### ✅ Signed PnL (MED-1 FIX VERIFIED)
Lines 111-116 correctly implement signed PnL:
- `signed_ret = ret * pos` — long: +ret, short: -ret, flat: 0
- `held = (new_pos == pos) & (pos != 0)` — only accumulate when holding
- `jnp.clip(new_pnl, -10.0, 10.0)` — bounded to value support range

**Status**: ✅ CORRECT — fixes MED-1 from previous review

---

### ✅ Tiled State (CRIT-3 FIX VERIFIED)
Lines 207-222 correctly tile state across time axis:
- `state_tiled = jnp.tile(state_at_t[:, None, :], (1, window_size, 1))` — current state only
- Lines 222-222 explain rationale clearly: "only know current state, not state history"

**Status**: ✅ CORRECT — no causal leakage

---

### ✅ Scheduled Sampling (CRIT-2 FIX VERIFIED)
Lines 153-177 implement 3-phase scheduled sampling:
- Phase 1 (0-30%): Oracle actions
- Phase 2 (30-70%): Mixed (oracle + model)
- Phase 3 (70-100%): Model actions

**Status**: ✅ CORRECT — addresses train/inference gap

---

### ⚠️ Multi-Scale Heat (GOOD DESIGN)
Lines 59-65 and 121-124 implement 3 decay rates:
- `heat_fast: 0.80` — half-life ≈ 3 steps
- `heat_med: 0.95` — half-life ≈ 14 steps
- `heat_slow: 0.99` — half-life ≈ 70 steps

**Status**: ✅ GOOD — addresses MED-2 from previous review

---

### ⚠️ lax.scan Instead of Python Loop (GOOD DESIGN)
Lines 78-139 use `jax.lax.scan` with JIT compilation:
- `@partial(jax.jit, static_argnames=['n_assets'])`
- Scan function updates state per timestep efficiently

**Status**: ✅ GOOD — addresses performance concern from mem-001 review

---

## Recommendations (Prioritized for v2)

### Must Fix Before Implementation (HIGH priority):

1. **Address CRIT-4 (Scheduled sampling cost)**: Implement cached predictions or reduce to 5% model-state episodes. Current 20% will slow training 2-3×.

### Should Fix During Implementation (MEDIUM priority):

2. **Reduce MED-3 (State corruption)**: Lower corruption from 15% → 5%, remove zero dropout, use position jitter instead of bit-flip.

3. **Evaluate MED-4 (Confidence channel)**: Consider making confidence channel optional or dropping it entirely.

### Nice to Have (LOW priority):

4. **Make Phase C optional**: Confidence channel should be a separate experiment after core implementation works.

---

## Implementation Order (Recommended)

Based on risks and dependencies:

1. **Phase A (Core, oracle state only)**: 50-100 epochs
   - This validates state channels work at all
   - Low risk, fast feedback
   - No scheduled sampling cost yet

2. **Phase B (Oracle independence, with scheduled sampling)**: 200-300 epochs
   - Start with 5% model-state (not 20%)
   - Add state corruption gradually (2% → 5%)
   - Monitor training speed vs Phase A

3. **Phase C (Confidence channel, optional)**: Separate experiment
   - Only if Phase A+B show promise
   - Compare with/without confidence
   - Ablation study

4. **Ablation study**: Final 100 epochs
   - Compare full state channels vs no state channels
   - Compare oracle-only vs scheduled sampling
   - Measure actual over-trading reduction

---

## Comparison: v1 vs v2

| Issue | v1 Status | v2 Status |
|-------|-----------|-----------|
| Position from returns (ambiguous) | ❌ CRIT-1 | ✅ Fixed (from actions) |
| Train/inference gap | ❌ CRIT-2 | ✅ Fixed (3-phase sampling) |
| Causal leakage (evolving state) | ❌ CRIT-3 | ✅ Fixed (tiled state) |
| Signed PnL bug | ❌ MED-1 | ✅ Fixed |
| Magic number decay rate | ❌ MED-2 | ✅ Fixed (3 scales) |
| Python loop | ⚠️ Slow | ✅ Fixed (lax.scan) |
| Scheduled sampling cost | N/A | ❌ CRIT-4 (new) |
| State corruption aggressive | N/A | ⚠️ MED-3 (new) |
| Confidence channel mismatch | N/A | ⚠️ MED-4 (new) |

---

## Conclusion

**Is task v2 ready to proceed?** ✅ **YES**, with modification for CRIT-4.

**Summary**:
- All HIGH issues from v1 review are fixed ✅
- 1 new HIGH issue (CRIT-4) must be addressed before Phase 2 implementation
- 2 new MEDIUM issues (MED-3, MED-4) should be tuned during implementation
- Confidence channel (Phase C) should be optional experiment

**Critical path**:
1. Implement Phase A with oracle state only (no scheduled sampling cost)
2. Test 50-100 epochs → verify state channels help
3. Before Phase 2, resolve CRIT-4 (use cached predictions or reduce to 5%)
4. Implement Phase 2 with conservative state corruption (5% not 15%)
5. Phase C only if Phase A+B successful

**Next steps**:
1. Update task document with CRIT-4 fix (cached predictions approach)
2. Reduce state corruption to 5% in task document
3. Mark confidence channel as optional experiment
4. Proceed with Phase A implementation

---

**Review completed**: 2026-02-21
**Task status**: READY for implementation with CRIT-4 modification
**Next review**: After Phase A (50-100 epochs) — verify state channels work before Phase 2
