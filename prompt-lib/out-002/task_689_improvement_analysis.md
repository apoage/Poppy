# TASK_689: Adversarial Review — Agent State Channels

**Date**: 2026-02-21
**Review Type**: Improvement Analysis + Better Approaches
**Reviewer**: Claude (Sonnet 4.6)

---

## Executive Summary

**Overall Assessment**: TASK_689 is well-structured and addresses a real bottleneck (model over-trading 7× vs oracle). The core concept of adding agent state channels is sound. However, several implementation details need clarification and alternative approaches could yield better results with less complexity.

**Key Findings**:
- **3 HIGH issues**: Train/inference distribution gap risk, position direction logic ambiguity, channel shape design inconsistency
- **2 MEDIUM issues**: Missing edge case handling, computational overhead concerns
- **1 LOW issue**: Documentation improvement needed
- **2 Alternative approaches** that may be superior to the proposed design

**Recommendation**: Proceed with implementation after addressing HIGH issues. Consider the proposed alternative approaches before committing to the current design.

---

## Critical Issues (CRIT/HIGH)

### CRIT-1: Position Direction Logic is Ambiguous and Potentially Wrong

**Description**:
The task defines `position_state` as {-1, 0, +1} for short/flat/long positions, but the computation uses `new_pos = np.where(sel > 0.5, np.where(ret > 0, 1.0, -1.0), 0.0)`. This logic is problematic:

1. It assumes oracle selections (`sel`) are binary (trade vs no-trade), but multipass oracle produces soft probabilities (0-1)
2. It ties position direction to the *next* return's sign, not the position opened by the oracle
3. For state-aware oracle (TASK_141), the oracle already tracks positions — why recompute?

**Why it matters**:
If the position state doesn't match what the oracle actually did, the model will learn incorrect state-action mappings. The state channels would provide noisy/incorrect feedback rather than helpful self-awareness.

**Evidence**:
- Line 39: "`position_state`: {-1, 0, +1} — Current position: short (-1), flat (0), long (+1)"
- Line 81: `new_pos = np.where(sel > 0.5, np.where(ret > 0, 1.0, -1.0), 0.0)`
- Line 55-57: Function takes `oracle_selections: (T, n_assets) binary mask from path solver` — but oracle can be soft with multipass

**What needs verification**:
- Read `mcts_trader/jax_agent/path_solver.py` lines 39-200 to understand:
  - Does `greedy_lookahead_path` return position-aware selections?
  - Does multipass oracle use hard thresholding (0.5) or soft probabilities?
  - What is the actual oracle position tracking logic?

**Proposed fix**:
If oracle tracks positions (likely with `state_aware_oracle: bool = True`), extract the actual position history directly rather than recomputing from selections + returns:

```python
# Option A: If oracle returns positions, use them directly
if hasattr(oracle, 'position_history'):
    position = oracle.position_history  # (T, n_assets) with -1/0/+1

# Option B: Track positions during oracle computation
# Modify greedy_lookahead_path to return (selections, positions) tuple
# This is more accurate than inferring from returns
```

**Severity**: HIGH — Incorrect position state channels could make training worse than no state channels

---

### CRIT-2: Train/Inference Distribution Gap is More Severe Than Acknowledged

**Description**:
The task acknowledges the train/inference gap but dismisses it with a "self-correcting property" claim. This is optimistic. The gap is actually two-fold:

1. **Oracle state vs model state**: Oracle trades at 3.9% rate → heat mostly 0-0.2. Model trades at 27.7% rate → heat often 0.5-0.9. The model will rarely see high-heat states during training.

2. **Oracle position accuracy vs model position accuracy**: Oracle's position_state is ground truth (correctly identifies profitable positions). Model's position_state comes from its own noisy predictions. Even if model knows "I'm in a position," it doesn't know if that position is the *correct* one.

The "self-correcting" claim assumes high heat naturally suppresses trades. But if the model learns "high heat = I'm already in a position," it might double-down on wrong positions rather than hold.

**Why it matters**:
If the model never sees the state distribution it will encounter during inference, it may behave unpredictably or fail to generalize. The "train/inference gap actually helps" argument contradicts standard ML wisdom and lacks empirical evidence.

**Evidence**:
- Line 173-174: "Train/inference distribution gap: Oracle state channels vs model state channels differ. The self-correcting property should help"
- Line 24: "oracle trade rate: 3.9%" vs "model trade rate: 27.7%" → 7× gap
- Line 130-132: "Why teacher forcing works here... At inference time, when the model over-trades, its own heat will be much higher than training distribution, naturally suppressing further trades"

**Proposed fix**:
Implement **scheduled mixing** (mentioned as mitigation but not detailed):

```python
def build_agent_state_channels(oracle_selections, oracle_positions, model_state_history=None,
                             epoch=0, total_epochs=500, mixing_schedule=True):
    """Build state channels with oracle-model mixing.

    During early training: 100% oracle state channels
    During late training: 100% model state channels (to match inference)
    Middle: Linear interpolation

    This bridges the train/inference gap.
    """
    # ... compute oracle state channels ...
    oracle_state = np.stack([position, heat, unrealized, hold_dur], axis=1)

    if not mixing_schedule or model_state_history is None:
        return oracle_state  # Pure oracle (early epochs)

    # Compute model state channels from model's own action history
    model_state = build_agent_state_channels_from_model_history(model_state_history)

    # Linear interpolation: 100% oracle → 50% oracle → 0% oracle
    mix_ratio = max(0, 1 - (epoch / total_epochs))  # 1 → 0
    mixed_state = mix_ratio * oracle_state + (1 - mix_ratio) * model_state

    return mixed_state
```

**Severity**: HIGH — Without proper mixing, the model may not learn to use state channels effectively at inference time

---

### CRIT-3: Channel Shape Design is Inconsistent with Temporal Encoding

**Description**:
The task proposes state channels with shape `(4, window_size, n_assets)` where the window_size dimension shows "state evolving through time." This is inconsistent with how the encoder currently works:

1. **V3 encoder (minGRU)** expects temporal sequences where each timestep contains independent observations
2. State channels at time t depend on the *entire action history* up to t, not just the observation window
3. Creating "evolving state" across the window requires looking ahead beyond what the model sees

Example: At time t=400 (observation window [336-400]), the `hold_duration` for t=340 depends on when the last trade was (could be t=200). This creates a **causal leakage** — the observation at t=400 contains information about t=200.

**Why it matters**:
If state channels are constructed with causal leakage, the model will learn to use future information, creating an unrealistic training signal that won't work at inference time.

**Evidence**:
- Line 34: "Observation shape changes: (27, window_size, n_assets) → (31, window_size, n_assets)"
- Line 44: "Each channel is (window_size, n_assets) — the model sees state evolving through the window"
- Lines 52-107: The loop `for t in range(T)` builds state for all timesteps, then slices per window. The computation at t depends on all prior t, which may be outside the window.

**Proposed fix**:
State channels should be **per-timestep, not per-window**:

```python
# WRONG: State channels as full (T, 4, n_assets) then sliced
state_channels = build_agent_state_channels(oracle_selections, ...)  # (T, 4, n_assets)
obs_window = obs[:, t-window_size:t, :]  # (27, window_size, n_assets)
state_window = state_channels[t-window_size:t, :, :]  # (4, window_size, n_assets) ← Causal leakage!
full_obs = jnp.concatenate([obs_window, state_window], axis=0)

# CORRECT: State channels for current timestep only
state_now = state_channels[t, :, :]  # (4, n_assets) — state at time t
# Expand across window (tile, not evolving)
state_tiled = jnp.tile(state_now[:, None, :], (1, window_size, 1))  # (4, window_size, n_assets)
full_obs = jnp.concatenate([obs_window, state_tiled], axis=0)
```

Rationale: The model should know "my current state is X," not see the history of how I got there. The temporal evolution should be learned from the observation windows (which already show price evolution), not from state channels.

**Severity**: HIGH — Causal leakage creates unrealistic training that won't transfer to inference

---

## Medium/Low Issues

### MED-1: Unrealized PnL Calculation has Edge Case Bugs

**Description**:
The `unrealized_pnl` computation has several edge cases not handled:

1. **Reset logic is wrong**: Line 92-94 resets PnL when position is held, opened, or closed. It should only reset on open/close, not hold.
2. **Sign handling**: For short positions, the PnL should be `entry_pnl - ret` (not just accumulate absolute value).
3. **Overflow risk**: `entry_pnl` can grow unbounded if the oracle holds for many steps without closing.

**Evidence**:
- Lines 89-94:
  ```python
  held = (new_pos == cur_pos) & (cur_pos != 0)
  entry_pnl = np.where(held, entry_pnl + np.abs(ret), 0.0)  # ← Always 0.0 if not held!
  ```

The issue is `np.where(held, X, 0.0)` sets PnL to 0.0 when `held` is False (i.e., when position just opened or closed). But when opening, we want PnL=0 (correct). When closing, we also want PnL=0 (correct). But the condition is wrong — it should be `entry_pnl = np.where(held, entry_pnl + signed_ret, 0.0)` where `signed_ret = ret if long else -ret`.

**Proposed fix**:
```python
for t in range(T):
    sel = oracle_selections[t]
    ret = returns[t]

    new_pos = np.where(sel > 0.5, np.where(ret > 0, 1.0, -1.0), 0.0)
    traded = (new_pos != cur_pos).astype(float)

    # Signed return based on position direction
    signed_ret = ret * cur_pos  # long: +ret, short: -ret, flat: 0

    # Update unrealized PnL
    held = (new_pos == cur_pos) & (cur_pos != 0)
    entry_pnl = np.where(held, entry_pnl + signed_ret, 0.0)

    # Clip to avoid unbounded growth
    entry_pnl = np.clip(entry_pnl, -10.0, 10.0)  # Match value range [-10, 10]
```

**Severity**: MEDIUM — Bugs in PnL computation will provide incorrect feedback, but model may still learn approximate patterns

---

### MED-2: Heat Decay Rate is Arbitrary and Not Learnable

**Description**:
The heat channel uses `decay=0.95` (half-life ≈ 14 steps). This is a magic number with no justification. Different assets/strategies may have different optimal heat decay rates.

**Evidence**:
- Line 57: `decay: heat decay rate` (parameter, not validated)
- Line 87: `cur_heat = cur_heat * decay + traded`
- Line 130: "Decay rate 0.95 means half-life ≈ 14 steps (14 minutes). Could also be made learnable."

**Proposed fix**:
Make decay learnable or experiment with multiple values:

```python
# Option A: Make decay a config parameter with multiple presets
heat_decay_configs = {
    'fast': 0.90,   # half-life ≈ 7 steps
    'medium': 0.95,  # half-life ≈ 14 steps
    'slow': 0.98,    # half-life ≈ 35 steps
}

# Option B: Learnable decay (advanced)
# Add a learnable scalar to the model that modulates heat
# But this adds complexity and may not be worth it

# Option C: Multi-heat channels (recommended)
heat_fast = compute_heat(oracle_selections, decay=0.90)
heat_slow = compute_heat(oracle_selections, decay=0.98)
# Model sees both and learns to weigh them
```

**Severity**: MEDIUM — Arbitrary decay may work but is suboptimal; easy to experiment with

---

### LOW-1: Documentation of "Why not X" Could be More Comprehensive

**Description**:
The "Design Choices" section addresses some alternatives but misses several relevant ones:

- **Why not separate trade detector + action selector?** (Two-stage architecture)
- **Why not use attention over action history instead of state channels?**
- **Why not use RL-style state tracking (MDP state) instead of channels?**

Adding these would strengthen the rationale for the chosen approach.

**Proposed fix**:
Add to "Design Choices" section:

```markdown
### Why not separate trade detector + action selector?
Two-stage approach (detect trade vs hold, then select action) would explicitly handle class imbalance. However:
1. Adds complexity and training instability (cascading errors)
2. State channels achieve similar self-regulation through the unified model
3. Current loss weighting already addresses imbalance

### Why not use attention over action history?
We could have the model attend over recent actions (e.g., last 20 steps). But:
1. Action history is discrete (320-dim sparse) — harder to learn smooth representations
2. State channels provide dense, continuous signals (heat, PnL, duration) that are easier to learn from
3. Adds computational overhead (attention over time vs per-asset state)

### Why not use RL-style state tracking?
In RL, the environment tracks state and provides it to the agent. Here:
1. We're training via imitation (supervised), not RL
2. State channels provide self-state without environment changes
3. Keeps architecture consistent with expert iteration (teacher forcing)
```

**Severity**: LOW — Nice-to-have documentation improvement, doesn't affect implementation

---

## Alternative Approaches (Not Issues, But Better Designs)

### ALT-1: Two-Stage Architecture: Trade Detector + Action Selector

**Hypothesis**:
The core problem is extreme class imbalance (96.1% holds, 3.9% trades). Instead of state channels + weighted loss, use a two-stage architecture:

1. **Stage 1: Binary classifier** — predict "trade vs hold" for each asset
   - Input: 27-channel observation
   - Output: 64 binary logits (trade/hold)
   - Loss: Balanced binary cross-entropy (50% trade examples via oversampling)

2. **Stage 2: Action selector** — predict action given trade decision
   - Input: 27-channel observation + Stage 1 trade probabilities
   - Output: 320-dim action logits
   - Only trained on examples where oracle trades

**Why this might be better**:
- Explicitly decouples "when to trade" from "what action to take"
- Stage 1 can learn "don't over-trade" without competing with Stage 2's action precision
- No need for adaptive trade weighting or state channels (simpler)
- More interpretable — we can monitor Stage 1's trade rate directly

**Implementation effort**: HIGH (need to modify model architecture, loss, training loop)

**Risk**: Cascading errors — if Stage 1 is wrong, Stage 2 has nothing to work with

**Priority**: 2 (try after state channels, if state channels don't work)

---

### ALT-2: Action History Attention Instead of State Channels

**Hypothesis**:
Instead of precomputing 4 state channels, let the model learn from recent action history via attention:

```python
# Track recent actions (shape: history_len, n_assets)
action_history = np.zeros((history_len, n_assets))

# During encoding:
action_embeddings = embed_actions(action_history)  # (history_len, n_assets, d_model)
# Concatenate to price observations or attend over them
```

**Why this might be better**:
- More flexible — model learns what to remember, not just 4 fixed features
- No need to design handcrafted features (heat, PnL, duration)
- Attention mechanism can learn cross-asset patterns in action history

**Why this might be worse**:
- Adds computational overhead (attention over time)
- Action space is discrete and sparse — harder to learn smooth representations
- Less interpretable than explicit state channels

**Implementation effort**: MEDIUM (modify encoder to include action history)

**Risk**: May not converge or may learn noisy patterns from sparse action history

**Priority**: 3 (experimental, worth trying if state channels don't work)

---

## Verification Checklist (Source Code Reading Required)

Before proceeding, verify these by reading the actual source code:

- [ ] Read `mcts_trader/jax_agent/path_solver.py` lines 39-200:
  - Does `greedy_lookahead_path` return position information?
  - Does multipass oracle use soft probabilities or hard selections?
  - What is the oracle's position tracking logic?

- [ ] Read `mcts_trader/jax_agent/muzero_model.py` lines 1-100:
  - Verify encoder accepts variable `n_channels` parameter
  - Check if observation shape changes require model initialization changes

- [ ] Read `train_easy.py` lines 150-250:
  - Verify oracle selection computation flow
  - Check where state channels should be inserted

- [ ] Read `easy_trainer.py` lines 40-80:
  - Verify `dummy_obs` shape and how it's used for JIT warmup
  - Check if state channels require trainer modifications

---

## Recommendations (Prioritized)

### Must Fix Before Implementation (HIGH priority):

1. **Fix CRIT-1 (Position direction logic)**: Verify oracle position tracking and use ground truth, not inferred from returns
2. **Fix CRIT-3 (Channel shape design)**: Use per-timestep state channels (tiled), not per-window evolving state
3. **Implement CRIT-2 (Train/inference mixing)**: Add scheduled mixing to bridge distribution gap

### Should Fix During Implementation (MEDIUM priority):

4. **Fix MED-1 (Unrealized PnL bugs)**: Correct sign handling and clipping
5. **Experiment with MED-2 (Heat decay rates)**: Test 0.90, 0.95, 0.98 in separate runs

### Nice to Have (LOW priority):

6. **Add LOW-1 documentation**: Expand "Design Choices" section with missing alternatives

### Consider After Implementation (Alternative approaches):

7. **Try ALT-1 (Two-stage architecture)** if state channels don't break the plateau
8. **Try ALT-2 (Action history attention)** if state channels work but want to improve further

---

## Conclusion

**Is the task ready to proceed?** PARTIALLY — needs fixes for HIGH priority issues before implementation.

**Summary**:
- Core concept (agent state channels) is sound and addresses real bottleneck
- 3 HIGH issues must be fixed: position logic, channel shape, train/inference mixing
- 2 alternative approaches (ALT-1, ALT-2) may be superior and worth considering
- Implementation is straightforward once design is corrected

**Next steps**:
1. Verify oracle position tracking by reading `path_solver.py`
2. Revise the task document with fixes for CRIT-1, CRIT-2, CRIT-3
3. Consider ALT-1 as alternative if state channels seem complex
4. Implement with scheduled mixing and proper channel design
5. Run validation experiment (50 epochs) to verify state channels help

---

**Review completed**: 2026-02-21
**Next review needed**: After fixing HIGH issues and before implementation begins
