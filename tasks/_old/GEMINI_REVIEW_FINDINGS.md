# Gemini Architecture Review - Findings & Action Plan

**Date:** December 21, 2025
**Reviewer:** Gemini (Large Context)
**Status:** Pending Implementation

---

## Critical Findings

### 1. Value Head Disconnection [HIGH PRIORITY]
**File:** `mcts_trader/jax_agent/trm_flax.py:94-99`

**Problem:** Value head consumes raw `state` from encoder, bypassing all H×L cycles of reasoning. Policy uses "slow thinking" (refined latent), value uses "fast thinking" (raw state).

**Impact:** Value predictions may be less accurate than they could be, causing MCTS to make suboptimal decisions.

**Current:**
```python
# Value head uses raw state
value = nn.Dense(features=value_hidden, name='value_head_1')(state)  # <-- raw state!
```

**Fix:** Feed refined `latent` to value head instead of `state`.

**Cross-validation:**
- Compare value-reward correlation before/after
- Track value_mse improvement
- A/B test on Level 0 synthetic

---

### 2. Latent State Drift in Dynamics [HIGH PRIORITY]
**File:** `mcts_trader/jax_agent/muzero_model.py:90-94`

**Problem:** Dynamics model returns unnormalized `next_state`. Over multiple MCTS unroll steps, latent representations can drift/explode.

**Impact:** Search instability, especially at deeper MCTS depths.

**Current:**
```python
next_state = out[:, :self.state_dim]  # Raw, unnormalized
reward = out[:, self.state_dim:]
return next_state, reward
```

**Fix:** Add LayerNorm to `next_state` output.

**Cross-validation:**
- Monitor latent state magnitude over 10+ unroll steps
- Compare MCTS search quality at depth 5 vs depth 10
- Track if self-play time stabilizes

---

### 3. Replay Buffer Wrap-around Bug [MEDIUM PRIORITY]
**File:** `mcts_trader/jax_agent/replay_buffer.py`

**Problem:** When buffer wraps around, sampled sequences may span two different episodes (end of old + start of new), creating corrupted training data.

**Impact:** Noisy gradients, potential training instability.

**Fix:** Validate that sampled trajectory indices don't cross wrap-around boundary.

**Cross-validation:**
- Add assertion/logging when wrap-around sampling occurs
- Compare gradient variance before/after fix
- Test with small buffer to force wrap-around

---

## Medium Findings

### 4. Unrolled vs True Recursive [MEDIUM PRIORITY]
**File:** `mcts_trader/jax_agent/trm_flax.py:66-87`

**Problem:** Each H×L iteration uses unique layer names, creating a deep feed-forward net (high params) rather than true recursion (shared weights, inductive bias).

**Impact:** High parameter count (good for capacity), but lacks recursive inductive bias (may hurt generalization).

**Options:**
- A) Keep current (high capacity, works for now)
- B) Share weights within L_cycles (reduce params, test generalization)
- C) Full transformer migration (architectural overhaul)

**Cross-validation:**
- Compare small model (shared weights) vs large (unrolled) on Level 2+
- Track overfitting metrics

---

### 5. Action Masking in MCTS Children [LOW PRIORITY]
**File:** `mcts_trader/jax_agent/mctx_search.py`

**Problem:** Root node is masked but child nodes rely on policy head to avoid invalid actions. If dynamics predicts high prob for invalid future action, search may explore it.

**Impact:** Minor - Gumbel MuZero generally handles this, but could waste search budget.

**Fix:** Pass action mask through recurrent inference or trust policy learned masking.

**Cross-validation:**
- Log invalid action selection rate in MCTS
- Only fix if >5% invalid actions observed

---

### 6. LayerNorm Alternative for Value Head [LOW PRIORITY]
**File:** `mcts_trader/jax_agent/trm_flax.py`

**Problem:** LayerNorm was removed entirely due to collapse. Could try LayerNorm on *input* to value head instead of intermediate layers.

**Cross-validation:**
- Test LayerNorm on input vs no LayerNorm
- Monitor value distribution entropy

---

### 7. Root Reward Consistency [LOW PRIORITY]
**File:** `mcts_trader/jax_agent/muzero_model.py:110-111`

**Problem:** `initial_inference` sets root reward as one-hot at center bin. Ensure `_scalar_to_categorical` in trainer handles zero consistently.

**Cross-validation:**
- Verify both produce identical output for reward=0
- Add unit test

---

## Proposed: Transformer Migration

**Rationale:** Current TRM uses custom recursive blocks. Migrating to standard transformer architecture would:
- Improve tensor shape handling (attention is batch-friendly)
- Enable positional encoding for temporal awareness
- Allow use of pre-built optimizations (Flash Attention, etc.)
- Make architecture more robust and maintainable

**Scope:**
- Replace TRMReasoning with TransformerReasoning
- Keep TRMEncoder (CNN) or replace with patch embedding
- Maintain same interface to MuZeroNetworks

**Cross-validation:**
- Parameter-matched comparison (same ~36M params)
- Compare training speed, value calibration, final performance

---

## Task Priority Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Value Head → Use Refined Latent | Small | High |
| 2 | Dynamics → Add LayerNorm | Small | High |
| 3 | Replay Buffer → Fix Wrap-around | Medium | Medium |
| 4 | Cross-validation Framework | Medium | High |
| 5 | Transformer Migration (TRM v2) | Large | High |
| 6 | Weight Sharing Option | Medium | Medium |
| 7 | Action Masking Audit | Small | Low |
| 8 | Root Reward Consistency | Small | Low |

---

## Implementation Notes

### For Task 1 (Value Head):
```python
# In TRMReasoning.__call__, change:
# FROM: value = nn.Dense(...)(state)
# TO:   value = nn.Dense(...)(latent)  # After all H×L refinement
```

### For Task 2 (Dynamics Normalization):
```python
# In MuZeroNetworks.dynamics, add:
next_state = nn.LayerNorm(epsilon=1e-5)(next_state)
```

### For Task 3 (Buffer Fix):
```python
# In JAXReplayBuffer.sample_batch, add boundary check:
# Ensure (start_idx + seq_len) doesn't cross write position
```

### For Task 5 (Transformer):
- New file: `mcts_trader/jax_agent/transformer_flax.py`
- Standard multi-head attention + FFN blocks
- Cross-attention between state and learned queries
