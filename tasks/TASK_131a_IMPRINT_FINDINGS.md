# TASK_131a: Imprint/BC System Findings & Debug Report

**Auditor**: Gemini
**Date**: 2026-02-01
**Status**: IDENTIFIED

## 1. Data Starvation (Circular Dependency)
**Location**: `train_jax_fast.py:600`
**Finding**: The callback `on_episode_complete_callback` contains a filter `should_store = agent_alpha > 0`.
**Issue**: In the early stages of training, the agent's performance is often negative or zero. Because of this filter, no expert demonstrations are added to the buffer.
**Impact**: The agent never learns from the expert because it's only allowed to "see" the expert's path if it has already beaten the market on its own. This creates a circular dependency that prevents the BC system from ever starting.

## 2. Zero Confidence / Priority (Starvation Part 2)
**Location**: `train_jax_fast.py:595`
**Finding**: `confidence` is calculated as `agent_alpha / optimal_alpha`.
**Issue**: Even if an episode barely passes the `agent_alpha > 0` filter, if the agent only captured 1% of the optimal alpha, the `confidence` is set to `0.01`.
**Impact**: The `UnifiedIndexBuffer` uses this confidence to set sampling priority: `priority = confidence * self.max_priority`. A confidence of 0.01 means the demo is 100x less likely to be sampled than a fresh agent experience.
**Correction**: The **Hindsight Optimal Path** is the ground truth expert. Its validity does not depend on the agent's performance. Confidence for hindsight demos should always be **1.0**.

## 3. Missing Logging Metrics
**Location**: `train_jax_fast.py:1150` (CSV Row Construction)
**Finding**: `expert_action_agreement` is calculated in `trainer.py` and returned in the metrics dict, but it is not added to the `csv_row` in the training loop.
**Impact**: We cannot monitor whether the policy is actually converging toward the expert actions over time.

## 4. Demo Count Stagnation
**Finding**: `demo_count` was observed to be stuck at 50.
**Analysis**: 50 matches `parallel_envs`. This confirms that only the very first batch of episodes (likely the first epoch) was processed, and subsequently, almost all episodes were rejected by the `agent_alpha > 0` filter.

## 5. Future Leakage in Expert Buffer (CRITICAL)
**Location**: `mcts_trader/jax_agent/expert_buffer.py:630` (`_encode_observations_vectorized`)
**Finding**: The buffer encodes observations for step `t` by passing `t` directly to `encoder_fn`.
**Issue**: `encoder_fn` (from `jax_encoder.py`) interprets its second argument as `window_start` (index of first element).
- **Result**: The encoder extracts prices `[t, t + window_size]`.
- **Reality**: This is **FUTURE LEAKAGE**. The agent sees the next 60 minutes of prices to decide the action at time `t`.
- **Mismatch**: During Self-Play (Rollout), the agent sees `[t - window_size, t]` (Past). During Expert Training (BC), it sees `[t, t + window_size]` (Future).
**Impact**: The agent learns a policy that relies on seeing the future. When deployed in self-play (where it sees the past), this policy is garbage.
**Correction**: In `expert_buffer.py`, calculate `window_start = jnp.maximum(steps - self.window_size, 0)` before passing to `encoder_fn`.

## Recommendations
1. **Remove Filter**: Delete the `should_store` logic in `train_jax_fast.py`.
2. **Force Confidence**: Set `confidence = 1.0` for all hindsight-derived demonstrations.
3. **Update CSV**: Add `metrics['expert_action_agreement']` to the `csv_row` in `train_jax_fast.py`.
4. **Fix Leakage**: Update `_encode_observations_vectorized` in `expert_buffer.py` to subtract `window_size` from step indices.