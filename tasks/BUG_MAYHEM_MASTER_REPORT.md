# BUG MAYHEM: Consolidated Audit Master Report

## Executive Summary
This report consolidates **22 unique issues** identified across the entire audit suite (including `127k*` deep dives). The training pipeline suffers from "silent killers" in data sampling, gradient flow, and reward scaling.

---

## I. CRITICAL (Training Breaking)

| Bug | File | Impact |
|:---|:---|:---|
| **Alpha-Relative Scale Mismatch** | `trainer.py` | 100x inflation/deflation of reward signals. |
| **BC Confidence Masking** | `trainer.py` | Zero-confidence samples dilute loss denominator, killing gradients. |
| **Demo Sampling Ignores Priority** | `expert_buffer.py` | `_sample_from_range` uses `np.random.choice` instead of segment tree. **PER is broken for demos.** |
| **Double Alpha Exponentiation** | `expert_buffer.py` | Priorities raised to `alpha` twice, flattening the distribution to near-uniform. |
| **PER Weights Unused** | `expert_buffer.py` | Training is biased toward high-priority samples without IS correction. |
| **PopArt Desync (Outside JIT)** | `trainer.py` | Value head correction and gradient updates are desynchronized. |
| **PopArt Cat. Head Incompatibility**| `popart.py` | Weight correction formula (`σ_old/σ_new`) is mathematically invalid for categorical logits. |
| **Raw Visit Counts** | `self_play.py` | Policy loss uses counts instead of distributions; gradient magnitude is wrong. |
| **Phantom Rewards** | `self_play.py` | Summing rewards past episode termination (penalizing dead agents). |
| **Missing Consistency Loss** | `trainer.py` | No latent self-supervision; dynamics drift over unroll steps. |

---

## II. HIGH (Stability & Correctness)

| Bug | File | Impact |
|:---|:---|:---|
| **Burn Sign Error** | `jax_trading_env.py` | Being flat in a bear market is rewarded as a "burn" (due to symlog logic). |
| **Inconsistent Return Metric** | `jax_trading_env.py` | Agent state doesn't reflect "burn" losses seen by the reward signal. |
| **Path Solver Transaction Costs** | `path_solver.py` | Default 10bps kills greedy oracle performance on minute data. |
| **Return Type Mismatch** | `self_play.py` | Comparing linear `market_return` to log `optimal_alpha`. |
| **Stratified IS Bias** | `expert_buffer.py` | IS weights ignore the 12.5% fixed demo sampling ratio. |
| **Weight Decay Missing** | `config.py` | Weights are effectively unregularized (defaulting to 0). |
| **Priority Scale Mismatch** | `expert_buffer.py` | TD-error (agent) priorities dwarf confidence (demo) priorities. |

---

## III. MEDIUM (Suboptimal / Dead Code)

| Bug | File | Impact |
|:---|:---|:---|
| **Unused Target Network** | `trainer.py` | Maintaining `target_params` is pure memory overhead. |
| **Div-by-Zero Risk** | `trainer.py` | Crash risk if `unroll_steps=1`. |
| **`jnp.where` Branching** | `trainer.py` | Evaluating both branches of Huber loss creates noisy gradients. |
| **Entropy Discontinuity** | `trainer.py` | Bonus "jump" at threshold causes oscillation. |
| **Path Solver Normalization** | `path_solver.py` | Greedy (Sum) vs Soft (Mean) scale inconsistency. |

---

## IV. OPERATIONAL

| Bug | File | Impact |
|:---|:---|:---|
| **JAX/XLA Deadlock** | `self_play.py` | Training hangs at epoch 90 during device-to-host sync. |

---

## Action Plan (Phase 1 Fixes)

1.  **Fix Expert Buffer:** Rewrite `_sample_from_range` to use segment tree and remove double alpha.
2.  **Fix Reward Scale:** Align `max_alpha` scaling in trainer.
3.  **Fix BC Loss:** Use masked mean for confidence weighting.
4.  **Fix PopArt:** Disable weight correction (incompatible) and move stats update inside JIT (or accept sync cost).
5.  **Fix MCTS Counts:** Normalize visit counts before loss computation.
6.  **Fix Phantom Rewards:** Apply mask before summing episode score.
