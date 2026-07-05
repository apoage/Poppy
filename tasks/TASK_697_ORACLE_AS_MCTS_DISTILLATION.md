# TASK_697: Oracle-as-MCTS Distillation Pipeline

**Status**: ON HOLD (architectural shift — not compatible with current easy_trainer path)
**Priority**: 3 (strategic, not urgent)
**Depends on**: TASK_696 fixes applied, Foxglove baseline recovered
**Blocks**: Poppy Phase 2 transition

---

## Motivation

Foxglove (Phase 1) trains TRM with flat oracle targets — one-hot action labels and scalar value.
Poppy (Phase 2) needs TRM to work with MCTS — visit count distributions, backed-up Q-values, multi-step dynamics.

There is a **representation gap**: TRM trained on flat labels learns classification features, not search-compatible features. When real MCTS runs, the TRM's latent space isn't organized for tree exploration. The network must re-learn representations during MCTS self-play, wasting the Phase 1 investment.

**Core idea**: Disguise the oracle as MCTS during Phase 1. Train TRM on MCTS-shaped targets from day one. Then freeze the TRM and bootstrap real MCTS with a network that already "speaks search."

---

## Architecture Overview

```
Phase A: Oracle-as-MCTS → Train full MuZero (encoder + TRM + dynamics + heads)
Phase B: Freeze encoder+TRM → Train MCTS search policy with real simulations
Phase C: Unfreeze all → Fine-tune end-to-end with MCTS self-play
```

---

## Phase A: Oracle-as-MCTS Training

### A.1: Oracle → MCTS Target Mapping

The oracle multipass (TASK_673) already generates N rollouts with noise perturbation. This maps naturally to MCTS simulations:

| Oracle Concept | MCTS Equivalent | Shape |
|---|---|---|
| N oracle passes (noise_scale=0.03) | N MCTS simulations | scalar |
| Per-pass asset selections (top-k) | Visited actions per simulation | (N, k) |
| Per-pass returns (PnL) | Backed-up leaf values | (N,) |
| Aggregated selection frequency | Visit count distribution | (n_actions,) |
| Mean return across passes | Root value estimate | scalar |
| Per-step oracle decisions | Dynamics transition targets | (T, n_actions) |

**Concrete mapping** (pseudocode):

```python
def oracle_to_mcts_targets(oracle_passes, n_actions=320):
    """Convert N oracle passes into MCTS-shaped training targets.

    Args:
        oracle_passes: list of N OracleResult, each with:
            - selections: (T, k) asset indices selected at each step
            - actions: (T, k) action types per selection (0=sell, 1=hold, 2=buy, ...)
            - returns: (T,) per-step PnL
            - total_return: scalar episode return

    Returns:
        mcts_targets: dict with MCTS-compatible target shapes
    """
    N = len(oracle_passes)

    # 1. Visit count distribution (policy target)
    #    Count how often each (asset, action) pair was selected across passes
    visit_counts = jnp.zeros(n_actions)
    for p in oracle_passes:
        for asset, action in zip(p.selections, p.actions):
            action_idx = asset * 5 + action  # or asset * 3 + action for Poppy-compat
            visit_counts = visit_counts.at[action_idx].add(1)

    policy_target = visit_counts / visit_counts.sum()  # Normalize to distribution

    # 2. Root value (value target)
    root_value = jnp.mean(jnp.array([p.total_return for p in oracle_passes]))

    # 3. Q-values per action (for auxiliary loss)
    #    Average return of passes that selected each action
    q_values = jnp.zeros(n_actions)
    q_counts = jnp.zeros(n_actions)
    for p in oracle_passes:
        for asset, action in zip(p.selections, p.actions):
            idx = asset * 5 + action
            q_values = q_values.at[idx].add(p.total_return)
            q_counts = q_counts.at[idx].add(1)
    q_values = q_values / jnp.maximum(q_counts, 1)

    # 4. Per-step policy targets (for dynamics training)
    #    At each timestep t, what was the visit distribution?
    step_policies = []  # (T, n_actions)
    for t in range(T):
        step_counts = aggregate_at_step(oracle_passes, t)
        step_policies.append(step_counts / step_counts.sum())

    return {
        'policy_target': policy_target,       # (n_actions,) — matches Poppy format
        'value_target': root_value,           # scalar — matches Poppy format
        'q_values': q_values,                 # (n_actions,) — auxiliary
        'step_policies': jnp.stack(step_policies),  # (T, n_actions) — dynamics
        'step_values': per_step_values,       # (T,) — dynamics value targets
    }
```

### A.2: Loss Function (MuZero-compatible)

Train with Poppy's loss structure from day one, using oracle-generated targets:

```python
total_loss = (
    policy_weight * CE(policy_logits, oracle_visit_distribution)    # Visit counts, not one-hot
  + value_weight  * hybrid_value_loss(value_logits, oracle_root_value)  # Categorical + MSE
  + reward_weight * CE(reward_logits, oracle_step_reward)           # Per-step (dynamics!)
  + consistency_weight * ||encode(s_t) - dynamics(h_{t-1}, a_{t-1})||²
)
```

**Key difference from current Foxglove**: This trains the **dynamics model** (recurrent_inference) too. Current Foxglove only trains initial_inference. The oracle's multi-step trajectory provides natural dynamics targets.

### A.3: Oracle Pass Count Schedule

Start with many passes (high-quality targets), reduce over time (force network to generalize):

```
Epoch   0-50:   N=32 passes (rich visit distribution, low variance)
Epoch  50-150:  N=16 passes (moderate)
Epoch 150-300:  N=8  passes (sparser, more noise — mimics real MCTS budget)
```

This teaches the network to work with both rich and sparse search statistics — matching real MCTS conditions where simulation budget varies.

### A.4: Action Space Alignment

Current mismatch:
- Foxglove: 5 actions × 64 assets = 320 (Sell, Hold, Buy, LimitSell, LimitBuy)
- Poppy: 3 actions × 64 assets = 192 (Sell, Hold, Buy)

**Decision needed**: Either:
- **Option A**: Train with 320 actions, project to 192 at handoff (merge Limit→Market)
- **Option B**: Train directly with 192 from the start (discard Limit orders in Phase A)
- **Option C**: Keep 320 throughout (Poppy absorbs Limit orders)

Recommendation: **Option B** — simpler, no projection loss, and Limit orders aren't useful until live trading anyway.

---

## Phase B: Freeze TRM → Bootstrap MCTS

### B.1: What to Freeze

```
FROZEN (from Phase A):
├── Encoder (observation → latent_state)
├── TRM Reasoning (H×L cycles of refinement)
├── Value head (latent → categorical value)
└── Policy head (latent → action logits / prior)

TRAINABLE (new):
├── Dynamics model (latent + action → next_latent + reward)
│   └── Already pre-trained in Phase A via oracle step targets
│   └── Fine-tune with real MCTS rollout consistency
└── MCTS search hyperparameters (temperature, exploration constants)
```

### B.2: MCTS Bootstrap Protocol

1. Load Phase A checkpoint (encoder + TRM + dynamics + heads)
2. Freeze encoder, TRM, value head, policy head
3. Run MCTS self-play with real simulations (32 sims initially)
4. The policy prior from the frozen network is already MCTS-shaped — search starts effective immediately
5. Train dynamics only: minimize `||encode(s_t) - dynamics(h_{t-1}, a_{t-1})||²`
6. Log: root value accuracy, visit distribution entropy, action diversity

**Expected benefit**: MCTS converges much faster because:
- Prior policy already predicts visit-count-like distributions (not flat one-hots)
- Value estimates are calibrated to search-compatible ranges
- Dynamics model has a head start from Phase A oracle trajectories

### B.3: Freeze Duration

Monitor these metrics to decide when to unfreeze:
- `dynamics_consistency_loss < 0.01` — dynamics model has converged
- `mcts_policy_improvement > 5%` — search is adding value over prior
- `value_prediction_error < 0.1` — value head is still accurate

If value head degrades (distribution shift from self-play vs oracle), unfreeze early.

---

## Phase C: End-to-End Fine-Tuning

### C.1: Gradual Unfreeze Schedule

```
Step 1: Unfreeze dynamics (if not already)
Step 2: Unfreeze value head + policy head (representation still frozen)
Step 3: Unfreeze TRM reasoning (careful — large param count)
Step 4: Unfreeze encoder (last, most fragile)
```

Each unfreeze step: reduce LR by 10× for newly unfrozen params (discriminative LR).

### C.2: This IS Standard ExIt

At this point, the system is doing standard Expert Iteration:
1. MCTS generates improved policy targets (expert)
2. Network trains on MCTS targets (imitation)
3. Better network → better MCTS priors → better targets → loop

The difference: TRM starts the ExIt loop with representations already shaped for search, instead of being shaped for classification.

---

## Why This Is an Architectural Shift

This task is **NOT compatible with the current easy_trainer pipeline** because:

1. **Different loss structure**: Foxglove uses flat CE + value loss. This uses MuZero's full loss (policy CE + categorical value + reward + consistency). Would need to replace `_train_step_impl` entirely.

2. **Dynamics training**: Foxglove only trains `initial_inference`. This requires `recurrent_inference` training — needs unroll loop, reward targets, consistency loss. Essentially the Poppy trainer's loss function.

3. **Oracle interface change**: Current oracle returns (selections, returns). This needs multi-pass results preserved individually (not averaged), with per-step granularity.

4. **Action space**: Need to decide 320 vs 192 before starting.

5. **Buffer structure**: Current episode buffer stores (observations, oracle_targets). This needs (observations, actions, rewards, visit_distributions, values, dones) — Poppy's replay buffer format.

**Recommended approach**: Build this as a new trainer (e.g., `bridge_trainer.py`) that uses Foxglove's data pipeline but Poppy's loss function and target structure. It bridges Phase 1→2 by speaking both languages.

---

## Implementation Estimate

| Component | Effort | Notes |
|---|---|---|
| `oracle_to_mcts_targets()` | 1 day | Core mapping function, needs JIT compatibility |
| `bridge_trainer.py` | 3-5 days | New trainer combining Foxglove data + Poppy loss |
| Dynamics training in bridge | 2 days | Unroll loop, reward targets, consistency |
| Pass count schedule | 0.5 day | Config + annealing logic |
| Phase B freeze/unfreeze | 1 day | Parameter partitioning, discriminative LR |
| Phase C ExIt loop | 2-3 days | Self-play integration, replay buffer |
| Testing + validation | 2 days | Smoke tests, metric comparison |
| **Total** | **~2 weeks** | |

---

## Success Criteria

1. Phase A model matches or exceeds current Foxglove agree_trade within same epoch budget
2. Phase B MCTS converges to useful policy within 50 self-play episodes (vs hundreds without pre-training)
3. Phase C end-to-end improves over Phase B by measurable margin
4. Total wall-clock time (A+B+C) is less than training Poppy from scratch

---

## Open Questions

1. **Oracle pass count vs MCTS simulation count**: Oracle runs 10 passes. Typical MCTS runs 32-128 simulations. Should we scale oracle passes up, or accept the distribution is coarser?

2. **Temperature sharpening**: Oracle multipass already has `oracle_sharpen_temperature=0.1`. Should we apply different temperatures to create visit distributions with varying entropy (curriculum)?

3. **Reward scaling**: Oracle returns are raw log-returns. Poppy scales rewards by 100× for categorical bins. Need consistent scaling from Phase A.

4. **When to start**: This makes most sense after TASK_696 fixes are validated and Foxglove baseline is recovered. No point building on broken foundations.

---

*Created: 2026-02-23*
*Status: ON HOLD — requires TASK_696 fixes first, plus architectural design review*
*Concept: Oracle multipass as degenerate MCTS → TRM learns search-shaped representations → freeze → bootstrap real MCTS*
