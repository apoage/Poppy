# TASK_666: Easy Mode Branch - Expert Iteration Architecture

**Date**: 2026-02-02
**Status**: PLANNING
**Priority**: HIGH - Path to working system
**Codename**: Order 666 (Execute the complexity)

---

## Overview

Branch codebase to implement **Expert Iteration (ExIt)** architecture:
- Phase 1: Supervised learning from Oracle (The Cheat)
- Phase 2: Frozen dynamics model (The Dream)
- Phase 3: Inference-time MCTS only (The Thought)

**Goal**: Working trading bot in 4 weeks, not 4 months.

---

## Branch Strategy

```bash
# Create branch from current state
git checkout -b easy-mode
# or with your backup system - snapshot current state as "hard-mode-backup"
```

**Keep main branch intact** - research codebase preserved for future.

---

## Phase 1: The Strip Down (Week 1)

### 1.1 Create Simplified Trainer

**New file**: `easy_trainer.py`

```python
class EasyTrainer:
    """Supervised-only trainer. No RL, no MCTS during training."""

    def __init__(self, model, config):
        self.model = model  # TRM encoder + policy/value heads
        self.optimizer = optax.adam(config.lr)

    def train_step(self, batch):
        """Pure supervised learning."""
        obs = batch['observations']
        oracle_actions = batch['optimal_actions']
        oracle_returns = batch['returns']

        # Forward
        policy_logits, value = self.model(obs)

        # Losses (stable, no bootstrapping)
        policy_loss = cross_entropy(policy_logits, oracle_actions)
        value_loss = mse(value, oracle_returns)

        return policy_loss + value_loss
```

### 1.2 Create Easy Mode Entry Point

**New file**: `train_easy.py`

```python
"""
Easy Mode Training - Expert Iteration

Usage:
  python train_easy.py --phase 1 --epochs 50  # Supervised
  python train_easy.py --phase 2 --epochs 20  # Dynamics (optional)
  python trade_easy.py --sims 50              # Inference with MCTS
"""
```

### 1.3 Strip These Components

| Component | Action | Reason |
|-----------|--------|--------|
| `trainer.py` MuZero loss | Remove | No online RL |
| Replay buffer PER | Simplify | No priority updates needed |
| PopArt | Remove | Oracle returns are stable |
| BC annealing | Remove | BC is 100% now |
| `_loss_fn` complexity | Gut it | Just CE + MSE |
| MCTS during training | Remove | Inference only |
| Prediction loss | Remove | Separate Phase 2 |

### 1.4 Keep These Components

| Component | Why |
|-----------|-----|
| `trm_flax.py` | Encoder architecture |
| `muzero_model.py` | Policy/value heads |
| `path_solver.py` | Oracle |
| `expert_buffer.py` | Batching (simplified) |
| `jax_encoder.py` | Observation encoding |
| Evaluation code | Need to measure performance |

---

## Phase 2: Oracle Data Pipeline (Week 1-2)

### 2.1 Bulk Oracle Generation

```python
# generate_oracle_data.py
"""Generate oracle labels for entire dataset upfront."""

def generate_oracle_dataset(price_data, output_path):
    """
    For each episode:
    1. Run path_solver to get optimal actions
    2. Compute realized returns
    3. Save (obs, action, return) tuples
    """
    for episode in price_data:
        optimal_path = compute_optimal_path(episode.log_returns)

        for step in range(len(episode)):
            save_sample(
                observation=encode(episode, step),
                action=optimal_path.actions[step],
                return_=optimal_path.returns[step],
            )
```

### 2.2 Data Sources

| Source | Episodes | Quality |
|--------|----------|---------|
| Historical (real) | ~1000 | High |
| Bella synthetic | ~10000 | Medium |
| L0-L5 curriculum | ~5000 | Training wheels |

### 2.3 Pre-encode Observations

From TASK_136a - encode observations once, store compressed:

```python
# Store (27, 64) not (27, 64, 64) - 64x smaller
oracle_dataset = {
    'observations': np.array(...),  # (N, 27, 64) compressed
    'actions': np.array(...),       # (N,) int
    'returns': np.array(...),       # (N,) float
}
np.savez('oracle_data.npz', **oracle_dataset)
```

---

## Phase 3: Supervised Training (Week 2)

### 3.1 Training Loop

```python
def train_phase1(config):
    model = create_model(config)
    trainer = EasyTrainer(model, config)
    dataset = load_oracle_dataset('oracle_data.npz')

    for epoch in range(config.epochs):
        for batch in dataset.batches(config.batch_size):
            loss = trainer.train_step(batch)

        # Evaluate against oracle agreement
        agreement = evaluate_oracle_agreement(model, val_set)
        print(f"Epoch {epoch}: loss={loss:.4f}, agreement={agreement:.1%}")

        if agreement > 0.85:
            print("Phase 1 complete - 85% oracle agreement")
            break

    save_checkpoint(model, 'phase1.ckpt')
```

### 3.2 Success Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Oracle agreement | >85% | Policy matches oracle |
| Value MSE | <0.1 | Return prediction accurate |
| Training loss | Converged | Not overfitting |

---

## Phase 4: Dynamics Model (Week 3, Optional)

### 4.1 Option A: Use Bella

If Bella (diff_teacher) is already trained:

```python
# Just load it
dynamics_model = load_bella_checkpoint('bella.ckpt')
dynamics_model.freeze()  # No gradients
```

### 4.2 Option B: Train Simple Dynamics

```python
class SimpleDynamics:
    """Predict next latent state from current state + action."""

    def __init__(self, state_dim):
        self.transition = nn.Dense(state_dim)
        self.reward_head = nn.Dense(1)

    def __call__(self, state, action):
        action_emb = self.action_embed(action)
        next_state = self.transition(concat(state, action_emb))
        reward = self.reward_head(next_state)
        return next_state, reward
```

### 4.3 Training

```python
def train_phase2(encoder, config):
    """Train dynamics with frozen encoder."""
    encoder.freeze()
    dynamics = SimpleDynamics(config.state_dim)

    for batch in dataset:
        state = encoder(batch['obs_t'])
        next_state_true = encoder(batch['obs_t+1'])

        next_state_pred, reward_pred = dynamics(state, batch['action'])

        loss = mse(next_state_pred, next_state_true) + mse(reward_pred, batch['reward'])
```

---

## Phase 5: Inference MCTS (Week 4)

### 5.1 Wire Up mctx

```python
def trade_with_mcts(model, dynamics, observation, config):
    """Inference-time search - no gradients."""

    # Get policy prior from Phase 1 model
    state = model.encode(observation)
    policy_logits, value = model.predict(state)

    # MCTS search using Phase 2 dynamics
    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=value,
        embedding=state,
    )

    policy_output = mctx.muzero_policy(
        params=dynamics.params,
        rng_key=rng,
        root=root,
        recurrent_fn=dynamics.recurrent_fn,
        num_simulations=config.sims,  # 50-100 on 3090
    )

    return policy_output.action
```

### 5.2 Inference Script

**New file**: `trade_easy.py`

```python
"""
Easy Mode Trading - Inference with MCTS

Usage:
  python trade_easy.py --checkpoint phase1.ckpt --sims 50
  python trade_easy.py --checkpoint phase1.ckpt --sims 0  # No MCTS baseline
"""

def main():
    model = load_checkpoint(args.checkpoint)
    dynamics = load_dynamics(args.dynamics_checkpoint)

    for observation in market_stream:
        if args.sims > 0:
            action = trade_with_mcts(model, dynamics, observation, args)
        else:
            action = model.predict(observation).argmax()  # Pure intuition

        execute_trade(action)
```

---

## File Structure After Branch

```
easy-mode/
├── train_easy.py           # Main entry point
├── trade_easy.py           # Inference with optional MCTS
├── generate_oracle_data.py # Bulk oracle generation
├── easy_trainer.py         # Simplified supervised trainer
├── easy_config.py          # Minimal config
│
├── mcts_trader/
│   ├── jax_agent/
│   │   ├── trm_flax.py         # KEEP - encoder
│   │   ├── muzero_model.py     # KEEP - heads only
│   │   ├── path_solver.py      # KEEP - oracle
│   │   ├── expert_buffer.py    # SIMPLIFY - just batching
│   │   ├── trainer.py          # REMOVE - replaced by easy_trainer
│   │   └── ...                 # Remove most complexity
│   └── env/
│       └── jax_encoder.py      # KEEP
│
├── diff_teacher/               # KEEP - Bella for Phase 2
│
└── checkpoints/
    ├── phase1.ckpt            # Supervised policy/value
    └── phase2.ckpt            # Frozen dynamics
```

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Strip down | `easy_trainer.py`, `train_easy.py` working |
| 2 | Data + Train | Oracle dataset, Phase 1 trained to 85%+ |
| 3 | Dynamics | Phase 2 trained or Bella integrated |
| 4 | MCTS | `trade_easy.py` with inference search |

---

## Success Metrics

### Phase 1 Complete When:
- [ ] Oracle agreement > 85%
- [ ] Value prediction MSE < 0.1
- [ ] Pure intuition beats random baseline

### Phase 2 Complete When:
- [ ] Next-state prediction loss converged
- [ ] Reward prediction reasonable
- [ ] Or: Bella works as drop-in

### Phase 3 Complete When:
- [ ] MCTS runs at inference (50 sims)
- [ ] MCTS improves over pure intuition (even slightly)
- [ ] Can trade on paper account

---

## Rollback Plan

If Easy Mode fails or you want to return to Hard Mode:

```bash
# Your research codebase is intact
git checkout main
# or restore from differential backup
```

Nothing is lost. Easy Mode is a branch, not a replacement.

---

## Why "Order 666"

Because we're executing the complexity. The intricate MuZero machinery served its purpose (learning), but now we strip it down to ship something that works.

The Jedi (complex RL) will return when we understand markets better.

First: prove the oracle is worth imitating.
