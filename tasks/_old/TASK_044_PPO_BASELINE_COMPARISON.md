# TASK_044: PPO Baseline for Algorithm Comparison

## Overview

Implement a standalone PPO (Proximal Policy Optimization) trading agent as a **lightweight comparison baseline** against the main MuZero+MCTS system. This lives in its own directory with zero dependencies on the main codebase.

## Motivation

1. **Speed Comparison**: PPO has no MCTS overhead - how much faster?
2. **Performance Baseline**: Does MuZero's planning actually help on simple patterns?
3. **Architecture Ablation**: Same TRM encoder vs simpler MLP - what matters?
4. **Debugging Reference**: Simpler algo = easier to debug training dynamics

## Requirements

- **Completely isolated** - own folder, own scripts, no imports from `mcts_trader/`
- **Reuse synthetic data generators** - can import from `mcts_trader/data/` OR copy them
- **Same evaluation protocol** - comparable metrics (reward, alpha, Sharpe, action dist)
- **JAX-native** - leverage existing JAX/Flax expertise, GPU acceleration

## Directory Structure

```
ppo_baseline/
├── README.md                    # Usage instructions
├── train_ppo.py                 # Main training script (standalone)
├── evaluate_ppo.py              # Evaluation script
├── ppo/
│   ├── __init__.py
│   ├── networks.py              # Actor-Critic networks (MLP or mini-TRM)
│   ├── agent.py                 # PPO agent (GAE, clipped loss)
│   ├── buffer.py                # Rollout buffer (simpler than MuZero replay)
│   └── utils.py                 # Helpers (symlog, normalization)
├── env/
│   ├── __init__.py
│   ├── trading_env.py           # Simplified trading env (copy or wrap)
│   └── synthetic.py             # Synthetic data generators (copy from main)
├── configs/
│   └── default.yaml             # Hyperparameters
├── checkpoints/                 # Saved models
└── logs/                        # Training logs + CSV
```

## Implementation Plan

### Phase 1: Environment Setup (Copy & Simplify)

```python
# env/trading_env.py
# Simplified version of VectorBTTradingEnv
# - Same state space (27×64×64 or flattened)
# - Same action space (5 actions × 64 assets = 320)
# - Same reward (symlog PnL)
# - Remove: limit orders complexity, TP/SL, multi-action points
```

### Phase 2: PPO Networks

```python
# ppo/networks.py
import jax.numpy as jnp
import flax.linen as nn

class ActorCritic(nn.Module):
    """Combined actor-critic for PPO."""
    action_dim: int = 320
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        # Option A: Simple MLP
        x = x.reshape((x.shape[0], -1))  # Flatten 27×64×64
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Actor head (policy)
        policy_logits = nn.Dense(self.action_dim)(x)

        # Critic head (value)
        value = nn.Dense(1)(x)

        return policy_logits, value.squeeze(-1)

class TRMActorCritic(nn.Module):
    """Actor-critic with TRM-style encoder for fair comparison."""
    action_dim: int = 320
    state_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # CNN encoder (same as TRM)
        x = nn.Conv(64, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.Conv(128, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (8, 8), (8, 8))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.state_dim)(x)

        # Simple heads (no recursive reasoning)
        policy_logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)

        return policy_logits, value.squeeze(-1)
```

### Phase 3: PPO Agent

```python
# ppo/agent.py
import jax
import jax.numpy as jnp
import optax

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return jnp.array(advantages)

def ppo_loss(params, apply_fn, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO clipped surrogate loss."""
    obs, actions, old_log_probs, advantages, returns = batch

    policy_logits, values = apply_fn({'params': params}, obs)

    # Policy loss (clipped)
    log_probs = jax.nn.log_softmax(policy_logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()

    ratio = jnp.exp(action_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # Value loss
    value_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
    probs = jax.nn.softmax(policy_logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    return total_loss, {'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': entropy}
```

### Phase 4: Training Loop

```python
# train_ppo.py
"""
Standalone PPO training for trading - comparison baseline.

Usage:
    python ppo_baseline/train_ppo.py --synthetic-level 0 --epochs 500

Compare with MuZero:
    python train_jax_fast.py --synthetic-level 0 --epochs 500
"""

import argparse
import jax
import jax.numpy as jnp
from ppo.networks import ActorCritic, TRMActorCritic
from ppo.agent import compute_gae, ppo_loss
from ppo.buffer import RolloutBuffer
from env.trading_env import TradingEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-level', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--parallel-envs', type=int, default=50)
    parser.add_argument('--network', choices=['mlp', 'trm'], default='mlp')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--name', type=str, default='ppo_baseline')
    args = parser.parse_args()

    # Create env
    env = TradingEnv(
        synthetic_level=args.synthetic_level,
        n_envs=args.parallel_envs,
        max_steps=args.steps,
    )

    # Create network
    if args.network == 'mlp':
        model = ActorCritic(action_dim=320)
    else:
        model = TRMActorCritic(action_dim=320)

    # Initialize
    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.ones((1, 27, 64, 64))
    params = model.init(rng, dummy_obs)['params']

    # Optimizer
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(args.epochs):
        # Collect rollouts
        rollout = collect_rollout(env, model, params, args.steps)

        # Compute advantages
        advantages = compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            gamma=args.gamma,
            lam=args.gae_lambda
        )
        returns = advantages + rollout['values']

        # PPO update (multiple epochs on same data)
        for _ in range(4):
            loss, metrics = train_step(params, opt_state, rollout, advantages, returns)

        # Log
        print(f"Epoch {epoch} | Reward: {rollout['total_reward']:.2f} | "
              f"Loss: {loss:.4f} | Entropy: {metrics['entropy']:.2f}")

if __name__ == '__main__':
    main()
```

### Phase 5: Evaluation & Comparison

```python
# evaluate_ppo.py
"""Compare PPO vs MuZero on same environments."""

def compare_algorithms():
    results = {}

    for level in [0, 1, 2, 3, 4]:
        # Load PPO checkpoint
        ppo_reward = evaluate_ppo(level)

        # Load MuZero checkpoint (if available)
        muzero_reward = evaluate_muzero(level)  # External call

        results[level] = {
            'ppo': ppo_reward,
            'muzero': muzero_reward,
            'speedup': ppo_time / muzero_time,
        }

    return results
```

## Key Hyperparameters

| Parameter | PPO Default | Notes |
|-----------|-------------|-------|
| Learning rate | 3e-4 | Standard PPO |
| Clip epsilon | 0.2 | Clipping range |
| GAE lambda | 0.95 | Advantage estimation |
| Discount (gamma) | 0.99 | Same as MuZero's 0.997 |
| Entropy coef | 0.01 | Same as MuZero |
| Value coef | 0.5 | Standard PPO |
| PPO epochs | 4 | Updates per rollout |
| Batch size | 64 | Mini-batch for updates |

## Expected Outcomes

### Speed
- PPO: ~5-10x faster per epoch (no MCTS)
- MuZero: More compute but potentially better sample efficiency

### Performance by Level
| Level | PPO Expected | MuZero Expected | Winner |
|-------|--------------|-----------------|--------|
| 0 (sine) | Good | Good | Tie |
| 1 (lagged) | Okay | Better | MuZero |
| 2 (dip) | Okay | Better | MuZero |
| 3 (trends) | Good | Good | Tie |
| 4 (rotation) | Struggle | Better | MuZero |

Hypothesis: MuZero's planning should help most on levels requiring multi-step reasoning (1, 2, 4).

## Success Criteria

1. **Working PPO** - Trains without NaN, learns on Level 0
2. **Fair Comparison** - Same env, same metrics, same compute budget
3. **Clear Insights** - Document where MuZero wins/loses vs PPO
4. **Speed Benchmark** - Measure wall-clock time per epoch

## Files to Create

- [ ] `ppo_baseline/README.md`
- [ ] `ppo_baseline/train_ppo.py`
- [ ] `ppo_baseline/evaluate_ppo.py`
- [ ] `ppo_baseline/ppo/__init__.py`
- [ ] `ppo_baseline/ppo/networks.py`
- [ ] `ppo_baseline/ppo/agent.py`
- [ ] `ppo_baseline/ppo/buffer.py`
- [ ] `ppo_baseline/ppo/utils.py`
- [ ] `ppo_baseline/env/__init__.py`
- [ ] `ppo_baseline/env/trading_env.py`
- [ ] `ppo_baseline/env/synthetic.py`
- [ ] `ppo_baseline/configs/default.yaml`

## Implementation Notes

### What to Copy from Main Codebase
1. Synthetic data generators (`mcts_trader/data/synthetic_market.py`)
2. Symlog transform
3. State encoder structure (if using TRM variant)

### What to Simplify
1. No replay buffer (PPO uses rollout buffer, discards after use)
2. No MCTS
3. No categorical value representation (just scalar)
4. No PopArt (optional, can add later)
5. No prediction head
6. Simpler checkpointing

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)

## Status

- [ ] Phase 1: Environment setup
- [ ] Phase 2: Networks
- [ ] Phase 3: PPO agent
- [ ] Phase 4: Training loop
- [ ] Phase 5: Comparison benchmarks
