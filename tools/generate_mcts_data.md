# generate_mcts_data.py

**MCTS Training Data Generator**

## Purpose

Generate MCTS training data by running self-play episodes.

## Usage

```bash
python tools/generate_mcts_data.py
```

## Function

1. Creates environment and model
2. Runs self-play episodes
3. Stores (observation, action, reward, value) tuples
4. Saves to replay buffer or file

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 100 | Number of episodes |
| `--steps` | 200 | Steps per episode |
| `--sims` | 32 | MCTS simulations |
| `--output` | data/ | Output directory |

## Output Format

```
data/
├── episode_0001.npz
├── episode_0002.npz
└── ...
```

Each file contains:
- `observations`: (steps, 27, 64, 64)
- `actions`: (steps,)
- `rewards`: (steps,)
- `values`: (steps,)

## Use Cases

- Pre-generate training data
- Offline analysis
- Debugging policies

## Dependencies

- mcts_trader.env.vectorbt_env
- mcts_trader.jax_agent

## Related Files

- `train_jax_fast.py` - Online training
