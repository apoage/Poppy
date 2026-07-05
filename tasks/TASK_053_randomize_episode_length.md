# TASK_053: Randomize Episode Length

## Status: COMPLETED (2026-01-23)

**Implementation**: `--episode-variance` CLI flag
- Located in: `cli.py`, `config.py`, `setup.py`, `env_factory.py`, `jax_trading_env.py`
- Usage: `--episode-variance 50` adds ±50 steps variance
## Priority: MEDIUM

## Problem

Fixed episode length (400 steps) allows end-of-episode exploitation:
- Agent piles into positions at step 395
- No need to exit - mark-to-market values count
- Real trading has no "episode end"

## Solution

```python
# train_jax_fast.py or env config
episode_length_base = 400
episode_length_variance = 50  # ±50 steps

actual_length = episode_length_base + np.random.randint(-episode_length_variance, episode_length_variance)

# Or use geometric distribution for truly unknown end
p_terminate = 0.003  # ~333 step average
done = done | (np.random.random() < p_terminate)
```

## CLI Arguments

```python
parser.add_argument('--episode-length-variance', type=int, default=50,
                   help='Random variance in episode length (0=fixed)')
parser.add_argument('--stochastic-termination', type=float, default=0.0,
                   help='Per-step termination probability (0=disabled)')
```

## Files

- `mcts_trader/jax_agent/jax_trading_env.py:1054` (done condition)
- `train_jax_fast.py` (episode loop)

## Testing

```python
def test_episode_lengths_vary():
    """Episodes should have different lengths."""
    lengths = [run_episode_and_get_length(env) for _ in range(100)]
    assert len(set(lengths)) > 10  # At least 10 different lengths
```
