# test_env_init.py

**Environment Initialization Test**

## Purpose

Test VectorBT trading environment initialization and basic operations.

## Usage

```bash
python tools/test_env_init.py
```

## Function

1. Creates VectorBTTradingEnv
2. Tests reset()
3. Tests step() with sample actions
4. Verifies observation and reward shapes

## Tests

| Test | Verification |
|------|--------------|
| Init | No exceptions |
| Reset | Returns valid obs dict |
| Step | Returns (obs, reward, term, trunc, info) |
| Obs shape | (27, 64, 64) |
| Action mask | (192,) boolean |

## Expected Output

```
Testing VectorBT environment...
  Init: ✅
  Reset: ✅ obs_shape=(27, 64, 64)
  Step: ✅ reward=-0.02, done=False
  Action mask: ✅ 192 actions, 128 valid
All tests passed!
```

## Dependencies

- mcts_trader.env.vectorbt_env

## Related Files

- `test_episode_length.py` - Episode tests
- `debug_env.py` - Old environment (deprecated)
