# test_episode_length.py

**Episode Length Configuration Test**

## Purpose

Test different episode length configurations to verify environment behavior.

## Usage

```bash
python tools/test_episode_length.py
```

## Function

1. Creates environment with different step limits
2. Runs test episodes
3. Verifies termination at correct step
4. Checks observation consistency

## Tests

| Test | Description |
|------|-------------|
| Short episode | 100 steps |
| Standard episode | 500 steps |
| Long episode | 2000 steps |
| Edge cases | 1 step, max steps |

## Expected Output

```
Testing episode lengths...
  100 steps: ✅ Terminated at step 100
  500 steps: ✅ Terminated at step 500
  2000 steps: ✅ Terminated at step 2000
All tests passed!
```

## Dependencies

- mcts_trader.env.vectorbt_env

## Related Files

- `test_env_init.py` - Environment init test
