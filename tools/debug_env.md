# debug_env.py

**DEPRECATED**

## Status

This tool is **deprecated**. It uses the old `TradingEnv` which has been replaced by `VectorBTTradingEnv`.

## Replacement

Use `test_env_init.py` instead for environment testing.

## Original Purpose

Test the old TradingEnv initialization and step operations.

## Migration

```bash
# Old (deprecated)
python tools/debug_env.py

# New
python tools/test_env_init.py
```

## Related Files

- `test_env_init.py` - New environment test
- `mcts_trader/env/vectorbt_env.py` - Current environment
