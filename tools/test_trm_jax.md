# test_trm_jax.py

**TRM Model Test**

## Purpose

Test TRM (Tiny Recursive Model) initialization and forward pass on JAX/GPU.

## Usage

```bash
python tools/test_trm_jax.py
```

## Function

1. Creates TRM model with default parameters
2. Initializes weights
3. Runs forward pass with dummy input
4. Verifies output shapes

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_dim` | 64 | Latent state dimension |
| `H_cycles` | 4 | Output refinement cycles |
| `L_cycles` | 6 | Latent refinement cycles |

## Expected Output

```
Testing TRM JAX implementation...
Creating model: state_dim=64, H_cycles=4, L_cycles=6
Initializing parameters...
  Param count: 7.3M
Running forward pass...
  Input: (4, 27, 64, 64)
  Output: latent (4, 64), policy (4, 192), value (4, 1)
✅ All tests passed
```

## Dependencies

- mcts_trader.jax_agent.muzero_model

## Related Files

- `test_mctx_jax.py` - Test full MCTS search
- `diagnose_jax_training.py` - Full diagnostics
