# test_model_init.py

**Model Initialization Test**

## Purpose

Test MuZero model initialization and verify parameter shapes.

## Usage

```bash
python tools/test_model_init.py
```

## Function

1. Creates MuZeroNetworks with default config
2. Initializes parameters
3. Verifies shapes for all components
4. Tests forward pass

## Components Tested

| Component | Purpose |
|-----------|---------|
| Encoder | Observation → Latent |
| Dynamics | Latent + Action → Next Latent |
| Prediction | Latent → Policy + Value |
| Representation | Initial latent extraction |

## Expected Output

```
Testing model initialization...
  Encoder: ✅ (27, 64, 64) → (64,)
  Dynamics: ✅ (64,) + action → (64,)
  Prediction: ✅ (64,) → policy(192) + value(1)
  Total params: 7.3M
All tests passed!
```

## Dependencies

- mcts_trader.jax_agent.muzero_model

## Related Files

- `test_trm_jax.py` - TRM-specific tests
- `test_mctx_jax.py` - Full MCTS test
