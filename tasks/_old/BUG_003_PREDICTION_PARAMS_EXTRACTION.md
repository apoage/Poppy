# Bug: Prediction Model Parameters Incorrectly Extracted

## Priority: CRITICAL
**Status:** FALSE POSITIVE - CODE IS CORRECT

## Analysis (2026-01-01)

Code review shows the implementation is correct:
1. `save_checkpoint()` saves `self.state` (includes all `params`)
2. `load_checkpoint()` restores the full TrainState
3. At line 1017: `trainer.state.params['price_predictor']` correctly extracts prediction params

The checkpoint save/load pipeline properly preserves prediction model parameters.
**Component:** `train_jax_fast.py`

## Problem Description

When loading checkpoints, the prediction model parameters are extracted from the wrong location in the parameter tree. The code assumes `pred_model_params` exists at the top level, but the actual structure has prediction params nested under `params['price_predictor']`.

### Location

```python
# train_jax_fast.py:1017-1024
if 'pred_model_params' in restored:
    pred_params = restored['pred_model_params']
    # ... tries to use pred_params directly
```

### Actual Parameter Structure

```python
# Checkpointed structure:
{
    'params': {
        'encoder': {...},
        'TRMReasoning': {...},
        'value_head': {...},
        'price_predictor': {      # <-- Prediction params are HERE
            'physics_core': {...},
            'neural_corrector': {...},
            'integration': {...}
        }
    },
    'opt_state': {...}
}
```

### Impact

- Prediction model never loads from checkpoint
- Physics parameters reset to random initialization
- Learned market dynamics are lost on resume
- Training continuity broken for prediction head

## Solution

Extract prediction params from the correct nested location:

```python
# train_jax_fast.py - checkpoint loading
if args.use_prediction_loss:
    # Prediction params are nested under main params
    main_params = restored.get('params', restored.get('model', {}))
    if 'price_predictor' in main_params:
        pred_params = main_params['price_predictor']
        print(f"Loaded prediction model from checkpoint")
    else:
        print(f"Warning: No prediction params in checkpoint, using random init")
```

## Files to Modify

| File | Change |
|------|--------|
| `train_jax_fast.py:1017-1024` | Fix parameter extraction path |

## Testing

1. Train with `--use-prediction-loss` for 100 epochs
2. Save checkpoint
3. Load checkpoint and verify prediction params match
4. Check physics parameters (alpha, beta, sigma) are preserved

## Verification Script

```python
import orbax.checkpoint as ocp

ckpt = ocp.StandardCheckpointer()
restored = ckpt.restore('/path/to/checkpoint')

# Check structure
print(restored.keys())
print(restored['params'].keys())
if 'price_predictor' in restored['params']:
    print("Found prediction params at params['price_predictor']")
    print(restored['params']['price_predictor'].keys())
```
