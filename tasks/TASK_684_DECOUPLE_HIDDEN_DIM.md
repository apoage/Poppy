# TASK_684: Decouple hidden_dim from state_dim in TRM

**Status**: DONE
**Priority**: 1 (prerequisite for TASK_683)
**Source**: Analysis 002 (Gemini) — "The Coupling Trap"

## Problem

`trm_flax.py` line 110 hardcodes:
```python
hidden_dim = self.state_dim * 16
```

This couples **representation bandwidth** (state_dim) with **compute capacity** (hidden_dim):
- state_dim=64 → hidden_dim=1024 (current, 14.6M params)
- state_dim=128 → hidden_dim=2048 (explosion to ~60M+ params)
- state_dim=256 → hidden_dim=4096 (impossible on 24GB)

We want wide state (128-256 dims per asset) with constrained compute (hidden_dim=1024-2048).

## Changes

### 1. `mcts_trader/jax_agent/trm_flax.py`

Add `hidden_dim` as class parameter to `TRMReasoning`:

```python
class TRMReasoning(nn.Module):
    state_dim: int = 64
    hidden_dim: int = None  # If None, defaults to state_dim * 16 (backward compat)
    ...

    def __call__(self, state, ...):
        hidden_dim = self.hidden_dim if self.hidden_dim is not None else self.state_dim * 16
```

Also update `TinyRecursiveTradingModel` to accept and pass `hidden_dim`.

### 2. `mcts_trader/jax_agent/muzero_model.py`

Add `hidden_dim` param to `MuZeroNetworks`, pass to TRM.

### 3. `easy_config.py`

`hidden_dim` field already exists (line 30) but is NOT used by TRM. Wire it through.

### 4. `easy_trainer.py`

Pass `hidden_dim` to `MuZeroNetworks`.

## Backward Compatibility

- `hidden_dim=None` defaults to old behavior (`state_dim * 16`)
- Existing checkpoints load fine (Flax ignores extra module params)
- Existing small/medium/large presets unchanged

## Verification

1. With `hidden_dim=None`: exact same param count as before (regression test)
2. With `hidden_dim=1024, state_dim=128`: verify param count is ~30M (not ~60M)
3. Run 5 epochs to confirm training works
