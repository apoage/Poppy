# Bug: State Dimension Hardcoded in Multiple Places

## Priority: MEDIUM
**Status:** DEFERRED

Design improvement - code works correctly with current structure.
Model sizes are passed via CLI `--model-size` which sets state_dim.
**Component:** Multiple files

## Problem Description

The state dimension (64/96/160) is hardcoded in multiple locations instead of being passed consistently from configuration.

### Locations

```python
# Various files may have:
state_dim = 64  # Hardcoded instead of from config

# Should be:
state_dim = config.state_dim  # From configuration
```

### Impact

- Changing model size requires edits in multiple files
- Easy to miss a hardcoded value
- Inconsistent behavior between components

## Solution

### Step 1: Audit all state_dim references

```bash
grep -rn "state_dim" mcts_trader/
grep -rn "= 64" mcts_trader/  # Look for magic numbers
```

### Step 2: Centralize configuration

```python
# config.py or similar
@dataclass
class ModelConfig:
    state_dim: int = 64

# All components receive config
class MuZeroNetwork(nn.Module):
    config: ModelConfig

    def setup(self):
        self.state_dim = self.config.state_dim
```

## Files to Modify

| File | Change |
|------|--------|
| Multiple | Replace hardcoded dimensions with config values |

## Testing

1. Train with different model sizes
2. Verify all components use correct dimension
3. No shape mismatch errors
