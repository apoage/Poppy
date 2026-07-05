# TASK 006: Document/Address TRM Weight Sharing Deviation from Paper

**Severity**: INFO (Architectural Decision)
**Location**: `mcts_trader/jax_agent/trm_flax.py:73-86`
**Status**: DONE (2025-12-29)
**Estimated Complexity**: Medium (if changing), Low (if documenting)

---

## Problem Description

The Tiny Recursive Model (TRM) implementation uses **UNIQUE WEIGHTS** for each refinement cycle, deviating from the original paper which suggests **SHARED WEIGHTS** across cycles.

### Current Implementation (`trm_flax.py:67-88`):
```python
for h in range(H_cycles):
    for l in range(L_cycles):
        combined = jnp.concatenate([state, latent], axis=-1)

        # UNIQUE weights per (h, l) combination
        latent_delta = nn.Dense(
            features=hidden_dim,
            name=f'latent_refiner_h{h}_l{l}_1'  # Different name = different weights!
        )(combined)
        # ...
```

### Paper Expectation (Conceptual):
```python
# Shared weights - same parameters reused across iterations
latent_refiner = nn.Dense(features=hidden_dim, name='latent_refiner')

for h in range(H_cycles):
    for l in range(L_cycles):
        latent_delta = latent_refiner(combined)  # Same weights every time
```

---

## Impact Analysis

### Current Approach (Unique Weights):

**Pros:**
- More expressivity (each cycle can learn different transformations)
- May capture progression from coarse to fine refinement
- Standard MLP behavior (predictable gradient flow)

**Cons:**
- More parameters (H×L more weight matrices)
- Loses "iterative computation" inductive bias
- May overfit on small datasets
- Deviates from paper's core insight

### Paper Approach (Shared Weights):

**Pros:**
- Fewer parameters (1 weight matrix reused)
- Enforces "thinking" as iteration, not depth
- Better generalization (weight sharing regularizes)
- True "recursive" computation

**Cons:**
- Less expressivity
- May need more cycles for same performance
- Gradient through many reuses can be noisy

---

## Parameter Count Comparison

For `state_dim=64`, `hidden_dim=1024`, `H_cycles=4`, `L_cycles=6`:

**Current (Unique Weights):**
- Per cycle: 2 Dense layers = (64+64)×1024 + 1024×64 = ~197K params
- Total: 24 cycles × 197K = **~4.7M params** (just for latent refiner)

**Shared Weights:**
- Single: 2 Dense layers = ~197K params
- Total: **~197K params** (regardless of cycles)

The current implementation has **24x more parameters** in the recursive block!

---

## Proposed Solutions

### Option A: Document as Intentional Deviation (Quick)

If the deviation is intentional and works well:

```python
class TRMReasoning(nn.Module):
    """
    Recursive reasoning block for policy and value.

    IMPLEMENTATION NOTE: This uses unique weights per (H, L) cycle,
    unlike the original TRM paper which uses shared weights. This
    increases parameter count but provides more expressivity for
    the trading domain. See TASK_006 for analysis.
    """
```

### Option B: Add Weight Sharing Option (Recommended)

Make it configurable to enable experimentation:

```python
class TRMReasoning(nn.Module):
    state_dim: int = 64
    share_weights: bool = False  # NEW: Enable paper-style weight sharing

    def setup(self):
        if self.share_weights:
            # Shared weights (paper style)
            self.latent_refiner_1 = nn.Dense(self.state_dim * 16)
            self.latent_refiner_ln = nn.LayerNorm()
            self.latent_refiner_2 = nn.Dense(self.state_dim)

            self.output_refiner_1 = nn.Dense(self.state_dim * 16)
            self.output_refiner_ln = nn.LayerNorm()
            self.output_refiner_2 = nn.Dense(self.n_actions)

    @nn.compact
    def __call__(self, state, H_cycles=4, L_cycles=6):
        for h in range(H_cycles):
            for l in range(L_cycles):
                if self.share_weights:
                    # Use shared layers
                    latent_delta = self.latent_refiner_1(combined)
                    latent_delta = self.latent_refiner_ln(latent_delta)
                    latent_delta = nn.gelu(latent_delta)
                    latent_delta = self.latent_refiner_2(latent_delta)
                else:
                    # Current behavior (unique weights)
                    latent_delta = nn.Dense(..., name=f'h{h}_l{l}')(combined)
```

### Option C: Full Paper Compliance (Major Refactor)

Implement TRM exactly as described in paper with proper residual structure.

---

## Implementation Plan (Option B)

### Step 1: Add `share_weights` Parameter (15 min)

```python
# trm_flax.py

class TRMReasoning(nn.Module):
    state_dim: int = 64
    n_actions: int = 192
    value_support_size: int = VALUE_SUPPORT_SIZE
    share_weights: bool = False  # NEW parameter
```

### Step 2: Implement Shared Weight Path (30 min)

Add `setup()` method for shared layers and conditional logic in `__call__()`.

### Step 3: Update Model Factory (10 min)

```python
# muzero_model.py

class MuZeroNetworks(nn.Module):
    share_trm_weights: bool = False  # Propagate to TRM

    def setup(self):
        self.trm = TinyRecursiveTradingModel(
            state_dim=self.state_dim,
            share_weights=self.share_trm_weights  # NEW
        )
```

### Step 4: Add CLI Flag (5 min)

```python
# train_jax_fast.py
parser.add_argument('--share-trm-weights', action='store_true',
                    help='Use shared weights in TRM (paper-compliant)')
```

### Step 5: Benchmark Comparison (1-2 hours)

Run experiments comparing:
- `--share-trm-weights` OFF (current, more params)
- `--share-trm-weights` ON (paper-style, fewer params)

---

## Acceptance Criteria

- [ ] Deviation documented in code comments
- [ ] Optional: `share_weights` parameter implemented
- [ ] Optional: CLI flag to toggle weight sharing
- [ ] Optional: Benchmark comparing both approaches

---

## Files to Modify

1. `mcts_trader/jax_agent/trm_flax.py` - Add weight sharing option
2. `mcts_trader/jax_agent/muzero_model.py` - Propagate parameter
3. `train_jax_fast.py` - Add CLI flag

---

## Research Reference

From "Tiny Recursive Models" paper:
> The key insight is that a single parameterized function f can be applied
> recursively to refine predictions, with weights shared across iterations.
> This provides an inductive bias toward iterative computation rather than
> simply adding depth.

The current implementation loses this inductive bias by using unique weights.

---

## Experiment Suggestion

Compare on synthetic Level 0 (simple, fast):

```bash
# Experiment A: Current (unique weights)
python train_jax_fast.py --synthetic-level 0 --epochs 100 --name trm_unique

# Experiment B: Shared weights
python train_jax_fast.py --synthetic-level 0 --epochs 100 --name trm_shared --share-trm-weights

# Compare:
# 1. Final reward
# 2. Training speed (shared should be faster)
# 3. Parameter count
# 4. Generalization to Level 1
```
