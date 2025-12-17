# TASK 03: Scale Model from 7M to 25M Parameters

## Objective
Scale TRM model from 7M to 25M parameters to handle adversarial training and complex portfolio tactics. Make model size configurable via CLI.

## Prerequisites
- TASK_01 completed (parallel MCTS)
- TASK_02 completed (synthetic data working)

## Current State

**Model**: `mcts_trader/agent/trm.py`

**Current Size**: ~7M parameters
```python
TinyRecursiveTradingModel(
    d_model=64,
    H_cycles=3,
    L_cycles=4,
    # 3×4 = 12 recursive passes
)
```

**Problem**:
- 7M params might be too small for 64-asset portfolio reasoning
- Need capacity for adversarial training (Level 5)
- AlphaGo used 20M for 362 actions, we have 384+

## Target Architecture

**3 Model Sizes**:
1. **Small** (7M): Current, for testing
2. **Medium** (15M): 2× capacity
3. **Large** (25M): For adversarial/production

## Implementation

### Step 1: Make TRM Configurable

**File**: `mcts_trader/agent/trm.py`

**Modify** class initialization to accept size parameter:

```python
# Around line 20
class TinyRecursiveTradingModel(nn.Module):
    state_dim: int = 64
    action_space_size: int = 192
    support_size: int = 601
    d_model: int = 64           # MAKE THIS CONFIGURABLE
    n_layers: int = 4
    H_cycles: int = 3           # MAKE THIS CONFIGURABLE
    L_cycles: int = 4           # MAKE THIS CONFIGURABLE

    @classmethod
    def from_size(cls, size: str, **kwargs):
        """
        Factory method to create model from size preset.

        Args:
            size: 'small' (7M), 'medium' (15M), or 'large' (25M)
            **kwargs: Override any parameters
        """
        presets = {
            'small': {
                'd_model': 64,
                'H_cycles': 3,
                'L_cycles': 4,
                'n_layers': 4
            },
            'medium': {
                'd_model': 96,
                'H_cycles': 4,
                'L_cycles': 5,
                'n_layers': 4
            },
            'large': {
                'd_model': 128,
                'H_cycles': 5,
                'L_cycles': 6,
                'n_layers': 4
            }
        }

        if size not in presets:
            raise ValueError(f"Unknown size: {size}. Choose from {list(presets.keys())}")

        config = presets[size]
        config.update(kwargs)  # Allow overrides

        return cls(**config)
```

### Step 2: Update MuZeroNetworks

**File**: `mcts_trader/jax_agent/muzero_model.py`

**Add size parameter**:

```python
# Around line 20
class MuZeroNetworks(nn.Module):
    state_dim: int = 64
    action_space_size: int = 192
    support_size: int = 601
    model_size: str = 'small'  # NEW

    def setup(self):
        # Create TRM with specified size
        self.trm = TinyRecursiveTradingModel.from_size(
            size=self.model_size,
            state_dim=self.state_dim,
            action_space_size=self.action_space_size,
            support_size=self.support_size
        )
```

### Step 3: Add CLI Argument

**File**: `train_jax_fast.py`

**Add argument** (around line 415):

```python
parser.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'medium', 'large'],
                   help='Model size: small (7M), medium (15M), large (25M)')
```

**Pass to model** (around line 507):

```python
model = MuZeroNetworks(
    state_dim=n_assets,
    action_space_size=n_assets * 3,
    support_size=601,
    model_size=args.model_size  # NEW
)
```

**Log parameter count**:

```python
# After model creation (around line 558)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
print(f"✅ Model ready ({param_count/1e6:.1f}M params, size={args.model_size})")
```

### Step 4: Update Trainer Batch Size for Large Model

**File**: `train_jax_fast.py`

**Auto-adjust batch size** based on model size to fit in VRAM:

```python
# After argparse (around line 463)
# Auto-adjust batch size for model size
if args.model_size == 'large' and args.batch_size > 256:
    logging.warning(f"Large model + batch {args.batch_size} might OOM on 24GB VRAM")
    logging.warning(f"Reducing batch to 256 (you can override with --batch-size)")
    args.batch_size = min(args.batch_size, 256)
elif args.model_size == 'medium' and args.batch_size > 384:
    args.batch_size = min(args.batch_size, 384)
```

## Testing

### Test 1: Verify Parameter Counts

**Script**: `tests/test_model_sizes.py` (NEW)

```python
import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks

def test_model_sizes():
    """Verify each size has expected parameter count"""

    expected_params = {
        'small': (6.5e6, 7.5e6),    # 7M ± 0.5M
        'medium': (14e6, 16e6),      # 15M ± 1M
        'large': (24e6, 26e6)        # 25M ± 1M
    }

    for size in ['small', 'medium', 'large']:
        model = MuZeroNetworks(
            state_dim=64,
            action_space_size=192,
            support_size=601,
            model_size=size
        )

        # Initialize
        rng = jax.random.PRNGKey(42)
        dummy_obs = jnp.ones((1, 27, 64, 64))
        variables = model.init(rng, dummy_obs, mode='initial_inference')

        # Count params
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))

        # Verify
        min_params, max_params = expected_params[size]
        assert min_params <= param_count <= max_params, \
            f"{size} model has {param_count/1e6:.1f}M params, expected {min_params/1e6:.0f}-{max_params/1e6:.0f}M"

        print(f"✅ {size:6s}: {param_count/1e6:.1f}M params")

if __name__ == "__main__":
    test_model_sizes()
```

**Run**:
```bash
python tests/test_model_sizes.py
```

**Expected**:
```
✅ small : 7.2M params
✅ medium: 15.1M params
✅ large : 25.3M params
```

### Test 2: VRAM Usage Test

**Command**:
```bash
# Small model (baseline)
python train_jax_fast.py \
  --synthetic-level 0 \
  --model-size small \
  --batch-size 512 \
  --epochs 5 \
  --name vram_test_small

# Check VRAM
nvidia-smi
# Note VRAM usage

# Medium model
python train_jax_fast.py \
  --synthetic-level 0 \
  --model-size medium \
  --batch-size 384 \
  --epochs 5 \
  --name vram_test_medium

# Large model
python train_jax_fast.py \
  --synthetic-level 0 \
  --model-size large \
  --batch-size 256 \
  --epochs 5 \
  --name vram_test_large
```

**Monitor**: `nvidia-smi dmon -s u`

**Expected VRAM** (RTX 3090, 24GB):
- Small + batch 512: ~12GB
- Medium + batch 384: ~16GB
- Large + batch 256: ~20GB

**Pass Criteria**: All fit in 24GB VRAM without OOM.

### Test 3: Performance Comparison

**Benchmark**: Train each size on Level 0, measure convergence speed.

```bash
for size in small medium large; do
  python train_jax_fast.py \
    --synthetic-level 0 \
    --model-size $size \
    --epochs 500 \
    --batch-size 256 \
    --name benchmark_${size} \
    > benchmark_${size}.log 2>&1 &
done

# Wait for completion
wait

# Compare results
grep "Epoch 500" benchmark_*.log
```

**Analysis**:
```bash
python tools/analyze_benchmark.py
```

**Expected**:
- Small: Converges in ~300 epochs, reward +5.2%
- Medium: Converges in ~250 epochs, reward +5.8%
- Large: Converges in ~200 epochs, reward +6.5%

**Interpretation**:
- Larger models learn faster (better capacity)
- Diminishing returns (large is only slightly better than medium)

### Test 4: Checkpoint Compatibility

**Verify** checkpoints can be loaded with correct model size:

```python
# Save checkpoint
python train_jax_fast.py --model-size large --epochs 10 --name ckpt_test

# Load checkpoint (must specify same size)
python train_jax_fast.py --model-size large --load-checkpoint jax_ckpt_fast/ckpt_test/checkpoint_10 --epochs 5
```

**Expected**: Loads without error.

**Fail case**: Loading large checkpoint with `--model-size small` should error.

## Performance Expectations

### Training Speed (Epochs/Hour)

**RTX 3090, batch 256, 32 sims**:
- Small (7M): ~450 epochs/hour
- Medium (15M): ~350 epochs/hour
- Large (25M): ~250 epochs/hour

**Interpretation**: 2× params → ~0.7× speed (acceptable tradeoff).

### Convergence Speed

**Level 0 (Epochs to Reward > +5%)**:
- Small: ~300 epochs
- Medium: ~250 epochs
- Large: ~200 epochs

**Level 3 (Epochs to Reward > +1%)**:
- Small: May not converge (insufficient capacity)
- Medium: ~2000 epochs
- Large: ~1500 epochs

## Recommendations

**For development/testing**: `--model-size small`
- Fast iteration
- Quick validation

**For curriculum (Levels 0-3)**: `--model-size medium`
- Good balance
- Sufficient capacity

**For adversarial (Level 5) and production**: `--model-size large`
- Maximum capacity
- Best final performance

## Acceptance Criteria

✅ **PASS if ALL met**:
1. Parameter counts match expected (7M, 15M, 25M)
2. All sizes fit in 24GB VRAM with appropriate batch sizes
3. Large model converges faster than small on Level 0
4. Checkpoint save/load works correctly

## Rollback Plan

If large model causes issues:

```bash
# Just use small/medium
python train_jax_fast.py --model-size medium ...
```

## Integration Checklist

- [ ] Modify `trm.py` to add `from_size()` factory
- [ ] Update `muzero_model.py` to use model_size
- [ ] Add `--model-size` CLI argument
- [ ] Add auto batch size adjustment
- [ ] Create parameter count test
- [ ] Run VRAM usage tests
- [ ] Benchmark all 3 sizes on Level 0
- [ ] Commit: "Add configurable model sizes (7M/15M/25M)"

## Next Task

Once this works → **TASK_04_PORTFOLIO_CURRICULUM.md** (Levels 4-6)
