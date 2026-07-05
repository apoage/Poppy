# TASK_111: Conv-Based Diffusion Model Training Integration

## Status: COMPLETED

## Goal
Train a 302M parameter conv-based flow matching model on historical market data using the `diff_teacher` pipeline, then use it for on-the-fly generation in `fast_generator.py` during RL training.

## Problem
We have two separate implementations:
1. `diff_teacher/` - Full training pipeline but uses transformer (O(n²) memory, 22M params)
2. `fast_generator.py` - Conv architecture (O(n) memory, 302M params) but no training loop

Need to combine: **conv architecture + diff_teacher training pipeline**

## Architecture Details

### Conv-Based Generator (from `fast_generator.py`)
```
Location: mcts_trader/jax_agent/fast_generator.py

- DilatedConvBlock: Causal 1D conv with exponential dilation (1,2,4,8,16,32)
- FastVelocityNet: Stack of DilatedConvBlocks with FiLM conditioning
- FastFlowGenerator: Flow matching wrapper

Config for 302M params:
  hidden_dim: 1536
  num_layers: 16
  ff_dim: 3072
  cond_dim: 768

Memory: ~5GB for 1440 steps × 15 envs × 64 assets
Latency: ~642ms per generation
```

### Training Pipeline (from `diff_teacher/`)
```
Location: diff_teacher/

- data_loader.py: RainbowTableLoader reads from training_schedule.json
- flow_matching.py: Flow matching loss (OT-CFM)
- teacher.py: Training orchestrator
- train.py: Entry point
- config.py: Hyperparameters
```

## Implementation Steps

### Step 1: Create Unified Architecture Module
Create `diff_teacher/conv_flow_matching.py`:
- Copy `DilatedConvBlock`, `CausalConv1D`, `FastVelocityNet` from `fast_generator.py`
- Add flow matching training methods (compute_loss, sample)
- Keep regime conditioning (5 regimes: crash, rally, sideways, volatile, calm)

### Step 2: Update Config
Modify `diff_teacher/config.py`:
```python
# Add conv-specific settings
flow_architecture: str = "conv"  # "conv" or "transformer"
flow_hidden_dim: int = 1536
flow_layers: int = 16
flow_kernel_size: int = 3
```

### Step 3: Update Training Script
Modify `diff_teacher/train.py`:
- Add flag `--architecture conv`
- Instantiate ConvFlowGenerator instead of transformer when flag is set
- Keep same training loop, loss, data loading

### Step 4: Update Checkpoint Format
Ensure checkpoint saves:
- Model architecture config (hidden_dim, num_layers, etc.)
- Trained weights
- Training metadata (epochs, loss history)

### Step 5: Create Inference Loader
Add to `mcts_trader/jax_agent/fast_generator.py`:
```python
def load_trained_generator(checkpoint_path: str) -> Tuple[FastFlowGenerator, Dict]:
    """Load trained weights from diff_teacher checkpoint."""
    # Load config and weights
    # Return model and params ready for inference
```

### Step 6: Integration with RL Training
Update `train_jax_fast.py`:
```python
--adversarial-checkpoint PATH  # Load trained generator weights
```

## File Changes Required

| File | Action | Description |
|------|--------|-------------|
| `diff_teacher/conv_flow_matching.py` | CREATE | Conv architecture with flow matching |
| `diff_teacher/config.py` | MODIFY | Add conv config options |
| `diff_teacher/train.py` | MODIFY | Support --architecture conv |
| `diff_teacher/teacher.py` | MODIFY | Use conv generator when configured |
| `fast_generator.py` | MODIFY | Add load_trained_generator() |
| `train_jax_fast.py` | MODIFY | Add --adversarial-checkpoint |

## Training Command (after implementation)
```bash
python -m diff_teacher.train \
    --rainbow \
    --architecture conv \
    --hidden-dim 1536 \
    --num-layers 16 \
    --batch-size 32 \
    --max-steps 100000 \
    --checkpoint-dir checkpoints/conv_generator
```

## Inference Command (after training)
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --adversarial-on-the-fly \
    --adversarial-checkpoint checkpoints/conv_generator/best \
    --name rl_with_trained_gen
```

## Data Available
- Database: `trading_data.db` (2013-2025, 12.3 years)
- Schedule: `training_schedule.json` (18 periods, 2018-2025)
- Assets: 64 top liquid pairs
- Windows: Millions available via RainbowTableLoader

## Quality Validation
After training, validate generator quality:
1. Visual inspection of generated price paths
2. Statistical tests (volatility distribution, autocorrelation)
3. Regime fidelity (crash should look like crash)
4. Cross-asset correlation structure

## Estimated Training Time
- 302M params, batch_size=32, 100k steps
- ~12 hours on RTX 3090

## Success Criteria
1. Training completes without OOM
2. Loss converges
3. Generated data passes statistical tests
4. RL agent trains successfully on generated data

## Dependencies
- Existing `fast_generator.py` conv architecture
- Existing `diff_teacher/` training pipeline
- `training_schedule.json` with valid periods
- `trading_data.db` with historical data

## Notes
- Conv architecture uses O(n) memory vs O(n²) for transformer
- 302M params fits in 5GB VRAM for inference
- Training may need gradient checkpointing for memory efficiency
- Keep both architectures available (conv for long sequences, transformer for short)

## Implementation Summary (2026-01-25)

### Files Created
- `diff_teacher/conv_flow_matching.py` - Conv-based flow matching generator with:
  - `CausalConv1D` - Causal 1D convolution with dilation
  - `DilatedConvBlock` - WaveNet-style block with FiLM conditioning
  - `ConvVelocityNet` - Full velocity network with exponential dilation pattern
  - `ConvFlowMatchingGenerator` - Training and inference wrapper
  - `flow_matching_loss()` - OT-CFM loss function

### Files Modified
- `diff_teacher/config.py` - Added conv architecture options:
  - `flow_architecture: str = "conv"` (or "transformer")
  - `flow_kernel_size: int = 3`
  - `flow_ff_dim: int = 3072`
  - `flow_cond_dim: int = 768`

- `diff_teacher/train.py` - Added CLI support:
  - `--architecture conv|transformer`
  - `--hidden-dim` (defaults: 1536 for conv, 512 for transformer)
  - `--flow-layers` (defaults: 16 for conv, 6 for transformer)

- `diff_teacher/teacher.py` - Uses conv or transformer based on config

- `mcts_trader/jax_agent/fast_generator.py` - Added checkpoint loading:
  - `load_trained_generator()` - Load diff_teacher checkpoint
  - `load_or_init_generator()` - Load or random init

- `mcts_trader/training/cli.py` - Added `--adversarial-checkpoint`

- `mcts_trader/training/config.py` - Added `adversarial_checkpoint` field

- `mcts_trader/training/setup.py` - `setup_fast_generator()` accepts checkpoint

- `train_jax_fast.py` - Uses checkpoint when specified

### Usage

**Training (12+ hours on RTX 3090):**
```bash
python -m diff_teacher.train \
    --rainbow \
    --architecture conv \
    --hidden-dim 1536 \
    --flow-layers 16 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/conv_generator
```

**Inference during RL training:**
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --adversarial-on-the-fly \
    --adversarial-checkpoint checkpoints/conv_generator/generator_best \
    --name rl_with_trained_gen
```
