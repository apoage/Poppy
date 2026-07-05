# Recovery Guide - Training System State

**Last Updated**: December 20, 2025, 23:59 UTC
**Status**: Value head collapse issue - aggressive label smoothing applied (50%)

---

## Current Problem

**Value head keeps collapsing** after training starts:

```
Warmup (before training): Peak prob=0.047 | Entropy=4.41  ← Perfect!
After 1 epoch training:   Peak prob=0.839 | Entropy=1.12  ← Collapsed!
```

Even with:
- ✅ Monte Carlo returns (no bootstrapping)
- ✅ Label smoothing 10%
- ✅ Support size 101
- ✅ Increased value loss weight (2.0)

**The value head still collapses to predicting constant -0.40 for all states.**

---

## Latest Fix Applied

**File**: `mcts_trader/jax_agent/trainer.py:89`

**Increased label smoothing from 0.1 → 0.5 (50%)**

```python
def _scalar_to_categorical(..., smoothing: float = 0.5):  # Was 0.1
    # 50% target distribution + 50% uniform
    uniform = jnp.ones((batch_size, support_size)) / support_size
    targets = (1.0 - smoothing) * targets + smoothing * uniform
```

**Expected Result**:
- Peak prob should stay <0.4 (not 0.8+)
- Entropy should stay >2.5 (not 1.1)
- Value head learns smooth distribution

---

## How to Restart Training

```bash
# Navigate to project
cd ~/Trader

# Activate environment (if needed)
source trader_env/bin/activate

# Delete old checkpoint (trained with bugs)
rm -rf jax_ckpt_fast/dojo_dynamic_L0_sine/

# Start fresh training
python dojo_dynamic_curriculum.py --level 0 --model-size large
```

---

## What to Watch For

### Good Signs ✅

**Value Head Staying Healthy**:
```
Epoch 6-10: Value Head: Peak bin=varies | Peak prob=0.2-0.4 | Entropy=2.5-3.5
```

**MC Targets Varying**:
```
[Train Iter 50] MC Targets: std >1.0
```

**Policy Entropy High**:
```
[MCTS Step 100] Entropy=3.8-4.3
```

**Sells Appearing**:
```
Epoch X: B/H/S: 150/180/70  ← 70 sells per 400 steps = 17.5%
```

### Bad Signs ❌

**Value Head Collapsing**:
```
Epoch 10: Value Head: Peak prob >0.7 | Entropy <1.5
```
→ Label smoothing still too low, increase to 0.7

**Zero Sells**:
```
Epoch X: B/H/S: 190/210/0  ← No sells
```
→ Concentration penalty too weak, increase to 0.15

**Entropy Collapse**:
```
[MCTS Step 100] Entropy <2.5
```
→ Increase gumbel_scale to 5.0

---

## Current Configuration

### Model & Training
- **Model size**: Large (160 state_dim, ~36M params)
- **Support size**: 101 bins (range -10 to +10)
- **Learning rate**: 1e-4
- **Batch size**: 512
- **Value loss weight**: 2.0
- **Label smoothing**: 0.5 (50%)
- **TD method**: Monte Carlo (no bootstrapping)

### MCTS Search
- **Simulations**: 32
- **Gumbel scale**: 3.0
- **Use Gumbel**: True

### Environment
- **Episode steps**: 400
- **Concentration penalty**: 0.10 per position over 16
- **Max leverage**: 4.0x (synthetic), 2.0x (real)
- **Closing bonus**: +0.10% for reducing positions

### Synthetic Data
- **Level**: 0 (sine waves)
- **Synthetic steps**: 10,000
- **Fee**: 0 bps
- **Slippage**: 0 bps

---

## Complete Fix History (Dec 20)

### Fixes Applied (In Order)

1. **Increased MCTS sims** (16 → 32)
2. **Strengthened concentration penalty** (0.05 → 0.10)
3. **Increased episode length** (200 → 400 steps)
4. **Added closing bonus** (+0.10% for sells that reduce positions)
5. **Dynamic leverage limits** (4.0x synthetic, 2.0x real)
6. **Reduced support size** (601 → 101 bins)
7. **Fixed support_size decoding** (hardcoded 601 → model.support_size)
8. **Increased Gumbel scale** (1.0 → 3.0)
9. **Switched to Monte Carlo returns** (removed bootstrapping)
10. **Increased value loss weight** (1.0 → 2.0)
11. **Added label smoothing 10%** (first attempt)
12. **Increased label smoothing to 50%** (latest fix)

---

## Monitoring Dashboard

### Real-time Logs
```bash
# Watch training progress
tail -f logs/training_dojo_dynamic_L0_sine.log

# Watch MCTS policy
tail -f logs/training_dojo_dynamic_L0_sine.log | grep "MCTS Step"

# Watch value head
tail -f logs/training_dojo_dynamic_L0_sine.log | grep "Value Head"

# Watch MC targets
tail -f logs/training_dojo_dynamic_L0_sine.log | grep "MC Targets"
```

### Visualization
```bash
# After 50+ epochs, generate plots
python tools/visualize_mcts_tree.py logs/training_dojo_dynamic_L0_sine.csv --save

# View plots
ls -lh plots/
```

---

## Key Metrics Explained

### Value Head Metrics
- **Peak bin**: Most probable bin (should vary by state)
  - Good: Changes frequently (40, 52, 48, 55, ...)
  - Bad: Always same bin (48, 48, 48, ...)

- **Peak prob**: Confidence in prediction
  - Good: 0.2-0.4 (uncertain, learning)
  - Bad: >0.7 (overconfident, collapsed)

- **Entropy**: Distribution spread
  - Good: 2.5-4.0 (smooth distribution)
  - Bad: <1.5 (peaked at single bin)

### MC Target Statistics
- **std (standard deviation)**: Target variance
  - Good: >1.0 (varying targets)
  - Bad: <0.5 (collapsed targets)

### Policy Metrics
- **S/H/B**: Sell/Hold/Buy probability sums
  - Good: S>0.10, balanced across actions
  - Bad: S<0.05 (avoiding sells)

- **Policy Entropy**: Exploration level
  - Good: 3.5-4.3 (exploring)
  - Bad: <2.5 (deterministic)

---

## If Value Head Still Collapses

### Option 1: Increase Label Smoothing Further
```python
# Edit mcts_trader/jax_agent/trainer.py:89
def _scalar_to_categorical(..., smoothing: float = 0.7):  # Was 0.5
```

### Option 2: Use KL Divergence Loss Instead of Cross-Entropy
Would require more code changes - not recommended yet.

### Option 3: Regularize Value Head Directly
Add entropy regularization to value predictions (penalize peaked outputs).

### Option 4: Increase Temperature in Value Softmax
Make value head output smoother before cross-entropy.

**Recommendation**: Try increasing smoothing to 0.7 first.

---

## If Sells Still Not Appearing

### Check Position Logs
Look for position accumulation:
```
[Step 50] Positions: 45/16 | Penalty: 2.90%
```

If positions >40, penalty should be forcing sells.

### Increase Concentration Penalty
```python
# Edit train_jax_fast.py:346
concentration_penalty = (n_positions - 16) * 0.15  # Was 0.10
```

### Check MCTS Exploration
If S<0.05 in MCTS logs, increase gumbel_scale:
```python
# Edit train_jax_fast.py:731
parser.add_argument('--gumbel-scale', type=float, default=5.0)  # Was 3.0
```

---

## Expected Timeline to Mastery

With all fixes working:
- **Epoch 1-50**: Exploration, sells appearing, value learning
- **Epoch 50-150**: Policy improving, balanced actions
- **Epoch 150-300**: Convergence to +2.5% reward (mastery)

**Target**: Reach mastery in 200-300 epochs (was 1000+ before fixes)

---

## Files to Check After Reboot

### Training Output
```bash
ls -lh logs/training_dojo_dynamic_L0_sine.csv
tail -20 logs/training_dojo_dynamic_L0_sine.csv
```

### Checkpoint
```bash
ls -lh jax_ckpt_fast/dojo_dynamic_L0_sine/
```

### Session Notes
```bash
cat TRAINING_DEBUG_SESSION_DEC20.md
```

---

## Quick Reference Commands

### Start Training
```bash
python dojo_dynamic_curriculum.py --level 0 --model-size large
```

### Resume from Checkpoint
```bash
python train_jax_fast.py --synthetic-level 0 --epochs 250 --batch-size 512 \
    --model-size large --name dojo_dynamic_L0_sine --fee 0 --slippage 0 \
    --load-checkpoint jax_ckpt_fast/dojo_dynamic_L0_sine
```

### Visualize Results
```bash
python tools/visualize_mcts_tree.py logs/training_dojo_dynamic_L0_sine.csv
```

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

---

## Critical Files Modified

### Core Training Logic
- `train_jax_fast.py` - Main training script
- `mcts_trader/jax_agent/trainer.py` - MuZero trainer (MC returns, label smoothing)
- `mcts_trader/jax_agent/mctx_search.py` - MCTS search (Gumbel MuZero)
- `mcts_trader/env/trading_env.py` - Trading environment (penalties, rewards)
- `mcts_trader/agent/portfolio.py` - Portfolio manager (leverage limits)

### Tools
- `tools/visualize_mcts_tree.py` - Visualization tool
- `dojo_dynamic_curriculum.py` - Curriculum manager

### Documentation
- `TRAINING_DEBUG_SESSION_DEC20.md` - Complete session notes
- `RECOVERY_GUIDE.md` - This file
- `PERFORMANCE_FIXES_SELLS.md` - Zero-sells fix documentation
- `ENTROPY_COLLAPSE_FIX.md` - Gumbel exploration fix
- `VALUE_LOSS_FIX.md` - Support size fix
- `MCTS_MONITORING_ADDED.md` - Monitoring features

---

## Known Issues

### Issue 1: Value Head Collapse
**Status**: Investigating
**Current Fix**: Label smoothing 50%
**If Still Fails**: Increase to 70% or try different loss function

### Issue 2: Epoch 6 Long Self-Play
**Status**: Normal (JAX JIT compilation on first training epoch)
**Impact**: First training epoch takes 40-50s, then drops to 6-7s

### Issue 3: Position Logs Not Showing
**Status**: Unknown - debug code added but not printing
**Workaround**: Monitor CSV sells count instead

---

## Success Criteria

### Phase 1: Value Head Health (Epochs 1-20)
- [ ] Value entropy stays >2.5
- [ ] Peak prob stays <0.5
- [ ] MC targets std >1.0
- [ ] Value loss 0.5-2.0 (not <0.3 or >5.0)

### Phase 2: Policy Learning (Epochs 20-100)
- [ ] Sells appearing (>50 per episode)
- [ ] Policy entropy 3.0-4.0
- [ ] Value-reward correlation >0.5
- [ ] Episode rewards improving

### Phase 3: Mastery (Epochs 100-300)
- [ ] Average reward (last 50 epochs) ≥ +2.5%
- [ ] Balanced actions (S:15-25%, H:50-60%, B:20-30%)
- [ ] Value-reward correlation >0.8
- [ ] Stable training (no collapse)

---

## Contact Points

### If Training Crashes
1. Check `logs/training_dojo_dynamic_L0_sine.log` for errors
2. Check GPU memory: `nvidia-smi`
3. Check for NaN losses in logs
4. Restart from last checkpoint

### If Results Look Wrong
1. Verify value head entropy >2.5
2. Check MC targets std >1.0
3. Verify sells are appearing
4. Check TRAINING_DEBUG_SESSION_DEC20.md for comparison

---

## Emergency Rollback

If 50% label smoothing breaks training completely:

### Revert to 10% Smoothing
```bash
# Edit mcts_trader/jax_agent/trainer.py:89
smoothing: float = 0.1  # Change from 0.5 back to 0.1
```

### Increase Value Loss Weight Instead
```bash
# Edit train_jax_fast.py:863
value_loss_weight=5.0,  # Change from 2.0 to 5.0
```

### Or Use Standard MuZero Instead of Gumbel
```bash
# Edit train_jax_fast.py:911
'use_gumbel': False,  # Change from True
'temperature': 1.5,
'dirichlet_alpha': 0.5,
```

---

## Summary

**Current Status**:
- Value head collapses after 1 training epoch despite MC returns
- Applied aggressive label smoothing (50%) to prevent collapse
- Need to restart and verify fix works

**Next Action**:
```bash
rm -rf jax_ckpt_fast/dojo_dynamic_L0_sine/
python dojo_dynamic_curriculum.py --level 0 --model-size large
```

**Watch For**:
- Value Head entropy staying >2.5 after epoch 10
- Peak prob staying <0.5
- Sells appearing (>50 per episode)

**Goal**:
- Reach +2.5% mastery in 200-300 epochs
- Maintain stable value head (no collapse)

---

**Last Updated**: December 20, 2025, 23:59 UTC
**Ready to Resume**: ✅ All fixes applied
**Restart Required**: ✅ Delete checkpoint first
