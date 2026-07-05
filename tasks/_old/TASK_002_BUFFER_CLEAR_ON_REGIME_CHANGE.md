# TASK 002: Clear Replay Buffer on Curriculum Regime Change

**Severity**: MEDIUM
**Location**: `train_jax_fast.py:346-369`, `dojo_dynamic_curriculum.py:160-163`
**Status**: DONE (2025-12-29)
**Estimated Complexity**: Low

---

## Problem Description

When loading a checkpoint to continue training on a new curriculum level (e.g., Level 0 -> Level 1), the replay buffer retains ALL old experiences from the previous level. This causes:

1. **Distribution mismatch**: Old experiences were collected under different market dynamics (e.g., sine waves vs lagged pairs)
2. **Policy confusion**: Model trains on mixed data from incompatible regimes
3. **Slow adaptation**: Old data dilutes new experiences, slowing learning

### Current Code (`train_jax_fast.py:346-369`):
```python
if args.load_checkpoint:
    restored_state = trainer.restore_checkpoint(abs_ckpt_path)
    if restored_state:
        trainer.state = restored_state
        state = restored_state
        # NOTE: replay_buffer is NEVER cleared!
```

### Current Code (`dojo_dynamic_curriculum.py:160-163`):
```python
if total_epochs_so_far > 0 and os.path.exists(checkpoint_dir):
    cmd.extend(['--load-checkpoint', checkpoint_dir])
    # Buffer state is not managed here
```

---

## Root Cause Analysis

The training script treats checkpoint loading as "resume from where we left off" rather than "transfer learning to new domain". For curriculum learning, we need:

1. **Keep model weights** (transfer learned representations)
2. **Clear replay buffer** (fresh experiences from new environment)
3. **Optionally reset optimizer** (see TASK_003)

---

## Proposed Solution

### Option A: Add `--clear-buffer` CLI Flag (Recommended)

```python
# train_jax_fast.py
parser.add_argument('--clear-buffer', action='store_true',
                    help='Clear replay buffer (use when changing curriculum levels)')

# After loading checkpoint:
if args.load_checkpoint:
    restored_state = trainer.restore_checkpoint(abs_ckpt_path)
    if restored_state:
        trainer.state = restored_state

        if args.clear_buffer:
            print("🧹 Clearing replay buffer for new regime...")
            replay_buffer.clear()  # Need to implement this
```

### Option B: Auto-Detect Regime Change

Detect if synthetic level changed and auto-clear:
```python
# Store current config in checkpoint metadata
checkpoint_metadata = {
    'synthetic_level': args.synthetic_level,
    'fee_bps': args.fee,
    ...
}

# On load, compare and clear if different
if loaded_metadata['synthetic_level'] != args.synthetic_level:
    print("⚠️ Curriculum level changed, clearing buffer...")
    replay_buffer.clear()
```

---

## Implementation Plan

### Step 1: Add `clear()` Method to Replay Buffer (10 min)

```python
# mcts_trader/jax_agent/replay_buffer.py

def clear(self):
    """Reset buffer to empty state, keeping allocated memory."""
    self.position = 0
    self.num_trajectories = 0
    self.total_steps_added = 0
    # Optionally zero out arrays (not strictly necessary)
    # self.observations = jnp.zeros_like(self.observations)
```

### Step 2: Add CLI Flag to Training Script (10 min)

```python
# train_jax_fast.py
parser.add_argument('--clear-buffer', action='store_true',
                    help='Clear replay buffer when loading checkpoint')
```

### Step 3: Update Dojo Manager (15 min)

```python
# dojo_dynamic_curriculum.py

def _train_phase(self, ...):
    cmd = [...]

    if total_epochs_so_far > 0 and os.path.exists(checkpoint_dir):
        cmd.extend(['--load-checkpoint', checkpoint_dir])

        # Clear buffer when advancing phases (optional, based on phase change)
        if self._is_phase_transition():
            cmd.append('--clear-buffer')
```

### Step 4: Add Warmup Period After Clear (10 min)

After clearing buffer, ensure enough new experiences before training:
```python
if args.clear_buffer:
    # Extend warmup since buffer is empty
    warmup_epochs = 10  # Instead of 5
```

---

## Acceptance Criteria

- [ ] `JAXReplayBuffer.clear()` method implemented
- [ ] `--clear-buffer` CLI flag added to `train_jax_fast.py`
- [ ] Dojo manager uses `--clear-buffer` on level transitions
- [ ] Buffer is verified empty after clear (add logging)
- [ ] Training resumes correctly after buffer clear

---

## Files to Modify

1. `mcts_trader/jax_agent/replay_buffer.py` - Add `clear()` method
2. `train_jax_fast.py` - Add CLI flag and clear logic
3. `dojo_dynamic_curriculum.py` - Pass flag on phase transitions

---

## Testing

```bash
# Test 1: Manual clear flag
python train_jax_fast.py --synthetic-level 1 --load-checkpoint jax_ckpt_fast/level0 --clear-buffer

# Verify in logs:
# "🧹 Clearing replay buffer for new regime..."
# "Buffer: 0 trajectories"

# Test 2: Dojo automatic clear
python dojo_dynamic_curriculum.py --level 1
# Should auto-clear when transitioning from level 0 checkpoint
```

---

## Risk Assessment

**Risk**: Clearing buffer too aggressively could slow learning by discarding useful experiences.

**Mitigation**:
- Only clear on LEVEL changes, not phase changes within same level
- Add `--keep-buffer` flag to override if needed
- Log buffer state before/after for debugging
