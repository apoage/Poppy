# TASK_068: Zombie Flag Restoration & Missing Features

**Priority**: CRITICAL
**Status**: COMPLETE
**Created**: 2026-01-22
**Completed**: 2026-01-22

## Context

During TASK_065/066 refactoring, many CLI flags were preserved but their implementation logic was removed from the training loop. This creates "zombie flags" that accept input but do nothing.

## Phase 1: CRITICAL - Target Network Restoration

### Problem
Target network (`target_params`) was removed from `TrainState` in `trainer.py`. This breaks Double Q-learning stability - the value target now moves constantly instead of being periodically updated.

### Files to Modify
- `mcts_trader/jax_agent/trainer.py`

### Changes Required
1. Add `target_params` field back to `TrainState` class
2. Initialize `target_params = params` in `init_train_state()`
3. Add conditional target update in `_train_step()`:
   ```python
   should_update_target = (new_step_count % self.target_update_interval) == 0
   new_target_params = jax.lax.cond(
       should_update_target,
       lambda: new_state.params,
       lambda: state.target_params
   )
   new_state = new_state.replace(target_params=new_target_params)
   ```

---

## Phase 2: HIGH - ReDo Application in Training Loop

### Problem
ReDo (dormant neuron recycling) is configured in trainer but never called from training loop.

### Files to Modify
- `train_jax_fast.py`

### Changes Required
1. Add ReDo application call after training step:
   ```python
   if getattr(config, 'redo', False) and hasattr(trainer, 'apply_redo'):
       if (epoch + 1) % getattr(config, 'redo_interval', 10) == 0:
           redo_stats = trainer.apply_redo()
           # Log stats
   ```
2. Add ReDo metrics to CSV row

---

## Phase 3: HIGH - SAC Entropy Integration

### Problem
SAC-style learnable entropy coefficient exists in `sac_entropy.py.bak` but never integrated.

### Files to Modify
- `mcts_trader/jax_agent/sac_entropy.py.bak` -> rename to `.py` (activate)
- `train_jax_fast.py`
- `mcts_trader/training/cli.py`
- `mcts_trader/training/config.py`

### Changes Required
1. Activate `sac_entropy.py` module
2. Add CLI flags: `--sac-target-entropy-ratio`, `--sac-alpha-lr`, `--sac-init-alpha`
3. Add config fields
4. Initialize `AdaptiveEntropyManager` when `--sac-entropy` enabled
5. Call `manager.update(policy_entropy)` each epoch
6. Use returned alpha instead of static `entropy_coef`

---

## Phase 4: MEDIUM - Zombie Flag Restoration

### 4.1 Data Regeneration (`--regenerate-data-every`)

**Problem**: Flag parsed but no logic to regenerate synthetic data.

**Changes**:
- Add CLI flag and config field
- Add logic in training loop:
  ```python
  if config.regenerate_data_every > 0 and (epoch + 1) % config.regenerate_data_every == 0:
      # Regenerate synthetic data for anti-memorization
      for penv in parallel_envs:
          penv.regenerate_data()
  ```

### 4.2 Early Graduation (`--early-graduation`)

**Problem**: Flag parsed but no logic to graduate when difficulty saturates.

**Changes**:
- Add CLI flag and config field
- Track consecutive epochs at max difficulty
- Trigger graduation when threshold met

### 4.3 Dirichlet Aggressive (`--dirichlet-aggressive`)

**Problem**: Flag parsed but doesn't override noise params.

**Changes**:
- Add logic after config parsing:
  ```python
  if config.dirichlet_aggressive:
      config.noise = 0.5
      config.noise_frac = 0.5
  ```

### 4.4 Adaptive Calibration (`--adaptive-calibration`)

**Problem**: Flag parsed but value calibration never runs.

**Changes**:
- Add CLI flags: `--adaptive-calibration`, related params
- Add calibration tracking variables
- Add periodic calibration update logic

### 4.5 Differential Reward (`--differential-reward`)

**Problem**: Flag parsed but reward not modified.

**Changes**:
- Modify reward computation in self_play or environment
- Subtract market baseline when flag enabled

---

## Phase 5: LOW - Logging Completeness

### 5.1 ReDo Metrics to CSV
- Add columns: `redo_dormant_neurons`, `redo_dormant_pct`, `redo_recycled_count`

### 5.2 Eval Results to CSV
- Add columns: `eval_level`, `eval_reward`, `eval_alpha`, `eval_pct_hold`

### 5.3 ATC Metrics to CSV
- Ensure `atc_pos_sim`, `atc_neg_sim` logged

---

## Execution Order

1. [CRITICAL] Phase 1: Target Network
2. [HIGH] Phase 2: ReDo Application
3. [HIGH] Phase 3: SAC Entropy
4. [MEDIUM] Phase 4.1: Data Regeneration
5. [MEDIUM] Phase 4.3: Dirichlet Aggressive (quick fix)
6. [MEDIUM] Phase 4.5: Differential Reward
7. [MEDIUM] Phase 4.2: Early Graduation
8. [MEDIUM] Phase 4.4: Adaptive Calibration
9. [LOW] Phase 5: Logging Completeness

## Verification

After each phase, run syntax check:
```bash
python -m py_compile train_jax_fast.py
python -m py_compile mcts_trader/jax_agent/trainer.py
```

Quick functional test:
```bash
python train_jax_fast.py --synthetic-level 0 --epochs 5 --name test_restore
```
