# BUGHUNT PROMPT: Critical Training Hang (TASK_128)

## Your Mission

Find and fix a critical bug that causes JAX training to hang indefinitely after 60+ epochs. This blocks ALL long-term training runs.

## The Bug

**Symptoms:**
- Training runs fine for 50-60 epochs
- At some point (epoch 60-90+), training freezes
- GPU utilization drops to 0%
- GPU memory stays high (17GB+)
- Process doesn't crash - just hangs forever
- No error message, no stack trace

**Reproduction:**
```bash
cd /home/lukas/Trader
python train_jax_fast.py \
  --name hang_debug \
  --synthetic-level 0 \
  --epochs 100 \
  --steps 200 \
  --parallel-envs 4 \
  --jax-env \
  --batch-size 512 \
  --use-prediction-loss \
  --imprint
```
Wait 30-60 minutes. Hang occurs around epoch 60-80.

## What You Need To Do

### Step 1: Understand the Training Loop
Read these files to understand the flow:
- `train_jax_fast.py` - Main training loop (start here)
- `mcts_trader/jax_agent/trainer.py` - Training step
- `mcts_trader/training/self_play.py` - Episode generation

### Step 2: Add Diagnostic Instrumentation
Add timing and logging to narrow down WHERE it hangs:
```python
# Example: Add to train_jax_fast.py main loop
print(f"[DEBUG] Epoch {epoch} - Starting self-play", flush=True)
trajectories = run_self_play_jax(...)
print(f"[DEBUG] Epoch {epoch} - Self-play done", flush=True)

print(f"[DEBUG] Epoch {epoch} - Starting train step", flush=True)
metrics = trainer.train_step(...)
print(f"[DEBUG] Epoch {epoch} - Train step done", flush=True)
```

### Step 3: Test Hypotheses
Run with different configurations to isolate:

1. **Without prediction loss:** Does it still hang?
   ```bash
   python train_jax_fast.py --epochs 100 --jax-env  # no --use-prediction-loss
   ```

2. **Without imprinting:** Does it still hang?
   ```bash
   python train_jax_fast.py --epochs 100 --jax-env --use-prediction-loss  # no --imprint
   ```

3. **Single env:** Does parallelism matter?
   ```bash
   python train_jax_fast.py --epochs 100 --jax-env --parallel-envs 1
   ```

4. **Smaller batch:** Is it memory pressure?
   ```bash
   python train_jax_fast.py --epochs 100 --jax-env --batch-size 256
   ```

### Step 4: Check Suspect Areas

**A. Memory leak detection:**
```python
# Add to training loop
import jax
mem = jax.devices()[0].memory_stats()
print(f"[MEM] Epoch {epoch}: {mem['bytes_in_use'] / 1e9:.2f} GB")
```

**B. NaN/Inf detection:**
```python
# Add after gradient computation in trainer.py
has_nan = jax.tree_util.tree_reduce(
    lambda a, b: a | b,
    jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), grads)
)
```

**C. Lock deadlock detection:**
Check `expert_buffer.py` - the `threading.Lock()` usage. Is it possible the lock is acquired but never released?

**D. JAX scan issues:**
Check all `jax.lax.scan` calls:
- `prediction/market_ode.py:443`
- `self_play.py` (episode generation)
- `mctx_search.py` (MCTS)

### Step 5: Find the Smoking Gun

The hang is likely one of:
1. **Infinite loop** in a scan body due to numerical issue
2. **Deadlock** in device-to-host transfer
3. **Memory exhaustion** causing silent failure
4. **Threading lock** never released
5. **XLA compilation** stuck in infinite recompilation

## Key Files

```
train_jax_fast.py                           # Main loop - START HERE
mcts_trader/jax_agent/trainer.py            # Training step
mcts_trader/training/self_play.py           # Self-play episode generation
mcts_trader/jax_agent/expert_buffer.py      # Expert buffer with threading lock
mcts_trader/jax_agent/prediction/market_ode.py  # Has jax.lax.scan
mcts_trader/jax_agent/replay_buffer.py      # Replay buffer
```

## Constraints

- Do NOT rewrite large portions of code
- Make minimal, targeted changes
- Add diagnostics first, fix second
- Test fix works for 100+ epochs

## Success

You're done when:
1. You've identified the exact line/function causing the hang
2. You've implemented a fix
3. Training runs 100+ epochs without hanging

## Notes

- See `tasks/TASK_128_TRAINING_HANG_BUGHUNT.md` for full analysis
- Previous audit: `tasks/TASK_127r_hang_at_epoch_90.md`
- This may be a JAX/XLA issue, not Python logic
- Memory pattern: grows from 8GB to 17GB before hang
