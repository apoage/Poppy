# TASK_128c: Training Hang Debug Plan

## Incident Analysis
- **Symptom:** Training hangs indefinitely at the start of Epoch 91 (during self-play phase).
- **Environment:** JAX/XLA with 4 parallel environments.
- **Resources:** GPU at 0% utilization, CPU idle.
- **Hypothesis:** Deadlock in device-to-host transfer (`fetch` in `run_self_play_jax`) or silent GPU kernel failure.

## Debugging Strategy

### 1. Instrumentation (Phase 1)
Add verbose logging to `train_jax_fast.py` to pinpoint exactly where the hang occurs.
- **Before Self-Play:** Log start and current GPU memory usage.
- **Inside Self-Play Loop:** Log start/end of JAX execution.
- **After Self-Play:** Log duration and successful completion.
- **Buffer/Train:** Log entry/exit points.

### 2. Instrumentation (Phase 2 - if Phase 1 inconclusive)
- **Chunking:** Break the 400-step `lax.scan` into smaller chunks (e.g., 50 steps) to see if it hangs at a specific step.
- **JAX Debug Print:** Insert `jax.debug.print` inside the kernel (performance cost, but visibility).

### 3. Potential Fixes
- **Timeout:** Implement a timeout wrapper for the self-play call (difficult with blocking JAX calls).
- **Resource Management:** Ensure no memory leaks (monitoring added in Phase 1).
- **Restart Logic:** If it's a driver flake, automated restart might be the only robust solution.

## Proposed Code Changes

### `train_jax_fast.py`
```python
# [HANG-DEBUG] Print before self-play
try:
    mem_stats = jax.devices()[0].memory_stats()
    mem_gb = mem_stats.get('bytes_in_use', 0) / 1e9 if mem_stats else 0
except:
    mem_gb = 0
print(f"[HANG-DEBUG] Epoch {epoch} START (GPU mem: {mem_gb:.2f}GB)", flush=True)
```

### `self_play.py`
Add prints before and after `jax.lax.scan` and `fetch`.

## Next Steps
1. Apply Phase 1 instrumentation.
2. Run reproduction command.
3. Analyze logs to confirm if hang is inside `run_self_play_jax`.
