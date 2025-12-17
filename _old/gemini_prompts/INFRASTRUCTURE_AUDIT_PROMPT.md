# Infrastructure Bottleneck Analysis Prompt

## Context
We are training a JAX-based MuZero trading agent (`train_jax_fast.py`).
*   **Model:** Tiny Recursive Model (TRM) + MCTS (`mctx`).
*   **Input:** (Batch 256, 27, 64, 64).
*   **Hardware:** 16-Core CPU, 46GB RAM, RTX 3090/4090 class GPU (24GB VRAM).
*   **Current Load:**
    *   **GPU:** ~60% Compute utilization. VRAM abundant.
    *   **CPU:** One core at 90-100%, others idle.
    *   **RAM:** Low usage (cached data fits easily).

## Recent Optimizations
1.  **RAM Caching:** `HistoricalMarket` loads all OHLCV data to a Numpy matrix (float32) at start. SQL queries are replaced by O(1) array lookups.
2.  **MCTS Sims:** Increased from 16 to 32.
3.  **Result:** Epoch time improved from ~21s to ~16s, but GPU is still not saturated.

## The Problem
The system appears **CPU Single-Thread Bound**. The 90% load on one core suggests the Python process (Environment Step -> MCTS Loop -> Gradient Update) is serial and bottlenecked by Python execution speed, preventing the GPU from receiving data fast enough to hit 100% utilization.

## Objective
Identify the specific function calls causing the bottleneck and propose a solution to saturate the GPU (target >90% util).
Alternatively, if the infrastructure is healthy, tune hyperparameters (Batch Size, Sims) to utilize the available headroom for better training (free performance).

## Investigation Steps

### 1. Profile the Python Process (The "Why")
Use `py-spy` or `cProfile` to visualize where the CPU time is going.

**Command:**
```bash
# Install py-spy
pip install py-spy

# Run training with profiler (sampling mode, low overhead)
# Replace PID with actual PID of train_jax_fast.py
py-spy record -o profile.svg --pid <PID>
# OR run directly
py-spy record -o profile.svg -- python train_jax_fast.py ...
```

**Hypothesis:**
*   Is it `env.step()` logic (even with RAM cache, Python loops are slow)?
*   Is it `jax.device_put` (moving data to GPU)?
*   Is it `mctx` overhead?

### 2. JAX Profiling (The "GPU View")
Capture a JAX trace to see gaps in GPU execution.

**Code Modification:**
Add this to `train_jax_fast.py`:
```python
jax.profiler.start_trace("/tmp/jax-trace")
# ... run a few epochs ...
jax.profiler.stop_trace()
```
**Analysis:** Open in TensorBoard. Look for white space (GPU idle) between XLA kernels.

### 3. Batch Size Scaling
If the GPU is 60% busy, can we simply double the workload?
*   **Experiment:** Increase `batch_size` from 256 -> 512 or 1024.
*   **Expected Outcome:** GPU util goes up, per-epoch time might stay same (throughput increases).

### 4. Parallel Environments (The "Real Fix")
If Python execution is the bottleneck, we must parallelize.
*   **Current:** `FastTradingEnv` (Serial).
*   **Proposed:** `VectorEnv` using `multiprocessing` or `Ray`.
    *   Run 8 `FastTradingEnv` instances in 8 separate processes.
    *   Main process collects 8 observations -> Stack -> Send to GPU.
    *   This utilizes the 15 idle cores.

### 5. Hyperparameter Tuning (Saturating the GPU)
Since the GPU is stable at 60% (Compute) with dips only between epochs, the system is likely **Latency Bound** or **Under-saturated** due to small model/batch size. We can "spend" the remaining 40% compute to get a smarter agent.

**Experiments:**
1.  **Increase Simulations:**
    *   Change `--sims 32` -> `--sims 64`.
    *   *Prediction:* GPU Load -> ~80-90%. Training time per epoch increases slightly, but agent quality improves drastically (deeper thought).
2.  **Increase Batch Size:**
    *   Change `batch_size=256` -> `512`.
    *   *Prediction:* GPU Load -> Higher. Training throughput (samples/sec) increases. Epoch time decreases.
3.  **Increase Model Size (Long Term):**
    *   Increase TRM `d_model` (e.g., 64 -> 128) or layers.
    *   *Prediction:* Uses more compute per sim.

## Actionable Questions
1.  Is `batch_size=256` simply too small for this GPU to bother hitting 100%?
2.  Is the `HistoricalMarket.get_current_data` still slow due to dictionary creation? (It creates a dict of 64 dicts *every step*).
    *   *Optimization:* Return raw Numpy arrays directly to JAX, skip dictionary creation entirely.
3.  Can we use `jax.pmap` to run MCTS on multiple GPU streams?

## Next Steps for the Agent
1.  Run `py-spy` on the running process.
2.  Analyze the SVG flame graph.
3.  If `dict` creation is the culprit -> Rewrite Env to be "JAX-Native" (Array in, Array out).
4.  If just serial processing -> Implement `VectorEnv`.
