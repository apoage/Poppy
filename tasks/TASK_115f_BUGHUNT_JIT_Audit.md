# TASK_115f: JAX Tracing & JIT Audits

**Status**: Pending
**Priority**: Medium
**Created**: 2026-01-25

## Identified Issues

### 1. Compile-Time Unrolling in Flow Matching
*   **File**: `diff_teacher/flow_matching.py`
*   **Issue**: The `sample` method uses a Python `for` loop (`for i in range(num_steps):`) for Euler integration.
*   **Impact**: If `num_steps` is large (e.g., 50-100), this causes the JIT compiler to unroll the loop, creating a massive computational graph, slow compilation, and potentially OOM during compilation.
*   **Fix**: Replace with `jax.lax.fori_loop` or `jax.lax.scan` to keep the loop structure in the compiled XLA graph.

### 2. Misleading Compile-Time Printing in Trainer
*   **File**: `mcts_trader/jax_agent/trainer.py`
*   **Issue**: In `_hybrid_value_loss`, debug printing is done via standard Python `print`:
    ```python
    if debug:
        print(f"    [VALUE DEBUG] Pred: ...")
    ```
*   **Impact**: When JIT-compiled, this prints *only* during the first tracing pass (showing Tracer info), and never during actual execution. This misleads the user.
*   **Fix**: Use `jax.debug.print` for runtime printing, or remove if unused.

### 3. Inefficient Metric Synchronization
*   **File**: `mcts_trader/jax_agent/trainer.py`
*   **Issue**: At the end of `train()`, metrics are converted to float one by one:
    ```python
    history[key] = [float(v) for v in history[key]]
    ```
*   **Impact**: `history[key]` contains JAX DeviceArrays. `float(v)` forces a device-to-host synchronization for *each element*. For 1000 steps, this triggers 1000 individual roundtrips, causing a noticeable hang at the end of training.
*   **Fix**: Stack arrays on device first, then pull to host once: `np.array(jnp.stack(history[key])).tolist()`.

## Plan
1.  Refactor `FlowMatchingGenerator.sample` to use `jax.lax.fori_loop`.
2.  Replace/remove `print` in `_hybrid_value_loss` in `trainer.py`.
3.  Optimize metric extraction in `trainer.py`.