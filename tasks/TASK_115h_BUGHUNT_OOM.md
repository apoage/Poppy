# Bug Hunt: OOM & Stability in Mamba

**Date**: 2026-01-25
**Scope**: `diff_teacher/mamba/`
**Focus**: Memory explosion, numerical instability, efficient training.

## Findings

#### BUG-001: Massive Memory Footprint in Selective Scan
**File**: `diff_teacher/mamba/ssm.py:85`
**Severity**: CRITICAL
**Status**: OPEN

```python
# Discretize A: exp(delta * A)
# Result: (batch, seq, d_inner, d_state)
delta_A = jnp.exp(jnp.einsum('bsd,dn->bsdn', delta, A))

# Discretize B*x: delta * B * x
# Result: (batch, seq, d_inner, d_state)
delta_B_x = jnp.einsum('bsd,bsn,bsd->bsdn', delta, B, x)
```

**Problem**: 
The parallel selective scan materializes full 4D tensors `(batch, seq, d_inner, d_state)`.
With default config: `B=64, L=512, D=1536 (768*2), N=64`.
Size: `64 * 512 * 1536 * 64 * 4 bytes` ≈ **12.8 GB per tensor**.
Two tensors (`delta_A`, `delta_B_x`) = **25.6 GB** just for inputs to `associative_scan`.
This will immediately OOM on most GPUs (including 24GB consumer cards) for a single layer.
Mamba's efficiency claims usually rely on kernel fusion or lower `d_state`.

**Fix**:
1.  **Reduce `d_state`**: Default is 64, but original Mamba paper uses `N=16`. This reduces memory by 4x (3.2 GB per tensor).
2.  **Gradient Checkpointing**: Essential. The intermediate tensors must not be stored for backprop. Use `jax.checkpoint`.
3.  **Chunking**: More complex, but splitting the sequence into chunks can reduce peak memory.

#### BUG-002: Missing Gradient Checkpointing in Deep Mamba Stack
**File**: `diff_teacher/mamba/mamba_flow.py:90`
**Severity**: CRITICAL
**Status**: OPEN

```python
        for i in range(self.n_layers):
            # ...
            # BiMamba
            h = BiMambaBlockWithNorm(...)(h, ...)
```

**Problem**:
The generator uses 24 layers of `BiMambaBlock`. Given BUG-001, the memory requirement scales linearly with layers if activations are stored.
24 layers * 2 directions * 12.8 GB = **Impossible**.
Even with `d_state=16`, it is `24 * 2 * 3.2 GB` = ~150 GB context.

**Fix**:
Apply `nn.remat` (Flax's checkpointing) to `BiMambaBlock` or the layer loop. This trades compute for memory (re-computes forward pass during backward).

#### BUG-003: BiMamba Doubles Memory Pressure
**File**: `diff_teacher/mamba/bimamba.py:48`
**Severity**: HIGH
**Status**: OPEN

```python
        # Forward pass (causal: left to right)
        h_fwd = MambaBlock(...)(x, cond)

        # Backward pass (anti-causal: right to left)
        x_rev = jnp.flip(x, axis=1)
        h_bwd = MambaBlock(...)(x_rev, cond)
```

**Problem**:
Bidirectional modeling doubles the cost. It instantiates two full `MambaBlock`s per layer.
Combined with BUG-001 and BUG-002, this guarantees OOM.

**Fix**:
Sequential processing or strict checkpointing is required.

#### BUG-004: Excessive Default State Dimension
**File**: `diff_teacher/mamba/mamba_block.py:34`
**Severity**: MEDIUM
**Status**: OPEN

```python
    d_state: int = 64      # SSM state dimension
```

**Problem**:
Default `d_state=64` is very high for a "latent" state in SSMs, which is multiplicative to memory usage in the parallel scan. Mamba paper suggests 16.

**Fix**:
Change default `d_state` to 16.

## Summary

| Bug ID | Severity | Description | Fix |
|--------|----------|-------------|-----|
| BUG-001 | **CRITICAL** | Huge tensor materialization in `ssm.py` | `d_state=16` + `remat` |
| BUG-002 | **CRITICAL** | No checkpointing in 24-layer stack | Add `nn.remat` |
| BUG-003 | HIGH | BiMamba doubles memory usage | `remat` |
| BUG-004 | MEDIUM | Default `d_state=64` too high | Set to 16 |

## Immediate Actions Required

1.  Modify `mamba_block.py` and `bimamba.py` to default `d_state=16`.
2.  Add `@nn.remat` decorator to `BiMambaBlock` or `MambaBlock`.

## Test Suite Construction

**Goal**: Create a comprehensive test suite for the Market Generator in `diff_teacher/tests/`.

### Required Tests
1.  **`test_mamba_shapes.py`**: 
    - Verify tensor shapes through BiMamba and Hybrid blocks.
2.  **`test_variable_horizon.py`**: 
    - Ensure model handles different sequence lengths (JIT check).
3.  **`test_overfit.py`**: 
    - Train on a single batch of sinewaves, ensure loss goes to near-zero.
4.  **`test_sampling.py`**: 
    - Run sampling loop, check output shapes and finite values.

### Execution Plan
- Create the 4 test files.
- Run them with `pytest diff_teacher/tests/`.
- Fix any immediate crashes (like shape mismatches).
- **Do NOT** fix complex logic bugs yet, just ensure the harness runs.

### Code Guidelines
- Use `jax.random.PRNGKey(42)`
- Use `model.init` and `model.apply`
- Mock data with `jax.random.normal`
- Assert shapes like `assert out.shape == (B, L, D)`
