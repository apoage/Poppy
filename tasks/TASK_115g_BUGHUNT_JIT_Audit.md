# TASK_115g: JAX/JIT Audit for Diff Teacher

**Status**: COMPLETED
**Owner**: Gemini
**Date**: 2026-01-25

## Goal
Audit `diff_teacher/` for JAX/JIT antipatterns, shape mismatches, and numerical stability issues.

## Findings

### Summary
| Bug ID | Severity | File | Status |
|--------|----------|------|--------|
| BUG-001 | **CRITICAL** | `diff_teacher/teacher.py` | OPEN |
| BUG-002 | **HIGH** | `diff_teacher/skeleton/encoder.py` | OPEN |
| BUG-003 | **HIGH** | `diff_teacher/skeleton/extraction.py` | OPEN |
| BUG-004 | **MEDIUM** | `diff_teacher/mamba/mamba_block.py` | OPEN |
| BUG-005 | **MEDIUM** | `diff_teacher/flow_matching.py` | OPEN |
| BUG-006 | LOW | `diff_teacher/flow_matching.py` | OPEN |

### Detail

#### BUG-001: Missing JIT Compilation on Training Steps
**File**: `diff_teacher/teacher.py:189` and `:221`
**Severity**: **CRITICAL**
**Status**: OPEN

```python
    def train_generator_step(self, latents, regimes=None):
        # ...
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.generator_state.params)
        self.generator_state = self.generator_state.apply_gradients(grads=grads)
```

**Problem**: The `train_generator_step` and `train_encoder_step` methods are not decorated with `@jax.jit`. `jax.value_and_grad` returns a function that is executed eagerly (or op-by-op) unless wrapped in a JIT context. This results in severe performance degradation compared to a compiled step. `DiffusionTeacher` is a class, so standard `@jax.jit` requires handling `self` (likely marking it static or refactoring).

**Fix**: Extract the update logic into a pure, static function (or `create_train_step` factory) that can be JIT-compiled, and call it from the method.

---

#### BUG-002: Non-deterministic Hashing in EventEncoder
**File**: `diff_teacher/skeleton/encoder.py:126`
**Severity**: **HIGH**
**Status**: OPEN

```python
    # Simple type hash
    type_idx = hash(event.event_type) % self.n_event_types
```

**Problem**: Python's `hash()` function is non-deterministic across process restarts (random seed). This means the model's behavior/embeddings for specific event strings will change every time the script is run, making the model unreproducible and potentially invalidating checkpoints if the mapping changes.

**Fix**: Use a deterministic hashing function (e.g., `zlib.adler32`) or a fixed string-to-index dictionary.

---

#### BUG-003: NumPy/JAX Mixing in Extraction
**File**: `diff_teacher/skeleton/extraction.py:75`
**Severity**: **HIGH** (if used in JIT), **NOTE** (if offline)
**Status**: OPEN

```python
def extract_correlation_clusters(returns: Array, n_clusters: int = 5) -> Array:
    # ...
    returns_np = np.array(returns)
    corr = np.corrcoef(returns_np.T)
```

**Problem**: `extract_correlation_clusters` casts a JAX Array to a NumPy array (`np.array(returns)`). If this function is ever called inside a JIT-compiled function (e.g., during on-the-fly skeleton extraction in a training loop), it will fail with a `TracerArrayConversionError`. Currently it appears to be used for offline extraction, but it's a dangerous trap.

**Fix**: Rewrite using `jnp` operations or clearly mark/docstring as CPU-only/Non-JIT.

---

#### BUG-004: Mamba Conv1D Implementation Mismatch
**File**: `diff_teacher/mamba/mamba_block.py:27`
**Severity**: **MEDIUM**
**Status**: OPEN

```python
        # 1D convolution
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding='VALID',
            feature_group_count=1,  # <--- HERE
        )(x_padded)
```

**Problem**: The Mamba architecture typically specifies a *depthwise* convolution (where `groups == features`) for the local context mixing. The current implementation uses a full dense convolution (`feature_group_count=1`). For a model dimension of 1024, this increases parameters from ~4k (depthwise) to ~4M (dense), significantly impacting compute and memory without the intended inductive bias.

**Fix**: Set `feature_group_count=self.features` for depthwise convolution.

---

#### BUG-005: Potential Recompilation in Flow Matching
**File**: `diff_teacher/flow_matching.py:387`
**Severity**: **MEDIUM**
**Status**: OPEN

```python
    def train_step(state, batch, key):
        # ...
        regime = batch.get("regime", None)
```

**Problem**: `batch` is a dictionary passed to a JIT-compiled function. JAX recompiles if the *structure* (keys) of the dictionary changes. If `regime` is sometimes present and sometimes not, this will trigger frequent recompilation.

**Fix**: Ensure data loader always provides the "regime" key, using `None` or a placeholder array consistently if missing.

---

#### BUG-006: Python Loops in Sampling
**File**: `diff_teacher/flow_matching.py:284`
**Severity**: LOW
**Status**: OPEN

```python
        for i in range(num_steps):
            t = jnp.full((shape[0],), i * dt)
            v = self.velocity_net(x, t, regime, training=False)
            x = x + dt * v
```

**Problem**: Uses a Python `for` loop for Euler integration. For small `num_steps` (1-10) this is fine (unrolled), but for accurate ODE solving (100+ steps), this creates a massive computation graph and increases compilation time/memory.

**Fix**: Use `jax.lax.scan` or `diffrax` for the integration loop.
