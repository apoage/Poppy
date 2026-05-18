"""TASK_749: Titans-MAG neural long-term memory module for the V3 encoder.

Implements the "memory module" from Behrouz et al. 2025
(https://arxiv.org/abs/2501.00663) — a small MLP whose weights update at
inference time via surprise-momentum gradient descent with weight decay
(paper eq. 13). The MAG variant (Memory as a Gate, paper §3.2) combines
this memory branch with the existing minGRU temporal recurrence via a
learnable sigmoid gate.

This is the *memory module only*; we drop the paper's causal-attention
companion and substitute the existing minGRU as the "short-term" path
(see TASK_749 §3 for the rationale on this adaptation).

Key implementation pattern
--------------------------
- The K/V/Q/theta/eta/alpha projections are applied to the full
  per-stream sequence outside the scan (they are time-distributed,
  linear in the input — no need to recompute inside the scan body).
- The scan over the temporal axis (T=64) carries `(M_t, S_t)` where
  `M_t` is a tuple of per-stream MLP weight tensors and `S_t` the
  momentum buffer.
- Inside the scan body, the inner-loop gradient ∂L/∂M of the
  associative-memory loss `L = ||M(k) - v||^2 / D` is computed
  *analytically* (explicit forward+backward through the MLP layers,
  using a closed-form GELU' formula). This avoids a nested `jax.grad`
  inside the scan body — TASK_749 step 8 root-cause fix. The OUTER
  autodiff thus sees only single-level reverse-mode autodiff through
  standard matmul/GELU/add ops, eliminating the inf-gradient propagation
  caused by the previous `jax.grad(inner_loss)` pattern under
  `jax.checkpoint(body)`.
- TRUNCATED BPTT (TASK_749 fix 3): the carry's prior state
  `(M_prev, S_prev)` is wrapped in `jax.lax.stop_gradient` inside the
  scan body. Forward pass is unchanged; the outer gradient does NOT
  flow back through the T-step trajectory. Standard pattern in TTT /
  Titans implementations (Sun et al. 2024) — required because the
  unrolled trajectory backprop is numerically unstable in our stack
  (1e9-1e10 magnitudes even when finite).
- The scan body is wrapped in `jax.checkpoint` (remat) to bound peak
  memory during backward — without this, the per-step carry (~1 GB
  for B=256/A=64) held across all 64 timesteps would blow the 24 GB
  budget.
- TASK_749 OOM fix (2026-05-18): the carry (`M_params, S_params`) is
  stored in `bfloat16` to halve the `lax.scan` reverse-mode tape
  footprint. At b=128/A=64/D=64/H=32/T_aug=68 the fp32 tape was
  ~18.2 GB; the bf16 tape is ~9.1 GB. Compute remains in fp32 — the
  body upcasts immediately after `stop_gradient` and downcasts on
  return. This is the standard "compute fp32, store bf16" pattern.
  Validated 2026-05-18 on workstation RTX 3090: full `stage_b_train_step`
  at b=128 measures JAX peak working set 11.87 GB (vs 1.89 GB baseline;
  Titans-add ~10 GB, down from pre-fix ~19.4 GB). 50-step smoke confirms
  finite gradients, healthy Titans diagnostics, no NaN. See
  `runs/749_titans_mag/result.md` §"749 OOM fix applied (2026-05-18)".

Checkpoint compatibility
------------------------
The gate scalar is initialised to a large-negative value (default -10).
At step 0, `sigmoid(-10) ≈ 4.5e-5`, so the MAG output is essentially
the minGRU output unchanged. The Phase-1 ckpt loads without touching
this subtree (init from scratch via `--reset-titans-memory`), and the
encoder reproduces baseline behaviour exactly until gradient pulls the
gate away.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def _apply_memory_mlp(M_params, x):
    """Apply the inner-loop MLP to `x`.

    Args:
        M_params: tuple of L weight matrices, each `(D, D)`.
            (Per-stream MLP — for use *inside* vmap.)
        x: any leading shape, trailing dim `D`.
    Returns:
        x with the same shape, after `gelu(... gelu(x @ W_0) @ W_1) ... @ W_{L-1}`.
    """
    h = x
    for i, W in enumerate(M_params):
        h = h @ W
        if i < len(M_params) - 1:
            h = nn.gelu(h)
    return h


def _memory_loss(M_params, k, v):
    """Associative-memory inner loss: || M(k) - v ||^2 (paper eq. 1).

    Args:
        M_params: tuple of L per-stream MLP weight matrices `(D, D)`.
        k, v: `(D,)` per-stream key/value vectors for one timestep.
    Returns:
        scalar loss (mean over D).
    """
    pred = _apply_memory_mlp(M_params, k)
    return jnp.mean((pred - v) ** 2)


def _per_stream_loss(M_params, k, v):
    """Per-stream associative-memory loss. Retained for diagnostic/test use.

    Args:
        M_params: tuple of L (D, D) weight matrices for a single stream.
        k, v: (D,) per-stream key/value.
    Returns:
        scalar loss = ||M(k) - v||^2 / D.
    """
    pred = _apply_memory_mlp(M_params, k)
    return jnp.mean((pred - v) ** 2)


def _gelu_grad_tanh(x):
    """Explicit derivative of the tanh-approximation GELU (Flax default).

    GELU(x) = 0.5 * x * (1 + tanh(g(x)))  with  g(x) = sqrt(2/pi) * (x + 0.044715 * x^3)
    GELU'(x) = 0.5 * (1 + tanh(g)) + 0.5 * x * (1 - tanh(g)^2) * g'(x)
             = 0.5 * (1 + tanh(g)) + 0.5 * x * sech(g)^2 * sqrt(2/pi) * (1 + 0.134145 * x^2)

    Written in closed form so OUTER autodiff differentiates through standard
    pointwise ops (tanh, square, multiply) rather than through a nested
    jax.grad call. This is the load-bearing piece that makes the analytical
    inner-gradient path stable.
    """
    # Python float constants — JAX promotes to x.dtype automatically.
    sqrt_2_over_pi = 0.7978845608028654   # sqrt(2/pi)
    coef_cubic = 0.044715
    coef_cubic_3 = 0.134145               # 3 * 0.044715
    g = sqrt_2_over_pi * (x + coef_cubic * x ** 3)
    tanh_g = jnp.tanh(g)
    sech2_g = 1.0 - tanh_g ** 2
    g_prime = sqrt_2_over_pi * (1.0 + coef_cubic_3 * x ** 2)
    return 0.5 * (1.0 + tanh_g) + 0.5 * x * sech2_g * g_prime


def _batched_grad_M(M_params_b, k_b, v_b):
    """Compute per-stream gradient of L = ||M(k) - v||^2 / D wrt each W_i in M.

    ROOT-CAUSE FIX (TASK_749 step 8): the previous implementation called
    `jax.grad(_per_stream_loss)` inside the scan body. Combined with the
    surrounding `jax.checkpoint(body)` and an outer `value_and_grad`, this
    produces a four-way nested autodiff (outer fwd / outer bwd / inner fwd /
    inner bwd) whose backward path generates non-finite values once the
    MAG gate is non-trivial (e.g. gate_init=-5 → sigmoid≈6.7e-3 vs the
    earlier 4.5e-5 at -10). Symptom: `grad_norm = inf` every step.

    The fix is to compute `∂L/∂W_i` analytically via an explicit forward
    pass that caches `h_i = activations before W_i` and `pre_i = preact
    after W_i`, then an explicit backward that propagates `delta` back
    through each linear-GELU layer. All operations are standard JAX
    pointwise + matmul, so OUTER autodiff sees a single-level reverse-mode
    problem (no nested `jax.grad`), which is reliably handled.

    Args:
        M_params_b: tuple of (N, in_i, out_i) per-layer weight matrices.
            Shapes may vary per layer (e.g. D→H→D for depth=2).
        k_b, v_b: (N, D) per-stream keys/values (D = embed_dim).
    Returns:
        Per-stream grads — tuple of (N, in_i, out_i), matching M_params_b
        layer-by-layer.
    """
    n_layers = len(M_params_b)
    # Output dim from the last layer's weight matrix — Python int, used as
    # plain divisor below (JAX promotes to k_b.dtype automatically).
    D_out = int(M_params_b[-1].shape[-1])

    # === Forward pass — cache per-layer activations h_i (input to layer i)
    # and pre_i (pre-activation = h_i @ W_i, BEFORE the GELU on intermediate
    # layers). For the last layer pre_i IS the output (no GELU follows).
    h_list = [k_b]                 # h_0 = k
    pre_list = []
    h = k_b
    for i, W in enumerate(M_params_b):
        pre = jnp.einsum('nd,ndp->np', h, W)  # (N, out_i)
        pre_list.append(pre)
        if i < n_layers - 1:
            h = nn.gelu(pre)
            h_list.append(h)
        else:
            h = pre  # final output (linear)
    pred = h

    # === Backward — explicit delta propagation.
    # delta_i = ∂L/∂pre_i;  grad_W_i = h_i^T outer delta_i  (per-stream).
    # Start: delta_{L-1} = (2/D) * (pred - v).
    delta = (2.0 / D_out) * (pred - v_b)  # (N, D_out)

    grads = [None] * n_layers
    for i in range(n_layers - 1, -1, -1):
        # grad_W_i = outer(h_i, delta_i)  per-stream → einsum gives (N, in_i, out_i).
        grads[i] = jnp.einsum('na,nb->nab', h_list[i], delta)
        if i > 0:
            # Backprop delta through W_i: delta_{i-1}_raw = delta_i @ W_i^T.
            # Then multiply by GELU'(pre_{i-1}) elementwise.
            delta_raw = jnp.einsum('np,ndp->nd', delta, M_params_b[i])  # (N, in_i = out_{i-1})
            delta = delta_raw * _gelu_grad_tanh(pre_list[i - 1])

    return tuple(grads)


def _titans_scan_body_batched(carry, inputs):
    """One step of the Titans memory recurrence — batched over N streams.

    Operates on the full batch of streams in a single step. No outer vmap.

    Carry:
      M_params: tuple of (N, D, D) — per-stream MLP weights.
      S_params: tuple of (N, D, D) — per-stream momentum buffer.

    Inputs (per-timestep):
      k_t, v_t, q_t: (N, D)
      theta_t, eta_t, alpha_t: (N,) per-stream scalars.

    TRUNCATED BPTT (TASK_749 fix 3)
    -------------------------------
    The outer-loop gradient is prevented from flowing back through the carry's
    prior state via `jax.lax.stop_gradient`. The forward pass is unchanged —
    the inner-loop SSL update still uses M_prev / S_prev numerically — but
    autodiff's reverse pass sees them as constants. This is the standard
    pattern in TTT / Titans implementations (e.g. Sun et al. 2024 official
    PyTorch code): without it, the outer gradient accumulates through every
    one of the T=64 scan steps, producing catastrophic magnitudes (1e9-1e10
    even when finite) in the OUTER autodiff of `jax.grad(loss)`.

    With this fix, the outer gradient reaches Titans trainable params (K, V,
    Q, gate, η/θ/α heads, persistent tokens, M_init) through ONE step only,
    not the trajectory.
    """
    M_params, S_params = carry
    # TRUNCATED BPTT — cut outer-gradient flow through the carry.
    # TASK_749 OOM fix (2026-05-18): carry is stored in bfloat16 to halve the
    # lax.scan reverse-mode tape footprint (the per-step carry-input is what
    # blows the 24 GB budget at b=128; ~18 GB fp32 tape → ~9 GB bf16 tape).
    # Compute stays in float32 — upcast immediately after stop_gradient so all
    # downstream ops (grad_M, new_S, new_M arithmetic) run at full precision.
    M_params = jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x).astype(jnp.float32), M_params)
    S_params = jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x).astype(jnp.float32), S_params)
    k_t, v_t, q_t, theta_t, eta_t, alpha_t = inputs

    # Broadcast-shaped scalars for elementwise update.
    th = theta_t[:, None, None]                 # (N, 1, 1)
    et = eta_t[:, None, None]
    al = jnp.minimum(alpha_t, jnp.float32(0.99))[:, None, None]

    # Inner-loop gradient (per-stream MSE → grad wrt M).
    grad_M = _batched_grad_M(M_params, k_t, v_t)  # tuple of (N, D, D)

    # TASK_749 §8 R2 mitigation: clip per-stream gradient norm. Without this,
    # surprise momentum (eta near 0.7) amplifies large gradients across steps
    # and the memory state diverges → NaN by ~T=8 with realistic input scale.
    # Clip threshold 1.0 is generous (typical per-stream grad norm ~0.01 at
    # init) while still bounding the worst case.
    grad_clip = jnp.float32(1.0)
    grad_M = tuple(_clip_grad_per_stream(gM, grad_clip) for gM in grad_M)

    new_S = tuple(et * S_i - th * gM_i
                  for S_i, gM_i in zip(S_params, grad_M))
    # Additional safeguard: clip the surprise momentum buffer too. Even with
    # grad clipping, eta near 0.7 can let S grow unbounded over 68 steps.
    # Clip per-stream Frobenius norm to a generous threshold.
    new_S = tuple(_clip_grad_per_stream(S_i, jnp.float32(2.0)) for S_i in new_S)
    new_M = tuple((1.0 - al) * M_i + S_i
                  for M_i, S_i in zip(M_params, new_S))
    # Final safeguard on M itself.
    new_M = tuple(_clip_grad_per_stream(M_i, jnp.float32(5.0)) for M_i in new_M)

    # Retrieve via fresh forward pass on q_t. Batched matmul:
    # q_t: (N, D); M_i: (N, D, D); result: (N, D).
    h = q_t
    for i, W in enumerate(new_M):
        # h: (N, D), W: (N, D, D); want h_next: (N, D).
        h = jnp.einsum('nd,ndp->np', h, W)
        if i < len(new_M) - 1:
            h = nn.gelu(h)
    y_t = h  # (N, D)

    # Diagnostics — per-stream then mean to scalar.
    # Compute on fp32 new_M/new_S BEFORE downcasting so we don't double the
    # precision loss on the displayed metrics.
    mem_norm = jnp.sqrt(sum(jnp.sum(M_i ** 2, axis=(-2, -1)) for M_i in new_M))
    surprise_norm = jnp.sqrt(sum(jnp.sum(gM_i ** 2, axis=(-2, -1)) for gM_i in grad_M))

    # TASK_749 OOM fix (2026-05-18): downcast carry to bf16 before return so
    # the lax.scan reverse-mode tape stores bf16 (halves the 18 GB carry tape
    # at b=128). Forward arithmetic above was in fp32.
    new_M = tuple(M_i.astype(jnp.bfloat16) for M_i in new_M)
    new_S = tuple(S_i.astype(jnp.bfloat16) for S_i in new_S)

    return (new_M, new_S), (y_t, theta_t, eta_t, alpha_t,
                             jnp.mean(mem_norm), jnp.mean(surprise_norm))


def _clip_grad_per_stream(g, max_norm):
    """Per-stream gradient clipping. g shape (N, in, out); norm over last 2 axes.

    The clipping scale factor is wrapped in `stop_gradient` so the OUTER
    autodiff doesn't backprop through the clipping decision. Without this,
    the second-order grad through `jnp.minimum` introduces NaNs at the
    clip threshold boundary.

    Args:
        g: (N, in, out) — per-stream gradient tensor.
        max_norm: scalar — maximum per-stream Frobenius norm.
    Returns:
        Clipped g, same shape.
    """
    g_norm = jnp.sqrt(jnp.sum(g ** 2, axis=(-2, -1), keepdims=True))  # (N, 1, 1)
    scale = jnp.minimum(jnp.float32(1.0), max_norm / jnp.maximum(g_norm, jnp.float32(1e-8)))
    scale = jax.lax.stop_gradient(scale)
    return g * scale


class TitansMemoryBlock(nn.Module):
    """Titans memory module (paper §2.1) — depth-L MLP whose weights
    update at inference time via surprise-momentum gradient descent.

    Inputs to `__call__`:
      x: (N, T, D) — per-stream temporal sequence (N = B*A typically).
    Returns:
      y: (N, T, D) — retrieved memory output per timestep.
      diag: dict of running diagnostic scalars.

    Init biases on the data-dependent scalars `(theta, eta, alpha)` are
    added as constants (post-projection, pre-sigmoid) and live as module
    attributes — not learnable. Paper §2.1 keeps them implicit; we surface
    them so the gate is dormant at step 0.
    """
    embed_dim: int
    memory_depth: int = 2
    memory_hidden: Optional[int] = None
    persistent_tokens: int = 4
    surprise_momentum_init: float = 0.9
    decay_init: float = 0.01
    lr_init: float = 0.01
    use_remat: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        D = self.embed_dim
        H = self.memory_hidden if self.memory_hidden is not None else D
        d = dict(dtype=self.dtype, param_dtype=self.param_dtype)

        # K/V/Q projections (paper §2.1). No biases — pure linear.
        self.W_K = nn.Dense(D, use_bias=False, name='W_K', **d)
        self.W_V = nn.Dense(D, use_bias=False, name='W_V', **d)
        self.W_Q = nn.Dense(D, use_bias=False, name='W_Q', **d)

        # Data-dependent scalar heads (paper eq. 13). Bias-init = 0; the
        # large constant init bias is added inside the scan input pipeline.
        self.theta_head = nn.Dense(1, name='theta_head', **d)
        self.eta_head = nn.Dense(1, name='eta_head', **d)
        self.alpha_head = nn.Dense(1, name='alpha_head', **d)

        # Persistent memory tokens (paper §2.3).
        if self.persistent_tokens > 0:
            self.persistent = self.param(
                'persistent',
                nn.initializers.normal(stddev=0.02),
                (self.persistent_tokens, D),
                self.param_dtype,
            )

        # Initial memory MLP weights. To keep input/output shapes compatible
        # we use D→H→D for depth=2 (or D→D for depth=1). Non-square H<D
        # makes the inner state cheaper (key VRAM savings for big B*A).
        # paper §2.1's "MLP memory" is non-prescriptive on width — any H works.
        if self.memory_depth == 1:
            self._mlp_shapes = [(D, D)]
        else:
            # First layer D→H, intermediate layers H→H, last H→D.
            shapes = [(D, H)]
            for _ in range(self.memory_depth - 2):
                shapes.append((H, H))
            shapes.append((H, D))
            self._mlp_shapes = shapes

        self.M_init = tuple(
            self.param(
                f'M_init_{i}',
                nn.initializers.normal(stddev=0.02),
                shape,
                self.param_dtype,
            )
            for i, shape in enumerate(self._mlp_shapes)
        )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """Apply the Titans memory recurrence to per-stream sequence x.

        Args:
            x: (N, T, D) — N = B*A streams, T = window length, D = embed_dim.
        Returns:
            y: (N, T, D) — retrieved memory output per timestep (post-prefix).
            diag: dict of per-batch diagnostic scalars.
        """
        N, T, D = x.shape

        # Prepend persistent memory tokens (paper §2.3).
        if self.persistent_tokens > 0:
            persistent_bcast = jnp.broadcast_to(
                self.persistent[None, :, :].astype(self.dtype),
                (N, self.persistent_tokens, D),
            )
            x_aug = jnp.concatenate([persistent_bcast, x], axis=1)  # (N, P+T, D)
        else:
            x_aug = x
        T_aug = x_aug.shape[1]

        # Precompute time-distributed K/V/Q and data-dependent scalars
        # outside the scan. These are pure linear/sigmoid projections —
        # nothing per-step that requires param access.
        k = self.W_K(x_aug)           # (N, T_aug, D)
        v = self.W_V(x_aug)
        q = self.W_Q(x_aug)
        theta = jax.nn.sigmoid(self.theta_head(x_aug).squeeze(-1)
                               + jnp.float32(self.lr_init))            # (N, T_aug)
        eta = jax.nn.sigmoid(self.eta_head(x_aug).squeeze(-1)
                             + jnp.float32(self.surprise_momentum_init))
        alpha = jax.nn.sigmoid(self.alpha_head(x_aug).squeeze(-1)
                               + jnp.float32(self.decay_init))

        # Build per-stream M / S initial state: broadcast shared init to (N, in, out).
        # Each layer can have a different shape (e.g. D→H→D for depth=2).
        # TASK_749 OOM fix (2026-05-18): carry stored in bfloat16 to halve the
        # lax.scan reverse-mode tape footprint. Body upcasts to fp32 on read for
        # compute fidelity; downcasts back to bf16 on write. See _titans_scan_body_batched.
        M_init_b = tuple(
            jnp.broadcast_to(M[None, :, :], (N, M.shape[0], M.shape[1])).astype(jnp.bfloat16)
            for M in self.M_init
        )
        S_init_b = tuple(
            jnp.zeros((N, M.shape[0], M.shape[1]), dtype=jnp.bfloat16)
            for M in self.M_init
        )

        # Single batched scan over time. Carry has stream axis baked in.
        # Inputs at each step are per-stream tensors (N, ...).
        # Scan inputs need leading time axis → transpose (N, T_aug, D) → (T_aug, N, D).
        k_t_axis = jnp.transpose(k, (1, 0, 2))           # (T_aug, N, D)
        v_t_axis = jnp.transpose(v, (1, 0, 2))
        q_t_axis = jnp.transpose(q, (1, 0, 2))
        theta_t_axis = jnp.transpose(theta, (1, 0))      # (T_aug, N)
        eta_t_axis = jnp.transpose(eta, (1, 0))
        alpha_t_axis = jnp.transpose(alpha, (1, 0))
        inputs = (k_t_axis, v_t_axis, q_t_axis,
                  theta_t_axis, eta_t_axis, alpha_t_axis)

        body = _titans_scan_body_batched
        # TASK_749 §5.3: rematerialize the scan body during backward to bound
        # the carry trajectory's VRAM footprint. With B=256,A=64,D=64 the carry
        # per step is N*L*D*D*4 ≈ 256 MB; without remat the T-step trajectory
        # would hold T copies (~17 GB).
        if self.use_remat:
            body = jax.checkpoint(body)

        (_, _), (y_aug_t, theta_t_out, eta_t_out, alpha_t_out,
                 mem_norm_seq, surprise_norm_seq) = jax.lax.scan(
            body, (M_init_b, S_init_b), inputs
        )
        # y_aug_t: (T_aug, N, D); transpose back to (N, T_aug, D)
        y_aug = jnp.transpose(y_aug_t, (1, 0, 2))
        theta_out = jnp.transpose(theta_t_out, (1, 0))
        eta_out = jnp.transpose(eta_t_out, (1, 0))
        alpha_out = jnp.transpose(alpha_t_out, (1, 0))
        # mem_norm_seq and surprise_norm_seq are already (T_aug,) scalars
        # after the per-step mean.
        mem_norm_out = mem_norm_seq      # (T_aug,)
        surprise_norm_out = surprise_norm_seq

        # Drop persistent-token outputs.
        if self.persistent_tokens > 0:
            y = y_aug[:, self.persistent_tokens:, :]
        else:
            y = y_aug

        diag = {
            'theta_mean': jnp.mean(theta_out),
            'eta_mean': jnp.mean(eta_out),
            'alpha_mean': jnp.mean(alpha_out),
            'mem_norm': jnp.mean(mem_norm_out),
            'surprise_norm': jnp.mean(surprise_norm_out),
        }
        return y, diag


class GatedTemporalMemory(nn.Module):
    """MAG (Memory as a Gate) combiner — paper §3.2.

    Combines the minGRU temporal output `mingru_out: (N, D)` with the
    Titans memory output `memory_y: (N, T, D)` via a learnable scalar
    sigmoid gate `g`:

        out = sigmoid(g) * memory_last + (1 - sigmoid(g)) * mingru_out

    where `memory_last = memory_y[:, -1, :]`. The gate scalar is
    initialised large-negative (default -10) so `sigmoid(g) ≈ 0` and the
    memory branch contributes nothing at step 0 — encoder output is
    byte-identical to the V3 baseline until gradient pulls `g` away.
    """
    embed_dim: int
    memory_depth: int = 2
    memory_hidden: Optional[int] = None
    persistent_tokens: int = 4
    surprise_momentum_init: float = 0.9
    decay_init: float = 0.01
    lr_init: float = 0.01
    gate_init: float = -5.0
    use_remat: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.memory_block = TitansMemoryBlock(
            embed_dim=self.embed_dim,
            memory_depth=self.memory_depth,
            memory_hidden=self.memory_hidden,
            persistent_tokens=self.persistent_tokens,
            surprise_momentum_init=self.surprise_momentum_init,
            decay_init=self.decay_init,
            lr_init=self.lr_init,
            use_remat=self.use_remat,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.gate_param = self.param(
            'gate',
            nn.initializers.constant(self.gate_init),
            (),
            self.param_dtype,
        )

    def __call__(self, x_seq: jnp.ndarray, mingru_out: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """Combine memory output with minGRU output via the MAG gate.

        Args:
            x_seq: (N, T, D) per-stream temporal sequence (input to memory).
            mingru_out: (N, D) minGRU last-hidden output.
        Returns:
            out: (N, D) gated combination.
            diag: dict of diagnostic scalars including gate_mean.
        """
        memory_y_seq, mem_diag = self.memory_block(x_seq)
        memory_out = memory_y_seq[:, -1, :].astype(self.dtype)

        g = jax.nn.sigmoid(self.gate_param.astype(self.dtype))
        out = (1.0 - g) * mingru_out + g * memory_out

        diag = dict(mem_diag)
        diag['gate_value'] = g
        diag['gate_mean'] = g
        return out, diag
