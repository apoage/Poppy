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
- Inside the scan body, `jax.grad` computes the inner-loop gradient of
  the associative-memory loss with respect to `M`. JAX handles the
  second-order autodiff natively when the outer-loop training loss
  backpropagates through the scan.
- The scan body is wrapped in `jax.checkpoint` (remat) to bound peak
  memory during backward — without this, the per-step carry (~1 GB
  for B=256/A=64) held across all 64 timesteps would blow the 24 GB
  budget.

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


def _titans_scan_body(carry, inputs):
    """One step of the Titans memory recurrence inside vmap.

    Carry (per-stream):
      M_params: tuple of (D, D) weight matrices.
      S_params: tuple of (D, D) momentum buffers.

    Inputs (per-stream, per-timestep, pre-projected):
      k_t, v_t, q_t: (D,) — keys/values/queries
      theta_t, eta_t, alpha_t: scalars — data-dependent updates
    """
    M_params, S_params = carry
    k_t, v_t, q_t, theta_t, eta_t, alpha_t = inputs

    # Clip alpha away from exactly 1 to keep memory non-degenerate.
    alpha_t = jnp.minimum(alpha_t, jnp.float32(0.99))

    # Inner-loop gradient of MSE wrt M (paper eq. 1 → grad → paper eq. 13).
    grad_M = jax.grad(_memory_loss, argnums=0)(M_params, k_t, v_t)

    # Momentum + decay update (paper eqs. 2, 3, 13).
    new_S = tuple(eta_t * S_i - theta_t * gM_i
                  for S_i, gM_i in zip(S_params, grad_M))
    new_M = tuple((1.0 - alpha_t) * M_i + S_i
                  for M_i, S_i in zip(M_params, new_S))

    # Retrieve via fresh forward pass on q_t (paper §2.1).
    y_t = _apply_memory_mlp(new_M, q_t)  # (D,)

    # Diagnostics — scalar per step (post-vmap they'll be per-stream).
    mem_norm = jnp.sqrt(sum(jnp.sum(M_i ** 2) for M_i in new_M))
    surprise_norm = jnp.sqrt(sum(jnp.sum(gM_i ** 2) for gM_i in grad_M))

    return (new_M, new_S), (y_t, theta_t, eta_t, alpha_t, mem_norm, surprise_norm)


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

        if H != D:
            raise NotImplementedError(
                f"titans memory: memory_hidden ({H}) != embed_dim ({D}) "
                "is reserved for future work. Leave memory_hidden=None."
            )

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

        # Initial memory MLP weights — one (D, D) per layer.
        self.M_init = tuple(
            self.param(
                f'M_init_{i}',
                nn.initializers.normal(stddev=0.02),
                (D, D),
                self.param_dtype,
            )
            for i in range(self.memory_depth)
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

        # Build per-stream M / S initial state: broadcast shared init to (N, D, D).
        M_init_b = tuple(
            jnp.broadcast_to(M[None, :, :], (N, D, D)).astype(self.dtype)
            for M in self.M_init
        )
        S_init_b = tuple(jnp.zeros((N, D, D), dtype=self.dtype) for _ in range(self.memory_depth))

        # Per-stream scan: T_aug steps, carries (M_params, S_params).
        # Inputs per step: k_t/v_t/q_t (D,), theta/eta/alpha scalars.
        body = _titans_scan_body
        if self.use_remat:
            body = jax.checkpoint(body)

        def per_stream(M_s, S_s, k_s, v_s, q_s, theta_s, eta_s, alpha_s):
            """Run the time-axis scan for a single stream."""
            inputs = (k_s, v_s, q_s, theta_s, eta_s, alpha_s)
            (_, _), (y_seq, theta_seq, eta_seq, alpha_seq,
                     mem_norm_seq, surprise_norm_seq) = jax.lax.scan(
                body, (M_s, S_s), inputs
            )
            return y_seq, theta_seq, eta_seq, alpha_seq, mem_norm_seq, surprise_norm_seq

        # vmap over the stream axis. Pytree (M_s, S_s) maps axis 0; arrays
        # k/v/q (N, T_aug, D) and scalars theta/eta/alpha (N, T_aug) map
        # axis 0 too.
        vmapped = jax.vmap(per_stream, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
        y_aug, theta_out, eta_out, alpha_out, mem_norm_out, surprise_norm_out = vmapped(
            M_init_b, S_init_b, k, v, q, theta, eta, alpha
        )

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
    gate_init: float = -10.0
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
