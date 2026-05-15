import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple

from .titans_memory import GatedTemporalMemory


class FiLMProjector(nn.Module):
    """TASK_741 (Mint): post-pool per-asset scalar injection via FiLM.

    Zero-init on final Dense: γ=0, β=0 at step 0 → output equals input.
    """
    embed_dim: int
    hidden_dim: int = 64
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, scalars: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        d = dict(dtype=self.dtype, param_dtype=self.param_dtype)
        h = nn.gelu(nn.Dense(self.hidden_dim, name='film_1', **d)(
            scalars.astype(self.dtype)))
        out = nn.Dense(2 * self.embed_dim,
                       kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros,
                       name='film_out', **d)(h)
        gamma, beta = jnp.split(out, 2, axis=-1)
        return gamma, beta


class StreamingAssetEncoder(nn.Module):
    """
    Asset-Identity Preserving Encoder with Temporal Processing (V3).

    Replaces V2's flatten(window×channels) → Dense with a minGRU recurrence
    that preserves temporal ordering. Cross-asset attention is unchanged.

    Architecture:
      1. Per-timestep feature projection: Dense(channels → embed_dim)
      2. minGRU temporal recurrence via associative_scan (parallel training)
      3. Asset identity embedding + cross-asset self-attention
      4. Global projection to state_dim

    Input shape: (batch, channels, window_size, n_assets)
    Output shape: (batch, state_dim)

    Compared to V2:
    - Preserves temporal ordering (V2 destroys it by flattening)
    - Fewer parameters (27→32 + 2×32→32 vs 1620→32)
    - Supports streaming inference via step() method
    """
    state_dim: int = 64
    n_assets: int = 64
    embed_dim: int = 32
    num_heads: int = 4
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    # TASK_741 (Mint): post-pool per-asset scalar injection
    per_asset_scalar_dims: int = 0
    film_hidden_dim: int = 64
    # TASK_749 (Titans-MAG): neural memory module on the temporal axis,
    # gated against the existing minGRU branch. Toggleable so existing
    # Phase-1 checkpoints load unchanged (use_titans_memory=False by default;
    # when True the gate is init large-negative → memory dormant at step 0).
    use_titans_memory: bool = False
    titans_memory_depth: int = 2
    titans_memory_hidden: Optional[int] = None
    titans_persistent_tokens: int = 4
    titans_surprise_momentum_init: float = 0.9
    titans_decay_init: float = 0.01
    titans_lr_init: float = 0.01
    titans_gate_init: float = -10.0

    def setup(self):
        d = dict(dtype=self.dtype, param_dtype=self.param_dtype)

        # Temporal layers (shared between __call__ and step)
        self.temporal_proj = nn.Dense(self.embed_dim, **d)
        self.gate = nn.Dense(self.embed_dim, **d)
        self.candidate = nn.Dense(self.embed_dim, **d)

        # Cross-asset layers
        self.pre_attn_norm = nn.LayerNorm(**d)
        self.cross_attn = nn.SelfAttention(num_heads=self.num_heads, **d)
        self.pre_proj_norm = nn.LayerNorm(**d)
        self.global_proj = nn.Dense(self.state_dim, **d)

        # TASK_741 (Mint): instantiate FiLM only when injection enabled
        if self.per_asset_scalar_dims > 0:
            self.film_projector = FiLMProjector(
                embed_dim=self.embed_dim,
                hidden_dim=self.film_hidden_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        # TASK_749 (Titans-MAG): instantiate the gated memory only when enabled.
        # Subtree key on the param dict is 'titans' (see __call__).
        if self.use_titans_memory:
            self.titans = GatedTemporalMemory(
                embed_dim=self.embed_dim,
                memory_depth=self.titans_memory_depth,
                memory_hidden=self.titans_memory_hidden,
                persistent_tokens=self.titans_persistent_tokens,
                surprise_momentum_init=self.titans_surprise_momentum_init,
                decay_init=self.titans_decay_init,
                lr_init=self.titans_lr_init,
                gate_init=self.titans_gate_init,
                use_remat=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True,
                 return_per_asset: bool = False,
                 per_asset_scalars: Optional[jnp.ndarray] = None):
        """Training mode: parallel over full window via associative_scan.

        Args:
            x: (batch, channels, window_size, n_assets) — e.g. (B, 27, 64, 64)
            return_per_asset: TASK_729c — if True, also return post-attention
                per-asset features (batch, n_assets, embed_dim). Mirrors V2.
            per_asset_scalars: TASK_741 (Mint) — optional (batch, n_assets, n_s)
                injected via FiLM after pre_proj_norm. None → no-op.
        Returns:
            state: (batch, state_dim)
            or (state, per_asset_feats) when return_per_asset=True.
        """
        batch, channels, window, n_assets = x.shape

        # (B, C, T, A) → (B, A, T, C)
        x = jnp.transpose(x, (0, 3, 2, 1))

        # Per-timestep projection: (B, A, T, C) → (B, A, T, D)
        x = self.temporal_proj(x)

        # Reshape for per-asset temporal processing: (B*A, T, D)
        x = x.reshape(batch * n_assets, window, self.embed_dim)

        # minGRU gates (input-only, no hidden state dependency)
        z = nn.sigmoid(self.gate(x))       # (B*A, T, D)
        h_tilde = jnp.tanh(self.candidate(x))  # (B*A, T, D)

        # Parallel scan: h_t = (1-z_t) * h_{t-1} + z_t * h_tilde_t
        h_final = _mingru_scan(z, h_tilde)  # (B*A, D)

        # TASK_749 (Titans-MAG): gate the minGRU output with Titans memory
        # operating on the per-stream sequence x: (B*A, T, D). At step 0
        # sigmoid(gate_init=-10) ≈ 4.5e-5, so h_final is essentially unchanged.
        if self.use_titans_memory:
            h_final, _titans_diag = self.titans(x, h_final)
            # _titans_diag = {theta_mean, eta_mean, alpha_mean, mem_norm,
            #   surprise_norm, gate_value, gate_mean}. We stash it in a Flax
            # sow-able slot so train_phase2.py can pick it up; for now we
            # rely on the loss-side to recompute these if logged.
            self.sow('intermediates', 'titans_diag', _titans_diag)

        # Reshape to (B, A, D)
        h = h_final.reshape(batch, n_assets, self.embed_dim)

        # Asset identity + attention + projection (shared path)
        state, per_asset_feats = self._post_recurrence(
            h, batch, n_assets, per_asset_scalars=per_asset_scalars
        )
        if return_per_asset:
            return state, per_asset_feats
        return state

    def step(
        self,
        x_t: jnp.ndarray,
        h_prev: Optional[jnp.ndarray] = None,
        per_asset_scalars: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Streaming mode: process single timestep, carry hidden state.

        Args:
            x_t: (batch, channels, n_assets) — single timestep
            h_prev: (batch, n_assets, embed_dim) — previous hidden state, or None
            per_asset_scalars: TASK_741 — optional (batch, n_assets, n_s) FiLM input.
        Returns:
            state: (batch, state_dim) — current latent state
            h_new: (batch, n_assets, embed_dim) — updated hidden state

        NOTE (TASK_749): when `use_titans_memory=True`, the streaming path
        falls back to the minGRU branch only — Titans memory is not
        applied here (it requires a temporal sequence, not a single
        timestep). The MAG gate is dormant at init so this is a no-op
        regression at step 0. Persistent cross-call memory is deferred
        (Candidate C in TASK_749 §2). For MCTS use of representation()
        the __call__ path (with full T-window) is used.
        """
        batch, channels, n_assets = x_t.shape

        # (B, C, A) → (B, A, C)
        x_t = jnp.transpose(x_t, (0, 2, 1))

        # Per-timestep projection: (B, A, C) → (B, A, D)
        x_t = self.temporal_proj(x_t)

        # minGRU recurrent step
        z = nn.sigmoid(self.gate(x_t))       # (B, A, D)
        h_tilde = jnp.tanh(self.candidate(x_t))  # (B, A, D)

        if h_prev is None:
            h_prev = jnp.zeros_like(h_tilde)

        h_new = (1 - z) * h_prev + z * h_tilde  # (B, A, D)

        # Asset identity + attention + projection (shared path)
        state, _ = self._post_recurrence(
            h_new, batch, n_assets, per_asset_scalars=per_asset_scalars
        )
        return state, h_new

    @nn.compact
    def _post_recurrence(
        self, h: jnp.ndarray, batch: int, n_assets: int,
        per_asset_scalars: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Shared post-recurrence: identity embed + attention + projection.

        Args:
            h: (batch, n_assets, embed_dim)
            per_asset_scalars: TASK_741 — (batch, n_assets, n_s) FiLM modulation
                applied post-LN, pre-projection. None → identity.
        Returns:
            state: (batch, state_dim)
            per_asset_feats: (batch, n_assets, embed_dim) — post-LN, pre-pool.
                TASK_729c: exposed for DirectionHead + PerAssetPolicyHead.
        """
        # Asset identity embedding
        pos_embed = self.param(
            'asset_identity_embed',
            nn.initializers.normal(stddev=0.02),
            (1, self.n_assets, self.embed_dim),
        )
        h = h + pos_embed[:, :n_assets, :]

        # Cross-asset self-attention
        h = h + self.cross_attn(self.pre_attn_norm(h))

        # LayerNorm before global pool — this is the per-asset feature surface
        per_asset_feats = self.pre_proj_norm(h)  # (batch, n_assets, embed_dim)

        # TASK_741 (Mint): FiLM injection of per-asset scalars on the post-LN
        # surface. Zero-init γ, β → byte-identical at step 0.
        if self.per_asset_scalar_dims > 0 and per_asset_scalars is not None:
            gamma, beta = self.film_projector(per_asset_scalars)
            per_asset_feats = (1.0 + gamma) * per_asset_feats + beta

        # Global projection
        state = self.global_proj(per_asset_feats.reshape(batch, -1))
        return state, per_asset_feats


def _mingru_scan(z: jnp.ndarray, h_tilde: jnp.ndarray) -> jnp.ndarray:
    """Parallel minGRU via associative scan.

    h_t = (1-z_t) * h_{t-1} + z_t * h_tilde_t  (with h_{-1} = 0)

    Rewritten as affine recurrence: h_t = a_t * h_{t-1} + b_t
    where a_t = (1-z_t), b_t = z_t * h_tilde_t

    Binary operator: (a2, b2) ⊕ (a1, b1) = (a2*a1, a2*b1 + b2)

    Args:
        z: (N, T, D) gate values in [0, 1]
        h_tilde: (N, T, D) candidate hidden states
    Returns:
        h_final: (N, D) last hidden state
    """
    a = 1 - z       # (N, T, D) decay
    b = z * h_tilde  # (N, T, D) input contribution

    # associative_scan needs scan axis as dim 0
    # Transpose: (N, T, D) → (T, N, D)
    a_t = jnp.transpose(a, (1, 0, 2))
    b_t = jnp.transpose(b, (1, 0, 2))

    def binary_op(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_r * a_l, a_r * b_l + b_r)

    _, h_all = jax.lax.associative_scan(binary_op, (a_t, b_t), axis=0)

    # h_all: (T, N, D) — take last timestep
    return h_all[-1]  # (N, D)
