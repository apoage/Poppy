import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Any, Optional, Dict, Union
from . import VALUE_SUPPORT_SIZE
from .trm_encoder_v2 import AssetTransformerEncoder
from .trm_encoder_v3 import StreamingAssetEncoder


class TRMEncoder(nn.Module):
    """
    CNN Encoder for TRM.

    Mixed Precision Support:
    - dtype: Compute dtype (bfloat16 for ~2x speedup on Ampere GPUs)
    - param_dtype: Parameter storage dtype (always float32 for stability)
    """
    state_dim: int = 64
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Cast input to compute dtype
        x = x.astype(self.dtype)

        # Scale CNN filters with state_dim
        conv1_features = self.state_dim        # 64 for 64, 96 for 96, 128 for 128
        conv2_features = self.state_dim * 2    # 128 for 64, 192 for 96, 256 for 128

        # Lightweight CNN encoder (with dtype support for mixed precision)
        x = nn.Conv(features=conv1_features, kernel_size=(3, 3), padding='SAME',
                    dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # -> 32x32

        x = nn.Conv(features=conv2_features, kernel_size=(3, 3), padding='SAME',
                    dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.gelu(x)

        # Adaptive pooling logic (simulated with avg pool)
        # Input 32x32 -> 4x4. Factor 8.
        x = nn.avg_pool(x, window_shape=(8, 8), strides=(8, 8))  # -> 4x4

        # Flatten and project to state_dim
        x = x.reshape((x.shape[0], -1))  # (batch, features*4*4)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        state = nn.Dense(features=self.state_dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return state

class TRMReasoning(nn.Module):
    """
    Recursive reasoning block for policy and value.

    TASK_006: Weight sharing option for paper compliance.
    - share_weights=False (default): Unique weights per (H, L) cycle (current behavior)
    - share_weights=True: Shared weights reused every iteration (paper-compliant, smaller model)

    Mixed Precision Support:
    - dtype: Compute dtype (bfloat16 for ~2x speedup on Ampere GPUs)
    - param_dtype: Parameter storage dtype (always float32 for stability)
    """
    state_dim: int = 64
    n_actions: int = 192
    value_support_size: int = VALUE_SUPPORT_SIZE  # Central config: 401 bins over [-10, 10] (PopArt normalized)
    share_weights: bool = False  # TASK_006: Paper-compliant weight sharing option
    # TASK_684: Decouple hidden_dim from state_dim. None = backward compat (state_dim * 16)
    hidden_dim: Optional[int] = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        """Initialize shared layers if weight sharing enabled."""
        if self.share_weights:
            hidden_dim = self.hidden_dim if self.hidden_dim is not None else self.state_dim * 16
            # Shared latent refiner (used for ALL L_cycles)
            self.shared_latent_1 = nn.Dense(features=hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)
            self.shared_latent_ln = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype)
            self.shared_latent_2 = nn.Dense(features=self.state_dim, dtype=self.dtype, param_dtype=self.param_dtype)

            # Shared output refiner (used for ALL H_cycles)
            self.shared_output_1 = nn.Dense(features=hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)
            self.shared_output_ln = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype)
            self.shared_output_2 = nn.Dense(features=self.n_actions, dtype=self.dtype, param_dtype=self.param_dtype)

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        H_cycles: int = 4,
        L_cycles: int = 6,
        return_intermediates: bool = False
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
               Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]]:
        """
        Forward pass with optional intermediate output capture for DS24 dashboard.

        Args:
            state: Input state tensor (batch, state_dim)
            H_cycles: Number of outer reasoning cycles
            L_cycles: Number of inner latent refinement cycles
            return_intermediates: If True, return dict with cycle deltas for visualization

        Returns:
            policy_logits, value, latent (and intermediates dict if requested)
        """
        # Cast input to compute dtype for mixed precision
        state = state.astype(self.dtype)

        batch_size = state.shape[0]
        # TASK_684: Configurable hidden_dim, decoupled from state_dim
        hidden_dim = self.hidden_dim if self.hidden_dim is not None else self.state_dim * 16

        # Track intermediates if requested (for DS24 dashboard)
        intermediates = {
            'latent_deltas': {},
            'output_deltas': {},
            'latent_norms': [],
            'latent_per_h': {},  # Latent state at end of each H-cycle (for cycle ATC)
        } if return_intermediates else None

        # Initialize latent and output with learned parameters (small normal, not zeros!)
        latent = self.param(
            'initial_latent',
            nn.initializers.normal(stddev=0.02),
            (self.state_dim,)
        )
        latent = jnp.broadcast_to(latent, (batch_size, self.state_dim)).astype(self.dtype)

        output = self.param(
            'initial_output',
            nn.initializers.normal(stddev=0.02),
            (self.n_actions,)
        )
        output = jnp.broadcast_to(output, (batch_size, self.n_actions)).astype(self.dtype)

        # Recursive refinement (TRM core algorithm)
        for h in range(H_cycles):
            # Refine latent representation
            for l in range(L_cycles):
                combined = jnp.concatenate([state, latent], axis=-1)

                if self.share_weights:
                    # SHARED: Same weights reused every iteration (paper-compliant)
                    latent_delta = self.shared_latent_1(combined)
                    latent_delta = self.shared_latent_ln(latent_delta)
                    latent_delta = nn.gelu(latent_delta)
                    latent_delta = self.shared_latent_2(latent_delta)
                else:
                    # UNIQUE: Different weights per iteration (current behavior)
                    latent_delta = nn.Dense(features=hidden_dim, name=f'latent_refiner_h{h}_l{l}_1',
                                           dtype=self.dtype, param_dtype=self.param_dtype)(combined)
                    latent_delta = nn.LayerNorm(epsilon=1e-5, name=f'latent_refiner_h{h}_l{l}_ln',
                                               dtype=self.dtype, param_dtype=self.param_dtype)(latent_delta)
                    latent_delta = nn.gelu(latent_delta)
                    latent_delta = nn.Dense(features=self.state_dim, name=f'latent_refiner_h{h}_l{l}_2',
                                           dtype=self.dtype, param_dtype=self.param_dtype)(latent_delta)

                # Capture intermediate delta for DS24
                if return_intermediates:
                    intermediates['latent_deltas'][f'h{h}_l{l}'] = latent_delta

                latent = latent + latent_delta  # Residual connection

            # Capture latent state at end of this H-cycle (for cycle ATC)
            if return_intermediates:
                intermediates['latent_per_h'][f'h{h}'] = latent

            # Refine output
            output_input = jnp.concatenate([latent, output], axis=-1)

            if self.share_weights:
                # SHARED: Same weights reused every H_cycle
                output_delta = self.shared_output_1(output_input)
                output_delta = self.shared_output_ln(output_delta)
                output_delta = nn.gelu(output_delta)
                output_delta = self.shared_output_2(output_delta)
            else:
                # UNIQUE: Different weights per H_cycle
                output_delta = nn.Dense(features=hidden_dim, name=f'output_refiner_h{h}_1',
                                       dtype=self.dtype, param_dtype=self.param_dtype)(output_input)
                output_delta = nn.LayerNorm(epsilon=1e-5, name=f'output_refiner_h{h}_ln',
                                           dtype=self.dtype, param_dtype=self.param_dtype)(output_delta)
                output_delta = nn.gelu(output_delta)
                output_delta = nn.Dense(features=self.n_actions, name=f'output_refiner_h{h}_2',
                                       dtype=self.dtype, param_dtype=self.param_dtype)(output_delta)

            # Capture output delta for DS24
            if return_intermediates:
                intermediates['output_deltas'][f'h{h}'] = output_delta
                intermediates['latent_norms'].append(jnp.linalg.norm(latent, axis=-1))

            output = output + output_delta  # Residual connection

        policy_logits = output

        # Value head - uses REFINED latent (after H×L reasoning cycles)
        # This ensures value predictions benefit from same "thinking" as policy
        # NOTE: NO LayerNorm here - it causes collapse to mean=0!
        value_hidden = self.state_dim * 4  # Increased capacity for value prediction
        value = nn.Dense(features=value_hidden, name='value_head_1',
                        dtype=self.dtype, param_dtype=self.param_dtype)(latent)
        value = nn.gelu(value)
        value = nn.Dense(features=value_hidden, name='value_head_2',
                        dtype=self.dtype, param_dtype=self.param_dtype)(value)
        value = nn.gelu(value)
        value = nn.Dense(features=self.value_support_size, name='value_head_3',
                        dtype=self.dtype, param_dtype=self.param_dtype)(value)

        # Cast outputs back to float32 for numerical stability in loss/softmax
        policy_logits = policy_logits.astype(jnp.float32)
        value = value.astype(jnp.float32)
        latent = latent.astype(jnp.float32)

        # Add final latent to intermediates
        if return_intermediates:
            intermediates['latent_final'] = latent
            return policy_logits, value, latent, intermediates

        return policy_logits, value, latent

class TinyRecursiveTradingModel(nn.Module):
    """
    Combined model for convenience (if needed).
    """
    state_dim: int = 64
    n_actions: int = 192
    value_support_size: int = VALUE_SUPPORT_SIZE  # Central config: 401 bins over [-10, 10] (PopArt normalized)
    encoder_version: int = 2  # 2=V2 (flatten), 3=V3 (minGRU temporal)
    # TASK_684: Decouple hidden_dim from state_dim. None = backward compat (state_dim * 16)
    hidden_dim: Optional[int] = None
    # TASK_683: Encoder width params (defaults match V3 encoder defaults)
    embed_dim: int = 32
    encoder_num_heads: int = 4
    # TASK_741 (Mint): post-pool per-asset scalar injection
    per_asset_scalar_dims: int = 0
    # TASK_801 (Token front-end): per-timestep token embedding. 0 ⇒ disabled,
    # canon untouched. Plumbed through to StreamingAssetEncoder.
    token_vocab_size: int = 0
    token_embed_dim: int = 16
    # TASK_749 (Titans-MAG): plumbed-through encoder Titans config.
    use_titans_memory: bool = False
    titans_memory_depth: int = 2
    titans_memory_hidden: Optional[int] = None
    titans_persistent_tokens: int = 4
    titans_surprise_momentum_init: float = 0.9
    titans_decay_init: float = 0.01
    titans_lr_init: float = 0.01
    titans_gate_init: float = -5.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        if self.encoder_version == 3:
            self.encoder = StreamingAssetEncoder(
                state_dim=self.state_dim,
                embed_dim=self.embed_dim,
                num_heads=self.encoder_num_heads,
                per_asset_scalar_dims=self.per_asset_scalar_dims,
                token_vocab_size=self.token_vocab_size,
                token_embed_dim=self.token_embed_dim,
                use_titans_memory=self.use_titans_memory,
                titans_memory_depth=self.titans_memory_depth,
                titans_memory_hidden=self.titans_memory_hidden,
                titans_persistent_tokens=self.titans_persistent_tokens,
                titans_surprise_momentum_init=self.titans_surprise_momentum_init,
                titans_decay_init=self.titans_decay_init,
                titans_lr_init=self.titans_lr_init,
                titans_gate_init=self.titans_gate_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.encoder = AssetTransformerEncoder(
                state_dim=self.state_dim,
                embed_dim=self.embed_dim,           # TASK_696: Was hardcoded 32 (config ignored)
                num_heads=self.encoder_num_heads,    # TASK_696: Was hardcoded 4 (config ignored)
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        self.reasoning = TRMReasoning(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            value_support_size=self.value_support_size,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

    def __call__(self, x: jnp.ndarray, H_cycles: int = 3, L_cycles: int = 4):
        state = self.encoder(x)
        return self.reasoning(state, H_cycles, L_cycles)

    def encode(self, x, per_asset_scalars=None, token_ids=None):
        if self.encoder_version == 3 and (per_asset_scalars is not None
                                          or token_ids is not None):
            return self.encoder(x, per_asset_scalars=per_asset_scalars,
                                token_ids=token_ids)
        return self.encoder(x)

    def encode_with_features(self, x, per_asset_scalars=None, token_ids=None):
        """TASK_729a: return (global_state, per_asset_feats).

        per_asset_feats is (batch, n_assets, embed_dim) — post-LN, pre-pool.
        Used by DirectionHead + PerAssetPolicyHead to bypass the 64→state_dim
        global bottleneck.

        TASK_729c: V3 (StreamingAssetEncoder) now also exposes per-asset feats
        at the same structural point (post cross-attn LayerNorm, pre global_proj).
        TASK_741 (Mint): per_asset_scalars optionally FiLM-injected at that point.
        """
        if self.encoder_version == 3:
            return self.encoder(x, return_per_asset=True,
                                per_asset_scalars=per_asset_scalars,
                                token_ids=token_ids)
        if self.encoder_version == 2:
            return self.encoder(x, return_per_asset=True)
        state = self.encoder(x)
        return state, None

    def compute_policy_value(self, state, H_cycles=3, L_cycles=4):
        return self.reasoning(state, H_cycles, L_cycles)

# Utility function to create model
def create_trm_model(state_dim: int = 64, n_actions: int = 192, dtype=jnp.float32) -> TinyRecursiveTradingModel:
    """Factory function to create TRM model

    Args:
        state_dim: Model state dimension (64/96/160 for small/medium/large)
        n_actions: Number of actions (default 192 = 64 assets × 3 actions)
        dtype: Compute dtype (jnp.float32 or jnp.bfloat16 for mixed precision)
    """
    return TinyRecursiveTradingModel(
        state_dim=state_dim,
        n_actions=n_actions,
        value_support_size=VALUE_SUPPORT_SIZE,  # Central config
        dtype=dtype
    )