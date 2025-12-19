import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class TRMEncoder(nn.Module):
    """
    CNN Encoder for TRM.
    """
    state_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Scale CNN filters with state_dim
        conv1_features = self.state_dim        # 64 for 64, 96 for 96, 128 for 128
        conv2_features = self.state_dim * 2    # 128 for 64, 192 for 96, 256 for 128

        # Lightweight CNN encoder
        x = nn.Conv(features=conv1_features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # -> 32x32

        x = nn.Conv(features=conv2_features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)

        # Adaptive pooling logic (simulated with avg pool)
        # Input 32x32 -> 4x4. Factor 8.
        x = nn.avg_pool(x, window_shape=(8, 8), strides=(8, 8))  # -> 4x4

        # Flatten and project to state_dim
        x = x.reshape((x.shape[0], -1))  # (batch, features*4*4)
        x = nn.LayerNorm(epsilon=1e-5)(x)  # Add Norm before projection
        state = nn.Dense(features=self.state_dim)(x)
        return state

class TRMReasoning(nn.Module):
    """
    Recursive reasoning block for policy and value.
    """
    state_dim: int = 64
    n_actions: int = 192
    value_support_size: int = 601

    @nn.compact
    def __call__(self, state: jnp.ndarray, H_cycles: int = 4, L_cycles: int = 6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size = state.shape[0]
        # Aggressive scaling to reach 15M/25M params
        hidden_dim = self.state_dim * 16

        # Initialize latent and output with learned parameters
        latent = self.param(
            'initial_latent',
            nn.initializers.zeros,
            (self.state_dim,)
        )
        latent = jnp.broadcast_to(latent, (batch_size, self.state_dim))

        output = self.param(
            'initial_output',
            nn.initializers.zeros,
            (self.n_actions,)
        )
        output = jnp.broadcast_to(output, (batch_size, self.n_actions))

        # Recursive refinement (TRM core algorithm)
        for h in range(H_cycles):
            # Refine latent representation
            for l in range(L_cycles):
                combined = jnp.concatenate([state, latent], axis=-1)

                # Latent refiner network
                latent_delta = nn.Dense(features=hidden_dim, name=f'latent_refiner_h{h}_l{l}_1')(combined)
                latent_delta = nn.LayerNorm(epsilon=1e-5, name=f'latent_refiner_h{h}_l{l}_ln')(latent_delta)
                latent_delta = nn.gelu(latent_delta)
                latent_delta = nn.Dense(features=self.state_dim, name=f'latent_refiner_h{h}_l{l}_2')(latent_delta)

                latent = latent + latent_delta  # Residual connection

            # Refine output
            output_input = jnp.concatenate([latent, output], axis=-1)

            output_delta = nn.Dense(features=hidden_dim, name=f'output_refiner_h{h}_1')(output_input)
            output_delta = nn.LayerNorm(epsilon=1e-5, name=f'output_refiner_h{h}_ln')(output_delta)
            output_delta = nn.gelu(output_delta)
            output_delta = nn.Dense(features=self.n_actions, name=f'output_refiner_h{h}_2')(output_delta)

            output = output + output_delta  # Residual connection

        policy_logits = output

        # Value head (separate from recursive policy)
        value_hidden = self.state_dim # Scale with state_dim
        value = nn.Dense(features=value_hidden, name='value_head_1')(state)
        value = nn.LayerNorm(epsilon=1e-5)(value)  # Add Norm
        value = nn.gelu(value)
        value = nn.Dense(features=self.value_support_size, name='value_head_2')(value)

        return policy_logits, value, latent

class TinyRecursiveTradingModel(nn.Module):
    """
    Combined model for convenience (if needed).
    """
    state_dim: int = 64
    n_actions: int = 192
    value_support_size: int = 601

    def setup(self):
        self.encoder = TRMEncoder(state_dim=self.state_dim)
        self.reasoning = TRMReasoning(
            state_dim=self.state_dim, 
            n_actions=self.n_actions,
            value_support_size=self.value_support_size
        )

    def __call__(self, x: jnp.ndarray, H_cycles: int = 3, L_cycles: int = 4):
        state = self.encoder(x)
        return self.reasoning(state, H_cycles, L_cycles)
    
    def encode(self, x):
        return self.encoder(x)
    
    def compute_policy_value(self, state, H_cycles=3, L_cycles=4):
        return self.reasoning(state, H_cycles, L_cycles)

# Utility function to create model
def create_trm_model(state_dim: int = 64, n_actions: int = 192) -> TinyRecursiveTradingModel:
    """Factory function to create TRM model"""
    return TinyRecursiveTradingModel(
        state_dim=state_dim,
        hidden_dim=128,
        n_actions=n_actions,
        value_support_size=601
    )