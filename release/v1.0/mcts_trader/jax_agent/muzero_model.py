import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple
from .trm_flax import TinyRecursiveTradingModel

class MuZeroNetworks(nn.Module):
    """
    Complete MuZero model with representation, dynamics, and prediction.
    Compatible with mctx library.
    """
    state_dim: int = 64
    action_space_size: int = 192
    support_size: int = 601

    def setup(self):
        """Initialize all three networks"""
        # Shared TRM backbone
        self.trm = TinyRecursiveTradingModel(
            state_dim=self.state_dim,
            n_actions=self.action_space_size,
            value_support_size=self.support_size
        )

        # Dynamics network components
        self.action_embedding = nn.Embed(
            num_embeddings=self.action_space_size,
            features=self.state_dim
        )

        # Transition function: (state, action) -> (next_state, reward)
        self.dynamics_net = nn.Sequential([
            nn.Dense(256),
            nn.LayerNorm(epsilon=1e-5),
            lambda x: nn.gelu(x),
            nn.Dense(self.state_dim + self.support_size)
        ])

    def representation(self, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Encode observation to latent state.

        Args:
            observation: (batch, 27, 64, 64)
        Returns:
            latent_state: (batch, state_dim)
        """
        return self.trm.encode(observation)

    def prediction(self, latent_state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict policy and value from latent state.

        Args:
            latent_state: (batch, state_dim)
        Returns:
            policy_logits: (batch, action_space_size)
            value: (batch, support_size)
        """
        policy_logits, value, _ = self.trm.compute_policy_value(
            latent_state, H_cycles=3, L_cycles=4
        )
        return policy_logits, value

    def dynamics(
        self,
        latent_state: jnp.ndarray,
        action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict next latent state and reward.

        Args:
            latent_state: (batch, state_dim)
            action: (batch,) integer actions
        Returns:
            next_latent_state: (batch, state_dim)
            reward: (batch, support_size)
        """
        # Embed action
        action_emb = self.action_embedding(action)

        # Concatenate state and action
        x = jnp.concatenate([latent_state, action_emb], axis=-1)

        # Predict next state and reward
        out = self.dynamics_net(x)
        next_state = out[:, :self.state_dim]
        reward = out[:, self.state_dim:]

        return next_state, reward

    def initial_inference(self, observation: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Initial inference for root node (Representation + Prediction).

        Args:
            observation: (batch, 27, 64, 64)
        Returns:
            dict with keys: latent_state, policy_logits, value, reward
        """
        latent_state = self.representation(observation)
        policy_logits, value = self.prediction(latent_state)

        # Reward at root is zero (one-hot at center of support)
        batch_size = observation.shape[0]
        reward = jnp.zeros((batch_size, self.support_size))
        reward = reward.at[:, self.support_size // 2].set(1.0)  # Center bin

        return {
            'latent_state': latent_state,
            'policy_logits': policy_logits,
            'value': value,
            'reward': reward
        }

    def recurrent_inference(
        self,
        latent_state: jnp.ndarray,
        action: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Recurrent inference for tree expansion (Dynamics + Prediction).

        Args:
            latent_state: (batch, state_dim)
            action: (batch,)
        Returns:
            dict with keys: latent_state, policy_logits, value, reward
        """
        next_latent_state, reward = self.dynamics(latent_state, action)
        policy_logits, value = self.prediction(next_latent_state)

        return {
            'latent_state': next_latent_state,
            'policy_logits': policy_logits,
            'value': value,
            'reward': reward
        }

    def __call__(self, observation=None, action=None, latent_state=None, mode: str = 'initial_inference') -> Dict[str, jnp.ndarray]:
        """
        Dispatch to specific inference method.
        """
        if mode == 'initialize_parameters':
            # Run both paths to initialize all parameters
            # Require observation and action to be passed
            out_init = self.initial_inference(observation)
            if action is None:
                # Create dummy action if not provided (index 0)
                batch_size = observation.shape[0]
                action = jnp.zeros((batch_size,), dtype=jnp.int32)
            
            # Use latent state from init for recurrent
            self.recurrent_inference(out_init['latent_state'], action)
            return out_init

        if mode == 'initial_inference':
            return self.initial_inference(observation)
        elif mode == 'recurrent_inference':
            # When called from mctx, the first positional arg (mapped to 'observation')
            # is actually the embedding/latent_state
            return self.recurrent_inference(observation, action)
        else:
            raise ValueError(f"Unknown mode: {mode}")
