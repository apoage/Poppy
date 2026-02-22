import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, Optional, Any
from .trm_flax import TinyRecursiveTradingModel
from . import VALUE_SUPPORT_SIZE


class MuZeroNetworks(nn.Module):
    """
    Complete MuZero model with representation, dynamics, and prediction.
    Compatible with mctx library.

    Extended with optional prediction head for auxiliary price prediction loss (Phase 2).

    Mixed Precision Support:
    - dtype: Compute dtype (bfloat16 for ~2x speedup on Ampere GPUs)
    - param_dtype: Parameter storage dtype (always float32 for stability)
    """
    state_dim: int = 64
    action_space_size: int = 192
    support_size: int = VALUE_SUPPORT_SIZE  # Central config: 401 bins over [-10, 10] (PopArt normalized)
    H_cycles: int = 4
    L_cycles: int = 6

    # Prediction model parameters (Phase 2)
    use_prediction_head: bool = False
    physics_only: bool = False  # If True, use only physics core (no neural corrector)
    n_assets: int = 64
    n_features: int = 10
    # Prediction head size parameters
    pred_embed_dim: int = 128      # Neural corrector embedding dimension
    pred_num_blocks: int = 2       # Number of transformer blocks
    pred_mlp_dim: int = 256        # FFN hidden dimension
    pred_n_factors: int = 8        # Physics correlation rank

    # Mixed precision support
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    # TASK_676: Encoder version (2=V2 flatten, 3=V3 minGRU temporal)
    encoder_version: int = 2

    # TASK_684: Decouple hidden_dim from state_dim. None = backward compat (state_dim * 16)
    hidden_dim: Optional[int] = None

    # TASK_683: Encoder width params
    embed_dim: int = 32
    encoder_num_heads: int = 4

    # TASK_695: Oracle mask prediction auxiliary head
    use_oracle_mask_head: bool = False
    oracle_mask_hidden_dim: int = 256

    def setup(self):
        """Initialize all three networks"""
        # Shared TRM backbone (with dtype support)
        self.trm = TinyRecursiveTradingModel(
            state_dim=self.state_dim,
            n_actions=self.action_space_size,
            value_support_size=self.support_size,
            encoder_version=self.encoder_version,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
            encoder_num_heads=self.encoder_num_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

        # Dynamics network components (with dtype support)
        # NOTE: Keep embedding in param_dtype (float32) to avoid scatter dtype warning
        # Cast to compute dtype after lookup in dynamics()
        self.action_embedding = nn.Embed(
            num_embeddings=self.action_space_size,
            features=self.state_dim,
            param_dtype=self.param_dtype
        )

        # Transition function: (state, action) -> (next_state, reward)
        # Note: nn.Sequential doesn't support dtype, so we define layers separately
        dynamics_hidden = self.state_dim * 8  # 512 for 64, 768 for 96, 1024 for 128
        self.dynamics_dense1 = nn.Dense(dynamics_hidden, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dynamics_ln = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dynamics_dense2 = nn.Dense(self.state_dim + self.support_size, dtype=self.dtype, param_dtype=self.param_dtype)

        # Normalize next_state to prevent drift over MCTS unroll steps
        self.dynamics_state_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype)

        # NEW: Prediction head for auxiliary price prediction (Phase 2)
        if self.use_prediction_head:
            # Import here to avoid circular imports
            from .prediction import NeuralMarketODE

            # Decoder: latent -> market state (LOG-SPACE)
            # Outputs: log_prices, log_returns, moving_avg
            self.latent_to_market = nn.Dense(self.n_assets * 3, dtype=self.dtype, param_dtype=self.param_dtype)

            # Price predictor (operates in LOG-SPACE)
            # Note: NeuralMarketODE has its own dtype handling
            self.price_predictor = NeuralMarketODE(
                n_assets=self.n_assets,
                n_features=self.n_features,
                dt=1.0,
                ode_method='euler',  # Use euler for speed in training
                use_action_conditioning=True,
                use_transformer=True,
                physics_only=self.physics_only,  # Disable neural corrector if set
                # Size parameters
                embed_dim=self.pred_embed_dim,
                num_transformer_blocks=self.pred_num_blocks,
                mlp_dim=self.pred_mlp_dim,
                n_factors=self.pred_n_factors,
            )

        # TASK_695: Oracle mask prediction auxiliary head
        if self.use_oracle_mask_head:
            self.oracle_mask_dense1 = nn.Dense(
                self.oracle_mask_hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)
            self.oracle_mask_dense2 = nn.Dense(
                self.n_assets, dtype=self.dtype, param_dtype=self.param_dtype)

    def representation(self, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Encode observation to latent state.

        Args:
            observation: (batch, 27, 64, 64)
        Returns:
            latent_state: (batch, state_dim)
        """
        return self.trm.encode(observation)

    def prediction(
        self,
        latent_state: jnp.ndarray,
        return_intermediates: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict policy and value from latent state.

        Args:
            latent_state: (batch, state_dim)
            return_intermediates: If True, return TRM cycle intermediates for DS24
        Returns:
            policy_logits: (batch, action_space_size)
            value: (batch, support_size)
            intermediates: (optional) Dict with TRM cycle deltas if return_intermediates=True
        """
        result = self.trm.reasoning(
            latent_state,
            H_cycles=self.H_cycles,
            L_cycles=self.L_cycles,
            return_intermediates=return_intermediates
        )

        if return_intermediates:
            policy_logits, value, _, intermediates = result
            return policy_logits, value, intermediates
        else:
            policy_logits, value, _ = result
            return policy_logits, value

    def oracle_mask_prediction(self, latent_state: jnp.ndarray) -> jnp.ndarray:
        """TASK_695: Predict oracle's top-k selection mask. Returns (batch, n_assets) logits."""
        x = self.oracle_mask_dense1(latent_state.astype(self.dtype))
        x = nn.relu(x)
        return self.oracle_mask_dense2(x).astype(jnp.float32)

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
        # Cast input to compute dtype
        latent_state = latent_state.astype(self.dtype)

        # Embed action and cast to compute dtype (embedding stored in float32)
        action_emb = self.action_embedding(action).astype(self.dtype)

        # Concatenate state and action
        x = jnp.concatenate([latent_state, action_emb], axis=-1)

        # Predict next state and reward (using explicit layers instead of Sequential)
        x = self.dynamics_dense1(x)
        x = self.dynamics_ln(x)
        x = nn.gelu(x)
        out = self.dynamics_dense2(x)

        next_state = out[:, :self.state_dim]
        reward = out[:, self.state_dim:]

        # Normalize next_state to prevent drift over multiple MCTS unroll steps
        next_state = self.dynamics_state_norm(next_state)

        # Cast outputs back to float32 for numerical stability
        next_state = next_state.astype(jnp.float32)
        reward = reward.astype(jnp.float32)

        return next_state, reward

    def initial_inference(
        self,
        observation: jnp.ndarray,
        return_intermediates: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Initial inference for root node (Representation + Prediction).

        Args:
            observation: (batch, 27, 64, 64)
            return_intermediates: If True, include TRM cycle intermediates for DS24
        Returns:
            dict with keys: latent_state, policy_logits, value, reward
            If return_intermediates, also includes 'trm_intermediates'
        """
        latent_state = self.representation(observation)

        if return_intermediates:
            policy_logits, value, intermediates = self.prediction(
                latent_state, return_intermediates=True
            )
        else:
            policy_logits, value = self.prediction(latent_state)

        # Reward at root is zero (one-hot at center of support)
        batch_size = observation.shape[0]
        reward = jnp.zeros((batch_size, self.support_size))
        reward = reward.at[:, self.support_size // 2].set(1.0)  # Center bin

        result = {
            'latent_state': latent_state,
            'policy_logits': policy_logits,
            'value': value,
            'reward': reward
        }

        if return_intermediates:
            result['trm_intermediates'] = intermediates

        # TASK_695: Oracle mask prediction
        if self.use_oracle_mask_head:
            result['oracle_mask'] = self.oracle_mask_prediction(latent_state)

        return result

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

        # BUG-017 FIX: Stop gradients from prediction head to dynamics/encoder
        # In standard MuZero, dynamics is trained via reward loss + consistency loss only.
        # Value/policy gradients should not flow back through dynamics to corrupt encoder.
        # Note: We still return the non-stopped latent_state for MCTS tree expansion.
        policy_logits, value = self.prediction(jax.lax.stop_gradient(next_latent_state))

        return {
            'latent_state': next_latent_state,
            'policy_logits': policy_logits,
            'value': value,
            'reward': reward
        }

    # =========================================================================
    # Prediction Head Methods (Phase 2)
    # =========================================================================

    def decode_latent_to_market(
        self,
        latent_state: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Decode latent state to market state (LOG-SPACE).

        Args:
            latent_state: (batch, state_dim)

        Returns:
            market_state: Dict with 'log_prices', 'log_returns', 'moving_avg'
        """
        if not self.use_prediction_head:
            raise ValueError("Prediction head not enabled")

        batch = latent_state.shape[0]
        decoded = self.latent_to_market(latent_state)
        decoded = decoded.reshape(batch, 3, self.n_assets)

        return {
            'log_prices': decoded[:, 0, :],     # LOG-SPACE
            'log_returns': decoded[:, 1, :],    # Log returns
            'moving_avg': decoded[:, 2, :],     # LOG-SPACE moving avg
        }

    def predict_log_prices(
        self,
        latent_state: jnp.ndarray,
        features: jnp.ndarray,
        noise: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        use_teacher_forcing: bool = False,
        teacher_log_prices: Optional[jnp.ndarray] = None,
        teacher_log_returns: Optional[jnp.ndarray] = None,
        teacher_moving_avg: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Predict next log-prices from latent state.

        Supports scheduled sampling: use teacher-forced market state
        instead of decoded state.

        Args:
            latent_state: (batch, state_dim)
            features: (batch, n_features)
            noise: (batch, n_assets)
            action: (batch,) - optional action for conditioning
            use_teacher_forcing: Whether to use teacher-forced market state
            teacher_log_prices: Ground truth log-prices
            teacher_log_returns: Ground truth log-returns
            teacher_moving_avg: Ground truth moving average

        Returns:
            pred_log_prices: (batch, n_assets) in LOG-SPACE
            aux: Dict with physics/neural components
        """
        if not self.use_prediction_head:
            raise ValueError("Prediction head not enabled")

        # Choose between decoded and teacher-forced market state
        if use_teacher_forcing and teacher_log_prices is not None:
            # Scheduled sampling: use ground truth
            log_prices = teacher_log_prices
            log_returns = teacher_log_returns
            moving_avg = teacher_moving_avg
        else:
            # Decode from latent
            market_state = self.decode_latent_to_market(latent_state)
            log_prices = market_state['log_prices']
            log_returns = market_state['log_returns']
            moving_avg = market_state['moving_avg']

        # Predict using NeuralMarketODE (operates in LOG-SPACE)
        pred_log_prices, aux = self.price_predictor(
            log_prices,
            log_returns,
            moving_avg,
            noise,
            features,
            action,
        )

        return pred_log_prices, aux

    def predict_multi_step(
        self,
        latent_state: jnp.ndarray,
        features: jnp.ndarray,
        rng: jnp.ndarray,
        action_sequence: Optional[jnp.ndarray] = None,
        num_steps: int = 5,
        teacher_sequence: Optional[jnp.ndarray] = None,
        teacher_log_returns: Optional[jnp.ndarray] = None,
        teacher_moving_avg: Optional[jnp.ndarray] = None,
        scheduled_sampling_prob: float = 0.0,
        neural_scale: float = 1.0,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Multi-step prediction for temporal consistency loss.

        TASK_035: Added neural_scale for physics warmup.
        TASK_106: Added initial state teacher forcing.

        Args:
            latent_state: (batch, state_dim)
            features: (batch, n_features)
            rng: Random key
            action_sequence: (batch, num_steps) - optional action sequence
            num_steps: Number of steps to predict
            teacher_sequence: (batch, num_steps+1, n_assets) - ground truth log-prices
            teacher_log_returns: (batch, n_assets) - ground truth initial log-returns
            teacher_moving_avg: (batch, n_assets) - ground truth initial moving avg
            scheduled_sampling_prob: Probability of using teacher forcing
            neural_scale: Scale for neural corrector [0, 1] for physics warmup.
                         0 = physics only, 1 = full neural contribution.

        Returns:
            predictions: (batch, num_steps, n_assets) predicted log-prices
            aux: Dict with step-wise auxiliary info
        """
        if not self.use_prediction_head:
            raise ValueError("Prediction head not enabled")

        batch = latent_state.shape[0]
        predictions = []

        # Decode initial market state from latent
        market_state = self.decode_latent_to_market(latent_state)
        current_log_prices = market_state['log_prices']
        current_log_returns = market_state['log_returns']
        current_moving_avg = market_state['moving_avg']

        # Check teacher_sequence availability at Python level (before tracing)
        has_teacher = teacher_sequence is not None

        # TASK_106: Apply teacher forcing to INITIAL state if available
        # This prevents the "zero-gradient trap" where physics params receive no gradients
        # because the initial state from the decoder is near zero.
        if has_teacher:
            rng, init_sample_rng = jax.random.split(rng)
            # Use same probability for initial state override
            use_teacher_init = jax.random.uniform(init_sample_rng, (batch,)) < scheduled_sampling_prob
            
            # Override initial prices with teacher (k=0)
            current_log_prices = jnp.where(
                use_teacher_init[:, None],
                teacher_sequence[:, 0, :],
                current_log_prices
            )
            
            # Override initial returns if provided
            if teacher_log_returns is not None:
                current_log_returns = jnp.where(
                    use_teacher_init[:, None],
                    teacher_log_returns,
                    current_log_returns
                )
                
            # Override initial moving average if provided
            if teacher_moving_avg is not None:
                current_moving_avg = jnp.where(
                    use_teacher_init[:, None],
                    teacher_moving_avg,
                    current_moving_avg
                )

        for k in range(num_steps):
            rng, noise_rng, sample_rng = jax.random.split(rng, 3)
            noise = jax.random.normal(noise_rng, (batch, self.n_assets))

            # Scheduled sampling: use jnp.where for JAX-compatible conditional
            if has_teacher:
                # BUG_008 FIX: Generate per-batch-element decisions, not scalar
                # This allows different batch elements to independently use teacher/decoded
                use_teacher = jax.random.uniform(sample_rng, (batch,)) < scheduled_sampling_prob
                # Get teacher values for this step
                teacher_prices_k = teacher_sequence[:, k, :]
                teacher_returns_k = jnp.where(
                    k > 0,
                    teacher_sequence[:, k, :] - teacher_sequence[:, k-1, :],
                    current_log_returns  # Use decoded/current returns for k=0 (already handled by init override)
                )
                
                # Update current state if using teacher for this step
                current_log_prices = jnp.where(
                    use_teacher[:, None],
                    teacher_prices_k,
                    current_log_prices
                )
                
                # For returns, we want teacher returns at step k.
                # For k=0, we use teacher_log_returns (if available) or compute from seq
                if k == 0 and teacher_log_returns is not None:
                     current_teacher_returns = teacher_log_returns
                else:
                     current_teacher_returns = teacher_returns_k
                     
                current_log_returns = jnp.where(
                    use_teacher[:, None],
                    current_teacher_returns,
                    current_log_returns
                )

            # Get action for this step
            action = action_sequence[:, k] if action_sequence is not None else None

            # Predict next step - TASK_035: Pass neural_scale for physics warmup
            next_log_prices, _ = self.price_predictor(
                current_log_prices,
                current_log_returns,
                current_moving_avg,
                noise,
                features,
                action,
                neural_scale=neural_scale,
            )

            predictions.append(next_log_prices)

            # Update for next iteration
            current_log_returns = next_log_prices - current_log_prices
            current_log_prices = next_log_prices
            current_moving_avg = 0.9 * current_moving_avg + 0.1 * current_log_prices

        return jnp.stack(predictions, axis=1), {'neural_scale': neural_scale}

    def __call__(
        self,
        observation=None,
        action=None,
        latent_state=None,
        mode: str = 'initial_inference',
        return_intermediates: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Dispatch to specific inference method.

        Args:
            observation: Input observation (for initial_inference)
            action: Action (for recurrent_inference)
            latent_state: Latent state (for recurrent_inference)
            mode: 'initial_inference', 'recurrent_inference', or 'initialize_parameters'
            return_intermediates: If True, return TRM cycle intermediates for DS24 dashboard
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

            # Initialize prediction head if enabled
            if self.use_prediction_head:
                dummy_features = jnp.zeros((observation.shape[0], self.n_features))
                dummy_noise = jnp.zeros((observation.shape[0], self.n_assets))
                self.predict_log_prices(
                    out_init['latent_state'],
                    dummy_features,
                    dummy_noise,
                    action,
                )

            # TASK_695: Initialize oracle mask head if enabled
            if self.use_oracle_mask_head:
                self.oracle_mask_prediction(out_init['latent_state'])

            return out_init

        if mode == 'initial_inference':
            return self.initial_inference(observation, return_intermediates=return_intermediates)
        elif mode == 'recurrent_inference':
            # When called from mctx, the first positional arg (mapped to 'observation')
            # is actually the embedding/latent_state
            return self.recurrent_inference(observation, action)
        elif mode == 'initial_inference_with_intermediates':
            # Explicit mode for DS24 dashboard logging
            return self.initial_inference(observation, return_intermediates=True)
        elif mode == 'prediction_only':
            # TASK_666: Skip representation, run prediction on pre-computed latent states
            # Used by Easy Mode for latent state caching optimization
            policy_logits, value = self.prediction(latent_state)
            batch_size = latent_state.shape[0]
            reward = jnp.zeros((batch_size, VALUE_SUPPORT_SIZE))
            reward = reward.at[:, VALUE_SUPPORT_SIZE // 2].set(1.0)
            result = {
                'latent_state': latent_state,
                'policy_logits': policy_logits,
                'value': value,
                'reward': reward,
            }
            # TASK_695: Oracle mask prediction
            if self.use_oracle_mask_head:
                result['oracle_mask'] = self.oracle_mask_prediction(latent_state)
            return result
        else:
            raise ValueError(f"Unknown mode: {mode}")
