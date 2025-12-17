import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
from typing import Dict, Tuple
import chex

from .muzero_model import MuZeroNetworks
from .mctx_search import create_mctx_search
from .replay_buffer import JAXReplayBuffer


class TrainState(train_state.TrainState):
    """Extended train state with additional metrics/target params"""
    target_params: flax.core.FrozenDict = None
    step_count: int = 0


class MuZeroTrainer:
    """
    Complete JAX MuZero training pipeline.
    Fully GPU-accelerated with JIT compilation.
    """

    def __init__(
        self,
        model: MuZeroNetworks,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        num_simulations: int = 32,
        batch_size: int = 256,
        unroll_steps: int = 5,
        td_steps: int = 10,
        discount: float = 0.997,
        value_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
        target_update_interval: int = 100
    ):
        self.model = model
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.discount = discount
        self.target_update_interval = target_update_interval

        # Loss weights
        self.value_loss_weight = value_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.policy_loss_weight = policy_loss_weight

        # Create MCTX search
        self.search = create_mctx_search(
            model,
            num_simulations=num_simulations,
            use_gumbel=True
        )

        # Create optimizer (AdamW with gradient clipping)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        )

        # Initialize training state
        self.state = None

        # JIT-compile training step
        self.train_step_fn = jax.jit(self._train_step)

    def init_train_state(self, rng: chex.PRNGKey, dummy_obs: jnp.ndarray) -> TrainState:
        """Initialize training state with model parameters"""
        # Initialize all parameters using mode='initialize_parameters'
        variables = self.model.init(rng, dummy_obs, mode='initialize_parameters')
        params = variables['params']

        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            target_params=params,  # Initialize target network
            tx=self.optimizer,
            step_count=0
        )

        self.state = state
        return state

    def _scalar_to_categorical(self, scalar: jnp.ndarray, support_size: int = 601) -> jnp.ndarray:
        """
        Convert scalar value to categorical distribution.

        Args:
            scalar: (batch,) scalar values in range [-10, 10]
            support_size: Number of bins

        Returns:
            logits: (batch, support_size) logits
        """
        # Clip to support range
        scalar = jnp.clip(scalar, -10, 10)

        # Map to bin indices
        support = jnp.linspace(-10, 10, support_size)
        bin_width = 20.0 / (support_size - 1)

        # Soft binning (distribute value between adjacent bins)
        indices = (scalar + 10.0) / bin_width
        lower_idx = jnp.floor(indices).astype(jnp.int32)
        upper_idx = jnp.clip(lower_idx + 1, 0, support_size - 1)

        # Interpolation weight
        weight_upper = indices - lower_idx
        weight_lower = 1.0 - weight_upper

        # Create target distribution
        batch_size = scalar.shape[0]
        targets = jnp.zeros((batch_size, support_size))
        targets = targets.at[jnp.arange(batch_size), lower_idx].set(weight_lower)
        targets = targets.at[jnp.arange(batch_size), upper_idx].add(weight_upper)

        return targets

    def _compute_target_value(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute n-step bootstrapped target values.

        Args:
            rewards: (batch, unroll_steps)
            values: (batch, unroll_steps)
            dones: (batch, unroll_steps)

        Returns:
            targets: (batch, unroll_steps) target values
        """
        batch_size, unroll_steps = rewards.shape

        # Compute n-step returns
        targets = jnp.zeros_like(rewards)

        for t in range(unroll_steps):
            # Bootstrap value (from td_steps ahead)
            bootstrap_idx = min(t + self.td_steps, unroll_steps - 1)
            bootstrap_value = values[:, bootstrap_idx]

            # Accumulated discounted rewards
            reward_sum = 0.0
            for k in range(min(self.td_steps, unroll_steps - t)):
                # Handle done? If done occurs, reward sum stops? 
                # Simplified for now: assume rewards are zero after done.
                reward_sum += (self.discount ** k) * rewards[:, t + k]

            # Target = sum of rewards + discounted bootstrap
            # Note: dones is 1 if episode ended at step t.
            # If done, bootstrap value is 0.
            targets = targets.at[:, t].set(
                reward_sum + (self.discount ** self.td_steps) * bootstrap_value * (1 - dones[:, bootstrap_idx])
            )

        return targets

    def _loss_fn(
        self,
        params: flax.core.FrozenDict,
        batch: Dict[str, jnp.ndarray],
        rng: chex.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute MuZero loss.

        Args:
            params: Model parameters
            batch: Training batch
            rng: Random key

        Returns:
            total_loss: Scalar loss
            metrics: Dict of metrics
        """
        observations = batch['observations']  # (batch, unroll_steps, 27, 64, 64)
        actions = batch['actions']  # (batch, unroll_steps)
        rewards = batch['rewards']  # (batch, unroll_steps)
        target_values = batch['values']  # (batch, unroll_steps)
        target_policies = batch['policy_probs']  # (batch, unroll_steps, 192)

        batch_size, unroll_steps = actions.shape[:2]

        # Initial inference
        initial_output = self.model.apply(
            {'params': params},
            observations[:, 0],  # First observation
            mode='initial_inference'
        )

        # Compute initial losses
        value_loss = optax.softmax_cross_entropy(
            initial_output['value'],
            self._scalar_to_categorical(target_values[:, 0])
        ).mean()

        policy_loss = optax.softmax_cross_entropy(
            initial_output['policy_logits'],
            target_policies[:, 0]
        ).mean()

        reward_loss = jnp.array(0.0)  # No reward at root

        # Unroll dynamics for subsequent steps
        latent_state = initial_output['latent_state']

        for t in range(1, unroll_steps):
            # Recurrent inference
            recurrent_output = self.model.apply(
                {'params': params},
                latent_state,
                actions[:, t - 1],
                mode='recurrent_inference'
            )

            # Update latent state
            latent_state = recurrent_output['latent_state']

            # Compute losses for this step
            value_loss += optax.softmax_cross_entropy(
                recurrent_output['value'],
                self._scalar_to_categorical(target_values[:, t])
            ).mean()

            policy_loss += optax.softmax_cross_entropy(
                recurrent_output['policy_logits'],
                target_policies[:, t]
            ).mean()

            reward_loss += optax.softmax_cross_entropy(
                recurrent_output['reward'],
                self._scalar_to_categorical(rewards[:, t])
            ).mean()

        # Average over unroll steps
        value_loss /= unroll_steps
        policy_loss /= unroll_steps
        reward_loss /= (unroll_steps - 1)  # No reward at t=0

        # Total weighted loss
        total_loss = (
            self.value_loss_weight * value_loss +
            self.policy_loss_weight * policy_loss +
            self.reward_loss_weight * reward_loss
        )

        metrics = {
            'total_loss': total_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'reward_loss': reward_loss
        }

        return total_loss, metrics

    def _train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: chex.PRNGKey
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """
        Single training step (JIT-compiled).

        Args:
            state: Training state
            batch: Training batch
            rng: Random key

        Returns:
            new_state: Updated training state
            metrics: Training metrics
        """
        # Compute gradients
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params, batch, rng)

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        # Update target network periodically
        new_step_count = state.step_count + 1
        should_update_target = (new_step_count % self.target_update_interval) == 0

        # Use jax.lax.cond for conditional update inside JIT
        new_target_params = jax.lax.cond(
            should_update_target,
            lambda: new_state.params,  # True branch: copy current params
            lambda: state.target_params # False branch: keep old target params
        )

        new_state = new_state.replace(
            target_params=new_target_params,
            step_count=new_step_count
        )

        # Add gradient norm to metrics
        metrics['grad_norm'] = optax.global_norm(grads)

        return new_state, metrics

    def train(
        self,
        replay_buffer: JAXReplayBuffer,
        num_iterations: int,
        rng: chex.PRNGKey
    ) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            replay_buffer: Replay buffer with trajectories
            num_iterations: Number of training iterations
            rng: Random key

        Returns:
            history: Dict of training metrics over time
        """
        history = {
            'total_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'reward_loss': [],
            'grad_norm': []
        }

        for iteration in range(num_iterations):
            # Sample batch from replay buffer
            rng, sample_rng = jax.random.split(rng)
            batch = replay_buffer.sample_batch(sample_rng)

            # Debug: Check for NaNs/Infs in batch
            for k, v in batch.items():
                if jnp.isnan(v).any() or jnp.isinf(v).any():
                    print(f"!!! CRITICAL BATCH FAILURE at iteration {iteration} !!!")
                    print(f"Key: {k}")
                    print(f"  Min: {jnp.min(v)}")
                    print(f"  Max: {jnp.max(v)}")
                    print(f"  Mean: {jnp.mean(v)}")
                    raise ValueError(f"NaNs/Infs detected in training batch key '{k}'!")

            # Training step
            rng, train_rng = jax.random.split(rng)
            self.state, metrics = self.train_step_fn(self.state, batch, train_rng)

            # Record metrics
            for key, value in metrics.items():
                if key in history:
                    history[key].append(float(value))

            if jnp.isnan(metrics['total_loss']):
                print(f"!!! NAN LOSS DETECTED at iteration {iteration} !!!")
                print(f"Value Loss: {metrics['value_loss']}")
                print(f"Policy Loss: {metrics['policy_loss']}")
                print(f"Reward Loss: {metrics['reward_loss']}")
                raise ValueError("Training stopped due to NaN loss.")

            # Log progress
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations} | "
                      f"Loss: {metrics['total_loss']:.4f} | "
                      f"Value: {metrics['value_loss']:.4f} | "
                      f"Policy: {metrics['policy_loss']:.4f} | "
                      f"Reward: {metrics['reward_loss']:.4f}")

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        from flax.training import checkpoints
        checkpoints.save_checkpoint(
            ckpt_dir=path,
            target=self.state,
            step=self.state.step_count,
            keep=3,
            overwrite=True
        )

    def load_checkpoint(self, path: str, step: int = None):
        """Load model checkpoint"""
        from flax.training import checkpoints
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=path,
            target=self.state,
            step=step
        )
        return restored_state

    def restore_checkpoint(self, path: str, step: int = None):
        """Alias for load_checkpoint"""
        return self.load_checkpoint(path, step)
