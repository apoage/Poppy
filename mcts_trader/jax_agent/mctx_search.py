import jax
import jax.numpy as jnp
import mctx
from typing import Callable, Tuple, Any
import chex

class MuZeroMCTXSearch:
    """
    GPU-accelerated MCTS using Google's mctx library.
    Fully JIT-compiled for maximum performance.
    """

    def __init__(
        self,
        model_apply_fn: Callable,
        num_simulations: int = 32,
        max_num_considered_actions: int = 16,
        discount: float = 0.997,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
        gumbel_scale: float = 1.0,
        value_scale: float = 1.0,
        value_shift: float = 0.0
    ):
        """
        Args:
            model_apply_fn: Model's apply function (from Flax)
            num_simulations: Number of MCTS simulations
            max_num_considered_actions: Top-K actions for Gumbel
            discount: Reward discount factor
            temperature: Sampling temperature (for standard MuZero)
            use_gumbel: Use Gumbel MuZero (recommended) vs standard PUCT
            dirichlet_alpha: Noise alpha for root prior (Standard MuZero only)
            exploration_fraction: Fraction of noise to add (Standard MuZero only)
            gumbel_scale: Scale of Gumbel noise (for Gumbel MuZero)
            value_scale: Calibration scale for value predictions (default 1.0)
            value_shift: Calibration shift for value predictions (default 0.0)
        """
        self.model_apply_fn = model_apply_fn
        self.num_simulations = num_simulations
        self.max_num_considered_actions = max_num_considered_actions
        self.discount = discount
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.gumbel_scale = gumbel_scale
        self.value_scale = value_scale
        self.value_shift = value_shift

        # JIT-compile search functions to XLA
        # static_argnums not needed for pure function call if self is mostly static config
        # But we'll wrap the call in a jit inside __call__ or init
        self.search_fn = jax.jit(self._search)

    def _root_fn(
        self,
        params: dict,
        observation: jnp.ndarray
    ) -> mctx.RootFnOutput:
        """
        Create root node from observation.
        Called once per search.
        """
        output = self.model_apply_fn(
            {'params': params},
            observation,
            mode='initial_inference'
        )

        # Convert value from categorical to scalar
        value_scalar = self._categorical_to_scalar(output['value'])

        return mctx.RootFnOutput(
            prior_logits=output['policy_logits'],
            value=value_scalar,
            embedding=output['latent_state']
        )

    def _recurrent_fn(
        self,
        params: dict,
        rng: chex.PRNGKey,
        action: jnp.ndarray,
        embedding: jnp.ndarray
    ) -> mctx.RecurrentFnOutput:
        """
        Recurrent dynamics function for tree expansion.
        Called num_simulations times per search.
        """
        output = self.model_apply_fn(
            {'params': params},
            embedding,
            action,
            mode='recurrent_inference'
        )

        # Convert categorical outputs to scalars
        reward_scalar = self._categorical_to_scalar(output['reward'])
        value_scalar = self._categorical_to_scalar(output['value'])

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward_scalar,
            discount=jnp.full_like(reward_scalar, self.discount),
            prior_logits=output['policy_logits'],
            value=value_scalar
        )
        return recurrent_output, output['latent_state']

    def _categorical_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Convert categorical representation to scalar.

        Args:
            logits: (batch, support_size) logits
        Returns:
            scalar: (batch,) scalar values
        """
        from . import get_value_support
        support = get_value_support()
        probs = jax.nn.softmax(logits, axis=-1)
        raw_value = jnp.sum(probs * support, axis=-1)

        # Apply calibration scaling (fixes slope < 1.0 issue)
        # Scale expands prediction range, shift adjusts center
        # Defaults: scale=1.0, shift=0.0 (no change) for backward compat
        scale = getattr(self, 'value_scale', 1.0)
        shift = getattr(self, 'value_shift', 0.0)
        return raw_value * scale + shift

    def _search(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """
        Run GPU-batched MCTS search.

        Args:
            params: Model parameters
            rng: JAX random key
            observation: (batch, 27, 64, 64)
            action_mask: (batch, action_space_size) optional mask

        Returns:
            actions: (batch,) selected actions
            policy_probs: (batch, action_space_size) visit count distribution
            search_stats: dict of search statistics
        """
        # Run initial inference on FULL BATCH at once (more efficient than vmap)
        root_output = self.model_apply_fn(
            {'params': params},
            observation,
            method='initial_inference'
        )
        root = mctx.RootFnOutput(
            prior_logits=root_output['policy_logits'],
            value=self._categorical_to_scalar(root_output['value']),
            embedding=root_output['latent_state']
        )

        # Apply action mask to root logits if provided
        if action_mask is not None:
            # Mask is 1 for valid, 0 for invalid
            # Set invalid logits to -inf
            min_logit = jnp.finfo(root.prior_logits.dtype).min
            # Ensure mask is broadcastable
            root_logits = jnp.where(action_mask.astype(bool), root.prior_logits, min_logit)
            
            # Re-create root with masked logits
            root = root.replace(prior_logits=root_logits)

        # Create recurrent function with bound parameters
        # mctx passes (action, embedding) where action/embedding are BATChED if called inside mctx's batch logic?
        # mctx docs: recurrent_fn(params, rng_key, action, embedding)
        # It expects inputs to be batched if we provide batched roots.
        
        recurrent_fn = lambda params, rng, action, embedding: self._recurrent_fn(
            params, rng, action, embedding
        )

        if self.use_gumbel:
            # Gumbel MuZero (recommended)
            policy_output = mctx.gumbel_muzero_policy(
                params=params, 
                rng_key=rng,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.num_simulations,
                max_num_considered_actions=self.max_num_considered_actions,
                qtransform=mctx.qtransform_by_parent_and_siblings,  # Improved Q-transform
                gumbel_scale=self.gumbel_scale
            )
        else:
            # Standard MuZero with PUCT
            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=rng,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.num_simulations,
                temperature=self.temperature,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_fraction=self.exploration_fraction,
                invalid_actions=~action_mask if action_mask is not None else None
            )

        # Extract results
        actions = policy_output.action
        policy_probs = policy_output.action_weights

        # Collect search statistics
        search_stats = {
            'root_value': root.value,
            'search_tree': policy_output.search_tree if hasattr(policy_output, 'search_tree') else None
        }

        return actions, policy_probs, search_stats

    def __call__(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Convenience method for calling search"""
        return self.search_fn(params, rng, observation, action_mask)


# Example usage
def create_mctx_search(model, **kwargs):
    """Factory function to create MCTX search"""
    return MuZeroMCTXSearch(
        model_apply_fn=model.apply,
        **kwargs
    )


def extract_mcts_diagnostics(
    policy_output,
    root_value: jnp.ndarray,
    actual_rewards: jnp.ndarray = None
) -> dict:
    """
    Extract MCTS diagnostics for DS24 dashboard.

    Args:
        policy_output: mctx PolicyOutput from search
        root_value: Root node values from search stats
        actual_rewards: Actual episode rewards (optional, for calibration)

    Returns:
        Dict with diagnostic metrics:
        - root_value_mean: Mean predicted root value
        - root_value_std: Std of root values
        - visit_entropy_mean: Mean entropy of visit counts (exploration indicator)
        - calibration_mse: MSE vs actual rewards (if provided)
    """
    diagnostics = {
        'root_value_mean': float(jnp.mean(root_value)),
        'root_value_std': float(jnp.std(root_value)),
        'visit_entropy_mean': 0.0,
        'calibration_mse': 0.0,
    }

    # Visit entropy (exploration indicator)
    # Higher entropy = more balanced exploration across actions
    if hasattr(policy_output, 'action_weights'):
        visit_counts = policy_output.action_weights
        # Normalize to probabilities
        visit_sum = jnp.sum(visit_counts, axis=-1, keepdims=True)
        probs = visit_counts / (visit_sum + 1e-8)
        # Compute entropy: -sum(p * log(p))
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
        diagnostics['visit_entropy_mean'] = float(jnp.mean(entropy))

    # Calibration error vs actual rewards
    if actual_rewards is not None:
        actual_rewards = jnp.array(actual_rewards)
        # Match shapes if needed
        if actual_rewards.shape != root_value.shape:
            # Take mean over unmatched dims
            if len(actual_rewards.shape) > len(root_value.shape):
                actual_rewards = jnp.mean(actual_rewards)
        diagnostics['calibration_mse'] = float(jnp.mean((root_value - actual_rewards) ** 2))

    return diagnostics
