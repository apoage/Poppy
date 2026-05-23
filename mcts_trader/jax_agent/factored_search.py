"""
TASK_724: Factored per-asset search for trading action selection.

Instead of running MCTS over 320 joint actions, evaluates each asset's
5 actions independently via 1-step dynamics lookahead + value prediction.

Key insight: per-asset decisions are mostly independent. The oracle selects
k=4 assets to trade, with each asset's action chosen based on its own
expected return. Joint MCTS can't discover this factored structure.

Interface matches MuZeroMCTXSearch for drop-in replacement in train_phase2.py.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple


class FactoredPerAssetSearch:
    """
    Per-asset 1-step lookahead search.

    For each (asset, action) pair:
        1. Compose joint action index: j = asset * 5 + action
        2. Run dynamics: next_latent, reward = dynamics(latent, j)
        3. Run value prediction on next_latent
        4. Q(asset, action) = reward + discount * value(next_latent)

    Returns per-asset softmax policy targets (B, 320) for training,
    same interface as MuZeroMCTXSearch.
    """

    def __init__(
        self,
        model_apply_fn: Callable,
        n_assets: int = 64,
        n_actions_per_asset: int = 5,
        discount: float = 0.95,
        temperature: float = 1.0,
        epsilon_greedy: float = 0.0,
        sub_batch_size: int = 4096,
        value_scale: float = 1.0,
        value_shift: float = 0.0,
    ):
        self.model_apply_fn = model_apply_fn
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        self.n_total_actions = n_assets * n_actions_per_asset  # 320
        self.discount = discount
        self.temperature = temperature
        self.epsilon_greedy = epsilon_greedy
        self.sub_batch_size = sub_batch_size
        self.value_scale = value_scale
        self.value_shift = value_shift

        self._all_actions = jnp.arange(self.n_total_actions, dtype=jnp.int32)
        self._search_fn = jax.jit(self._search)

    def _compute_q_values(
        self,
        params: dict,
        latent: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Q-values for all 320 (asset, action) pairs via 1-step lookahead."""
        from . import get_value_support
        support = get_value_support()

        B = latent.shape[0]
        n_total = self.n_total_actions

        # Expand: (B, dim) → (B*320, dim)
        latent_expanded = jnp.repeat(latent, n_total, axis=0)
        actions_expanded = jnp.tile(self._all_actions, B)

        total = B * n_total
        sub_batch = self.sub_batch_size

        all_rewards = []
        all_values = []

        for start in range(0, total, sub_batch):
            end = min(start + sub_batch, total)
            lat_chunk = latent_expanded[start:end]
            act_chunk = actions_expanded[start:end]

            out = self.model_apply_fn(
                {'params': params},
                lat_chunk,
                act_chunk,
                mode='recurrent_inference'
            )

            # Reward: categorical → scalar (NO PopArt — rewards aren't normalized)
            reward_probs = jax.nn.softmax(out['reward'], axis=-1)
            reward_scalar = jnp.sum(reward_probs * support, axis=-1)
            all_rewards.append(reward_scalar)

            # Value: categorical → scalar + PopArt denormalization
            value_probs = jax.nn.softmax(out['value'], axis=-1)
            value_raw = jnp.sum(value_probs * support, axis=-1)
            value_denorm = value_raw * self.value_scale + self.value_shift
            all_values.append(value_denorm)

        rewards_flat = jnp.concatenate(all_rewards)
        values_flat = jnp.concatenate(all_values)

        # Q = R + γ * V
        q_flat = rewards_flat + self.discount * values_flat

        # Reshape: (B*320,) → (B, 64, 5)
        q_values = q_flat.reshape(B, self.n_assets, self.n_actions_per_asset)
        rewards = rewards_flat.reshape(B, self.n_assets, self.n_actions_per_asset)

        return q_values, rewards

    def _q_to_policy(self, q_values: jnp.ndarray,
                     valid_mask: jnp.ndarray = None) -> jnp.ndarray:
        """Convert per-asset Q-values to policy targets via temperature-scaled softmax.

        Each group of 5 actions sums to 1.0 (per-asset). Total sums to 64.0.

        TASK_767: valid_mask, when given, is a (B, 64, 5) {0,1} float array.
        The epsilon-greedy uniform component is spread over the VALID actions
        only (uniform = valid_mask / count_valid) instead of all 5 — otherwise
        the eps mix would re-inject eps/5 probability back into the masked
        limit actions. valid_mask=None keeps the original all-5 uniform.

        TASK_771/F1 (P0): the previous implementation subtracted the per-asset
        mean before softmax for "numerical stability" — but softmax is shift-
        invariant (softmax(x − c) ≡ softmax(x)) and `jax.nn.softmax` already
        performs its own internal max-shift (x − x.max()). Under the 767 mask
        the masked entries are driven to -1e9; the explicit mean step then
        rounded the per-asset mean to ~-4e8 in float32, shifting the valid
        Q's by +4e8 (ULP = 32 at that magnitude, swamping |Q| ≤ 20). All
        valid q_centered entries collapsed to the same float32 value, making
        the softmax output uniform across valid actions and silently
        degrading the policy training target to [1/3, 1/3, 1/3, 0, 0] per
        asset. Removing the mean-shift restores the Q-weighted softmax; the
        canon (action_mask=None) path is mathematically identical because
        softmax absorbs any per-row constant shift (the float32 arithmetic
        differs by ULP-scale ≤ noise floor of training-pipeline scheduling).
        """
        targets = jax.nn.softmax(q_values / self.temperature, axis=-1)  # (B, 64, 5)

        if self.epsilon_greedy > 0:
            if valid_mask is None:
                uniform = jnp.ones_like(targets) / self.n_actions_per_asset
            else:
                # Uniform over valid actions only — masked actions get 0.
                n_valid = jnp.maximum(
                    jnp.sum(valid_mask, axis=-1, keepdims=True), 1.0
                )
                uniform = valid_mask / n_valid
            targets = (1 - self.epsilon_greedy) * targets + self.epsilon_greedy * uniform

        # Flatten: (B, 64, 5) → (B, 320)
        return targets.reshape(-1, self.n_total_actions)

    def _search(
        self,
        params: dict,
        rng: jnp.ndarray,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Run factored per-asset search. Same interface as MuZeroMCTXSearch.

        TASK_767: when action_mask is provided (1 = valid, 0 = invalid, shape
        (B, 320)), the Q-value of every masked (asset, action) pair is driven
        to a large finite negative before the per-asset softmax and the
        action-selection argmax. This zeroes the masked actions' probability
        mass in the policy target and makes argmax never pick them — the
        factored-search equivalent of an MCTS expansion guard. action_mask=None
        leaves the path bit-identical to pre-767 behavior.
        """
        # 1. Initial inference
        root_out = self.model_apply_fn(
            {'params': params},
            observation,
            method='initial_inference'
        )
        latent = root_out['latent_state']
        root_value_logits = root_out['value']

        # Root value (PopArt denormalized)
        from . import get_value_support
        support = get_value_support()
        root_value_raw = jnp.sum(jax.nn.softmax(root_value_logits, axis=-1) * support, axis=-1)
        root_value = root_value_raw * self.value_scale + self.value_shift

        # 2. Q-values for all 320 (asset, action) pairs
        q_values, rewards = self._compute_q_values(params, latent)

        # TASK_767: mask invalid (asset, action) Q-values to -1e9. After
        # TASK_771/F1 removed the explicit mean-shift in _q_to_policy, the
        # -1e9 sentinel only feeds `jax.nn.softmax`'s internal max-shift,
        # which gives exp(-1e9 - Q_max) = 0 (underflow) at masked positions
        # while preserving full precision at valid positions. -1e9 is still
        # safely ~1e8 below any real Q (reward+γV ∈ ~[-10,10]) and remains
        # well inside float32 dynamic range; argmax-on-q_values continues to
        # exclude limit indices.
        valid_mask_r = None
        if action_mask is not None:
            mask_r = action_mask.reshape(-1, self.n_assets, self.n_actions_per_asset)
            valid_mask_r = mask_r.astype(jnp.float32)
            q_values = jnp.where(mask_r.astype(bool), q_values, -1e9)

        # 3. Per-asset policy targets (eps-uniform spread over valid actions)
        policy_probs = self._q_to_policy(q_values, valid_mask_r)

        # 4. Action selection: asset with highest Q-advantage over HOLD
        q_advantage = q_values - q_values[:, :, 1:2]  # Advantage over HOLD
        best_per_asset = jnp.argmax(q_values, axis=-1)  # (B, 64)
        max_advantage_asset = jnp.argmax(jnp.max(q_advantage, axis=-1), axis=-1)  # (B,)
        actions = max_advantage_asset * self.n_actions_per_asset + \
                  best_per_asset[jnp.arange(latent.shape[0]), max_advantage_asset]

        # 5. Diagnostics
        search_stats = {
            'root_value': root_value,
            'q_mean': jnp.mean(q_values),
            'q_std': jnp.std(q_values),
            'q_advantage_mean': jnp.mean(jnp.max(q_advantage, axis=-1)),
            'reward_mean': jnp.mean(rewards),
            'n_active_assets': jnp.mean(jnp.sum(jnp.argmax(q_values, axis=-1) != 1, axis=-1)),
        }

        return actions, policy_probs, search_stats

    def __call__(
        self,
        params: dict,
        rng: jnp.ndarray,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Convenience method matching MuZeroMCTXSearch interface.

        TASK_767: action_mask (1 = valid, 0 = invalid) is forwarded to the
        JIT'd search so masked (asset, action) pairs are excluded from the
        policy target and action selection. When action_mask is None the call
        is bit-identical to pre-767 behavior (same 3-arg JIT trace).
        """
        if action_mask is None:
            # Canon path — 3-arg JIT call, byte-identical to pre-767 code.
            return self._search_fn(params, rng, observation)
        return self._search_fn(params, rng, observation, action_mask)
