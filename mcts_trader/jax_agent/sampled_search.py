"""TASK_828 Stage 2 — Sampled-MuZero MCTS over joint action-VECTORS.

The scalar mctx search (``mctx_search.py``, CANON) holds a single categorical
over the 320 joint (asset, action) indices — it cannot represent the 5^n_assets
joint action-vector space a portfolio rebalance lives in. This module adds a
NEW, flag-gated search (Sampled MuZero, Hubert et al. 2021, arXiv:2104.06303):

  1. At the root, SAMPLE K candidate action-VECTORS from the factored policy
     head's per-asset marginals (independent per-asset categorical draws). K is
     a config parameter (default 16). Duplicates are allowed (i.i.d. sampling).
  2. Treat those K sampled vectors as the node's action set — a reduced MDP with
     a {0..K-1} action space. Run mctx's Gumbel-MuZero search / Sequential
     Halving + qtransform machinery over the K candidates (REUSED unchanged from
     canon; ``mctx_search.py`` is not imported or modified here → the canon
     scalar path is trivially bit-identical).
  3. Expand the tree with the SAMPLED VECTOR (not a scalar index): the
     recurrent_fn gathers candidate k's vector and calls the model's
     ``recurrent_inference_vector`` (dynamics_vector embeds the (n_assets,)
     vector — muzero_model.py). Reward/value therefore condition on the whole
     rebalance.
  4. CORRECT visit counts for the sampling distribution (§ _empirical_prior):
     the intractable joint prior π(a)=Πᵢπᵢ(aᵢ) is replaced by the EMPIRICAL
     sample-based prior β̂(a)=n_a/K, expressed per-slot as a uniform 1/K prior
     (log-uniform prior_logits). Because the K slots were drawn i.i.d. from β,
     duplicate vectors accumulate n_a/K → the search operates with an effective
     prior ≈ β WITHOUT ever evaluating the joint. This is THE Sampled-MuZero
     correction (Hubert 2021 §3, PUCT/empirical-prior variant).

Returns, per batch element: the selected action-VECTOR (B, n_assets) ∈ {0..4},
the K candidate vectors (B, K, n_assets), and the visit distribution over the
sampled set (B, K) — the raw material for the Stage-3 factored policy target.

Design calls flagged for the verify panel / operator (see task Factory notes):
  * DEPTH / tree reuse: interior nodes reuse the ROOT's K sampled vectors as
    their action set (mctx requires a fixed action space across depth). Full
    per-node resampling is a documented follow-up. With num_simulations modest
    the search is dominated by root Sequential-Halving over the sampled set
    (Sampled-MuZero root planning), which is the load-bearing behavior here.
  * PRIOR: uniform-per-slot empirical prior at ALL depths (the samples ARE the
    prior; the model policy enters only via the sampling proposal). Alternative:
    importance-weighted per-candidate logp(vec) interior priors — not used, to
    keep the correction single-sourced and unbiased.
  * EMBEDDING: joint one-hot → single Dense (VectorActionEmbedding). See
    muzero_model.py for the rationale.

CPU / fixture-only for tests. Never GPU-hours in the factory loop.
"""

import jax
import jax.numpy as jnp
import mctx
from typing import Callable, Tuple, Any
import chex


class SampledMuZeroVectorSearch:
    """Sampled-MuZero MCTS over joint per-asset action-vectors (flag-gated).

    Drop-in-adjacent to ``MuZeroMCTXSearch`` but returns an action VECTOR
    (B, n_assets) rather than a scalar action index. Requires a model built with
    ``use_vector_action=True`` (so ``recurrent_inference_vector`` /
    ``dynamics_vector`` exist).
    """

    def __init__(
        self,
        model_apply_fn: Callable,
        n_assets: int = 64,
        n_actions_per_asset: int = 5,
        num_samples: int = 16,          # K — candidate action-vectors per root
        num_simulations: int = 32,
        max_num_considered_actions: int = 16,
        discount: float = 0.997,
        gumbel_scale: float = 1.0,
        value_scale: float = 1.0,
        value_shift: float = 0.0,
    ):
        self.model_apply_fn = model_apply_fn
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        self.num_samples = num_samples          # K (static)
        self.num_simulations = num_simulations
        # Sequential Halving cannot consider more actions than we sampled.
        self.max_num_considered_actions = min(max_num_considered_actions,
                                              num_samples)
        self.discount = discount
        self.gumbel_scale = gumbel_scale
        self.value_scale = value_scale
        self.value_shift = value_shift

        self.search_fn = jax.jit(self._search)

    # ---- value/reward categorical → scalar (mirrors mctx_search.py canon) ----

    def _categorical_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        from . import get_value_support
        support = get_value_support()
        probs = jax.nn.softmax(logits, axis=-1)
        raw_value = jnp.sum(probs * support, axis=-1)
        return raw_value * self.value_scale + self.value_shift

    def _reward_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        from . import get_value_support
        support = get_value_support()
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * support, axis=-1)

    # ---- sampling + Sampled-MuZero visit-count correction --------------------

    def _sample_action_vectors(
        self, rng: chex.PRNGKey, policy_logits: jnp.ndarray
    ) -> jnp.ndarray:
        """Draw K action-VECTORS from the factored policy's per-asset marginals.

        policy_logits: (B, n_assets*A) — the flattened factored policy head, laid
        out [asset, action] (index = asset*A + action, matching the env decode
        asset=idx//A, action=idx%A). Reshape → (B, n_assets, A); each per-asset
        row is an independent categorical marginal. K i.i.d. joint samples are
        drawn by sampling every asset independently.

        Returns (B, K, n_assets) int32 in {0..A-1}.
        """
        B = policy_logits.shape[0]
        pa_logits = policy_logits.reshape(B, self.n_assets, self.n_actions_per_asset)
        # (K, B, n_assets) independent per-asset categorical draws → transpose.
        samples = jax.random.categorical(
            rng, pa_logits, axis=-1, shape=(self.num_samples, B, self.n_assets)
        )
        return jnp.transpose(samples, (1, 0, 2)).astype(jnp.int32)  # (B, K, n_assets)

    def _empirical_prior(
        self, sampled_vecs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sampled-MuZero sample-based prior + per-candidate empirical mass β̂.

        The proposal is β = the factored policy π. K vectors are drawn i.i.d.
        from β. The Sampled-MuZero correction replaces the intractable joint
        prior π(a)=Πᵢπᵢ(aᵢ) with the empirical frequency β̂(a)=n_a/K. In the
        STATIC-shape (fixed-K) representation this is a uniform 1/K prior over
        the K slots (log-uniform prior_logits); identical vectors then accumulate
        n_a/K mass naturally inside mctx's visit allocation — recovering β̂
        WITHOUT evaluating the joint.

        Returns:
            prior_logits: (B, K) — log-uniform (all zeros ⇒ softmax = 1/K). This
                is what the search consumes as the corrected prior.
            beta_hat:     (B, K) — per-slot per-candidate empirical mass n_a/K
                (n_a = number of the K slots whose vector equals this slot's).
                Exposed for diagnostics + the Stage-3 training-target correction;
                unit-tested against hand-worked tiny examples.
        """
        B, K, _ = sampled_vecs.shape
        prior_logits = jnp.zeros((B, K), dtype=jnp.float32)  # uniform 1/K
        # eq[b,i,j] = (vector at slot i == vector at slot j)
        eq = jnp.all(sampled_vecs[:, :, None, :] == sampled_vecs[:, None, :, :],
                     axis=-1)                                  # (B, K, K)
        counts = jnp.sum(eq.astype(jnp.int32), axis=-1)        # (B, K): n_a per slot
        beta_hat = counts.astype(jnp.float32) / float(K)       # (B, K)
        return prior_logits, beta_hat

    # ---- search --------------------------------------------------------------

    def _search(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Run Sampled-MuZero vector search.

        Returns:
            action_vec:   (B, n_assets) int32 — selected candidate vector
            sampled_vecs: (B, K, n_assets) int32 — the K candidate vectors
            visit_probs:  (B, K) — visit distribution over the sampled set
            search_stats: dict
        """
        sample_rng, search_rng = jax.random.split(rng)

        # 1. Root inference → factored policy, value, latent.
        root_out = self.model_apply_fn(
            {'params': params}, observation, method='initial_inference'
        )
        policy_logits = root_out['policy_logits']            # (B, n_assets*A)
        latent = root_out['latent_state']                    # (B, state_dim)
        root_value = self._categorical_to_scalar(root_out['value'])  # (B,)

        # 2. Sample K candidate action-vectors from the per-asset marginals.
        sampled_vecs = self._sample_action_vectors(sample_rng, policy_logits)  # (B,K,A_n)

        # 3. Sampled-MuZero corrected prior over the K slots (uniform 1/K).
        prior_logits, beta_hat = self._empirical_prior(sampled_vecs)  # (B,K),(B,K)

        root = mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=root_value,
            embedding=latent,
        )

        B = observation.shape[0]

        def recurrent_fn(params_, rng_, action, embedding):
            # action: (B,) index into the K sampled candidates. Gather each
            # batch element's chosen action-VECTOR and expand via the vector
            # dynamics path (embeds the vector, not a scalar index).
            idx = action.astype(jnp.int32)[:, None, None]          # (B,1,1)
            vec = jnp.take_along_axis(sampled_vecs, idx, axis=1)[:, 0, :]  # (B,A_n)
            out = self.model_apply_fn(
                {'params': params_}, embedding, vec,
                mode='recurrent_inference_vector'
            )
            reward = self._reward_to_scalar(out['reward'])
            value = self._categorical_to_scalar(out['value'])
            # Interior prior over the (reused) K sampled slots = same empirical
            # uniform prior (documented simplification — see module docstring).
            interior_prior = jnp.zeros((B, self.num_samples), dtype=jnp.float32)
            recurrent_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, self.discount),
                prior_logits=interior_prior,
                value=value,
            )
            return recurrent_output, out['latent_state']

        # 4. Gumbel-MuZero search / Sequential Halving over the K candidates —
        #    the same gumbel + qtransform machinery as the canon scalar search.
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=search_rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.num_simulations,
            max_num_considered_actions=self.max_num_considered_actions,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            gumbel_scale=self.gumbel_scale,
        )

        selected_idx = policy_output.action                  # (B,)
        visit_probs = policy_output.action_weights           # (B, K)

        # 5. Map the selected candidate index back to its action VECTOR.
        sel = selected_idx.astype(jnp.int32)[:, None, None]  # (B,1,1)
        action_vec = jnp.take_along_axis(sampled_vecs, sel, axis=1)[:, 0, :]  # (B,A_n)

        search_stats = {
            'root_value': root_value,
            'beta_hat': beta_hat,
            'selected_idx': selected_idx,
            'visit_entropy': -jnp.sum(
                visit_probs * jnp.log(jnp.maximum(visit_probs, 1e-10)), axis=-1),
        }
        return action_vec, sampled_vecs, visit_probs, search_stats

    def __call__(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """JIT-compiled Sampled-MuZero vector search."""
        return self.search_fn(params, rng, observation)

    def raw_search(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Non-JIT search (for use inside an outer lax.scan / debugging)."""
        return self._search(params, rng, observation)


def create_sampled_vector_search(model, **kwargs):
    """Factory: build a SampledMuZeroVectorSearch from a MuZeroNetworks model.

    The model MUST be built with use_vector_action=True.
    """
    return SampledMuZeroVectorSearch(model_apply_fn=model.apply, **kwargs)
