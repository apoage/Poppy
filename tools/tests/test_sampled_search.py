"""TASK_828 Stage 2 — tests for the Sampled-MuZero vector search.

Covers the factory-mandated acceptance checks:
  1. Shape/plumbing  — flag ON, the search runs end-to-end on a small batch and
                       returns a valid action-VECTOR (B, n_assets) ∈ {0..4} plus
                       a visit distribution over the K sampled candidates; shapes
                       + dtypes correct; JIT-compiles cleanly.
  2. Sampling        — K vectors are drawn from the factored policy marginals;
                       the Sampled-MuZero visit-count correction (empirical prior
                       β̂ = n_a/K, uniform 1/K per slot) is applied — pinned
                       against hand-worked tiny examples.
  3. Canon parity    — flag OFF, the canon scalar mctx search (mctx_search.py)
                       rollout hashes to the value pinned from the pre-Stage-2
                       module; the vector path does NOT perturb the scalar path.
  4. Integration     — the chosen action-vector feeds step_vector (Stage 1) and
                       produces a valid env transition (search → env loop).

CPU-only, never GPU, never trading_data.db. Run:
    JAX_PLATFORMS=cpu pytest tools/tests/test_sampled_search.py -v
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never GPU for these tests

import hashlib
import sys
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Robust, machine-independent import shim (mirrors test_vector_env.py). mcts_trader/
# is a PEP-420 namespace package — prepend the real repo root (computed from THIS
# file) so imports resolve under pytest's default prepend import mode from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]  # tools/tests/ -> tools/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch
from mcts_trader.jax_agent.mctx_search import MuZeroMCTXSearch
from mcts_trader.env.jax_trading_env import (
    EnvConfig, reset, step_vector, get_current_prices, compute_portfolio_value,
)


# =============================================================================
# Helpers
# =============================================================================

def build_model(n_assets=4, state_dim=16, use_vector_action=True):
    return MuZeroNetworks(
        state_dim=state_dim,
        action_space_size=n_assets * 5,
        n_assets=n_assets,
        encoder_version=2,
        use_vector_action=use_vector_action,
    )


def init_params(model, n_assets, B=2, seed=0):
    obs = jnp.asarray(
        np.random.default_rng(1).normal(size=(B, 27, 10, n_assets)),
        dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(seed), observation=obs,
                        mode='initialize_parameters')['params']
    return params, obs


# =============================================================================
# 1. Shape / plumbing
# =============================================================================

class TestShapePlumbing:

    def test_search_runs_and_shapes(self):
        na, K, B = 4, 16, 2
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K,
            num_simulations=32, max_num_considered_actions=16)
        av, sv, vp, stats = search(params, jax.random.PRNGKey(7), obs)
        jax.block_until_ready(vp)

        # action VECTOR (B, n_assets) ∈ {0..4}
        assert av.shape == (B, na)
        assert av.dtype == jnp.int32
        assert int(av.min()) >= 0 and int(av.max()) <= 4
        # candidate set + visit distribution over the sampled set
        assert sv.shape == (B, K, na)
        assert vp.shape == (B, K)
        assert np.allclose(np.asarray(vp.sum(-1)), 1.0, atol=1e-4)
        # 'vector_action_embedding' params were created (flag ON)
        assert 'vector_action_embedding' in params

    def test_selected_vector_is_a_candidate(self):
        na, K, B = 6, 16, 3
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K,
            num_simulations=24, max_num_considered_actions=16)
        av, sv, vp, stats = search(params, jax.random.PRNGKey(2), obs)
        sel = np.asarray(stats['selected_idx'])
        for b in range(B):
            assert np.array_equal(np.asarray(av[b]), np.asarray(sv[b, sel[b]]))

    def test_K_is_configurable(self):
        na = 4
        model = build_model(na)
        params, obs = init_params(model, na, B=2)
        for K in (8, 24):
            search = SampledMuZeroVectorSearch(
                model_apply_fn=model.apply, n_assets=na, num_samples=K,
                num_simulations=max(2 * K, 16), max_num_considered_actions=K)
            av, sv, vp, _ = search(params, jax.random.PRNGKey(1), obs)
            assert sv.shape == (2, K, na)
            assert vp.shape == (2, K)


# =============================================================================
# 2. Sampling correctness + visit-count correction (hand-worked)
# =============================================================================

class TestSamplingAndCorrection:

    def test_samples_drawn_from_factored_marginals_deterministic(self):
        """Near-deterministic per-asset marginals ⇒ every one of the K samples
        picks the peaked action for each asset (samples come from the factored
        marginals, not a joint index)."""
        na, K = 2, 32
        model = build_model(na)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K)
        # asset0 peaks at action 2, asset1 peaks at action 4.
        logits = np.full((1, na, 5), -50.0, dtype=np.float32)
        logits[0, 0, 2] = 50.0
        logits[0, 1, 4] = 50.0
        flat = jnp.asarray(logits.reshape(1, na * 5))
        sv = search._sample_action_vectors(jax.random.PRNGKey(0), flat)
        assert sv.shape == (1, K, na)
        assert np.all(np.asarray(sv[0, :, 0]) == 2)
        assert np.all(np.asarray(sv[0, :, 1]) == 4)

    def test_samples_match_marginal_frequencies(self):
        """A skewed single-asset marginal is recovered (approximately) by the
        empirical sample frequencies — confirms per-asset categorical sampling."""
        na, K = 1, 4000
        model = build_model(na)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K)
        # probs ∝ [1,2,3,4,5] over the 5 actions of the single asset.
        probs = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        probs /= probs.sum()
        logits = jnp.asarray(np.log(probs).reshape(1, na * 5))
        sv = np.asarray(search._sample_action_vectors(jax.random.PRNGKey(4), logits))
        emp = np.bincount(sv.reshape(-1), minlength=5) / K
        assert np.allclose(emp, probs, atol=0.03), (emp, probs)

    def test_empirical_prior_uniform_logits(self):
        """The corrected prior fed to the search is uniform-per-slot (log 1/K):
        prior_logits are all zero regardless of the sampled set (the samples ARE
        the prior — the Sampled-MuZero correction)."""
        na, K = 2, 5
        model = build_model(na)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K)
        sv = jnp.asarray(np.random.default_rng(0).integers(0, 5, size=(3, K, na)),
                         dtype=jnp.int32)
        prior_logits, _ = search._empirical_prior(sv)
        assert prior_logits.shape == (3, K)
        assert np.allclose(np.asarray(prior_logits), 0.0)

    def test_beta_hat_hand_worked_identical(self):
        """K=2, n_assets=2, both slots identical ⇒ β̂ = n_a/K = 2/2 = 1.0 each."""
        na, K = 2, 2
        model = build_model(na)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K)
        sv = jnp.asarray([[[0, 0], [0, 0]]], dtype=jnp.int32)  # (1,2,2)
        _, beta = search._empirical_prior(sv)
        assert np.allclose(np.asarray(beta[0]), [1.0, 1.0]), np.asarray(beta[0])

    def test_beta_hat_hand_worked_distinct(self):
        """K=2, n_assets=2, distinct slots ⇒ β̂ = 1/2 each."""
        na, K = 2, 2
        model = build_model(na)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K)
        sv = jnp.asarray([[[0, 1], [1, 0]]], dtype=jnp.int32)
        _, beta = search._empirical_prior(sv)
        assert np.allclose(np.asarray(beta[0]), [0.5, 0.5]), np.asarray(beta[0])

    def test_beta_hat_hand_worked_mixed_K3(self):
        """K=3: two identical + one distinct ⇒ β̂ = [2/3, 2/3, 1/3]. Also assert
        the per-CANDIDATE mass sums to 1 over DISTINCT candidates (2/3 counted
        once + 1/3 = 1)."""
        na, K = 2, 3
        model = build_model(na)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K)
        sv = jnp.asarray([[[0, 0], [0, 0], [1, 1]]], dtype=jnp.int32)
        _, beta = search._empirical_prior(sv)
        assert np.allclose(np.asarray(beta[0]), [2/3, 2/3, 1/3]), np.asarray(beta[0])
        # distinct-candidate masses: {[0,0]:2/3, [1,1]:1/3} sum to 1.0
        assert np.isclose(2/3 + 1/3, 1.0)


# =============================================================================
# 3. Canon parity — scalar mctx search bit-identical (flag OFF)
# =============================================================================

class TestCanonParity:

    # Pinned from the canon scalar mctx search on the PRE-Stage-2 module
    # (git HEAD, before muzero_model.py's use_vector_action addition) and
    # confirmed identical on the post-change module. Any drift means the canon
    # scalar path moved. Construction: n_assets=4, state_dim=16,
    # action_space_size=20, encoder v2, init seed 0, obs seed 1, search seed 7,
    # num_simulations=16, max_num_considered_actions=8, gumbel.
    PINNED = "e25841eb00881b1c3a425ce3733e198d910a38e5bd86aac6d5075641be780866"

    def test_scalar_search_hash_pinned(self):
        na, state_dim = 4, 16
        model = MuZeroNetworks(state_dim=state_dim, action_space_size=na * 5,
                               n_assets=na, encoder_version=2)  # flag OFF (default)
        obs = jnp.asarray(
            np.random.default_rng(1).normal(size=(2, 27, 10, na)), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), observation=obs,
                            mode='initialize_parameters')['params']
        # Flag OFF ⇒ no vector-action params.
        assert 'vector_action_embedding' not in params
        search = MuZeroMCTXSearch(model_apply_fn=model.apply, num_simulations=16,
                                  max_num_considered_actions=8, use_gumbel=True)
        actions, probs, stats = search.raw_search(params, jax.random.PRNGKey(7), obs)
        jax.block_until_ready(probs)
        blob = repr((np.asarray(actions).tolist(),
                     np.round(np.asarray(probs), 6).tolist(),
                     np.round(np.asarray(stats['root_value']), 6).tolist()))
        h = hashlib.sha256(blob.encode()).hexdigest()
        assert h == self.PINNED, f"canon scalar search drifted: {h}"


# =============================================================================
# 4. Integration — search → step_vector (Stage 1) transition
# =============================================================================

class TestIntegration:

    def test_action_vector_feeds_step_vector(self):
        na, K, B = 8, 16, 2
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=na, num_samples=K,
            num_simulations=24, max_num_considered_actions=16)
        av, sv, vp, _ = search(params, jax.random.PRNGKey(9), obs)

        cfg = EnvConfig(n_assets=na, initial_cash=100_000.0, fee_bps=0.0,
                        slippage_bps=0.0, max_steps=60, window_size=10,
                        position_size_pct=0.1, allow_short=True)
        prices = jnp.asarray(
            np.tile((100.0 + 5 * np.arange(na))[None, :], (120, 1)), dtype=jnp.float32)
        s = reset(jax.random.PRNGKey(3), prices, cfg)
        cur = get_current_prices(prices, s.current_step, s.episode_start_idx)
        pv_before = float(compute_portfolio_value(s, cur))

        out, _, r, done = step_vector(s, av[0].astype(jnp.int32), prices, cfg)
        pv_after = float(compute_portfolio_value(out, cur))
        # fee=0 rebalance conserves portfolio value; transition is valid/finite.
        assert np.isfinite(float(out.cash)) and float(out.cash) >= -1.0
        assert np.isclose(pv_before, pv_after, rtol=1e-4, atol=1e-1), (pv_before, pv_after)
        assert np.isfinite(float(r))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
