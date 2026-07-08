"""TASK_838 — tests for per-node action resampling in the sampled vector search.

Covers the ON behavior of ``SampledMuZeroVectorSearch(per_node_resampling=True)``:
  1. Shape/plumbing  — ON runs end-to-end; action vector, candidate set and
                       visit distribution keep the canon shapes/dtypes; the
                       selected vector is still one of the ROOT candidates.
  2. Resampling      — interior nodes genuinely get DIFFERENT sampled sets than
                       the root when ON (the whole point); the root node's
                       stored set equals the returned sampled_vecs.
  3. Correction      — the per-node Sampled-MuZero prior stays exact: every
                       node's K-slot prior is uniform 1/K (zero logits), because
                       each node's samples are drawn i.i.d. from that node's own
                       proposal (β=π per node — Hubert et al. 2021).
  4. Determinism     — bitwise-identical outputs given the same seed (the
                       interior rng is a pre-registered fold_in derivation of
                       the mctx expand rng).
  5. Isolation       — ON does not consume any new top-level rng: the ROOT
                       candidate set is bitwise-identical ON vs OFF at equal
                       seed; OFF stats carry no per-node diagnostics keys.

The OFF bit-identity gate lives in tools/tests/test_pernode_parity.py (pinned
pre-edit); the existing 11 Stage-2 tests in test_sampled_search.py are
untouched. CPU-only, never GPU, never trading_data.db. Run:
    JAX_PLATFORMS=cpu pytest tools/tests/test_pernode_resampling.py -q
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never GPU for these tests
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # tools/tests/ -> tools/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.sampled_search import (
    SampledMuZeroVectorSearch, create_sampled_vector_search,
)


# =============================================================================
# Helpers (test_sampled_search.py pattern)
# =============================================================================

def build_model(n_assets=4, state_dim=16):
    return MuZeroNetworks(
        state_dim=state_dim,
        action_space_size=n_assets * 5,
        n_assets=n_assets,
        encoder_version=2,
        use_vector_action=True,
    )


def init_params(model, n_assets, B=2, seed=0):
    obs = jnp.asarray(
        np.random.default_rng(1).normal(size=(B, 27, 10, n_assets)),
        dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(seed), observation=obs,
                        mode='initialize_parameters')['params']
    return params, obs


def make_search(model, na, K=8, sims=8, on=True, **kw):
    return SampledMuZeroVectorSearch(
        model_apply_fn=model.apply, n_assets=na, num_samples=K,
        num_simulations=sims, max_num_considered_actions=K,
        per_node_resampling=on, **kw)


# =============================================================================
# 1. Shape / plumbing (ON)
# =============================================================================

class TestOnShapePlumbing:

    def test_on_runs_and_shapes(self):
        na, K, B, sims = 4, 8, 2, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = make_search(model, na, K=K, sims=sims, on=True)
        av, sv, vp, stats = search.raw_search(params, jax.random.PRNGKey(7), obs)
        jax.block_until_ready(vp)
        assert av.shape == (B, na) and av.dtype == jnp.int32
        assert int(av.min()) >= 0 and int(av.max()) <= 4
        assert sv.shape == (B, K, na)
        assert vp.shape == (B, K)
        assert np.allclose(np.asarray(vp.sum(-1)), 1.0, atol=1e-4)
        # ON-only diagnostics present with tree shapes (N = sims + 1 nodes).
        assert stats['tree_cand_vecs'].shape == (B, sims + 1, K, na)
        assert stats['tree_children_prior_logits'].shape == (B, sims + 1, K)
        assert stats['tree_node_visits'].shape == (B, sims + 1)

    def test_on_selected_vector_is_a_root_candidate(self):
        """Per-node resampling only changes INTERIOR action sets; the applied
        action still comes from the root's own K candidates."""
        na, K, B = 6, 8, 3
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = make_search(model, na, K=K, sims=8, on=True)
        av, sv, vp, stats = search.raw_search(params, jax.random.PRNGKey(2), obs)
        sel = np.asarray(stats['selected_idx'])
        for b in range(B):
            assert np.array_equal(np.asarray(av[b]), np.asarray(sv[b, sel[b]]))

    def test_on_jit_compiles(self):
        na, K = 4, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=2)
        search = make_search(model, na, K=K, sims=8, on=True)
        av, sv, vp, _ = search(params, jax.random.PRNGKey(5), obs)  # jitted path
        jax.block_until_ready(vp)
        assert av.shape == (2, na)

    def test_on_via_factory_with_mask_and_book(self):
        """Factory passthrough + root mask/book path runs; root-masked actions
        never appear in the ROOT candidate set (interior nodes are unmasked by
        design — planning candidates, never executed)."""
        na, K, B = 4, 8, 2
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = create_sampled_vector_search(
            model, n_assets=na, num_samples=K, num_simulations=8,
            max_num_considered_actions=K, hold_bias=1.0,
            per_node_resampling=True)
        assert search.per_node_resampling is True
        mask = np.ones((B, na * 5), dtype=np.float32)
        mask[:, 0 * 5 + 0] = 0.0   # forbid SELL on asset 0 at the root
        book = jnp.ones((B, na), dtype=jnp.int32)
        av, sv, vp, _ = search.raw_search(
            params, jax.random.PRNGKey(11), obs,
            action_mask=jnp.asarray(mask), book=book, deterministic=True)
        assert not np.any(np.asarray(sv)[:, :, 0] == 0)   # root set respects mask
        assert not np.any(np.asarray(av)[:, 0] == 0)


# =============================================================================
# 2. Interior nodes genuinely resample
# =============================================================================

class TestInteriorResampling:

    def test_interior_nodes_differ_from_root(self):
        """ON: every expanded interior node carries its own sampled set; with a
        fresh-init policy + dither these differ from the root's set (fixed seed
        ⇒ deterministic assertion). Node 0 is the root; nodes 1..sims are the
        per-simulation expansions (mctx expands exactly one node per sim)."""
        na, K, B, sims = 4, 8, 2, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = make_search(model, na, K=K, sims=sims, on=True)
        av, sv, vp, stats = search.raw_search(params, jax.random.PRNGKey(3), obs)
        tree_cand = np.asarray(stats['tree_cand_vecs'])       # (B, N, K, na)
        root_set = np.asarray(sv)                             # (B, K, na)
        # Root node stores the returned root candidate set.
        assert np.array_equal(tree_cand[:, 0], root_set)
        # Every interior node's set differs from the root's.
        for node in range(1, sims + 1):
            assert not np.array_equal(tree_cand[:, node], root_set), (
                f"interior node {node} reused the root candidate set under ON")

    def test_interior_nodes_differ_from_each_other(self):
        """Distinct expand rngs ⇒ distinct per-node samples (fixed seed)."""
        na, K, B, sims = 4, 8, 1, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = make_search(model, na, K=K, sims=sims, on=True)
        _, _, _, stats = search.raw_search(params, jax.random.PRNGKey(6), obs)
        tree_cand = np.asarray(stats['tree_cand_vecs'])[0]    # (N, K, na)
        sets = [tree_cand[n].tobytes() for n in range(1, sims + 1)]
        assert len(set(sets)) == len(sets), "two interior nodes drew identical sets"

    def test_off_has_no_pernode_diagnostics(self):
        """OFF: root-reuse — no per-node candidate sets exist and the stats
        dict must stay byte-compatible with canon (no new keys)."""
        na, K = 4, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=2)
        search = make_search(model, na, K=K, sims=8, on=False)
        _, _, _, stats = search.raw_search(params, jax.random.PRNGKey(3), obs)
        assert set(stats.keys()) == {
            'root_value', 'beta_hat', 'selected_idx', 'visit_entropy'}


# =============================================================================
# 3. Per-node Sampled-MuZero correction stays exact
# =============================================================================

class TestPerNodeCorrection:

    def test_prior_uniform_per_node(self):
        """Every node's K-slot prior is uniform 1/K (zero logits): each node's
        slots are i.i.d. draws from that node's own proposal, so the
        sample-based prior β̂=n_a/K is exactly the uniform slot prior per node
        (Hubert et al. 2021 correction, applied node-locally)."""
        na, K, B, sims = 4, 8, 2, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = make_search(model, na, K=K, sims=sims, on=True)
        _, _, _, stats = search.raw_search(params, jax.random.PRNGKey(9), obs)
        prior = np.asarray(stats['tree_children_prior_logits'])   # (B, N, K)
        visits = np.asarray(stats['tree_node_visits'])            # (B, N)
        # Expanded nodes (visits > 0) — includes the root — carry zero logits
        # ⇒ softmax = 1/K per slot, summing to 1 per node.
        expanded = visits > 0
        assert expanded.any()
        assert np.allclose(prior[expanded], 0.0)
        probs = np.exp(prior[expanded]) / np.exp(prior[expanded]).sum(-1, keepdims=True)
        assert np.allclose(probs, 1.0 / K)
        assert np.allclose(probs.sum(-1), 1.0)

    def test_root_beta_hat_still_empirical(self):
        """The root β̂ diagnostic (n_a/K) is untouched by the ON path."""
        na, K = 2, 4
        model = build_model(na)
        search = make_search(model, na, K=K, sims=4, on=True)
        sv = jnp.asarray([[[0, 0], [0, 0], [1, 1], [2, 2]]], dtype=jnp.int32)
        _, beta = search._empirical_prior(sv)
        assert np.allclose(np.asarray(beta[0]), [0.5, 0.5, 0.25, 0.25])


# =============================================================================
# 4. Determinism given seed
# =============================================================================

class TestDeterminism:

    def test_on_deterministic_given_seed(self):
        na, K, B, sims = 4, 8, 2, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = make_search(model, na, K=K, sims=sims, on=True)
        outs = []
        for _ in range(2):
            av, sv, vp, stats = search.raw_search(params, jax.random.PRNGKey(21), obs)
            jax.block_until_ready(vp)
            outs.append((np.asarray(av).tobytes(), np.asarray(sv).tobytes(),
                         np.asarray(vp).tobytes(),
                         np.asarray(stats['tree_cand_vecs']).tobytes()))
        assert outs[0] == outs[1], "ON search is not deterministic given the seed"

    def test_on_different_seed_different_tree(self):
        na, K, B, sims = 4, 8, 1, 8
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        search = make_search(model, na, K=K, sims=sims, on=True)
        _, _, _, s1 = search.raw_search(params, jax.random.PRNGKey(0), obs)
        _, _, _, s2 = search.raw_search(params, jax.random.PRNGKey(1), obs)
        assert (np.asarray(s1['tree_cand_vecs']).tobytes()
                != np.asarray(s2['tree_cand_vecs']).tobytes())


# =============================================================================
# 5. RNG isolation — ON adds no new top-level rng consumption
# =============================================================================

class TestRngIsolation:

    def test_root_candidates_identical_on_vs_off(self):
        """Same seed ⇒ the ROOT candidate set is bitwise-identical ON vs OFF:
        the top-level (sample, dither, search) split is untouched; interior
        resampling rng derives ONLY from mctx's expand keys."""
        na, K, B = 4, 8, 2
        model = build_model(na)
        params, obs = init_params(model, na, B=B)
        rng = jax.random.PRNGKey(17)
        s_off = make_search(model, na, K=K, sims=8, on=False)
        s_on = make_search(model, na, K=K, sims=8, on=True)
        _, sv_off, _, _ = s_off.raw_search(params, rng, obs)
        _, sv_on, _, _ = s_on.raw_search(params, rng, obs)
        assert np.array_equal(np.asarray(sv_off), np.asarray(sv_on))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
