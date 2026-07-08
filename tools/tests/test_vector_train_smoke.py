"""TASK_828 Stage 3 §4b — flag-ON e2e vector-action smoke (CPU).

Runs the vector-action training path end-to-end on CPU (self-play → replay buffer →
trainer) plus a direct search probe, asserting the plumbing is shape-correct, finite,
and mask-respecting. This is a SHAPES + FINITE + NO-CRASH gate ONLY — no accuracy or
hash claim (the gumbel dither + K-sampling are nondeterministic, so there is nothing
to pin).

Mirrors the in-process harness structure of ``test_vector_train_parity.py`` but with
``vector_action=True, vector_k=8``. Kept tiny for a fast deterministic-enough CPU run.

Run: JAX_PLATFORMS=cpu pytest tools/tests/test_vector_train_smoke.py -q
"""
import os
# CPU / fixture-only; never GPU in the loop, never trading_data.db.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys

import numpy as np
import jax
import jax.numpy as jnp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_SEED = 0
_N_ASSETS = 64
_A = 5          # n_actions_per_asset
_VK = 8         # sampled candidate action-vectors per root
_CAPACITY = 5000


def _setup():
    from mcts_trader.training.config import TrainingConfig
    from mcts_trader.training.setup import (
        setup_jax_environment, setup_model_and_trainer,
    )
    config = TrainingConfig(
        name="vec_smoke", synthetic_level=0, seed=_SEED,
        parallel_envs=2, steps=16, epochs=2, sims=8, batch_size=32,
        model_size="small",
    )
    config.use_jax_env = True
    config.vector_action = True
    config.vector_k = _VK
    states, prices, env_cfg = setup_jax_environment(config, seed=_SEED)
    model, trainer = setup_model_and_trainer(config, _N_ASSETS)
    return config, states, prices, env_cfg, model, trainer


def test_vector_train_smoke():
    from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
    from mcts_trader.jax_agent.sampled_search import create_sampled_vector_search
    from mcts_trader.training.self_play import (
        run_self_play_jax, reconstruct_observations,
    )

    config, states, prices, env_cfg, model, trainer = _setup()
    n_assets = _N_ASSETS

    # ---- flag threaded through the model ----
    assert model.use_vector_action, "setup did not enable model.use_vector_action"
    assert trainer.vector_action, "setup did not enable trainer.vector_action"
    assert env_cfg.vector_action, "setup did not enable env_config.vector_action"

    # ---- Sampled-MuZero vector search ----
    search_obj = create_sampled_vector_search(
        model, n_assets=n_assets, n_actions_per_asset=_A,
        num_samples=_VK, num_simulations=config.sims,
        max_num_considered_actions=min(_VK, 16), discount=config.discount,
        gumbel_scale=config.gumbel_scale, value_scale=config.value_scale,
        value_shift=config.value_shift,
    )
    search_raw = search_obj.raw_search

    # ---- replay buffer stores (capacity, 64) action VECTORS + (capacity, 64, 5) marginals ----
    buf = JAXReplayBuffer(
        capacity=_CAPACITY, observation_shape=(27, n_assets, n_assets),
        action_size=n_assets * _A, batch_size=config.batch_size,
        n_assets=n_assets, vector_action=True,
    )
    assert buf.action_vecs.shape == (_CAPACITY, n_assets), buf.action_vecs.shape
    assert buf.policy_marginals.shape == (_CAPACITY, n_assets, _A), buf.policy_marginals.shape

    rng = jax.random.PRNGKey(_SEED)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.state = trainer.init_train_state(init_rng, dummy_obs)
    popart_state = trainer.popart.state if getattr(trainer, "popart", None) else None

    # ---- self-play emits action_vecs (n_steps, 64) + policy_marginals (n_steps, 64, 5) ----
    rng, sp_rng = jax.random.split(rng)
    results, _scores = run_self_play_jax(
        states, prices, env_cfg, search_raw, model.apply,
        trainer.state.params, sp_rng,
        steps=config.steps, support_size=model.support_size,
        discount=config.discount, popart_state=popart_state,
        vector_action=True,
    )
    assert len(results) > 0, "self-play returned no trajectories"
    for traj in results:
        assert 'action_vecs' in traj and 'policy_marginals' in traj, \
            "vector self-play must emit action_vecs + policy_marginals"
        av = np.asarray(traj['action_vecs'])
        pm = np.asarray(traj['policy_marginals'])
        assert av.ndim == 2 and av.shape[1] == n_assets, f"action_vecs shape {av.shape}"
        assert pm.ndim == 3 and pm.shape[1:] == (n_assets, _A), f"policy_marginals shape {pm.shape}"
        assert av.shape[0] == pm.shape[0], "action_vecs / marginals step mismatch"
        # per-asset action-types are legal codes {0..4}
        assert av.min() >= 0 and av.max() < _A, "illegal per-asset action-type in action_vecs"
        # factored marginals normalise per asset (fp16 storage → loose atol)
        sums = pm.sum(axis=-1)
        assert np.all(np.isfinite(pm)), "non-finite policy_marginals"
        assert np.allclose(sums, 1.0, atol=2e-3), \
            f"policy_marginals not per-asset-normalised (max dev {np.abs(sums - 1).max():.3e})"

    # ---- buffer add_trajectory routes vectors through the vector arrays ----
    for traj in results:
        if "observations" not in traj and "window_starts" in traj:
            traj["observations"] = reconstruct_observations(
                traj, n_assets=n_assets, window_size=60)
        buf.add_trajectory(
            jnp.array(traj["observations"]), jnp.array(traj["action_vecs"]),
            jnp.array(traj["rewards"]), jnp.array(traj["values"]),
            jnp.array(traj["policy_marginals"]), jnp.array(traj["dones"]),
        )

    # ---- trainer runs recurrent_inference_vector + returns FINITE losses ----
    rng, tr_rng = jax.random.split(rng)
    hist = trainer.train(buf, num_iterations=2, rng=tr_rng)
    for key in ("total_loss", "value_loss", "policy_loss", "reward_loss"):
        vals = hist.get(key, [])
        assert len(vals) > 0, f"trainer recorded no {key}"
        arr = np.asarray(vals, dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"{key} not finite: {vals}"

    # ---- direct search probe: mask enforcement + deterministic visit-argmax ----
    rng, s_rng = jax.random.split(rng)
    B = 2
    obs = jnp.ones((B, 27, n_assets, n_assets))
    # Forbid action-type 0 (SELL) on every asset (asset-major (B, n_assets*A) mask).
    mask = (jnp.ones((B, n_assets, _A), dtype=jnp.int8)
            .at[..., 0].set(0)
            .reshape(B, n_assets * _A))
    book = jnp.ones((B, n_assets), dtype=jnp.int32)  # v1 book = HOLD
    action_vec, sampled_vecs, visit_probs, stats = search_raw(
        trainer.state.params, s_rng, obs, mask, book, True)  # deterministic=True

    sv = np.asarray(sampled_vecs)                    # (B, K, n_assets)
    assert sv.min() >= 0 and sv.max() < _A, "search returned an out-of-range per-asset slot"
    assert not np.any(sv == 0), "search sampled a masked-out per-asset slot (SELL forbidden)"

    # deterministic=True selects the visit-argmax candidate vector.
    vp = np.asarray(visit_probs)                     # (B, K)
    amax = np.argmax(vp, axis=-1)
    av_sel = np.asarray(action_vec)                  # (B, n_assets)
    for b in range(B):
        assert np.array_equal(av_sel[b], sv[b, amax[b]]), \
            "deterministic eval action != visit-argmax candidate"
    assert np.all(np.isfinite(np.asarray(stats['root_value']))), "non-finite root_value"


if __name__ == "__main__":
    test_vector_train_smoke()
    print("PASS")
