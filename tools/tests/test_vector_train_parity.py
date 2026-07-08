"""TASK_828 Stage 3 — Canon-parity gate for the vector-action training build.

This is the LOAD-BEARING canon gate: with ``--vector-action`` OFF, the scalar
training path must remain BIT-IDENTICAL to canon. We prove it by running a tiny
but REAL self-play + learn loop (the same components ``train_jax_fast.py`` wires:
``setup_jax_environment`` → ``setup_model_and_trainer`` → ``create_mctx_search`` →
``run_self_play_jax`` → ``JAXReplayBuffer`` → ``MuZeroTrainer.train``) and hashing
the resulting params + step_count + per-epoch losses.

``PINNED`` was captured on the pre-Stage-3 HEAD (commit + JAX version recorded
below) BEFORE any Stage-3 edit. After the Stage-3 edits land, the OFF path must
still hash to ``PINNED`` — that is the bit-identity proof. The vector branches are
all gated behind a STATIC Python bool captured at trace/construction time, so XLA
dead-code-eliminates them when OFF and the traced graphs are unchanged.

Two variants (spec §4a): the in-process run AND a fresh-subprocess run must BOTH
equal ``PINNED`` — the subprocess catches any cross-process non-determinism that
would silently invalidate the pin.

Run:  JAX_PLATFORMS=cpu pytest tools/tests/test_vector_train_parity.py -q
Pin:  JAX_PLATFORMS=cpu python tools/tests/test_vector_train_parity.py   # prints hash
"""
import os
# CPU / fixture-only; never GPU in the loop, never trading_data.db.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hashlib
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp

# Repo root on sys.path so `python tools/tests/...` works standalone.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---------------------------------------------------------------------------
# PINNED canon hash — captured on HEAD BEFORE any Stage-3 edit.
#   commit:      60717ae9e38e1b311039e9e2a1a66825c3492ce1
#   jax version: 0.6.2
# If this breaks after a Stage-3 edit, that is a REAL canon regression — find the
# reorder / extra rng draw / param-leaf change; do NOT re-pin to hide it.
PINNED = "a0627ce09aa688c39c23d94b7d997818a876cacdd5a097655c26e84ef7934754"
PIN_COMMIT = "60717ae9e38e1b311039e9e2a1a66825c3492ce1"
PIN_JAX_VERSION = "0.6.2"

# Micro-config knobs (spec §4a). Kept small for a fast CPU run; determinism does
# not depend on the counts, only on the fixed seed.
_SEED = 0
_N_ASSETS = 64
_NUM_TRAIN_ITERS = 4  # trainer.train iterations per epoch (real learn steps)


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)


def run_micro_training(vector_action: bool = False, vector_k: int = 8):
    """Run a tiny deterministic self-play + learn loop; return (params, step, losses).

    Mirrors the canon ``train_jax_fast.py`` wiring for the simplest path (shared
    synthetic prices, no prediction/imprint/differential/epsilon extras). With
    ``vector_action=False`` this exercises the exact scalar components the Stage-3
    edits touch, so an unchanged hash proves those edits are OFF-inert.
    """
    from mcts_trader.training.config import TrainingConfig
    from mcts_trader.training.setup import (
        setup_jax_environment, setup_model_and_trainer,
    )
    from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
    from mcts_trader.jax_agent.mctx_search import create_mctx_search
    from mcts_trader.training.self_play import (
        run_self_play_jax, reconstruct_observations,
    )

    _seed_everything(_SEED)
    n_assets = _N_ASSETS

    config = TrainingConfig(
        name="parity_micro",
        synthetic_level=0,
        seed=_SEED,
        parallel_envs=2,
        steps=16,
        epochs=3,
        sims=8,
        batch_size=32,
        model_size="small",
    )
    config.use_jax_env = True
    # Set the flag on the config object either way; on the OFF path all Stage-3
    # getattr(config,'vector_action',False) reads return False → canon behavior.
    config.vector_action = vector_action
    config.vector_k = vector_k

    states, prices, env_cfg = setup_jax_environment(config, seed=_SEED)
    model, trainer = setup_model_and_trainer(config, n_assets)

    if vector_action:
        from mcts_trader.jax_agent.sampled_search import create_sampled_vector_search
        search_obj = create_sampled_vector_search(
            model, n_assets=n_assets, n_actions_per_asset=5,
            num_samples=vector_k, num_simulations=config.sims,
            max_num_considered_actions=min(vector_k, 16), discount=config.discount,
            gumbel_scale=config.gumbel_scale, value_scale=config.value_scale,
            value_shift=config.value_shift,
        )
    else:
        search_obj = create_mctx_search(model, num_simulations=config.sims, use_gumbel=True)
    search_raw = search_obj.raw_search

    # Buffer: only pass vector_action when ON, so the OFF construction is byte-for-
    # byte the same call as pre-Stage-3 (keeps the pin valid before/after edits).
    buf_kwargs = dict(
        capacity=5000,
        observation_shape=(27, n_assets, n_assets),
        action_size=n_assets * 5,
        batch_size=config.batch_size,
        n_assets=n_assets,
    )
    if vector_action:
        buf_kwargs["vector_action"] = True
    replay_buffer = JAXReplayBuffer(**buf_kwargs)

    rng = jax.random.PRNGKey(_SEED)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.state = trainer.init_train_state(init_rng, dummy_obs)

    popart_state = trainer.popart.state if getattr(trainer, "popart", None) else None

    sp_extra = {"vector_action": True} if vector_action else {}
    losses = {}
    for epoch in range(config.epochs):
        rng, sp_rng = jax.random.split(rng)
        results, _scores = run_self_play_jax(
            states, prices, env_cfg, search_raw, model.apply,
            trainer.state.params, sp_rng,
            steps=config.steps,
            support_size=model.support_size,
            discount=config.discount,
            popart_state=popart_state,
            **sp_extra,
        )
        for traj in results:
            if "observations" not in traj and "window_starts" in traj:
                traj["observations"] = reconstruct_observations(
                    traj, n_assets=n_assets, window_size=60)
            if vector_action:
                replay_buffer.add_trajectory(
                    jnp.array(traj["observations"]), jnp.array(traj["action_vecs"]),
                    jnp.array(traj["rewards"]), jnp.array(traj["values"]),
                    jnp.array(traj["policy_marginals"]), jnp.array(traj["dones"]),
                )
            else:
                replay_buffer.add_trajectory(
                    jnp.array(traj["observations"]), jnp.array(traj["actions"]),
                    jnp.array(traj["rewards"]), jnp.array(traj["values"]),
                    jnp.array(traj["policy_probs"]), jnp.array(traj["dones"]),
                )
        rng, tr_rng = jax.random.split(rng)
        hist = trainer.train(replay_buffer, num_iterations=_NUM_TRAIN_ITERS, rng=tr_rng)
        for key in ("total_loss", "value_loss", "policy_loss", "reward_loss"):
            vals = hist.get(key, [])
            losses[f"e{epoch}_{key}"] = float(np.mean(vals)) if len(vals) else 0.0

    jax.block_until_ready(trainer.state.params)
    return trainer.state.params, int(trainer.state.step_count), losses


def hash_result(params, step_count: int, losses: dict) -> str:
    """SHA-256 over sorted param leaves (bytes) + step_count + per-epoch losses."""
    h = hashlib.sha256()
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    leaves_sorted = sorted(leaves, key=lambda kv: jax.tree_util.keystr(kv[0]))
    for path, leaf in leaves_sorted:
        arr = np.ascontiguousarray(np.asarray(leaf))
        h.update(jax.tree_util.keystr(path).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    h.update(str(int(step_count)).encode())
    for k in sorted(losses):
        h.update(k.encode())
        h.update(np.asarray(losses[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def compute_off_hash() -> str:
    params, step_count, losses = run_micro_training(vector_action=False)
    return hash_result(params, step_count, losses)


# ---------------------------------------------------------------------------
# Pytest gates
# ---------------------------------------------------------------------------
def test_canon_parity_inprocess():
    """OFF path in-process hash must equal the HEAD pin."""
    h = compute_off_hash()
    assert h == PINNED, (
        f"CANON PARITY BROKEN (in-process): got {h}, expected {PINNED}.\n"
        "A Stage-3 edit perturbed the scalar path — find the reorder / extra rng "
        "draw / param-leaf change. Do NOT re-pin to hide it."
    )


def test_canon_parity_subprocess():
    """OFF path re-run in a fresh subprocess must also equal the HEAD pin
    (cross-process determinism — validates the pin itself)."""
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--print-hash"],
        capture_output=True, text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    h = out.stdout.strip().splitlines()[-1].strip()
    assert h == PINNED, (
        f"CANON PARITY BROKEN (subprocess): got {h}, expected {PINNED}.\n{out.stderr}"
    )


if __name__ == "__main__":
    _h = compute_off_hash()
    if "--print-hash" in sys.argv:
        # Machine-readable: last line is the hash (used by the subprocess gate).
        print(_h)
    else:
        print(f"commit      : {PIN_COMMIT}")
        print(f"jax version : {jax.__version__}")
        print(f"OFF hash    : {_h}")
        print(f"PINNED      : {PINNED}")
        print("MATCH" if _h == PINNED else "MISMATCH (update PINNED if this is the HEAD pin)")
