"""TASK_838 — Canon-parity pin for the Sampled-MuZero VECTOR search (per-node resampling).

This is the LOAD-BEARING canon gate for TASK_838: with ``per_node_resampling``
OFF (the default), the vector search must remain BIT-IDENTICAL to the tree it
shipped on (Stage 2/3, root-reuse of the K sampled vectors at every interior
node). We prove it with a deterministic rollout harness: fixed seeds, a small
fresh-init model, fixed observations; run ``SampledMuZeroVectorSearch.raw_search``
N steps across three constructor/argument configs (covering both
``_build_proposal`` branches, mask/book on and off, deterministic on and off)
and SHA-256 the raw bytes of sampled_vecs + visit_probs + selected action_vecs
+ root values.

``PINNED`` was captured on the PRE-TASK_838 tree (commit + JAX version recorded
below) BEFORE any edit to ``sampled_search.py``. After the TASK_838 edits land,
the OFF path must still hash to ``PINNED`` — that is the bit-identity proof.
The per-node-resampling branches are gated behind a STATIC Python bool captured
at construction time, so XLA dead-code-eliminates them when OFF, the traced
graphs are unchanged, and the OFF path consumes NO new rng.

Two variants (mirrors test_vector_train_parity.py): the in-process run AND a
fresh-subprocess run must BOTH equal ``PINNED`` — the subprocess catches any
cross-process non-determinism that would silently invalidate the pin.

Run:  JAX_PLATFORMS=cpu pytest tools/tests/test_pernode_parity.py -q
Pin:  JAX_PLATFORMS=cpu python tools/tests/test_pernode_parity.py   # prints hash
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
# PINNED canon hash — captured on HEAD BEFORE any TASK_838 edit.
#   commit:      2396c55c4d70e862e041f88e2cf4162aab70e557
#   jax version: 0.6.2
# If this breaks after a TASK_838 edit, that is a REAL canon regression — find
# the reorder / extra rng draw / shape change; do NOT re-pin to hide it.
PINNED = "22b0c7f577ac830a0eba6326518ce74fff2c994e6518d1c30ee17bed4598f23e"
PIN_COMMIT = "2396c55c4d70e862e041f88e2cf4162aab70e557"
PIN_JAX_VERSION = "0.6.2"

# Harness constants. Determinism depends only on the fixed seeds, not the sizes.
_N_ASSETS = 4
_B = 2
_N_STEPS = 3   # sequential raw_search calls per config (distinct fold_in rngs)


def _build_model_and_params():
    """Small fresh-init vector-action model + fixed observation (test_sampled_search
    pattern: init seed 0, obs seed 1)."""
    from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
    model = MuZeroNetworks(
        state_dim=16,
        action_space_size=_N_ASSETS * 5,
        n_assets=_N_ASSETS,
        encoder_version=2,
        use_vector_action=True,
    )
    obs = jnp.asarray(
        np.random.default_rng(1).normal(size=(_B, 27, 10, _N_ASSETS)),
        dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), observation=obs,
                        mode='initialize_parameters')['params']
    return model, params, obs


def _configs():
    """Three (constructor kwargs, call kwargs) configs covering the search surface:
    A — defaults (dithered-book, no book/mask, exploration pick);
    B — hold_bias + book + action_mask + deterministic (eval-style, book branch);
    C — iid-beta proposal scheme (no dither branch)."""
    na = _N_ASSETS
    # Mask: everything legal except (asset0, SELL=0) and (asset2, LIMIT_BUY=4);
    # HOLD stays legal everywhere (carry-note 1 semantics).
    mask = np.ones((_B, na * 5), dtype=np.float32)
    mask[:, 0 * 5 + 0] = 0.0
    mask[:, 2 * 5 + 4] = 0.0
    book = jnp.ones((_B, na), dtype=jnp.int32)  # v1 book = HOLD(=1) per asset
    return [
        (dict(num_samples=8, num_simulations=16, max_num_considered_actions=8),
         dict(action_mask=None, book=None, deterministic=False)),
        (dict(num_samples=8, num_simulations=16, max_num_considered_actions=8,
              hold_bias=1.5, dither_scale=0.5, proposal_scheme='dithered-book'),
         dict(action_mask=jnp.asarray(mask), book=book, deterministic=True)),
        (dict(num_samples=8, num_simulations=16, max_num_considered_actions=8,
              proposal_scheme='iid-beta'),
         dict(action_mask=None, book=None, deterministic=False)),
    ]


def compute_off_hash() -> str:
    """Deterministic rollout hash of the vector search with default (OFF) flags.

    SHA-256 over, for each config x step: selected action_vec bytes,
    sampled_vecs bytes, visit_probs raw float32 bytes, root_value raw float32
    bytes. Raw bytes = maximally sensitive: ANY numeric perturbation breaks it.
    """
    from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch

    model, params, obs = _build_model_and_params()
    h = hashlib.sha256()
    for ci, (ctor_kw, call_kw) in enumerate(_configs()):
        search = SampledMuZeroVectorSearch(
            model_apply_fn=model.apply, n_assets=_N_ASSETS,
            n_actions_per_asset=5, **ctor_kw)
        for step in range(_N_STEPS):
            rng = jax.random.fold_in(jax.random.PRNGKey(123), 100 * ci + step)
            av, sv, vp, stats = search.raw_search(params, rng, obs, **call_kw)
            jax.block_until_ready(vp)
            h.update(f"cfg{ci}_step{step}".encode())
            h.update(np.ascontiguousarray(np.asarray(av, dtype=np.int32)).tobytes())
            h.update(np.ascontiguousarray(np.asarray(sv, dtype=np.int32)).tobytes())
            h.update(np.ascontiguousarray(np.asarray(vp, dtype=np.float32)).tobytes())
            h.update(np.ascontiguousarray(
                np.asarray(stats['root_value'], dtype=np.float32)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pytest gates
# ---------------------------------------------------------------------------
def test_pernode_off_parity_inprocess():
    """OFF path in-process hash must equal the pre-edit pin."""
    h = compute_off_hash()
    assert h == PINNED, (
        f"CANON PARITY BROKEN (in-process): got {h}, expected {PINNED}.\n"
        "A TASK_838 edit perturbed the default vector-search path — find the "
        "reorder / extra rng draw / shape change. Do NOT re-pin to hide it."
    )


def test_pernode_off_parity_subprocess():
    """OFF path re-run in a fresh subprocess must also equal the pin
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
        print("MATCH" if _h == PINNED else "MISMATCH (update PINNED if this is the pre-edit pin)")
