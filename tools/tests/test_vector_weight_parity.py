"""TASK_840 — 839-BUDGET-PATH parity pin for the VECTOR env step.

Companion to ``test_vector_step_parity.py``. TASK_840 adds a SECOND flag,
``vector_weight_rebalance`` (default OFF). Two OFF-paths must stay bit-identical
after the 840 edits land:

  * canon  (``vector_budget_step``=OFF, ``vector_weight_rebalance``=OFF) — pinned
    by ``test_vector_step_parity.py`` at ``48e1f467…`` (still MATCHes).
  * 839    (``vector_budget_step``=ON,  ``vector_weight_rebalance``=OFF) — pinned
    HERE. Adding the weight flag must not perturb the shipped 839 budget sizing.

``PINNED`` was captured on the PRE-TASK_840 tree (commit + JAX version below)
BEFORE any edit to ``jax_trading_env.py``. It uses the EXACT rollout specs of the
canon harness (imported, not copied — zero drift): same seeds, price walks,
action vectors, injected positions; only the config differs by
``vector_budget_step=True``. After the 840 edits, the budget-ON / weight-OFF path
must still hash to ``PINNED``. Do NOT re-pin to hide a break.

Two variants (in-process + fresh subprocess) — the subprocess catches any
cross-process non-determinism that would silently invalidate the pin.

Run:  JAX_PLATFORMS=cpu pytest tools/tests/test_vector_weight_parity.py -q
Pin:  JAX_PLATFORMS=cpu python tools/tests/test_vector_weight_parity.py   # prints hash
"""
import os
# CPU / fixture-only; never GPU in the loop, never trading_data.db.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Repo root + this dir on sys.path so the standalone run and the sibling-module
# import both work (PEP-420 ns pkg; no __init__.py in tools/tests).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the CANON harness's rollout specs VERBATIM — importing (not copying)
# guarantees the two pins exercise the identical states, so any divergence is
# purely the budget-sizing branch, never a harness drift.
from test_vector_step_parity import _price_walk, _configs, _inject_positions

# ---------------------------------------------------------------------------
# PINNED 839-budget-path hash — captured on HEAD BEFORE any TASK_840 edit.
#   commit:      55a1d6a4588943ee620e5c3c5692c5ca4cf92ab4
#   jax version: 0.6.2
# If this breaks after a TASK_840 edit, adding the weight flag perturbed the
# shipped 839 budget path — find the reorder / extra rng / shape / sizing change;
# do NOT re-pin to hide it.
PINNED = "4f8b692ead92248f16850f4874d9b9fca7d85fb9447fe26721442d9f164c19d3"
PIN_COMMIT = "55a1d6a4588943ee620e5c3c5692c5ca4cf92ab4"
PIN_JAX_VERSION = "0.6.2"


def _cfg_budget(na, fee_bps=0.0, slippage_bps=0.0, allow_short=True, tp_sl=False):
    """Same as the canon harness's _cfg, but with the 839 budget flag ON.

    Deliberately does NOT set vector_weight_rebalance — relies on its default so
    this exact harness runs identically on the pre- and post-TASK_840 tree.
    """
    from mcts_trader.env.jax_trading_env import EnvConfig
    return EnvConfig(
        n_assets=na, initial_cash=100_000.0,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        max_steps=400, window_size=10, position_size_pct=0.1,
        allow_short=allow_short, tp_sl_enabled=tp_sl,
        vector_budget_step=True,
    )


def compute_budget_on_hash() -> str:
    """Deterministic rollout hash of step_vector with vector_budget_step=ON.

    Identical structure to test_vector_step_parity.compute_off_hash, only the
    config's budget flag differs -> isolates the 839 budget sizing path.
    """
    from mcts_trader.env.jax_trading_env import (
        reset, step_vector, get_current_prices, compute_portfolio_value,
    )
    h = hashlib.sha256()
    for name, na, cfg_kw, seed, action_vecs, (longs, shorts) in _configs():
        cfg = _cfg_budget(na, **cfg_kw)
        prices = _price_walk(T=200, na=na, seed=seed)
        state = reset(jax.random.PRNGKey(seed), prices, cfg)
        if longs or shorts:
            state = _inject_positions(state, longs, shorts)
        for si, av in enumerate(action_vecs):
            av_j = jnp.asarray(av, dtype=jnp.int32)
            cur = get_current_prices(prices, state.current_step, state.episode_start_idx)
            pv_before = compute_portfolio_value(state, cur)
            state, _obs, reward, done = step_vector(state, av_j, prices, cfg)
            jax.block_until_ready(state.cash)
            h.update(f"{name}_step{si}".encode())
            for arr, dt in [
                (state.cash, np.float32), (state.pos_active, np.int32),
                (state.pos_side, np.int32), (state.pos_size, np.float32),
                (state.pos_entry, np.float32), (state.order_active, np.int32),
                (state.order_side, np.int32), (state.order_size, np.float32),
                (state.order_price, np.float32), (reward, np.float32),
                (done, np.int32), (pv_before, np.float32),
            ]:
                h.update(np.ascontiguousarray(np.asarray(arr, dtype=dt)).tobytes())
    return h.hexdigest()


def test_vector_budget_path_parity_inprocess():
    """budget-ON / weight-OFF in-process hash must equal the pre-edit pin."""
    h = compute_budget_on_hash()
    assert h == PINNED, (
        f"839-BUDGET PARITY BROKEN (in-process): got {h}, expected {PINNED}.\n"
        "A TASK_840 edit perturbed the shipped 839 budget path. Do NOT re-pin to hide it."
    )


def test_vector_budget_path_parity_subprocess():
    """budget-ON path re-run in a fresh subprocess must also equal the pin."""
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--print-hash"],
        capture_output=True, text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    h = out.stdout.strip().splitlines()[-1].strip()
    assert h == PINNED, (
        f"839-BUDGET PARITY BROKEN (subprocess): got {h}, expected {PINNED}.\n{out.stderr}"
    )


if __name__ == "__main__":
    _h = compute_budget_on_hash()
    if "--print-hash" in sys.argv:
        print(_h)
    else:
        print(f"commit      : {PIN_COMMIT}")
        print(f"jax version : {jax.__version__}")
        print(f"budget hash : {_h}")
        print(f"PINNED      : {PINNED}")
        print("MATCH" if _h == PINNED else "MISMATCH (update PINNED only if this is the pre-edit pin)")
