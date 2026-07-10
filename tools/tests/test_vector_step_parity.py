"""TASK_839 — Canon-parity pin for the VECTOR env step (`step_vector`, budget sizing).

This is the LOAD-BEARING canon gate for TASK_839: with ``vector_budget_step``
OFF (the default), ``step_vector`` must remain BIT-IDENTICAL to the tree it
shipped on (the single-snapshot simultaneous-rebalance sizing where combined
demand can overdraw cash and ``process_margin_call`` resolves it). We prove it
with a deterministic rollout harness: fixed seeds, fixed price walks, fixed
per-step action vectors, a set of configs that DELIBERATELY exercise the
overdraft/margin path the ON branch changes (many simultaneous buys, many
opening shorts, the ~uniform 50/50 buy/short regime the env-wall ledger says
bankrupts at step 0), plus a mixed market/limit/hold config and one carrying
pre-existing positions; then SHA-256 the raw bytes of cash + every position
array + every order array + reward + done + portfolio value at every step.

``PINNED`` was captured on the PRE-TASK_839 tree (commit + JAX version recorded
below) BEFORE any edit to ``jax_trading_env.py``. After the TASK_839 edits land,
the OFF path must still hash to ``PINNED`` — that is the bit-identity proof.
The budget-sizing branch is gated behind a STATIC field on EnvConfig read inside
``step_vector``; with it OFF the traced graph is unchanged and the OFF path
consumes NO new rng and creates NO new arrays.

Two variants (mirrors test_pernode_parity.py / test_vector_train_parity.py): the
in-process run AND a fresh-subprocess run must BOTH equal ``PINNED`` — the
subprocess catches any cross-process non-determinism that would silently
invalidate the pin.

Run:  JAX_PLATFORMS=cpu pytest tools/tests/test_vector_step_parity.py -q
Pin:  JAX_PLATFORMS=cpu python tools/tests/test_vector_step_parity.py   # prints hash
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

# Repo root on sys.path so `python tools/tests/...` works standalone (PEP-420 ns pkg).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# PINNED canon hash — captured on HEAD BEFORE any TASK_839 edit.
#   commit:      39ae0bf5e7c20a584e06e1e472ca3272602f637b
#   jax version: 0.6.2
# If this breaks after a TASK_839 edit, that is a REAL canon regression — find
# the reorder / extra rng draw / shape change / sizing change on the OFF path;
# do NOT re-pin to hide it.
PINNED = "48e1f467c4e7e37316e3555015a89ec8e266297c8515e55c66ea4b54a536d25b"
PIN_COMMIT = "39ae0bf5e7c20a584e06e1e472ca3272602f637b"
PIN_JAX_VERSION = "0.6.2"


def _price_walk(T, na, seed):
    rng = np.random.default_rng(seed)
    walk = 100.0 + np.cumsum(rng.normal(0, 1.0, size=(T, na)), axis=0).clip(-40, 40)
    walk = walk + (np.arange(T)[:, None] * 0.02)
    return jnp.asarray(np.maximum(walk, 1.0), dtype=jnp.float32)


def _cfg(na, fee_bps=0.0, slippage_bps=0.0, allow_short=True, tp_sl=False):
    from mcts_trader.env.jax_trading_env import EnvConfig
    # NB: deliberately does NOT set vector_budget_step — relies on the default so
    # this exact harness runs identically on the pre- and post-edit tree.
    return EnvConfig(
        n_assets=na, initial_cash=100_000.0,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        max_steps=400, window_size=10, position_size_pct=0.1,
        allow_short=allow_short, tp_sl_enabled=tp_sl,
    )


def _inject_positions(state, longs, shorts):
    na = int(state.pos_active.shape[0])
    pos_active = np.zeros(na, dtype=bool)
    pos_side = np.zeros(na, dtype=np.int32)
    pos_size = np.zeros(na, dtype=np.float32)
    pos_entry = np.zeros(na, dtype=np.float32)
    for i, (entry, size) in longs.items():
        pos_active[i], pos_side[i], pos_size[i], pos_entry[i] = True, 1, size, entry
    for i, (entry, size) in shorts.items():
        pos_active[i], pos_side[i], pos_size[i], pos_entry[i] = True, 2, size, entry
    return state.replace(
        pos_active=jnp.asarray(pos_active), pos_side=jnp.asarray(pos_side),
        pos_size=jnp.asarray(pos_size), pos_entry=jnp.asarray(pos_entry),
    )


def _configs():
    """Rollout specs, each: (name, na, cfg_kwargs, seed, per-step action vectors, injected posns).

    Chosen to hammer exactly the accounting the ON branch rewrites:
      A overdraft-buys : 16 simultaneous buys (~1.6x cash) -> overdraft + margin resolve
      B shorts-heavy   : 16 opening shorts -> margin-reserve overdraft
      C uniform-5050   : ~13 buys + ~13 shorts of 24 (the env-wall step-0 bankruptcy regime)
      D mixed-limit    : buy/sell/hold/limit mix with fees + pre-existing long & short
    """
    na_a = 16
    buy_all = np.full(na_a, 2, dtype=np.int32)
    na_b = 16
    short_all = np.zeros(na_b, dtype=np.int32)  # 0 == sell/open-short (no pre-existing longs)
    na_c = 24
    unif = np.array([2 if i % 2 == 0 else 0 for i in range(na_c)], dtype=np.int32)
    na_d = 12
    rng_d = np.random.default_rng(909)
    mixed = [np.asarray(rng_d.integers(0, 5, size=na_d), dtype=np.int32) for _ in range(6)]
    return [
        ("A_overdraft_buys", na_a, dict(fee_bps=0.0),
         31, [buy_all] * 5, ({}, {})),
        ("B_shorts_heavy", na_b, dict(fee_bps=10.0, slippage_bps=5.0),
         32, [short_all] * 5, ({}, {})),
        ("C_uniform_5050", na_c, dict(fee_bps=16.0, slippage_bps=5.0),
         33, [unif] * 6, ({}, {})),
        ("D_mixed_limit_posns", na_d, dict(fee_bps=16.0, slippage_bps=5.0, tp_sl=True),
         34, mixed, ({0: (90.0, 20.0), 3: (110.0, 15.0)}, {1: (105.0, 12.0), 5: (95.0, 8.0)})),
    ]


def compute_off_hash() -> str:
    """Deterministic rollout hash of step_vector with default (OFF) budget sizing.

    Raw bytes of cash / pos_* / order_* / reward / done / portfolio_value at
    every (config, step) -> maximally sensitive; ANY numeric perturbation breaks it.
    """
    from mcts_trader.env.jax_trading_env import (
        reset, step_vector, get_current_prices, compute_portfolio_value,
    )
    h = hashlib.sha256()
    for name, na, cfg_kw, seed, action_vecs, (longs, shorts) in _configs():
        cfg = _cfg(na, **cfg_kw)
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


def test_vector_step_off_parity_inprocess():
    """OFF path in-process hash must equal the pre-edit pin."""
    h = compute_off_hash()
    assert h == PINNED, (
        f"CANON PARITY BROKEN (in-process): got {h}, expected {PINNED}.\n"
        "A TASK_839 edit perturbed the default step_vector path. Do NOT re-pin to hide it."
    )


def test_vector_step_off_parity_subprocess():
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
        print("MATCH" if _h == PINNED else "MISMATCH (update PINNED only if this is the pre-edit pin)")
