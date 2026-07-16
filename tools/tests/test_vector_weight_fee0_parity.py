"""TASK_842 — WEIGHT-ON-at-fee0 parity pin for the VECTOR env step.

Third companion pin (with test_vector_step_parity.py = canon, and
test_vector_weight_parity.py = 839-budget path). TASK_842 adds the
canon-equivalent explicit close-fee (``closed_notional * fee_bps/1e4``) to the
TASK_840 weight-rebalance re-trim. That term is EXACTLY 0 when ``fee_bps == 0``,
so the weight-ON path at fee0 must stay BIT-IDENTICAL after the 842 edit — this
is the "provable no-op for the RUNNING phase-1c (cost0)" guarantee in DECISIONS
D9. This pin locks it.

``PINNED`` was captured on the PRE-TASK_842 tree (commit + JAX version below)
BEFORE any edit. It reuses the canon harness's rollout specs VERBATIM (imported,
not copied — zero drift), only the config differs: ``vector_weight_rebalance=True``
and ``fee_bps`` FORCED to 0 (slippage kept — the new close-fee term is
independent of slippage, so this still exercises a rich weight-ON path). After
the 842 edits, this hash must be unchanged. Do NOT re-pin to hide a break.

Run:  JAX_PLATFORMS=cpu pytest tools/tests/test_vector_weight_fee0_parity.py -q
Pin:  JAX_PLATFORMS=cpu python tools/tests/test_vector_weight_fee0_parity.py   # prints hash
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the canon harness's rollout specs VERBATIM (same seeds/walks/actions/injected posns).
from test_vector_step_parity import _price_walk, _configs, _inject_positions

# ---------------------------------------------------------------------------
# PINNED weight-ON/fee0 hash.
#   commit:      8f76a71886af61a39e8ff0f9fd1b90ea3df3a119  (TASK_843 rebase)
#   jax version: 0.6.2
# TASK_843 (D12) RE-BASED this pin ON PURPOSE: incremental netting + snap-to-flat
# change the weight-ON re-trim ARITHMETIC (delta-only trades vs the old full
# close+reopen; a full residual close on snap), which alters the fee0 ON-path
# state even though every EXPLICIT fee term is still ×0 at fee0. The prior pin
#   1c50d58c570d10cfe237e4b7c630cd01058a7b5a68766eb0001a14e301474c20  (TASK_842)
# was captured on the clean tree pre-edit; the value below is the honest hash of
# the NEW intended behavior on the SAME deterministic rollout. The canon-OFF pin
# (test_vector_step_parity.py) and the 839-budget pin (test_vector_weight_parity.py)
# are UNCHANGED and still pass — only this ON-path pin moves. Do NOT re-pin to hide
# an unintended break.
PINNED = "6293668f034ef42f7fdd71005d625c4958e189705256463aae6eb49d4c3d90bf"
PIN_COMMIT = "8f76a71886af61a39e8ff0f9fd1b90ea3df3a119"
PIN_JAX_VERSION = "0.6.2"


def _cfg_weight_fee0(na, fee_bps=0.0, slippage_bps=0.0, allow_short=True, tp_sl=False):
    """Canon harness's _cfg, but weight-rebalance ON and fee_bps FORCED to 0.

    fee_bps is deliberately dropped from the incoming kwargs (forced 0) so the
    842 close-fee term is ×0; slippage_bps is kept to exercise the exec-price path.
    Does NOT set vector_weight_unit/max/deadband — relies on sentinels/defaults.
    """
    from mcts_trader.env.jax_trading_env import EnvConfig
    return EnvConfig(
        n_assets=na, initial_cash=100_000.0,
        fee_bps=0.0, slippage_bps=slippage_bps,
        max_steps=400, window_size=10, position_size_pct=0.1,
        allow_short=allow_short, tp_sl_enabled=tp_sl,
        vector_weight_rebalance=True,
    )


def compute_weight_fee0_hash() -> str:
    """Deterministic rollout hash of step_vector with weight-rebalance ON, fee0."""
    from mcts_trader.env.jax_trading_env import (
        reset, step_vector, get_current_prices, compute_portfolio_value,
    )
    h = hashlib.sha256()
    for name, na, cfg_kw, seed, action_vecs, (longs, shorts) in _configs():
        cfg = _cfg_weight_fee0(na, **cfg_kw)
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


def test_weight_fee0_parity_inprocess():
    """weight-ON / fee0 in-process hash must equal the pre-edit pin."""
    h = compute_weight_fee0_hash()
    assert h == PINNED, (
        f"WEIGHT-ON-FEE0 PARITY BROKEN (in-process): got {h}, expected {PINNED}.\n"
        "A TASK_842 edit perturbed the weight path at fee0 (would corrupt the phase-1c cost0 "
        "battery). The close-fee term must be exactly 0 when fee_bps==0. Do NOT re-pin to hide it."
    )


def test_weight_fee0_parity_subprocess():
    """weight-ON / fee0 re-run in a fresh subprocess must also equal the pin."""
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--print-hash"],
        capture_output=True, text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    h = out.stdout.strip().splitlines()[-1].strip()
    assert h == PINNED, (
        f"WEIGHT-ON-FEE0 PARITY BROKEN (subprocess): got {h}, expected {PINNED}.\n{out.stderr}"
    )


if __name__ == "__main__":
    _h = compute_weight_fee0_hash()
    if "--print-hash" in sys.argv:
        print(_h)
    else:
        print(f"commit      : {PIN_COMMIT}")
        print(f"jax version : {jax.__version__}")
        print(f"wt-fee0 hash: {_h}")
        print(f"PINNED      : {PINNED}")
        print("MATCH" if _h == PINNED else "MISMATCH (update PINNED only if this is the pre-edit pin)")
