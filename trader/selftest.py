#!/usr/bin/env python3
"""TASK_820 self-tests — ST1-ST6 ported verbatim from runs/819_single_asset/portfolio.self_tests
(construction validated before trusting any result), plus ST7 for the risk overlay. ST1-ST7 are
hermetic and forced onto the numpy backend: these are the determinism reference, and the
planted-panel magnitudes match the 819 RESULT.md values. ST8 (TASK_847) is a JAX property test
and is import-GUARDED so it can only ever SKIP, never red the hermetic numpy gate.

  ST1 annualization identity (_sharpe == mean/std*sqrt(525600/period))
  ST2 planted trend: downtrend book shorts the crash (gross+, long-only B&H dies)
  ST3 diversification: N-asset portfolio Sharpe >> mean single-asset (noise averaging)
  ST4 inverse-vol risk parity: quiet asset overweighted, equal risk contribution
  ST5 sub-cost wiggle: random walk -> gross~0, net@21 mean<0, cost-monotone
  ST6 leak-free: wrecking close>t leaves positions<=t byte-identical
  ST7 shadow-DD throttle: V-shaped dd recovers -> scale re-raises; legacy mode dead-states
  ST8 TASK_847 policy-prior candidates: argmax@slot0, temp monotonicity, explore slots, dedup
      (JAX; import-guarded — SKIPS with notice if the JAX stack is unavailable)
"""

import numpy as np

from . import config
from . import backtest as bt
from .strategy import (build_positions, trend_signal, rolling_vol, first_active,
                       present_mask, weights)
from .config import VOL_FLOOR


def self_tests():
    config.set_backend("numpy")   # determinism reference
    COSTS_BP = (0, 3, 8, 21)
    msgs = []

    # 1. annualization identity (inherits backtest._sharpe)
    rng = np.random.RandomState(0)
    x = rng.randn(500) * 0.01 + 0.001
    expect = np.mean(x) / np.std(x) * np.sqrt(525600 / 60.0)
    assert abs(bt._sharpe(x, 60) - expect) < 1e-9
    msgs.append("[ST1] annualization identity OK (_sharpe == mean/std*sqrt(525600/period_min))")

    # 2. planted DOWNTREND panel (the bear judge): TSMOM shorts -> portfolio gross +, long-only
    #    equal-weight B&H dies (-), portfolio beats B&H.
    rng = np.random.RandomState(3)
    A = 12
    drift = np.concatenate([np.full(120, 0.004), np.full(120, -0.004)])
    base = np.cumsum(np.tile(drift, 12))
    logc = (base[:, None] + np.cumsum(rng.randn(base.size, A) * 0.004, axis=0))
    pos, pres, w, warm = build_positions(logc, [20], "equal", window=20)
    res = bt.regime_series(logc, pos, pres, 0, logc.shape[0], H=5, warm=warm)
    g, tu, bh = res
    sh_g, sh_bh = bt._sharpe(g, 5), bt._sharpe(bh, 5)
    logp_down = np.cumsum(np.full(4000, -0.0006) + rng.randn(4000) * 0.004)
    logc_d = logp_down[:, None] + np.cumsum(rng.randn(4000, A) * 0.003, axis=0)
    posd, presd, _, warmd = build_positions(logc_d, [20], "equal", window=20)
    gd, tud, bhd = bt.regime_series(logc_d, posd, presd, 0, logc_d.shape[0], H=10, warm=warmd)
    shd_g, shd_bh = bt._sharpe(gd, 10), bt._sharpe(bhd, 10)
    assert shd_g > 0.5, f"downtrend portfolio gross Sharpe should be + (shorts), got {shd_g}"
    assert shd_bh < 0.0, f"downtrend long-only B&H should die (-), got {shd_bh}"
    assert shd_g > shd_bh, "portfolio TSMOM must beat long-only B&H in a downtrend (bear signature)"
    assert sh_g > 1.0 and sh_g > abs(sh_bh), f"square-wave portfolio should dominate B&H, g={sh_g} bh={sh_bh}"
    msgs.append(f"[ST2] planted trend: downtrend portfolio gross {shd_g:+.2f}>0 shorts the crash vs "
                f"long-only B&H {shd_bh:+.2f}<0; square-wave portfolio {sh_g:+.2f} dominates B&H {sh_bh:+.2f}")

    # 3. DIVERSIFICATION: common trend + INDEPENDENT idiosyncratic noise; portfolio Sharpe >>
    #    mean single-asset Sharpe.
    rng = np.random.RandomState(5)
    A = 24
    common = np.cumsum(np.tile(np.concatenate([np.full(60, 0.003), np.full(60, -0.003)]), 30))
    idio = np.cumsum(rng.randn(common.size, A) * 0.010, axis=0)
    logc = common[:, None] + idio
    sig = trend_signal(logc, [20])
    T = logc.shape[0]
    t = np.arange(40, T - 5, 5)
    fwd = logc[t + 5] - logc[t]
    g_asset = sig[t] * fwd
    single_sh = np.array([bt._sharpe(g_asset[:, i], 5) for i in range(A)])
    port = np.nanmean(g_asset, axis=1)
    port_sh = bt._sharpe(port, 5)
    mean_single = float(np.nanmean(single_sh))
    assert port_sh > 1.3 * mean_single, \
        f"diversification must lift Sharpe: portfolio {port_sh:.2f} vs mean-single {mean_single:.2f}"
    msgs.append(f"[ST3] DIVERSIFICATION: {A}-asset portfolio Sharpe {port_sh:+.2f} >> mean single-asset "
                f"{mean_single:+.2f} (x{port_sh/max(mean_single,1e-9):.2f}) — noise averaging confirmed")

    # 4. inverse-vol RISK PARITY: quiet asset overweighted, equal risk contribution.
    rng = np.random.RandomState(7)
    base = np.cumsum(np.full(2000, 0.0005))
    a_hi = base + np.cumsum(rng.randn(2000) * 0.020)
    a_lo = base + np.cumsum(rng.randn(2000) * 0.004)
    logc = np.column_stack([a_hi, a_lo])
    rv = rolling_vol(logc, 30)
    fa = first_active(logc)
    pres = present_mask(logc, rv, fa, 30, 30)
    w = weights(pres, rv, "invvol")
    t0 = 1500
    risk = w[t0] * rv[t0]
    assert w[t0, 1] > w[t0, 0], f"inverse-vol must overweight the quiet asset, got {w[t0]}"
    assert abs(risk[0] - risk[1]) / max(risk.mean(), 1e-12) < 0.05, \
        f"inverse-vol risk contributions must be ~equal, got {risk}"
    msgs.append(f"[ST4] risk parity: invvol weights {w[t0,0]:.3f}/{w[t0,1]:.3f} (hi/lo vol) -> equal risk "
                f"contrib {risk[0]:.4f}/{risk[1]:.4f}")

    # 5. sub-cost wiggle: random-walk panel -> gross ~0, turnover>0, net@21 mean<0, net<gross,
    #    cost-monotone.
    rng = np.random.RandomState(11)
    logc = np.cumsum(rng.randn(30000, 16) * 0.001, axis=0)
    pos, pres, w, warm = build_positions(logc, [20], "equal", window=20)
    g, tu, bh = bt.regime_series(logc, pos, pres, 0, logc.shape[0], H=10, warm=warm)
    sh_g = bt._sharpe(g, 10)
    sh_n = bt._sharpe(bt.net_series(g, tu, 21), 10)
    assert np.mean(tu) > 0, "random walk should incur turnover"
    assert sh_n < sh_g + 1e-9, "cost must not increase Sharpe"
    assert np.mean(bt.net_series(g, tu, 21)) < 0, "sub-cost wiggle net mean must be < 0"
    shs = [bt._sharpe(bt.net_series(g, tu, c), 10) for c in COSTS_BP]
    assert all(shs[i] >= shs[i + 1] - 1e-9 for i in range(len(shs) - 1)), f"net Sharpe must be monotone in cost {shs}"
    msgs.append(f"[ST5] sub-cost wiggle: random-walk portfolio gross {sh_g:+.2f}~0, net@21 {sh_n:+.2f}<gross, "
                f"net mean<0; cost-monotone {[round(s,2) for s in shs]} OK")

    # 6. LEAK-FREE: positions at decision t use only close<=t.
    rng = np.random.RandomState(13)
    logc = np.cumsum(rng.randn(2000, 8) * 0.01, axis=0)
    pos, pres, w, warm = build_positions(logc, [7, 14, 30], "invvol", window=30)
    t0 = 900
    logc2 = logc.copy()
    logc2[t0 + 1:] += np.cumsum(rng.randn(logc.shape[0] - t0 - 1, 8) * 0.05, axis=0)
    pos2, pres2, w2, _ = build_positions(logc2, [7, 14, 30], "invvol", window=30)
    assert np.array_equal(pos[:t0 + 1], pos2[:t0 + 1], equal_nan=True), "LEAK: future altered past positions"
    lab1 = logc[t0 + 1] - logc[t0]
    lab2 = logc2[t0 + 1] - logc2[t0]
    assert not np.allclose(lab1, lab2), "label must use the future (sanity)"
    msgs.append("[ST6] leak-free: wrecking close>t leaves positions<=t byte-identical; the forward label moves OK")

    # 7. SHADOW-DD THROTTLE (dead-state fix, coordinator directive 2026-07-05): a planted
    #    V-shaped underlying drawdown-then-recovery. shadow_scale mode must throttle DOWN
    #    (1.0 -> 0.5 -> 0.25, never flat) and RE-RAISE to 1.0 after the shadow curve
    #    recovers; the legacy realized_flat mode provably reaches the dead state (flat at
    #    dd <= -35%, realized equity freezes, NEVER re-enters). Assert the difference.
    from . import risk as riskmod
    from .config import RiskConfig
    raw_net = np.concatenate([np.full(40, -0.02), np.full(80, +0.02)])  # V: -55% dd, full recovery
    target = np.array([0.05])
    # (a) shadow_scale: drive with the shadow (underlying overlay-OFF) curve
    rc_sh = RiskConfig(enabled=True, dd_mode="shadow_scale")
    sh_eq = sh_hwm = 1.0
    prev = np.zeros(1)
    scales = []
    for r in raw_net:
        sh_eq *= float(np.exp(r))
        sh_hwm = max(sh_hwm, sh_eq)
        prev, info = riskmod.apply_overlay(target, prev, sh_eq, sh_hwm, 0, 0, rc_sh)
        scales.append(info["dd_scale"])
    assert min(scales) == rc_sh.dd_deep_scale, f"deep throttle must reach {rc_sh.dd_deep_scale}, got {min(scales)}"
    assert 0.5 in scales, "throttle must pass through the 0.5 level"
    assert 0.0 not in scales, "shadow_scale must NEVER go fully flat"
    assert scales[-1] == 1.0, f"scale must re-raise to 1.0 after recovery, got {scales[-1]}"
    assert abs(prev[0] - target[0]) < 1e-12, "positions must be fully restored after recovery"
    # (b) realized_flat (legacy): the throttled book's own equity drives it -> dead state
    rc_rf = RiskConfig(enabled=True, dd_mode="realized_flat")
    eq = hwm = 1.0
    prev2 = np.zeros(1)
    scales2 = []
    for r in raw_net:
        held_frac = prev2[0] / target[0]              # realized book participates at held fraction
        eq *= float(np.exp(r * held_frac))
        hwm = max(hwm, eq)
        prev2, info2 = riskmod.apply_overlay(target, prev2, eq, hwm, 0, 0, rc_rf)
        scales2.append(info2["dd_scale"])
    assert 0.0 in scales2, "realized_flat must go flat in the planted crash"
    assert scales2[-1] == 0.0 and prev2[0] == 0.0, \
        "realized_flat dead state: must STAY flat through the recovery (equity frozen)"
    assert scales[-1] != scales2[-1], "modes must differ after recovery (the fix)"
    msgs.append(f"[ST7] shadow-DD throttle: V-shaped dd -55%->recovery; shadow_scale walks "
                f"1.0->0.5->{rc_sh.dd_deep_scale} and RE-RAISES to {scales[-1]:.1f} (pos restored); "
                f"legacy realized_flat dead-states at scale {scales2[-1]:.1f} — difference asserted")

    # 8. TASK_847 policy-prior candidate-set properties (Sampled-MuZero root proposal).
    #    The dithered-book proposal is policy-BLIND; the policy-prior scheme lays the K
    #    candidates out as [argmax | tempered marginal samples | N dither explore slots]
    #    so the learned prior is representable in the search. Asserts, on planted logits:
    #    argmax vector present@slot0; temperature monotonicity; explore slots honored; dedup.
    #    JAX-import-GUARDED: ST1-ST7 above are hermetic numpy; if the JAX/flax stack
    #    cannot import (no GPU/CUDA, missing dep), ST8 SKIPS with notice — it must NEVER
    #    red the numpy determinism gate the factory runs (`python -m trader selftest`).
    try:
        import jax
        import jax.numpy as jnp
        from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch
    except ImportError as e:
        msgs.append(f"[ST8] TASK_847 policy-prior candidates: SKIPPED (JAX stack "
                    f"unavailable: {type(e).__name__}: {e}) — ST1-ST7 numpy gate unaffected")
        return msgs

    nA, Av, Kv, Nv = 8, 5, 16, 2                      # n_assets, actions/asset, K, explore N
    rng8 = np.random.RandomState(17)
    base = (rng8.randn(1, nA, Av) * 0.1).astype(np.float32)
    for i in range(nA):
        base[0, i, i % Av] += 6.0                     # sharp per-asset argmax at action i%Av
    policy_logits = jnp.asarray(base.reshape(1, nA * Av))
    expect_argmax = np.array([i % Av for i in range(nA)], dtype=np.int32)

    def _mk(scheme, temp=1.0, explore=Nv, K=Kv, dither=1.0):
        return SampledMuZeroVectorSearch(
            model_apply_fn=None, n_assets=nA, n_actions_per_asset=Av,
            num_samples=K, num_simulations=8, max_num_considered_actions=min(K, 8),
            dither_scale=dither,
            proposal_scheme=scheme, proposal_temp=temp, explore_slots=explore)

    # (a) policy-argmax vector GUARANTEED present, and at candidate slot 0.
    sp = _mk('policy-prior', temp=1.0)
    cA = np.asarray(sp.build_root_candidates(policy_logits, rng=jax.random.PRNGKey(3)))
    assert cA.shape == (1, Kv, nA), f"candidate set must be (1,K,n_assets), got {cA.shape}"
    assert np.array_equal(cA[0, 0], expect_argmax), "policy-argmax vector must be candidate 0"
    assert any(np.array_equal(cA[0, k], expect_argmax) for k in range(Kv)), \
        "policy-argmax vector must be present in the candidate set"

    # (b) temperature monotonicity: colder temp -> fewer distinct sampled candidates.
    def _distinct(temp):
        c = np.asarray(_mk('policy-prior', temp=temp).build_root_candidates(
            policy_logits, rng=jax.random.PRNGKey(9)))
        return len({tuple(c[0, k]) for k in range(Kv)})
    d_cold, d_hot = _distinct(0.05), _distinct(50.0)
    assert d_cold <= d_hot, f"distinct candidates must not shrink with temperature: {d_cold} > {d_hot}"
    assert d_cold <= 1 + Nv, f"near-zero temp: sampled slots collapse to argmax (distinct<=1+N), got {d_cold}"
    assert d_hot > d_cold, f"hot temp must spread the candidate set: {d_hot} !> {d_cold}"

    # (c) explore slots honored: with N=0 + cold temp EVERY candidate is the argmax (no
    #     dither injection); adding strongly-dithered explore slots injects non-argmax
    #     candidates while the cold sampled slots stay pinned to the argmax — so the only
    #     source of non-argmax candidates is the explore machinery.
    c0 = np.asarray(_mk('policy-prior', temp=0.05, explore=0).build_root_candidates(
        policy_logits, rng=jax.random.PRNGKey(9)))
    assert all(np.array_equal(c0[0, k], expect_argmax) for k in range(Kv)), \
        "explore=0 + cold temp: every candidate must be the argmax vector"
    c4 = np.asarray(_mk('policy-prior', temp=0.05, explore=4, dither=50.0).build_root_candidates(
        policy_logits, rng=jax.random.PRNGKey(9)))
    n_nonargmax = sum(not np.array_equal(c4[0, k], expect_argmax) for k in range(Kv))
    assert 0 < n_nonargmax <= 4, \
        f"explore slots (and only they) must inject non-argmax candidates, got {n_nonargmax}"

    # (d) dedup: identical candidates are COUNTED by the empirical prior (beta_hat=n_a/K),
    #     not dropped (fixed-K static shape). All-argmax -> beta_hat==1; one outlier -> 1/K.
    _, beta_all = _mk('policy-prior', explore=0)._empirical_prior(jnp.asarray(c0))
    assert np.allclose(np.asarray(beta_all), 1.0), "all-identical candidates -> beta_hat==1"
    mixed = c0.copy()
    mixed[0, 0, 0] = (expect_argmax[0] + 1) % Av          # perturb slot 0 to a distinct vector
    _, beta_mix = _mk('policy-prior', explore=0)._empirical_prior(jnp.asarray(mixed))
    beta_mix = np.asarray(beta_mix)
    assert abs(beta_mix[0, 0] - 1.0 / Kv) < 1e-6, "distinct outlier slot must get beta_hat==1/K"
    assert abs(beta_mix[0, 1] - (Kv - 1) / Kv) < 1e-6, "the duplicated block must get (K-1)/K"

    # canon parity of the scheme dispatch: non-policy-prior schemes ignore the new knobs
    # (proposal_temp / explore_slots) — same candidates regardless of their values.
    cD1 = np.asarray(_mk('dithered-book', temp=1.0, explore=2).build_root_candidates(
        policy_logits, rng=jax.random.PRNGKey(1)))
    cD2 = np.asarray(_mk('dithered-book', temp=9.9, explore=7).build_root_candidates(
        policy_logits, rng=jax.random.PRNGKey(1)))
    assert np.array_equal(cD1, cD2), "dithered-book must ignore the TASK_847 policy-prior knobs"

    msgs.append(f"[ST8] TASK_847 policy-prior candidates: argmax@slot0 present; temp-monotone "
                f"distinct {d_cold}<={d_hot}; explore slots inject {n_nonargmax} dithered cands; "
                f"dedup beta_hat exact; dithered-book ignores the new knobs")
    return msgs
