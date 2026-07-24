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
  ST9 TASK_851 true-env leaf: R=0/tail degenerate cases, R=3 matches an independently-walked
      true-env rollout, canon 'model' path is a bit-identical no-op for the new kwargs
      (JAX; same import guard as ST8)
  ST10 TASK_851 fix pass: unique-per-env prices dispatch to batch_step_vector_unique
      (correctness-judge MAJOR regression guard; JAX, same import guard as ST8)
  ST11 TASK_854 prior-weight on Gumbel candidate scoring: pw=0 => proposal/candidate draw
      invariant to policy_logits (pure-Q/E1); pw=1.0 => bit-identical to today; linear
      interpolation between; value_scale fail-loud in the inert true-env no-tail combo
      (JAX; same import guard as ST8)
  ST12 TASK_857 vhat-distilled deployable leaf: net_dependent True at pw0 (the v̂ leaf is the
      learned component) vs the vacuous true-env-no-tail pw0 (False); build_search maps
      'vhat-distilled' to the canon model leaf (bit-identical search); distill freeze partition
      trains dynamics+value only (JAX; same import guard as ST8)
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

    # 9. TASK_851 true-env leaf: formula properties on the shipped ``_true_env_leaf_values``
    #    helper (root-planning leaf reward/value for --leaf-eval true-env). A tiny synthetic
    #    env + a hand-scripted dummy model (peaked categorical logits -> an EXACT known
    #    decoded value) let the properties be checked against closed-form expectations,
    #    not just "runs without crashing". Same JAX-import guard as ST8 (never reds the
    #    numpy determinism gate).
    from mcts_trader.env.jax_trading_env import EnvConfig, reset, batch_step_vector

    nA9, T9 = 3, 40
    rng9 = np.random.RandomState(5)
    prices9_np = (100.0 * np.cumprod(
        1 + rng9.randn(T9, nA9) * 0.01, axis=0)).astype(np.float32)
    prices9 = jnp.asarray(prices9_np)
    cfg9 = EnvConfig(n_assets=nA9, initial_cash=100_000.0, fee_bps=5.0, slippage_bps=2.0,
                     window_size=5, position_size_pct=0.2, allow_short=True,
                     tp_sl_enabled=False, max_steps=100, vector_action=True)
    state9 = reset(jax.random.PRNGKey(0), prices9, cfg9)
    state9 = jax.tree_util.tree_map(lambda x: x[None], state9)      # batch B=1
    state9 = state9.replace(episode_start_idx=jnp.array([10]), current_step=jnp.array([0]))

    support_size9 = 401
    VHAT_IDX, VHAT_VAL = 300, -10.0 + 300 * (20.0 / 400.0)   # support[300] == +5.0 exactly
    state_dim9 = 8

    def _peak(B, idx=VHAT_IDX):
        row = jnp.full((support_size9,), -50.0, dtype=jnp.float32).at[idx].set(50.0)
        return jnp.tile(row[None, :], (B, 1))

    def _dummy_model(variables, *args, method=None, mode=None):
        # Mirrors the model_apply_fn call conventions _search/_true_env_leaf_values use.
        if method == 'initial_inference':
            B = args[0].shape[0]
            return {'policy_logits': jnp.zeros((B, nA9 * 5), dtype=jnp.float32),
                    'latent_state': jnp.zeros((B, state_dim9), dtype=jnp.float32),
                    'value': _peak(B)}
        if mode == 'recurrent_inference_vector':
            latent = args[0]
            return {'reward': _peak(latent.shape[0]), 'value': _peak(latent.shape[0]),
                    'latent_state': latent}
        raise ValueError(f"unexpected dummy model call: method={method} mode={mode}")

    Kv9 = 4
    latent0_9 = jnp.zeros((1, state_dim9), dtype=jnp.float32)
    book9 = jnp.ones((1, nA9), dtype=jnp.int32)
    sr9, dr9 = jax.random.split(jax.random.PRNGKey(11))
    proposal9 = SampledMuZeroVectorSearch(
        model_apply_fn=_dummy_model, n_assets=nA9, n_actions_per_asset=5,
        num_samples=Kv9, num_simulations=8, max_num_considered_actions=Kv9,
        discount=0.99)._build_proposal(
            jnp.zeros((1, nA9 * 5), dtype=jnp.float32), None, book9, dr9)
    samples9 = jax.random.categorical(sr9, proposal9, axis=-1, shape=(Kv9, 1, nA9))
    sampled_vecs9 = jnp.transpose(samples9, (1, 0, 2)).astype(jnp.int32)  # (1,K,nA)

    def _te_search(R, tail):
        return SampledMuZeroVectorSearch(
            model_apply_fn=_dummy_model, n_assets=nA9, n_actions_per_asset=5,
            num_samples=Kv9, num_simulations=8, max_num_considered_actions=Kv9,
            discount=0.99, leaf_eval='true-env', leaf_rollout_r=R, leaf_vhat_tail=tail)

    # (a) R=0, tail OFF -> V==0 exactly (Q==r0, "pure truncated return" degenerate case).
    _, v_r0_off = _te_search(0, False)._true_env_leaf_values(
        {}, state9, prices9, cfg9, latent0_9, sampled_vecs9)
    assert np.allclose(np.asarray(v_r0_off), 0.0), \
        f"R=0, tail OFF must give V==0 (pure r0), got {np.asarray(v_r0_off)}"

    # (b) R=0, tail ON -> V==v̂ exactly (the "R=0 degenerates to r0+γ·v̂" property).
    _, v_r0_on = _te_search(0, True)._true_env_leaf_values(
        {}, state9, prices9, cfg9, latent0_9, sampled_vecs9)
    assert np.allclose(np.asarray(v_r0_on), VHAT_VAL, atol=1e-3), \
        f"R=0, tail ON must give V==v̂=={VHAT_VAL}, got {np.asarray(v_r0_on)}"

    # (c) R=3, tail OFF -> V matches an INDEPENDENTLY-walked true-env HOLD rollout
    #     (r0 + the 831 build_search_step/eval_candidate formula, ported verbatim).
    R9 = 3
    leaf_r_r3, leaf_v_r3 = _te_search(R9, False)._true_env_leaf_values(
        {}, state9, prices9, cfg9, latent0_9, sampled_vecs9)
    vec0_9 = sampled_vecs9[:, 0, :]
    s1_9, _, r0_ref, _ = batch_step_vector(state9, vec0_9, prices9, cfg9)
    hold9 = jnp.ones((1, nA9), dtype=jnp.int32)
    sN9, disc_sum = s1_9, 0.0
    for t in range(R9):
        sN9, _, rt, _ = batch_step_vector(sN9, hold9, prices9, cfg9)
        disc_sum += (0.99 ** t) * float(np.asarray(rt)[0])
    assert abs(float(np.asarray(leaf_r_r3)[0, 0]) - float(np.asarray(r0_ref)[0])) < 1e-5, \
        "leaf reward must equal the true-env r0 for candidate 0"
    assert abs(float(np.asarray(leaf_v_r3)[0, 0]) - disc_sum) < 1e-4, \
        f"R=3 tail OFF: V must equal the independently-walked HOLD rollout, " \
        f"got {float(np.asarray(leaf_v_r3)[0, 0])} vs ref {disc_sum}"

    # (d) canon 'model' path is a true no-op for the new kwargs: an explicit env_state/
    #     prices/env_config of None reproduces the default call byte-for-byte.
    obs9 = jnp.zeros((1, 27, nA9, nA9), dtype=jnp.float32)
    mask9 = jnp.ones((1, nA9 * 5), dtype=jnp.float32)
    search_canon = SampledMuZeroVectorSearch(
        model_apply_fn=_dummy_model, n_assets=nA9, n_actions_per_asset=5,
        num_samples=Kv9, num_simulations=8, max_num_considered_actions=Kv9, discount=0.99)
    av_a, _, vp_a, _ = search_canon({}, jax.random.PRNGKey(2), obs9, mask9, book9, True)
    av_b, _, vp_b, _ = search_canon(
        {}, jax.random.PRNGKey(2), obs9, mask9, book9, True,
        env_state=None, prices=None, env_config=None)
    assert np.array_equal(np.asarray(av_a), np.asarray(av_b)), \
        "default leaf_eval='model' must be a bit-identical no-op for the new TASK_851 kwargs"
    assert np.allclose(np.asarray(vp_a), np.asarray(vp_b)), \
        "default leaf_eval='model' visit_probs must be unaffected by the new TASK_851 kwargs"

    msgs.append("[ST9] TASK_851 true-env leaf: R=0 tail-OFF degenerates to V==0 (Q==r0); "
                "R=0 tail-ON degenerates to V==v̂ exactly; R=3 tail-OFF matches an "
                "independently-walked true-env HOLD rollout; canon 'model' path unaffected "
                "by the new kwargs (bit-identical no-op)")

    # 10. TASK_851 fix pass (correctness-judge MAJOR): _true_env_leaf_values must dispatch
    #     on unique-vs-shared prices the SAME way self_play.py's real env step already does
    #     (self_play.py:699-708, driven by --unique-data-per-env). Pre-fix, a (n_envs,
    #     n_steps, n_assets) unique-price batch crashed batch_step_vector with an opaque
    #     broadcast-shape error; this pins the fixed dispatch AND that the un-dispatched
    #     call still fails the same way (proving the test would have caught the regression).
    from mcts_trader.env.jax_trading_env import batch_reset_unique, batch_step_vector

    nB10 = 2                                            # n_envs, each with its OWN prices
    rng10 = np.random.RandomState(9)
    prices10_np = np.stack([
        (100.0 * np.cumprod(1 + rng10.randn(T9, nA9) * 0.01, axis=0)).astype(np.float32)
        for _ in range(nB10)
    ], axis=0)                                          # (B, T, n_assets) — unique per env
    prices10 = jnp.asarray(prices10_np)
    keys10 = jax.random.split(jax.random.PRNGKey(1), nB10)
    state10 = batch_reset_unique(keys10, prices10, cfg9)
    state10 = state10.replace(episode_start_idx=jnp.array([10, 10]),
                              current_step=jnp.array([0, 0]))

    search_te10 = _te_search(2, True)
    latent10 = jnp.zeros((nB10, state_dim9), dtype=jnp.float32)
    sr10, dr10 = jax.random.split(jax.random.PRNGKey(21))
    book10 = jnp.ones((nB10, nA9), dtype=jnp.int32)
    proposal10 = search_te10._build_proposal(
        jnp.zeros((nB10, nA9 * 5), dtype=jnp.float32), None, book10, dr10)
    samples10 = jax.random.categorical(sr10, proposal10, axis=-1, shape=(Kv9, nB10, nA9))
    sampled_vecs10 = jnp.transpose(samples10, (1, 0, 2)).astype(jnp.int32)

    # (a) the fixed dispatch runs clean on unique per-env prices, finite outputs.
    leaf_r10, leaf_v10 = search_te10._true_env_leaf_values(
        {}, state10, prices10, cfg9, latent10, sampled_vecs10)
    assert np.all(np.isfinite(np.asarray(leaf_r10))), "unique-prices leaf reward non-finite"
    assert np.all(np.isfinite(np.asarray(leaf_v10))), "unique-prices leaf value non-finite"

    # (b) matches an independently-walked reference via the SAME unique-prices step fn.
    from mcts_trader.env.jax_trading_env import batch_step_vector_unique
    vec0_10 = sampled_vecs10[:, 0, :]
    _, _, r0_ref10, _ = batch_step_vector_unique(state10, vec0_10, prices10, cfg9)
    assert np.allclose(np.asarray(leaf_r10[:, 0]), np.asarray(r0_ref10), atol=1e-5), \
        "unique-prices leaf reward must match an independent batch_step_vector_unique call"

    # (c) the un-dispatched call (what pre-fix code always did) fails on the SAME 3-D
    #     prices — proving this test exercises the exact bug the fix closes.
    try:
        batch_step_vector(state10, vec0_10, prices10, cfg9)
        raise AssertionError(
            "expected batch_step_vector to fail on 3-D unique prices (regression guard); "
            "if this now succeeds, the env API changed underneath ST10 — re-check")
    except AssertionError:
        raise
    except Exception:
        pass   # expected: broadcast-shape crash on 3-D prices fed to the SHARED-price step

    msgs.append("[ST10] TASK_851 fix pass: _true_env_leaf_values dispatches unique-per-env "
                "(B,T,A) prices to batch_step_vector_unique (finite outputs, matches an "
                "independent reference call); confirmed the un-dispatched shared-price step "
                "still fails on the same input (regression guard for the correctness-judge "
                "MAJOR finding)")

    # 11. TASK_854 (D21) prior-weight on the Gumbel candidate-scoring / proposal path.
    #     The learned policy prior enters the vector search ONLY through candidate
    #     generation (the mctx RootFnOutput prior is the uniform empirical 1/K prior —
    #     _empirical_prior — at every depth), so --prior-weight scales the policy_logits
    #     term of the dithered-book proposal: score = prior_weight·policy_logits +
    #     hold_bias·book + dither·g. Asserts the three pre-registered properties (task
    #     scope 3): prior_weight 0 => scoring invariant to logits; 1.0 => byte-identical
    #     to today; monotone (linear) interpolation between. Same JAX guard as ST8-ST10.
    nA11, Av11, K11 = 6, 5, 12
    rng11 = np.random.RandomState(23)
    logits_a = jnp.asarray((rng11.randn(1, nA11 * Av11) * 2.0).astype(np.float32))
    logits_b = jnp.asarray((rng11.randn(1, nA11 * Av11) * 2.0).astype(np.float32))  # different prior
    book11 = jnp.ones((1, nA11), dtype=jnp.int32)
    dr11 = jax.random.PRNGKey(31)

    def _mk_pw(pw):
        return SampledMuZeroVectorSearch(
            model_apply_fn=None, n_assets=nA11, n_actions_per_asset=Av11,
            num_samples=K11, num_simulations=8, max_num_considered_actions=min(K11, 8),
            hold_bias=3.0, dither_scale=1.0, proposal_scheme='dithered-book',
            prior_weight=pw)

    # Reference (no mask so the score is the clean linear form): proposals share the SAME
    # dither rng, so everything except prior_weight·policy_logits is constant across weights.
    prop_a = {pw: np.asarray(_mk_pw(pw)._build_proposal(logits_a, None, book11, dr11))
              for pw in (0.0, 0.25, 0.5, 1.0)}
    prop_b0 = np.asarray(_mk_pw(0.0)._build_proposal(logits_b, None, book11, dr11))

    # (a) prior_weight=0 => candidate scoring INVARIANT to policy_logits (two different
    #     priors give an identical proposal — hence identical candidate distribution).
    assert np.allclose(prop_a[0.0], prop_b0, atol=1e-6), \
        "prior_weight=0 must make the proposal invariant to policy_logits (pure-Q selection)"

    # (b) prior_weight=1.0 => byte-identical to today: the proposal minus the policy-free
    #     baseline recovers EXACTLY the reshaped policy logits (the canon pre-854 term).
    logits_a_grid = np.asarray(logits_a).reshape(1, nA11, Av11)
    assert np.allclose(prop_a[1.0] - prop_a[0.0], logits_a_grid, atol=1e-5), \
        "prior_weight=1.0 must reproduce today's full policy_logits term (bit-identical)"

    # (c) monotone (linear) interpolation: proposal(pw) - proposal(0) == pw·(proposal(1) - proposal(0)).
    delta = prop_a[1.0] - prop_a[0.0]
    for pw in (0.25, 0.5):
        assert np.allclose(prop_a[pw] - prop_a[0.0], pw * delta, atol=1e-5), \
            f"prior_weight interpolation must be linear/monotone at pw={pw}"

    # (d) sampled candidate SET is invariant to logits at prior_weight=0 (the scoring
    #     property carried through the actual Gumbel-max candidate draw, same rng).
    sr11 = jax.random.PRNGKey(41)
    def _draw(prop):
        s = jax.random.categorical(sr11, jnp.asarray(prop), axis=-1, shape=(K11, 1, nA11))
        return np.asarray(jnp.transpose(s, (1, 0, 2)))
    assert np.array_equal(_draw(prop_a[0.0]), _draw(prop_b0)), \
        "prior_weight=0 candidate draw must not depend on policy_logits"

    # (e) TASK_854 scope 2: value_scale is fail-loud in the provably-inert true-env no-tail
    #     combo, and truthfully accepted everywhere it bites (model leaf; true-env + tail).
    def _ctor(**kw):
        return SampledMuZeroVectorSearch(
            model_apply_fn=None, n_assets=nA11, n_actions_per_asset=Av11,
            num_samples=K11, num_simulations=8, max_num_considered_actions=min(K11, 8), **kw)
    try:
        _ctor(leaf_eval='true-env', leaf_vhat_tail=False, value_scale=2.0)
        raise AssertionError("value_scale!=1 with true-env no-tail must fail-loud (TASK_854)")
    except ValueError:
        pass
    _ctor(leaf_eval='true-env', leaf_vhat_tail=True, value_scale=2.0)   # tail routes it — OK
    _ctor(leaf_eval='model', value_scale=2.0)                          # model leaf uses it — OK
    _ctor(leaf_eval='true-env', leaf_vhat_tail=False, value_scale=1.0)  # default — OK

    msgs.append("[ST11] TASK_854 prior-weight: pw=0 proposal+candidate-draw invariant to "
                "policy_logits (pure-Q); pw=1.0 recovers today's policy term exactly "
                "(bit-identical); linear/monotone interpolation at pw={0.25,0.5}; "
                "value_scale fail-loud in the inert true-env no-tail combo, accepted where "
                "it bites")

    # 12. TASK_857 vhat-distilled deployable leaf + distill freeze partition. The deployable
    #     v̂-leaf read is eval_vector.py --leaf-eval vhat-distilled: MECHANICALLY the canon
    #     model leaf (value head + dynamics, no lookahead), so it must (a) grade as
    #     net_dependent even at prior_weight=0 (the v̂ leaf IS the learned component) while the
    #     truly-vacuous true-env-no-tail pw0 config stays False, and (b) build a search that is
    #     BIT-IDENTICAL to --leaf-eval model on the same checkpoint/rng. (c) the offline distill
    #     freezes everything except the dynamics + value-head leaves. Reuses ST9's tiny dummy
    #     model / obs / mask; same JAX guard (ST8) so it can only ever SKIP, never red numpy.
    import os as _os
    import argparse as _argparse
    import importlib.util as _ilu
    _repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    def _load_by_path(name, relpath):
        spec = _ilu.spec_from_file_location(name, _os.path.join(_repo, relpath))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    ev = _load_by_path("eval_vector857", "experiments/835_stage4_eval/eval_vector.py")
    di = _load_by_path("distill857", "experiments/857_vhat_distill/distill.py")

    # (a) net-dependence: vhat-distilled True even at pw0 / non-policy proposal; the vacuous
    #     true-env-no-tail pw0 config is the ONE False (the 856 grading hole) — regression guard
    #     that the new mode is genuinely distinguished from it.
    def _nda(**kw):
        d = dict(operator="search", prior_weight=0.0, vector_proposal_scheme="dithered-book",
                 leaf_eval="model", leaf_vhat_tail=True)
        d.update(kw)
        return ev.operator_net_dependent(_argparse.Namespace(**d))
    assert _nda(leaf_eval="vhat-distilled", leaf_vhat_tail=False) is True, \
        "vhat-distilled must grade net_dependent=True even at prior_weight=0 (the v̂ leaf is learned)"
    assert _nda(leaf_eval="true-env", leaf_vhat_tail=False) is False, \
        "true-env no-tail pw0 dithered-book must stay net_dependent=False (856 vacuous guard)"
    assert _nda(leaf_eval="model") is True and \
        _nda(leaf_eval="true-env", leaf_vhat_tail=True) is True, \
        "model leaf and true-env WITH v̂ tail must both grade net_dependent=True"

    # (b) build_search('vhat-distilled') is bit-identical to build_search('model'): it maps to
    #     the model leaf in the sampled_search constructor, and the search selects the same
    #     vector + visit probs on the fixed ST9 dummy model / rng.
    class _Shim:                                   # create_sampled_vector_search reads model.apply
        apply = staticmethod(_dummy_model)

    def _bargs(leaf_eval):
        return _argparse.Namespace(
            vector_k=Kv9, sims=8, discount=0.99, gumbel_scale=3.0, value_scale=1.0,
            value_shift=0.0, vector_hold_bias=3.0, vector_dither_scale=1.0,
            vector_proposal_scheme="dithered-book", vector_proposal_temp=1.0,
            vector_proposal_explore_slots=2, prior_weight=1.0,
            vector_per_node_resample=False, leaf_eval=leaf_eval, leaf_rollout_r=3,
            leaf_vhat_tail=True)
    s_model = ev.build_search(_Shim(), _bargs("model"), nA9)
    s_vhat = ev.build_search(_Shim(), _bargs("vhat-distilled"), nA9)
    assert s_vhat.leaf_eval == "model", \
        "build_search must map leaf_eval='vhat-distilled' to the canon 'model' leaf"
    avm, _, vpm, _ = s_model({}, jax.random.PRNGKey(7), obs9, mask9, book9, True)
    avv, _, vpv, _ = s_vhat({}, jax.random.PRNGKey(7), obs9, mask9, book9, True)
    assert np.array_equal(np.asarray(avm), np.asarray(avv)), \
        "vhat-distilled search must select the SAME vector as the model-leaf search (bit-identical)"
    assert np.allclose(np.asarray(vpm), np.asarray(vpv)), \
        "vhat-distilled search visit_probs must match the model-leaf search (bit-identical)"

    # (c) distill freeze partition: dynamics + value-head trainable; encoder / reasoning refiners
    #     / policy output / reward (scalar action_embedding) frozen.
    assert di._is_trainable(("dynamics_dense1", "kernel"))
    assert di._is_trainable(("vector_action_embedding", "vec_act_proj", "kernel"))
    assert di._is_trainable(("trm", "reasoning", "value_head_2", "kernel"))
    assert not di._is_trainable(("trm", "encoder", "global_state_projection", "kernel"))
    assert not di._is_trainable(("trm", "reasoning", "output_refiner_h0_2", "kernel"))
    assert not di._is_trainable(("trm", "reasoning", "latent_refiner_h0_l0_1", "kernel"))
    assert not di._is_trainable(("action_embedding", "embedding"))

    msgs.append("[ST12] TASK_857 vhat-distilled: net_dependent True at pw0 (v̂ leaf is learned) "
                "while the vacuous true-env-no-tail pw0 stays False; build_search maps it to the "
                "canon model leaf (bit-identical selected vector + visit probs vs --leaf-eval "
                "model); distill freeze partition trains dynamics+value only")
    return msgs
