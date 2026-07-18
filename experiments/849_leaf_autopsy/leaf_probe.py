#!/usr/bin/env python3
"""TASK_849 / Read B — leaf-eval autopsy: WHY does the vector search's selection layer
reject the in-set policy-argmax prior?

Context (LEDGER 832/E5-autopsy → 832/E5-inversion): under proposal_scheme='policy-prior'
the policy-argmax vector is GUARANTEED in every root candidate set (TASK_847, 40/40) — yet
deterministic search-eval of the frozen imprint still scores −7.84 net@21 while the SAME
policy-argmax replayed directly scores +0.839. The selection layer ranks churny candidates
above the prior. This probe is a READ-ONLY diagnostic instrument that measures WHERE and BY
HOW MUCH: it dumps, per decision state, every candidate's prior mass / reward-model r̂ /
value-net leaf v̂ / Q = r̂+γv̂ / Gumbel selection score / visit weight / selected? PLUS the
TRUE one-step env reward per candidate (env step on a copy — the calibration column). The
aggregate report says which signal misranks the prior (→ value-target repair vs reward-model
repair vs selection-mechanics), the measurement Amendment 6 gates the next training spend on.

ZERO canon / training-path code diffs. Nothing here edits the env, model, search, or the 835
harness — every piece of the model/env/search/panel machinery is IMPORTED from
``experiments/835_stage4_eval/eval_vector.py`` (which TASK_850 owns edits to this run) and
from the 832 imprint runner, exactly as 835/848 reuse them. This module only OBSERVES:

  * TRAJECTORY (which states we visit): the 848 policy-argmax operator — replay the
    per-asset argmax vectors (imprint.policy_argmax_actions) through the REAL vector env
    (jit_batch_step_vector), the deterministic D17 operator. The env is stepped ONLY by the
    trajectory; the search is never allowed to step it.
  * OBSERVER (at every Nth state, default N=25): rebuild the leak-free obs/mask/book exactly
    as 835's rollout does, run the SHIPPED SampledMuZeroVectorSearch (deterministic=True) to
    get the real K candidate set + visit weights + selected vector, then score each candidate
    through the model's own ``recurrent_inference_vector`` (the exact depth-1 leaf eval the
    search's recurrent_fn runs) for r̂/v̂/Q, and step a COPY of the env under each candidate
    for the TRUE one-step reward. Both proposal schemes are supported (--scheme policy-prior
    | dithered-book); policy-prior guarantees the prior is in the set.

Fixture smoke: ``python leaf_probe.py --smoke`` — fresh-init UNTRAINED model on the fixture,
REAL save_checkpoint→check_checkpoint_arch→load_checkpoint roundtrip, full autopsy schema on
the fixture minute-slice. PASS = clean run + schema emitted + all dumped Q/v̂/r̂ finite +
the policy-argmax candidate present & flagged in 100% of dumped states (acceptance i). Untrained
numbers are near-noise by construction — no threshold applied (plumbing proof, not a verdict).

FACTORY / SCHEDULED split (835 house style): this module + the fixture smoke are the factory
deliverable. The real autopsy (frozen runs/832_e5/ckpt_imprint_full, fee16 ExIt env, both
schemes) is a SCHEDULED run — never inside the factory loop.
"""
import os
# CPU-only, deterministic (fixture smoke / scheduled job) — set BEFORE any jax import,
# mirroring eval_vector.py's header so importing it here is a no-op override.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import sys
from datetime import date

import numpy as np
import jax
import jax.numpy as jnp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- reuse the 835 scaffolding VERBATIM (import by path; never edit it — TASK_850 owns it) --
EV_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")


def _load_eval_vector():
    spec = importlib.util.spec_from_file_location("eval_vector835", EV_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load_eval_vector()

# Constants / helpers lifted from 835 (single-sourced — no re-derivation).
FIXTURE_DB = ev.FIXTURE_DB
N_ACTIONS_PER_ASSET = ev.N_ACTIONS_PER_ASSET
HOLD = ev.HOLD
FEE_BPS, SLIP_BPS = ev.FEE_BPS, ev.SLIP_BPS
SMOKE_SYMBOLS = ev.SMOKE_SYMBOLS
SMOKE_START, SMOKE_END = ev.SMOKE_START, ev.SMOKE_END
SMOKE_BAR_MINUTES = ev.SMOKE_BAR_MINUTES
SMOKE_MAX_DECISIONS = ev.SMOKE_MAX_DECISIONS
SMOKE_STRIDE = 8                       # smoke: tighter than the N=25 default so ~5 states dump

VALID_SCHEMES = ("policy-prior", "dithered-book")   # the two schemes this instrument scopes

# mctx gumbel_muzero_policy sigma-transform defaults (used only to RECONSTRUCT the
# deterministic part of the Sequential-Halving selection score for the trace column;
# eval selection itself is visit-argmax, see the schema note).
MCTX_MAXVISIT_INIT = 50.0
MCTX_SIGMA_VALUE_SCALE = 0.1


# =============================================================================
# small numeric helpers (JSON-safe; correlation over the K candidates)
# =============================================================================

def _f(x):
    return ev._f(x)


def _avg_rank(x):
    """Average (tie-aware) ranks of x, ascending (rank 1 = smallest). For Spearman."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0            # 1-based average rank for the tie block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else None


def _spearman(x, y):
    return _pearson(_avg_rank(x), _avg_rank(y))


def _rank_of(value, arr):
    """1-based rank of ``value`` among ``arr`` under DESCENDING order (rank 1 = best/largest).
    rank = 1 + (#entries strictly greater than value). If ``value`` equals one of the arr
    entries (the in-set case) that entry does not count as strictly greater, so the rank
    reflects its own position. Returned rank lives in [1, len(arr)+1]."""
    arr = np.asarray(arr, dtype=np.float64)
    return int(1 + np.sum(arr > value))


# =============================================================================
# the autopsy walk
# =============================================================================

def _slot_role(scheme, k, K, N):
    """Human-readable role of candidate slot k under the proposal scheme (TASK_847 layout)."""
    if scheme == "policy-prior":
        if k == 0:
            return "argmax"
        if k >= K - N:
            return "explore"
        return "sample"
    return "dither"                       # dithered-book: every slot is a dither sample


def autopsy(label, model, params, search_obj, panel, args):
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = panel
    warm = int(env_cfg0.window_size)
    n_assets = int(env_cfg0.n_assets)
    n_decisions = panel_info["n_decisions"]
    discount = float(args.discount)
    A = N_ACTIONS_PER_ASSET
    K = int(args.vector_k)
    N = int(args.vector_proposal_explore_slots)
    ts_dec = row_ts[warm:warm + n_decisions]

    # autopsy env = the panel env at the requested cost tier (default fee16+slip5 = the ExIt set)
    env_cfg = env_cfg0.replace(fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps))

    # empty-panel fail-loud (acceptance ii): build_panel already refuses n_decisions<2, but
    # guard here too so the instrument never emits a zero-state report.
    if n_decisions < 2:
        raise RuntimeError(f"empty/degenerate panel: n_decisions={n_decisions} (<2); widen the "
                           f"--start/--end window so at least one decision state exists.")

    # TRAJECTORY: the 848 policy-argmax operator (imprint.policy_argmax_actions on training obs).
    imp = ev.load_imprint_module()
    obs_all = imp.encode_training_obs(prices, warm, n_decisions, env_cfg0)     # (N,27,A_n,A_n)
    policy_actions = imp.policy_argmax_actions(model, params, obs_all, n_assets)  # (N,A_n) {0..4}

    enc_cfg = ev.EncoderConfig(n_assets=n_assets, window_size=warm, n_channels=27)  # 835 rollout
    book = jnp.ones((1, n_assets), dtype=jnp.int32)                # v1 book = HOLD (DEC-3)

    infer = jax.jit(lambda o: model.apply({"params": params}, o, method="initial_inference"))

    @jax.jit
    def score_candidates(latent, vecs):
        """Depth-1 leaf eval for each candidate — the EXACT computation the search's
        recurrent_fn runs (recurrent_inference_vector → reward/value categorical → scalar,
        via the search object's own decoders). latent:(1,D), vecs:(M,A_n) → r̂:(M,), v̂:(M,)."""
        M = vecs.shape[0]
        lat = jnp.repeat(latent, M, axis=0)
        out = model.apply({"params": params}, lat, vecs, mode="recurrent_inference_vector")
        r = search_obj._reward_to_scalar(out["reward"])
        v = search_obj._categorical_to_scalar(out["value"])
        return r, v

    key0 = jax.random.PRNGKey(args.eval_seed)
    state = state_init
    states_dump = []
    sel_match = 0
    pa_in_set = 0
    all_finite = True
    pa_present = 0

    for k in range(n_decisions):
        if k % args.stride == 0:
            # --- leak-free obs / mask / book, mirror of eval_vector.rollout step A ---
            price_idx = state.episode_start_idx + state.current_step        # (1,)
            window_starts = price_idx - warm
            obs = ev.jit_batch_encode_simple(prices, window_starts, enc_cfg)
            current_prices = prices[price_idx]                              # (1,A_n)
            mask = ev.jit_batch_compute_action_mask(
                state, current_prices, env_cfg, args.max_positions, args.min_cash_threshold)

            # --- the REAL shipped search (deterministic visit-argmax selection) ---
            step_rng = jax.random.fold_in(key0, k)
            action_vec, sampled_vecs, visit_probs, stats = search_obj(
                params, step_rng, obs, mask, book, True)                    # deterministic=True

            # --- root inference for latent/policy (same params/obs the search used) ---
            root_out = infer(obs)
            latent = root_out["latent_state"]
            policy_logits = root_out["policy_logits"]

            vecs = sampled_vecs[0]                                          # (K,A_n)
            r_hat, v_hat = score_candidates(latent, vecs)
            Q = r_hat + discount * v_hat
            r_hat = np.asarray(r_hat); v_hat = np.asarray(v_hat); Q = np.asarray(Q)
            beta_hat = np.asarray(stats["beta_hat"][0])                     # per-candidate prior mass
            aw = np.asarray(visit_probs[0])                                 # mctx action_weights
            visit_ct = aw * float(search_obj.num_simulations)              # approx visit counts
            vecs_np = np.asarray(vecs)

            # --- the policy-argmax (in-set) prior vector: masked per-asset argmax = search slot 0 ---
            mask_bool = mask.reshape(1, n_assets, A).astype(bool)
            pa_logits = jnp.where(mask_bool, policy_logits.reshape(1, n_assets, A), -1e9)
            pa_vec = jnp.argmax(pa_logits, axis=-1).astype(jnp.int32)       # (1,A_n)
            pa_np = np.asarray(pa_vec[0])
            pa_r, pa_v = score_candidates(latent, pa_vec[0][None, :])
            pa_r = float(np.asarray(pa_r)[0]); pa_v = float(np.asarray(pa_v)[0])
            pa_Q = pa_r + discount * pa_v

            is_pa = np.all(vecs_np == pa_np[None, :], axis=1)              # per-candidate flag
            pa_idx = int(np.argmax(is_pa)) if is_pa.any() else None
            in_set = bool(is_pa.any())
            pa_in_set += int(in_set)
            pa_present += 1                                                 # pa always dumped

            # When the prior IS a candidate (policy-prior: slot pa_idx), report its OWN in-set
            # row from the M=16 batched call for the r̂/v̂/Q columns and the rank. The separately
            # recomputed pa_r/pa_v/pa_Q above is an M=1 unbatched inference and is NOT bit-identical
            # to the batched row (~1e-7, one-directional): using it would rank the prior below a
            # bit-identical copy of itself in the near-tie regime this autopsy exists to resolve
            # (MAJOR fix). The recomputed M=1 score is the honest best only when the prior is
            # ABSENT (dithered-book) — see schema_notes.
            if in_set:
                pa_r = float(r_hat[pa_idx]); pa_v = float(v_hat[pa_idx]); pa_Q = float(Q[pa_idx])

            # --- selection cross-check (acceptance iii): our visit-argmax == operator's vector ---
            sel_idx = int(np.argmax(aw))
            selection_matches = bool(np.all(np.asarray(action_vec[0]) == vecs_np[sel_idx]))
            sel_match += int(selection_matches)

            # --- TRUE one-step env reward per candidate (env step on a COPY; the real trajectory
            #     state is untouched) — the 834-style calibration, now per candidate ---
            true_rewards = np.empty(K, dtype=np.float64)
            for j in range(K):
                _s2, _obs2, rew, _d = ev.jit_batch_step_vector(
                    state, vecs[j][None, :], prices, env_cfg)
                true_rewards[j] = float(np.asarray(rew).reshape(-1)[0])
            _s2, _obs2, pa_rew, _d = ev.jit_batch_step_vector(state, pa_vec, prices, env_cfg)
            pa_true = float(np.asarray(pa_rew).reshape(-1)[0])

            # --- Gumbel selection-score reconstruction (trace column only; NOT a gated signal).
            #     Deterministic part of mctx's Sequential-Halving root score: prior_logits (uniform
            #     ⇒ 0) + σ(completedQ), σ = (maxvisit_init + max_visits)·value_scale·minmax(Q). The
            #     ±Gumbel(gumbel_scale) exploration noise mctx adds internally is not reproduced;
            #     eval selection is visit-argmax, so this diagnoses the systematic ranking only. ---
            qspan = max(Q.max() - Q.min(), 1e-8)
            q_norm = (Q - Q.min()) / qspan
            gumbel_score = (MCTX_MAXVISIT_INIT + float(visit_ct.max())) * MCTX_SIGMA_VALUE_SCALE * q_norm

            # finiteness (acceptance i)
            if not (np.isfinite(r_hat).all() and np.isfinite(v_hat).all() and np.isfinite(Q).all()
                    and np.isfinite([pa_r, pa_v, pa_Q]).all()):
                all_finite = False

            # per-candidate rows
            cands = []
            for j in range(K):
                cands.append({
                    "idx": j,
                    "slot_role": _slot_role(args.vector_proposal_scheme, j, K, N),
                    "vector": vecs_np[j].astype(int).tolist(),
                    "is_policy_argmax": bool(is_pa[j]),
                    "prior_mass": _f(beta_hat[j]),
                    "r_hat": _f(r_hat[j]),
                    "v_hat": _f(v_hat[j]),
                    "Q": _f(Q[j]),
                    "gumbel_score": _f(gumbel_score[j]),
                    "action_weight": _f(aw[j]),
                    "visit_count_approx": _f(visit_ct[j]),
                    "selected": bool(j == sel_idx),
                    "true_reward": _f(true_rewards[j]),
                })

            # rank of the in-set prior under each signal (where the prior loses)
            rank_by = {
                "prior": _rank_of(beta_hat[pa_idx] if in_set else 0.0, beta_hat),
                "r_hat": _rank_of(pa_r, r_hat),
                "v_hat": _rank_of(pa_v, v_hat),
                "Q": _rank_of(pa_Q, Q),
                "visits": _rank_of(visit_ct[pa_idx] if in_set else 0.0, visit_ct),
            }

            sel_vec = vecs_np[sel_idx]
            flip_frac = float(np.mean(sel_vec != pa_np))

            states_dump.append({
                "decision_k": int(k),
                "row_ts": int(ts_dec[k]) if k < len(ts_dec) else None,
                "policy_argmax": {
                    "vector": pa_np.astype(int).tolist(),
                    "in_candidate_set": in_set,
                    "candidate_idx": pa_idx,
                    "r_hat": _f(pa_r), "v_hat": _f(pa_v), "Q": _f(pa_Q),
                    "true_reward": _f(pa_true),
                    "rank_by": rank_by,
                },
                "selected": {
                    "candidate_idx": sel_idx,
                    "vector": sel_vec.astype(int).tolist(),
                    "is_policy_argmax": bool(np.all(sel_vec == pa_np)),
                    "flip_fraction_vs_prior": flip_frac,
                    "matches_operator_vector": selection_matches,
                    "true_reward": _f(true_rewards[sel_idx]),
                },
                "candidates": cands,
            })

        # step the TRAJECTORY with the 848 policy-argmax vector (env advances here only)
        state, _, _, done = ev.jit_batch_step_vector(
            state, jnp.asarray(policy_actions[k], dtype=jnp.int32)[None, :], prices, env_cfg)
        if bool(np.asarray(done)[0]):
            break

    if not states_dump:
        raise RuntimeError("no decision states were dumped — panel too short for the given "
                           "--stride; lower --stride or widen the window.")

    n_states = len(states_dump)
    aggregate = _aggregate(states_dump, args)
    aggregate["selection_crosscheck"] = {
        "selected_matches_operator_frac": sel_match / n_states,
        "n_states": n_states,
    }
    aggregate["policy_argmax_in_candidate_set_frac"] = pa_in_set / n_states

    acceptance_i = {
        "schema_emitted": True,
        "n_states_dumped": n_states,
        "all_Q_vhat_rhat_finite": bool(all_finite),
        "policy_argmax_present_and_flagged_frac": pa_present / n_states,
        "policy_argmax_present_100pct": bool(pa_present == n_states),
        # under policy-prior the prior is GUARANTEED in the set (TASK_847); assert it here.
        "policy_argmax_in_set_100pct": bool(pa_in_set == n_states),
    }

    return {
        "tag": args.tag,
        "scheme": args.vector_proposal_scheme,
        "checkpoint": label["path"], "checkpoint_label": label["name"],
        "db": ev._display_path(args.db), "window": [args.start, args.end],
        "bar_minutes": args.bar_minutes, "eval_seed": args.eval_seed,
        "stride": args.stride, "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "panel": panel_info,
        "operator": {
            "trajectory": "policy-argmax replay (848 operator; imprint.policy_argmax_actions)",
            "observer_scheme": args.vector_proposal_scheme,
            "vector_k": K, "sims": args.sims, "explore_slots": N,
            "proposal_temp": args.vector_proposal_temp,
            "hold_bias": args.vector_hold_bias, "dither_scale": args.vector_dither_scale,
            "gumbel_scale": args.gumbel_scale, "discount": discount,
            "model_size": args.model_size, "h_cycles": args.h_cycles, "l_cycles": args.l_cycles,
            "deterministic": True, "selection": "visit_probs argmax (eval)",
        },
        "schema_notes": {
            "prior_mass": "per-candidate empirical mass β̂=n_a/K (sampled_search._empirical_prior)",
            "r_hat": "reward-model reward, recurrent_inference_vector at the candidate (depth-1 leaf)",
            "v_hat": "value-net leaf value, post dynamics_vector (recurrent_inference_vector)",
            "Q": "r̂ + γ·v̂ (γ=discount)",
            "gumbel_score": ("RECONSTRUCTED deterministic part of the mctx Sequential-Halving root "
                             "score: uniform prior_logits + σ(completedQ), σ=(maxvisit_init+"
                             "max_visits)·value_scale·minmax(Q), mctx defaults 50.0/0.1. The ±Gumbel"
                             "(gumbel_scale) exploration noise is NOT reproduced — eval selection is "
                             "visit-argmax, so this column diagnoses the systematic ranking only."),
            "action_weight": "mctx policy_output.action_weights over the K candidates",
            "visit_count_approx": "action_weight × num_simulations (approximate visit count)",
            "true_reward": "TRUE one-step env reward from stepping a COPY of the env under the "
                           "candidate vector (jit_batch_step_vector; trajectory state untouched)",
            "rank_by": "1-based DESCENDING rank of the policy-argmax vector's signal value among "
                       "the K candidates (rank 1 = best). When the prior IS in the set (policy-"
                       "prior), r̂/v̂/Q use the prior's OWN in-set candidate row from the M=16 "
                       "batched call (an in-set candidate can never rank > K); prior/visits use its "
                       "candidate mass/visits. When the prior is ABSENT (dithered-book) the r̂/v̂/Q "
                       "values are a SEPARATELY recomputed M=1 reference score (batched-vs-unbatched "
                       "XLA differs ~1e-7, so a ranked-against-K reference can read K+1) and "
                       "prior/visits are 0 (never sampled).",
            "candidate_draw": "the K-candidate set at each state is drawn with "
                              "jax.random.fold_in(PRNGKey(eval_seed), k); the trajectory is the 848 "
                              "policy-argmax replay (not an 835 search rollout), so there is no "
                              "canonical 835-search draw to match — both are valid deterministic draws.",
        },
        "n_states_dumped": n_states,
        "acceptance_i": acceptance_i,
        "aggregate": aggregate,
        "states": states_dump,
    }


def _aggregate(states_dump, args):
    """(a) rank distribution of the policy-argmax under each signal; (b) per-candidate signal
    vs TRUE one-step reward correlation (834-style calibration); (c) selected-candidate churn."""
    signals = ("prior", "r_hat", "v_hat", "Q", "visits")
    # (a) rank distribution
    rank_dist = {}
    K = int(args.vector_k)
    for s in signals:
        ranks = np.array([st["policy_argmax"]["rank_by"][s] for st in states_dump], dtype=np.int64)
        hist = {str(r): int((ranks == r).sum()) for r in range(1, K + 2)}
        rank_dist[s] = {
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "frac_rank1": float(np.mean(ranks == 1)),
            "worst_rank": int(ranks.max()),
            "histogram": {r: c for r, c in hist.items() if c > 0},
        }

    # (b) correlation of each candidate signal vs the TRUE one-step reward, per state → mean
    sig_key = {"prior": "prior_mass", "r_hat": "r_hat", "v_hat": "v_hat", "Q": "Q",
               "visits": "visit_count_approx"}
    corr = {}
    for s in signals:
        pear, spear = [], []
        for st in states_dump:
            xs = np.array([c[sig_key[s]] for c in st["candidates"]], dtype=np.float64)
            ys = np.array([c["true_reward"] for c in st["candidates"]], dtype=np.float64)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() < 2:
                continue
            p = _pearson(xs[ok], ys[ok])
            sp = _spearman(xs[ok], ys[ok])
            if p is not None:
                pear.append(p)
            if sp is not None:
                spear.append(sp)
        corr[s] = {
            "pearson_mean": _f(np.mean(pear)) if pear else None,
            "spearman_mean": _f(np.mean(spear)) if spear else None,
            "n_states": len(pear),
        }

    # (c) selected-candidate churn vs the prior vector
    flips = np.array([st["selected"]["flip_fraction_vs_prior"] for st in states_dump],
                     dtype=np.float64)
    sel_is_pa = np.array([st["selected"]["is_policy_argmax"] for st in states_dump])
    churn = {
        "flip_fraction_vs_prior_mean": float(flips.mean()),
        "flip_fraction_vs_prior_median": float(np.median(flips)),
        "flip_fraction_vs_prior_max": float(flips.max()),
        "selected_is_policy_argmax_frac": float(np.mean(sel_is_pa)),
    }
    return {"rank_distribution": rank_dist,
            "signal_true_reward_correlation": corr,
            "churn": churn}


# =============================================================================
# printed report
# =============================================================================

def print_report(r):
    a = r["aggregate"]
    print("=" * 92)
    print(f"LEAF-EVAL AUTOPSY ({r['tag']}) — {r['checkpoint_label']}  scheme={r['scheme']}")
    print(f"panel: {r['panel']['n_assets']} assets × {r['panel']['n_rows']} rows "
          f"(bar_minutes={r['bar_minutes']}), warm={r['panel']['warm']}, "
          f"decisions={r['panel']['n_decisions']}, stride={r['stride']} → "
          f"{r['n_states_dumped']} states; fee={r['fee_bps']}bp slip={r['slippage_bps']}bp; "
          f"eval_seed={r['eval_seed']}")
    ai = r["acceptance_i"]
    print(f"acceptance(i): states={ai['n_states_dumped']} finite={ai['all_Q_vhat_rhat_finite']} "
          f"pa_present_100%={ai['policy_argmax_present_100pct']} "
          f"pa_in_set_100%={ai['policy_argmax_in_set_100pct']}")
    print(f"acceptance(iii): selection matches operator on "
          f"{a['selection_crosscheck']['selected_matches_operator_frac']*100:.1f}% of states")
    print("-" * 92)
    print("(a) rank of the policy-argmax prior under each signal (rank 1 = best; lower=prior wins):")
    print(f"    {'signal':>8} | {'mean':>6} {'median':>7} {'frac@1':>7} {'worst':>6} | histogram")
    for s in ("prior", "r_hat", "v_hat", "Q", "visits"):
        rd = a["rank_distribution"][s]
        hist = " ".join(f"{k}:{v}" for k, v in rd["histogram"].items())
        print(f"    {s:>8} | {rd['mean_rank']:>6.2f} {rd['median_rank']:>7.1f} "
              f"{rd['frac_rank1']:>7.2f} {rd['worst_rank']:>6d} | {hist}")
    print("-" * 92)
    print("(b) per-candidate signal vs TRUE one-step reward correlation (mean over states):")
    print(f"    {'signal':>8} | {'pearson':>8} {'spearman':>9} {'n':>4}")
    for s in ("prior", "r_hat", "v_hat", "Q", "visits"):
        cc = a["signal_true_reward_correlation"][s]
        print(f"    {s:>8} | {ev._fmt(cc['pearson_mean']):>8} {ev._fmt(cc['spearman_mean']):>9} "
              f"{cc['n_states']:>4}")
    print("-" * 92)
    ch = a["churn"]
    print(f"(c) selected-vs-prior churn: flip_frac mean {ch['flip_fraction_vs_prior_mean']:.3f} "
          f"median {ch['flip_fraction_vs_prior_median']:.3f} max {ch['flip_fraction_vs_prior_max']:.3f}; "
          f"selected==prior {ch['selected_is_policy_argmax_frac']*100:.1f}% of states")
    print("=" * 92)


# =============================================================================
# driver
# =============================================================================

def _build_args():
    ap = argparse.ArgumentParser(
        description="TASK_849 leaf-eval autopsy — read-only per-state selection trace")
    ap.add_argument("--db", default=FIXTURE_DB)
    ap.add_argument("--periods-file", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--bar-minutes", type=int, default=1440)
    ap.add_argument("--n-assets", type=int, default=64)
    ap.add_argument("--checkpoint", default=None, help="checkpoint dir (frozen ckpt to autopsy)")
    ap.add_argument("--smoke", action="store_true",
                    help="fixture smoke: fresh-init untrained model + save->arch-check->load")
    ap.add_argument("--smoke-ckpt-dir", default=None)
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=25,
                    help="dump every Nth decision state (default 25; the smoke overrides to 8)")
    # observer proposal scheme — restricted to the two this instrument scopes (typo = fail-loud)
    ap.add_argument("--scheme", dest="vector_proposal_scheme", default="policy-prior",
                    choices=list(VALID_SCHEMES),
                    help="observer proposal scheme: policy-prior (prior guaranteed in set) or "
                         "dithered-book (policy-blind canon)")
    # cost tier for the autopsy env (default = the fee16 ExIt set)
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS, dest="fee_bps")
    ap.add_argument("--slippage-bps", type=float, default=SLIP_BPS, dest="slippage_bps")
    ap.add_argument("--position-size-pct", type=float, default=0.1, dest="position_size_pct")
    # run shapes — 828 Stage-3 battery manifest defaults (mirror eval_vector.main)
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--h-cycles", type=int, default=4)
    ap.add_argument("--l-cycles", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--vector-k", type=int, default=16)
    ap.add_argument("--vector-hold-bias", type=float, default=3.0)
    ap.add_argument("--vector-dither-scale", type=float, default=1.0)
    ap.add_argument("--vector-proposal-temp", type=float, default=1.0,
                    dest="vector_proposal_temp")
    ap.add_argument("--vector-proposal-explore-slots", type=int, default=2,
                    dest="vector_proposal_explore_slots")
    ap.add_argument("--vector-per-node-resample", action="store_true",
                    dest="vector_per_node_resample")
    ap.add_argument("--vector-budget-step", action="store_true", dest="vector_budget_step")
    ap.add_argument("--vector-budget-pct", type=float, default=0.99, dest="vector_budget_pct")
    ap.add_argument("--vector-weight-rebalance", action="store_true",
                    dest="vector_weight_rebalance")
    ap.add_argument("--vector-weight-unit", type=float, default=-1.0, dest="vector_weight_unit")
    ap.add_argument("--vector-weight-max", type=float, default=4.0, dest="vector_weight_max")
    ap.add_argument("--vector-rebalance-deadband", type=float, default=-1.0,
                    dest="vector_rebalance_deadband")
    ap.add_argument("--discount", type=float, default=0.997)
    ap.add_argument("--gumbel-scale", type=float, default=3.0)
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--value-shift", type=float, default=0.0)
    ap.add_argument("--max-positions", type=int, default=16)
    ap.add_argument("--min-cash-threshold", type=float, default=100.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def main():
    args = _build_args()
    # fail-loud on scheme typo (belt-and-suspenders over argparse choices)
    if args.vector_proposal_scheme not in VALID_SCHEMES:
        raise RuntimeError(f"--scheme '{args.vector_proposal_scheme}' invalid; "
                           f"expected one of {VALID_SCHEMES}")
    if args.max_decisions is not None and args.max_decisions < 0:
        args.max_decisions = None
    if args.symbols:
        args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.periods_file:
        with open(args.periods_file) as fh:
            first = json.load(fh)["periods"][0]
        args.start = args.start or first["start_date"]
        args.end = args.end or first["end_date"]
        if args.symbols is None:
            args.symbols = first.get("assets")
    ev._validate_eval_date(args.start, "start")
    ev._validate_eval_date(args.end, "end")

    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    if args.smoke:
        args.db = FIXTURE_DB
        args.symbols = args.symbols or list(SMOKE_SYMBOLS)
        args.start = args.start or SMOKE_START
        args.end = args.end or SMOKE_END
        if args.bar_minutes == 1440:
            args.bar_minutes = SMOKE_BAR_MINUTES
        if args.n_assets == 64:
            args.n_assets = len(args.symbols)
        if args.max_decisions is None:
            args.max_decisions = SMOKE_MAX_DECISIONS
        if args.stride == 25:
            args.stride = SMOKE_STRIDE
        is_fixture = True

    args.tag = args.tag or ("fixture-smoke" if (args.smoke and is_fixture)
                            else "fixture" if is_fixture else "real")
    out_dir = args.out_dir or (EXP_DIR if is_fixture
                               else os.path.join(_REPO, "runs", "849_leaf_autopsy"))
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        result = _run_smoke(args)
    else:
        if not args.checkpoint:
            raise RuntimeError("no checkpoint given (use --checkpoint or --smoke)")
        ev.check_checkpoint_arch(args.checkpoint, args)          # fail-loud arch guard (acceptance ii)
        model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
        ev.load_checkpoint_into(trainer, args.checkpoint)
        search_obj = ev.build_search(model, args, args.n_assets)
        panel = ev.build_panel(args)
        label = ev._ckpt_label(args.checkpoint)
        result = autopsy(label, model, trainer.state.params, search_obj, panel, args)

    out_name = f"leaf_autopsy_{args.tag}_{result['scheme']}.json"
    out_path = os.path.join(out_dir, out_name)
    result["out_pointer"] = os.path.relpath(out_path, _REPO)
    result["ledger_row"] = _ledger_row(result)
    print_report(result)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(f"\n[849] wrote {out_path}")
    print("[849] LEDGER row:")
    print(result["ledger_row"])


def _run_smoke(args):
    """Fresh-init untrained model + REAL save->arch-check->load roundtrip, then the autopsy
    (plumbing proof — untrained near-noise, no thresholds)."""
    print("[smoke] building fresh-init model/trainer (seed 0) ...")
    model_a, trainer_a = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
    ckpt_dir = os.path.abspath(args.smoke_ckpt_dir or os.path.join(EXP_DIR, "smoke_ckpt"))
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[smoke] save_checkpoint → {ckpt_dir}")
    trainer_a.save_checkpoint(ckpt_dir)
    print("[smoke] building second fresh-init trainer (seed 1) + arch-check + load_checkpoint ...")
    model_b, trainer_b = ev.build_model_and_trainer(args, args.n_assets, init_seed=1)
    ev.check_checkpoint_arch(ckpt_dir, args)                      # exercise the arch guard on smoke
    differ_before = not ev._trees_allclose(trainer_a.state.params, trainer_b.state.params)
    ev.load_checkpoint_into(trainer_b, ckpt_dir)
    match_after = ev._trees_allclose(trainer_a.state.params, trainer_b.state.params)
    print(f"[smoke] roundtrip: differ_before={differ_before} match_after={match_after}")
    if not (differ_before and match_after):
        raise RuntimeError(f"checkpoint roundtrip FAILED: differ_before={differ_before} "
                           f"match_after={match_after}")
    search_obj = ev.build_search(model_b, args, args.n_assets)
    panel = ev.build_panel(args)
    label = {"path": ev._display_path(ckpt_dir),
             "name": "untrained_fresh_init(save->load roundtrip)"}
    result = autopsy(label, model_b, trainer_b.state.params, search_obj, panel, args)
    result["roundtrip"] = {"ckpt_dir": ev._display_path(ckpt_dir),
                           "params_differ_before_load": bool(differ_before),
                           "params_match_after_load": bool(match_after)}
    return result


def _ledger_row(r):
    a = r["aggregate"]
    rd = a["rank_distribution"]
    cc = a["signal_true_reward_correlation"]
    ch = a["churn"]
    smoke_note = (" FIXTURE-SMOKE — plumbing proof, untrained near-noise, no threshold; real "
                  "autopsy numbers come from the scheduled run." if r["tag"].startswith("fixture")
                  else " Scheduled-run numbers.")
    return (
        f"| 849/readB | {date.today().isoformat()} | H (leaf-eval) | "
        f"Leaf-eval autopsy {r['checkpoint_label']} scheme={r['scheme']} "
        f"({r['n_states_dumped']} states, stride {r['stride']}) — where does the in-set prior "
        f"lose? | prior-rank mean Q {rd['Q']['mean_rank']:.1f} v̂ {rd['v_hat']['mean_rank']:.1f} "
        f"visits {rd['visits']['mean_rank']:.1f}; corr(Q,true) {ev._fmt(cc['Q']['spearman_mean'])} "
        f"corr(v̂,true) {ev._fmt(cc['v_hat']['spearman_mean'])}; sel-churn "
        f"{ch['flip_fraction_vs_prior_mean']:.2f}.{smoke_note} | {r['out_pointer']} |"
    )


if __name__ == "__main__":
    main()
