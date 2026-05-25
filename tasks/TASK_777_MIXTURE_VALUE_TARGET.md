# TASK_777 — Mixture Value Target (α·oracle + (1−α)·nstep)

**Status**: IMPLEMENTED 2026-05-25 (workstation worktree `worktree-agent-ad35b376f93d08f62`). Code flag-gated, default OFF, canon bit-identical. Awaiting GPU smoke-gate rerun + server launch per `experiments/777_mixture_value/SPEC.md`.
**Class**: Planning-side value-target reformulation. Same mechanism class as 746 (oracle), 747 (td0), 734 (flow-matching). **Counter-example bucket to the `[[classifier-alpha-anticorrelation]]` pattern.**
**Created**: 2026-05-25. Derived from `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.1 (primary variant); re-scoped per ship-path ceremony (`lore/ship_path_recommendations_2026-05-25.md` candidate #3).

---

## Hypothesis

746_value_oracle's paired Δα +0.72pp (5-seed, post-752 paired baselines — see `runs/746_value_oracle/result.md:160-168`) is the **only positive paired-Δα signal in the value-target family** that survives proper 5-seed pairing. The 5-seed STD is 2.28pp; the per-seed paired range spans `(−3.99pp, +8.69pp)` (s7 vs s23). The variance signature is consistent with a **variance-limited** mechanism, not a bias-limited one.

A convex combination
```
td_target_raw = α · oracle_scaled + (1 − α) · nstep_target_raw
```
interpolates between the bias-correct-but-high-variance oracle leg (α=1.0, reproduces 746 bit-identical) and the bias-biased-but-low-variance nstep leg (α=0.0, reproduces a1b1 nstep bit-identical). Both legs are already computed in production code paths; the mixture is a pure value-target reformulation with no encoder/classifier/dynamics touch.

**Falsifiable predictions**:
1. If the +0.72pp signal is **variance-limited**, intermediate α should produce mean paired Δα **higher than 746's +0.72pp** at lower STD. STD reduction at best-α cell below 746's 2.28pp is the mechanism diagnostic.
2. If the +0.72pp signal is **bias-limited** (pure oracle is the V* signal that any blending dilutes), the sweep is monotone in α with peak at α=1.0 — informative, confirms 746 was the best-in-family.
3. If the sweep is **flat or negative** for all α ∈ {0.25, 0.5, 0.75}, the value-target mechanism class is closed (combined with 747 td0 ≈ flat, 746 +0.72pp marginal). Document the closure.

---

## Mechanism class + anti-correlation check

**Class**: planning-side / value-target. The change is to **what the model believes is reachable** (the value target the value head trains against), not to **how confidently it predicts the next bar** (the policy head's output).

**Anti-correlation risk: LOW.** Same class as 746 (oracle), 747 (td0), 734 (flow-matching). 746's per-period breakdown showed bear/ftx_crash gains and no trade-rate explosion — the opposite signature from the anti-correlation pattern (`project_classifier_alpha_anti_correlation.md:9-32`: loss DOWN + Bal.Acc UP → trade-rate explodes → Δα collapses). The mixture sits in the same surface as 746; the anti-correlation guard rail is the same as 746's (watch trade-rate at per-period level; abort cell if it explodes).

**Counter-example status**: this task is in the same surface as the only two campaign mechanisms with paired Δα > 0 (746 +0.72pp; 734 +0.20pp at 5-seed — see `project_flow_matching_promising.md` for 734's status; both planning-side).

---

## Code change (~50 LOC, flag-gated, default OFF, canon bit-identical)

All changes are additive. No edits to existing oracle / nstep / td0 branches. Canon (`value_target_mode in {"oracle", "nstep", "td0"}`) is bit-identical.

| File | Lines (post-patch) | Change |
|---|---|---|
| `phase2_config.py:209-227` | +19 LOC (docs + field) | Extend `value_target_mode` docstring with "mixture" branch. Add `value_mixture_alpha: float = 0.5` field with `[0, 1]` range. |
| `phase2_config.py:312-326` | +14 LOC (validators) | Extend `__post_init__`: `value_target_mode ∈ {"oracle","nstep","td0","mixture"}` and `value_mixture_alpha ∈ [0, 1]`. Mirrors the TASK_771/N1 `value_clip_mode` validator. |
| `train_phase2_cli.py:252-272` | +13 LOC | Extend `--value-target-mode choices=` to include `"mixture"`. Add `--value-mixture-alpha` (float, default None → config default 0.5). Add `--oracle-horizon` (int, default None → config default 400) for the secondary §756b variant. |
| `train_phase2.py:1051-1055` (closure capture) | +5 LOC | `value_mixture_alpha = getattr(config, 'value_mixture_alpha', 0.5)` — captured at JIT-trace time, baked into HLO, α changes re-JIT once per cell. Same pattern as `value_clip_mode` / `value_target_mode`. |
| `train_phase2.py:1292-1332` (new `elif mixture:` block) | +41 LOC including comments | New value-loss branch placed between `nstep` and `td0`. Oracle leg mirrors the `oracle` branch (same clip-gating); nstep leg mirrors the `nstep` branch (same `out_tN` denorm, same clip-gating); blend in pre-PopArt raw space; PopArt normalize + categorical clip identical to the other branches. |
| `train_phase2.py:2612-2627` (arg translation) | +16 LOC | `--value-mixture-alpha` override with type-correctness guard ("`mixture` mode required") + `--oracle-horizon` override. Same pattern as `--value-clip-mode`. |

**Total**: +108 LOC including docs; functional surface ~50 LOC. Diff visible via `git diff master -- train_phase2.py train_phase2_cli.py phase2_config.py`.

**Variants** (production cells):

| cell | value_target_mode | value_mixture_alpha | seeds |
|---|---|---:|---|
| 777-α0.25 | mixture | 0.25 | 7, 13, 23 |
| 777-α0.50 | mixture | 0.50 | 7, 13, 23 |
| 777-α0.75 | mixture | 0.75 | 7, 13, 23 |
| **baseline a1b1** | nstep | — (n/a) | 7, 13, 23 (reuse `runs/760_train_mcts_sims/a1b1_sims32_seed_*/`) |
| **reference 746** | oracle | — (n/a) | 7, 13, 23 (and 31, 42 for variance reference); reuse `runs/746_value_oracle/oracle_seed_*/` |

α∈{0.0, 1.0} explicitly NOT in production sweep — those are smoke-only (reproduce 746 / a1b1 already at hand).

---

## Sanity gate (smoke)

Two bit-identity gates, both **CRITICAL** per TASK_756 §7a operator commitment.

### Smoke gate 1 — `value_target_mode="oracle"` is bit-identical post-patch

The oracle branch in `train_phase2.py:1262-1274` is **untouched** by the patch (verified via diff: the patch only inserts an `elif value_target_mode == "mixture":` block between the existing `nstep` and `else: td0` branches). The `if value_target_mode == "oracle":` dispatch still routes to the original oracle code path.

**Static-trace verification** (no GPU needed):
- `git diff master -- train_phase2.py` shows the oracle branch (lines 1262-1274) is byte-identical pre- and post-patch.
- `python -c "import ast; ast.parse(open('train_phase2.py').read())"` — parse-clean.
- `python -c "from phase2_config import Phase2Config; Phase2Config()"` — no validator regressions on canon defaults.

**GPU verification** (deferred to server smoke):
- Pre-patch baseline: `value_target_mode=oracle`, seed=7, 1 epoch × 10 steps, log `value_loss / td_target_mean / td_target_std / grad_norm / pa_mean / pa_std` to CSV.
- Post-patch run with identical args: diff CSV column-by-column; max abs diff ≤ 1e-6 across all step rows. Following the 769 bit-identity-gate protocol (`experiments/769_value_clip/SPEC.md:114-129`).

**Status (worktree)**: PASS by static trace — the oracle branch is byte-identical to pre-patch source. GPU bit-identity rerun deferred to server smoke (workstation has no spare GPU during this session).

### Smoke gate 2 — `value_target_mode="mixture"` with `value_mixture_alpha=1.0` reproduces oracle bit-identical

This is the **critical** gate — catches sign or clip-order bugs in the mixture insertion.

**Static-trace argument**:

Mixture branch at α=1.0:
```python
oracle_scaled = oracle_values * value_scale
if value_clip_mode == "pre_popart":
    oracle_scaled = jnp.clip(oracle_scaled, -10.0, 10.0)
# Nstep leg computes but the blend collapses it out
out_tN = model.apply(...)               # computed but unused after blend
v_tN = ...
v_tN_raw = v_tN * popart_std + popart_mean
nstep_target_raw = nstep_rewards + nstep_gamma_N * v_tN_raw
if value_clip_mode == "pre_popart":
    nstep_target_raw = jnp.clip(nstep_target_raw, -10.0, 10.0)
td_target_raw = 1.0 * oracle_scaled + 0.0 * nstep_target_raw
              # = oracle_scaled + 0.0
              # = oracle_scaled  (IEEE 754: finite x + 0.0 = x exactly,
              #                   finite y * 0.0 = 0.0)
td_target_norm = (td_target_raw - popart_mean) / jnp.maximum(popart_std, 1e-8)
td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
```

Oracle branch:
```python
oracle_scaled = oracle_values * value_scale
if value_clip_mode == "pre_popart":
    oracle_scaled = jnp.clip(oracle_scaled, -10.0, 10.0)
td_target_norm = (oracle_scaled - popart_mean) / jnp.maximum(popart_std, 1e-8)
td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
td_target_raw = oracle_scaled
```

Both produce identical `td_target_norm`, `td_target_raw`, and (since `out_tN` is a forward pass on stopped params with the same observations across both branches but unused in oracle) all downstream metrics. The α=1.0 mixture is `td_target_raw = 1.0 * oracle_scaled + 0.0 * nstep_target_raw`; IEEE 754 mandates this equals `oracle_scaled` exactly for any finite `nstep_target_raw` (and `nstep_target_raw` is clipped to `[-10, 10]` in pre_popart mode = canon, finite by construction).

**GPU verification** (deferred):
- Run `value_target_mode=oracle`, seed=7, 1 epoch × 10 steps → CSV A.
- Run `value_target_mode=mixture --value-mixture-alpha 1.0`, seed=7, identical other args → CSV B.
- `diff` on `value_loss / td_target_mean / td_target_std / grad_norm / pa_mean / pa_std` columns: max abs diff ≤ 1e-6 (cuDNN noise floor; 6th-sig-fig agreement).

**Status (worktree)**: PASS by static trace + IEEE 754 argument. The mixture branch (lines 1292-1332) at α=1.0 produces `td_target_raw = oracle_scaled` exactly. GPU bit-identity rerun deferred to server smoke per the 769 protocol.

### Smoke gate 3 (bonus) — α=0.0 reproduces nstep bit-identical

Symmetric to gate 2. The blend `0.0 * oracle_scaled + 1.0 * nstep_target_raw = nstep_target_raw`, and `nstep_target_raw` in the mixture branch is computed identically to `td_target_raw` in the nstep branch (same `obs_tN`, same `out_tN` apply, same `v_tN_raw` denorm, same conditional clip). **PASS** by static trace.

---

## Promotion gate (per universal constraints)

- **Paired Δα ≥ +3pp** vs 760 a1b1 sims=32 baseline, **no seed < −5pp**, **eval-MCTS sims=16 joint scorecard** (`benchmark_easy.py --eval-mcts-sims 16 joint`).
- 3-seed screen (seeds 7, 13, 23). If best-α cell passes → 5-seed extension (add 31, 42). Promotion only after 5-seed clears with no seed < −5pp.

**Note on baseline pairing**: TASK_756 spec recommends pairing against 746 (the +1pp gate maps to ~+3.7pp vs a1b1 given 746's +0.72pp). Per universal-constraint ship-rule, the **canonical** gate is **paired vs 760 a1b1 sims=32** — the deploy-truthful baseline since 2026-05-19 (`project_single_mechanism_exhausted.md`). 746 is the **reference for variance comparison**, not the paired baseline.

---

## Mechanism diagnostic (not a gate, but informative)

**STD-across-seeds at best-α cell vs 746's STD 2.28pp**:
- If best-α STD < 2.28pp AND mean > +0.72pp → **variance-limited hypothesis confirmed**. The mixture reduces variance and lifts mean — promote if it clears +3pp; if it lifts mean but doesn't clear +3pp, document as the cleanest variant-of-variant evidence on the stack.
- If best-α STD ≥ 2.28pp → mixing didn't reduce variance; α just slides between two known points (746 at α=1.0, a1b1 at α=0.0). Closes the variance-reduction subhypothesis.
- If sweep is monotone in α with peak at α=0.75 → 746 (α=1.0 endpoint) was near-optimal; no interior gain. Closes the mixture roadmap.

---

## Anti-correlation early-exit

Per `project_classifier_alpha_anti_correlation.md:9-32`, the canonical anti-correlation signature is: ΔBal.Acc > 0 AND ΔAlpha < 0 with a trade-rate explosion at per-period level. The value-target class has historically been counter-example (746: ΔBal.Acc −2.34pp, ΔAlpha +2.54pp at 3-seed; trade-rate not explosive).

**Early-exit rule for each variant cell**:
- If E50 (epoch 50) of seed 7 shows `policy_loss < 0.7 * baseline_E50_policy_loss` AND `trade_rate > 2 * canon_trade_rate` → flag as anti-correlation leakage, abort cell. Same protocol as 738/741 abort rules per TASK_756 §2 anti-corr framing.
- Risk is **LOW** — neither leg (oracle scale, nstep bootstrap) touches the policy head's softmax. The hypothesis is that this guard rail never trips on a mixture cell; if it does, the value target is somehow feeding back into the classifier (would require investigation — likely PopArt running statistics drift).

---

## Out of scope (explicit)

- **DO NOT** modify `path_solver.py` (oracle path computation fixed; any oracle-formulation change is a separate task).
- **DO NOT** add a new value-head architecture (PopArt 401-bin head fixed per `mcts_trader/CLAUDE.md`).
- **DO NOT** touch MCTS search budget (`mcts_sims`, `factored_temperature`, depth all fixed at 760 a1b1 sims=32 baseline values).
- **DO NOT** introduce a schedule on α (annealing α high→low). Static α only; α schedules are TASK_756d if 777 promotes.
- **DO NOT** combine with TASK_734 flow_matching or TASK_695 oracle-mask-head in this task. Stacking is a Tier-3 follow-up after each individually verifies on the canon scorecard.
- **DO NOT** include `--oracle-smoothing` / EWMA changes (TASK_784 / TASK_756c remains a separate task with its own batch-builder edits at `train_phase2.py:619-624`).

---

## References

- `lore/ship_path_recommendations_2026-05-25.md` §3 — TASK_756 mixture value target as candidate #3 in the ship-path triage.
- `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.1 — original mixture spec, §7a smoke-gate protocol (bit-identity gates).
- `runs/746_value_oracle/result.md` — 5-seed paired Δα +0.72pp, STD 2.28pp; the variance signature that motivates this task.
- `runs/760_train_mcts_sims/` — canonical a1b1 sims=32 baseline (the paired comparison).
- `runs/762_reward_shaping/` + `experiments/769_value_clip/SPEC.md:114-129` — bit-identity smoke-gate protocol (the deploy-safety pattern for value-pipeline patches).
- `mcts_trader/jax_agent/path_solver.py:39-150` — oracle infrastructure (untouched).
- `train_easy.py:595, 725, 762-766` — `oracle_cum_pnl` precompute (untouched).
- `train_phase2.py:618-633` — value-target batch fields (`oracle_values`, `nstep_rewards`; both already consumed, no precompute change).
- `train_phase2.py:1042, 1050, 1055` — closure-time capture of `value_target_mode`, `value_clip_mode`, `value_mixture_alpha` (the patch adds the third).
- `train_phase2.py:1257-1340` — value-loss dispatch (the patch inserts a new `elif mixture:` block).
- `phase2_config.py:209-228, 286-326` — config field + validator (the patch extends both).
- `train_phase2_cli.py:252-272` — CLI flag block (the patch extends the choices + adds two new flags).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — LOW-risk justification.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — campaign-state context (single-mechanism class closed under eval-MCTS recal; 777 is the final small-perturbation probe in the value-target family).
- `experiments/777_mixture_value/SPEC.md` — server launch + wall-time plan + diagnostics.

---

## Implementation log

**2026-05-25** — Implemented in workstation worktree `worktree-agent-ad35b376f93d08f62` per ship-path ceremony assignment. All three files edited:

- `phase2_config.py`: +19/0 LOC (field + docstring); +14/0 LOC (validators in `__post_init__`).
- `train_phase2_cli.py`: +13/3 LOC (extend choices in `--value-target-mode`, add `--value-mixture-alpha` + `--oracle-horizon` with helps; updated value-target-mode help).
- `train_phase2.py`: +5/0 LOC (closure capture at `:1051-1055`); +41/0 LOC (mixture branch at `:1292-1332`); +16/0 LOC (arg translation at `:2612-2627`).

**Validation**:
- `python -c "import ast; ast.parse(open('train_phase2.py').read())"` — PASS.
- `python -c "from phase2_config import Phase2Config; ..."` — PASS for default, mixture-mode + α∈{0.0, 0.25, 0.5, 0.75, 1.0}, AND raises on α=1.5 / α=−0.1 / mode='garbage'.
- CLI surface: `--value-target-mode {oracle,nstep,td0,mixture}`, `--value-mixture-alpha VALUE_MIXTURE_ALPHA`, `--oracle-horizon ORACLE_HORIZON` all appear in `--help` output.

**Bit-identity gates**: PASS by static trace + IEEE 754 argument. GPU smoke deferred to server (worktree has no spare GPU during this session); the bit-identity rerun follows the 769 protocol (`experiments/769_value_clip/SPEC.md:114-129`) and produces a CSV `diff` ≤ 1e-6 against the pre-patch oracle run at the same seed/args.

**Deploy contract**: Sub-agent worktree-only. NO `deploy.sh` call. NO push. Commit in worktree only. Main agent owns the deploy.
