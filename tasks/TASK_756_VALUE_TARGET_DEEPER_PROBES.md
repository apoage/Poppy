# TASK_756 — Value-Target Deeper Probes (Oracle Variants)

**Status**: DRAFTED 2026-05-17. SPEC stub awaiting smoke gate + queue placement.
**Class**: Planning-side / value-target reformulation. Successor to 746_value_oracle (5-seed mean Δα +2.23pp vs proxy baseline, all-positive but below +3pp promotion bar).
**Mechanism class**: Planning-side, NOT classifier-side. Counter-example to the anti-correlation pattern (`memory/project_classifier_alpha_anti_correlation.md`).

---

## Priority framing — read first

This task is the **highest-priority pivot** if Titans-MAG @ b128 (TASK_749, currently in flight, ETA early 2026-05-18 UTC) does not promote. Rationale:

- 746_value_oracle is the **strongest single signal of the entire Stage-B campaign** (`runs/746_value_oracle/result.md:3,30-35`). 5-seed extension landed all-positive, all-period-positive, STD 2.28pp (below the documented single-seed noise floor) — but Δα +2.23pp vs proxy baseline did not clear the +3pp promotion bar.
- 747_value_td0 came back **flat** (~ +0.16pp from internal review — see "TASK_754 status reconciliation" backlog). After 747's null result, the value-target roadmap was provisionally closed. The 746 5-seed re-opens it because oracle, specifically, is the *only* configuration in the value-target family with consistently positive Δα. Closing the entire family on td0's null was premature for the oracle branch.
- Bella-side tasks (TASK_753 Bella-as-data, TASK_754 trajectory-shape distill) are exploring a **different mechanism class** (encoder-side auxiliary signal). TASK_756 doubles down on the only mechanism class that has produced positive Alpha on this stack.

---

## 1. Goal

Extend 746's positive signal. Find a value-target variant of the form "use a richer / better value target than the n-step TD bootstrap" that **clears the +3pp paired-Δα promotion bar**.

Concrete deliverable: a 3-seed paired-Δα verdict on one primary variant (mixture target, §4), plus follow-up TASK_756b/c stubs for two secondary variants (longer oracle horizon, smoothed oracle).

---

## 2. Hypothesis

746 oracle works because the oracle PnL target encodes the *cumulative reachable return* from the current state, while n-step TD encodes the *N-step discounted reward sum bootstrapped via the value head's own estimate*. The oracle target is bias-correct (it is the true V* under the path-solver policy) but high-variance per-window; the n-step target is low-bias *to its own bootstrap* but inherits the value head's drift.

Three independent reasons each variant might amplify the +2.23pp signal:

1. **Mixture (primary)**: oracle alone has high per-window variance — the variance shows as the STD 2.28pp across 5 seeds, and in the per-period spread (s23 +8.69pp vs s7 −3.99pp on 3-seed paired Δα, `runs/746_value_oracle/result.md:13-23`). A convex combination `α · oracle + (1−α) · nstep` linearly interpolates between the bias-correct-but-variance-heavy oracle and the bias-biased-but-low-variance n-step. If the +2.23pp signal is variance-limited (rather than bias-limited), the mixture should produce a Δα peak at intermediate α with mean above 746's +2.23pp. If the signal is bias-limited, the peak is at α=1 (pure oracle, i.e. 746) and we have a clean confirmation that 746 is near-optimal in this family.

2. **Longer oracle horizon (secondary, TASK_756b)**: 746 uses `oracle_value_horizon=400` (`phase2_config.py:160`, the episode length). This is essentially "PnL to end of episode". Sweep N=200/400/800 (shorter, default, longer) to find whether the current N is at a sweet spot or at a corner. Short horizon = closer to n-step in bias (more dependent on local PnL); long horizon = closer to terminal-PnL planning (more global signal, but also more "infinite" relative to MCTS planning depth of ~5 steps).

3. **Smoothed oracle (secondary, TASK_756c)**: replace the point-in-time oracle PnL difference with an EWMA over a window of oracle PnL deltas. Reduces the per-window peakiness that may be driving the s7 −3.99pp outlier. Risk: smoothing may attenuate the very signal that's producing +2.23pp.

Anti-correlation framing (`memory/project_classifier_alpha_anti_correlation.md:9-32`): all three variants live in the **planning-side mechanism class** (they change *what the model believes is reachable*, not *how confidently it predicts the next bar*). The expected signature on a positive run is: loss roughly flat or slightly higher, Bal.Acc roughly flat, Δα positive — same as 734_flow_matching's signature (loss +0.06, Bal.Acc +1.36, Δα +3.38). If we see "loss DOWN, Bal.Acc UP, Δα collapse" on any of these variants, it's a signal that the variant is leaking into the classifier-side mechanism class and we should abort early.

**Pre-registration / falsification**: if 756 also lands in [+1, +3]pp on the paired-3-seed gate, the value-target roadmap is *genuinely* closed. Two consecutive runs with marginal-positive Δα in the value-target family, with no path to +3pp, is enough evidence to retire the family. The third variant (756c smoothed) is then NOT pursued. Document this commitment in `runs/756_value_mixture/result.md` at write-time.

---

## 3. Background / current state

Files and lines load-bearing for this task. All citations verified against current code 2026-05-17.

### Oracle infrastructure (already exists; do NOT modify)

- **Oracle path computation**: `mcts_trader/jax_agent/path_solver.py:39-150` — `greedy_lookahead_path` produces selections, per-step PnL, and `cumulative_pnl`. Lookahead is `oracle_lookahead=20` (`easy_config.py:66`). **This is the BC-target lookahead, NOT the value horizon.**
- **Oracle cum-PnL precompute**: `train_easy.py:595, 725, 762-766` — `oracle_cum_pnl` is computed once per slot at `_make_slot` time and persists in RAM for the slot's lifetime. Shape `(T-1,)` per slot. This is the *only* numerical artifact this task needs to read from — both oracle and nstep value targets are derived from it.
- **Oracle value-target read site**: `train_phase2.py:619-624` — `oracle_value_horizon` (default 400) controls the look-ahead: `remaining = cum_pnl[t + horizon] - cum_pnl[t]`.
- **N-step value-target read site**: `train_phase2.py:626-633` — discounted sum of per-step `cum_pnl[t+1] - cum_pnl[t]`, length `nstep_horizon=20`. Both targets are float32 in the batch dict (`'oracle_values'`, `'nstep_rewards'`).
- **Value-loss branch site**: `train_phase2.py:1094-1126` — the if/elif/else dispatch on `value_target_mode`. **This is the only place where the value-target value is consumed inside the train step.** Mixture should land here as a third branch (or a small inlined combination).
- **Config field**: `phase2_config.py:160-164` — `oracle_value_horizon: int = 400`, `nstep_horizon: int = 20`, `value_target_mode: str = "oracle"`.
- **CLI surface (existing)**: `train_phase2.py:2094-2100` — `--value-target-mode`, `--nstep-horizon`, `--td-discount`. Argument translation `train_phase2.py:2406-2411`.

### Why this is the right entry point

The mixture target consumes **two values already in the batch dict** (`oracle_values` and `nstep_rewards`). No precompute change. No path-solver change. No new state in `_make_slot`. The longer-horizon variant is a `oracle_value_horizon` config change → already an int parameter, no code change at all on the read side (only need to expose it as a CLI flag). The smoothed variant needs a small numpy EWMA in the batch builder, ~10 LOC.

### Anti-correlation context

- `memory/project_classifier_alpha_anti_correlation.md` — three confirmed instances (741 mint, 738 legacy_27ch, 737 no_aux_solo). The pattern: lower loss / higher Bal.Acc → trade-rate explosion → Δα collapse. **Counter-example**: 734_flow_matching (planning-side, +3.38pp). 746 oracle is the second counter-example (+2.23pp).

---

## 4. Design

### 4.1 Primary variant — mixture target (selected, TASK_756 main)

**Math.** For each window the batch already carries:

- `oracle_values[i]` — oracle remaining PnL over `oracle_value_horizon=400` steps, clipped to `[-10, 10]` post-`value_scale` (`train_phase2.py:1096-1100`).
- `nstep_rewards[i]` — discounted sum of per-step PnL over `nstep_horizon=20`, clipped to `[-10, 10]` (`train_phase2.py:629-632`).

Mixture target with bootstrap-aware blend (so the gradient is well-defined for transitions oracle cannot predict):

```
# Existing branches (kept unchanged for reproducibility of 746):
# - oracle: td_target_raw = oracle_scaled
# - nstep:  td_target_raw = nstep_rewards + γ^N * V(s_{t+N})
#
# New branch:
elif value_target_mode == "mixture":
    α = config.value_mixture_alpha    # static, in [0, 1]
    oracle_scaled = oracle_values * value_scale
    oracle_scaled = jnp.clip(oracle_scaled, -10.0, 10.0)
    # Nstep bootstrap term — identical to the "nstep" branch above
    out_tN = model.apply({'params': stopped_params}, observation=obs_tN,
                         mode='initial_inference')
    support = get_value_support()
    v_tN = jnp.sum(jax.nn.softmax(out_tN['value'], axis=-1) * support, axis=-1)
    v_tN_raw = v_tN * popart_std + popart_mean
    nstep_target_raw = nstep_rewards + nstep_gamma_N * v_tN_raw
    nstep_target_raw = jnp.clip(nstep_target_raw, -10.0, 10.0)
    # Blend (linear in raw space, before PopArt normalisation):
    td_target_raw = α * oracle_scaled + (1.0 - α) * nstep_target_raw
    td_target_norm = (td_target_raw - popart_mean) / jnp.maximum(popart_std, 1e-8)
    td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
```

Both oracle and nstep are pre-clipped to `[-10, 10]`, so the convex combination stays in `[-10, 10]` without further clipping (PopArt normalisation happens once at the end). The α=1 limit is byte-identical to the `oracle` branch (reproduces 746). The α=0 limit is byte-identical to the `nstep` branch (reproduces a1b1). This is load-bearing for reproducibility checks (see §7 smoke).

**α sweep**: pre-register **α ∈ {0.3, 0.5, 0.7}** for the 3-seed screening. Three points, not five, to keep the wall time bounded. Rationale: α=0.3 weights nstep > oracle (the "anchor at 746's predecessor"), α=0.5 is the symmetric mixture, α=0.7 weights oracle > nstep (the "anchor at 746"). If the Δα signal is monotone in α (i.e. signal is bias-driven), we'll see it across these three. If it peaks at intermediate α (variance-driven), we'll see it at α=0.5. We do NOT sweep α=0 or α=1 here — those reproduce existing runs.

### 4.2 Secondary variant A — longer oracle horizon (TASK_756b stub)

Sweep `oracle_value_horizon ∈ {200, 400 (default), 800}`. Implementation: add `--oracle-horizon` CLI flag, wire to `config.oracle_value_horizon`. No code change to the read site (it already reads from config). Storage impact: ZERO — `oracle_cum_pnl` is already computed for the full episode; horizon is just an index offset at batch-build time (`train_phase2.py:621-624`).

**Why not primary**: the current default 400 = episode length is already an extreme (PnL-to-end). N=800 would clamp at episode end for all decisions ≥ epoch_length-800; effectively a no-op for most windows. N=200 may attenuate the "global return" signal that 746 captures. Both endpoints are likely worse than 400; we expect a flat-or-negative sweep. Run it for completeness; do NOT lead the campaign with it.

### 4.3 Secondary variant B — smoothed oracle (TASK_756c stub)

EWMA over the per-step oracle PnL deltas, with smoothing constant λ ∈ {0.9, 0.95, 0.99}. Implementation: in `train_phase2.py:619-624`, replace the point-difference `cum_pnl[t+h] − cum_pnl[t]` with an EWMA over `cum_pnl[t : t+h]` deltas. ~15 LOC of numpy at batch-build time. CLI flag `--oracle-smoothing λ`; λ=0.0 reproduces 746 (no smoothing).

**Why not primary**: §2 hypothesis 3 — smoothing may attenuate the very signal driving +2.23pp. The per-window peakiness of oracle PnL is *informative*: it tells the value head when a window is "good vs bad" in the path-solver's eyes. EWMA flattens that. Run after the mixture variant if the value-target roadmap is still alive.

### 4.4 Why mixture is the primary

- **Most likely to produce a +Δα peak above 746** on the central hypothesis (variance-limited oracle). If the +2.23pp signal is variance-limited, a convex blend with low-variance n-step should reduce per-seed STD and raise the mean.
- **Zero risk of leaking into classifier-side mechanism class** — both oracle and nstep are value-target reformulations, not encoder/classifier signals.
- **Cleanest implementation**: ~15 LOC in the train-step, ~10 LOC for CLI / config. No precompute change, no batch dict change, no new state.
- **Reproducibility of 746 preserved**: α=1.0 limit is byte-identical to oracle branch. Reviewer can verify with a 5-step diff smoke (see §7).

---

## 5. CLI surface

New flags, added next to the existing `--value-target-mode` block at `train_phase2.py:2094-2100`. Defaults preserve all existing behaviour byte-identical.

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--value-target-mode {oracle,nstep,td0,mixture}` | str | `oracle` (config default) | Add `mixture` as a fourth choice. Implementation lives in `train_phase2.py:1094-1126` as a new `elif` branch. |
| `--value-mixture-alpha` | float | `0.5` | Blending weight α for `mixture` mode. Must be in `[0, 1]`. Ignored unless `--value-target-mode mixture`. **No CLI override** of α via mid-training — static for the whole run. |
| `--oracle-horizon` | int | `None` → `phase2_config.oracle_value_horizon` (400) | Override `config.oracle_value_horizon`. Effective for `oracle` and `mixture` modes (which both read `oracle_values`). No effect on `nstep` or `td0`. **TASK_756b dependency.** |
| `--oracle-smoothing` | float | `0.0` (off) | EWMA smoothing constant λ for oracle PnL difference. λ=0.0 reproduces 746 byte-identical. Implementation in `train_phase2.py:619-624`. **TASK_756c dependency.** |

**Composition with `--value-target-mode oracle`**: the new flags are *additive* — `--value-target-mode oracle` continues to work, `--value-target-mode mixture --value-mixture-alpha 1.0` is also a valid path to 746 (use as a reproducibility smoke). Argument-translation block at `train_phase2.py:2406-2411` extends symmetrically.

---

## 6. Implementation steps

Numbered, with target files and rough LOC. Most infrastructure already exists.

1. **Add `mixture` to the value-target-mode choice list** (~3 LOC). `train_phase2.py:2095` — extend `choices=['oracle', 'nstep', 'td0']` to include `'mixture'`. Add the flag help text.

2. **Add three new CLI flags** (~12 LOC). `train_phase2.py:2094-2100` block:
   - `--value-mixture-alpha` (float, default None → config default).
   - `--oracle-horizon` (int, default None → config default).
   - `--oracle-smoothing` (float, default None → config default).

3. **Extend `Phase2Config`** (~6 LOC). `phase2_config.py:160-164`:
   - Add `value_mixture_alpha: float = 0.5`.
   - Add `oracle_smoothing: float = 0.0`. (`oracle_value_horizon` already exists.)

4. **Argument translation block** (~9 LOC). `train_phase2.py:2406-2411`: wire the three new CLI overrides into `config`. Validate `0.0 ≤ value_mixture_alpha ≤ 1.0`; raise `ValueError` on out-of-range.

5. **Implement the `mixture` value-loss branch** (~15 LOC). `train_phase2.py:1094-1126`: add the `elif value_target_mode == "mixture":` block exactly as written in §4.1. The block reuses `obs_tN`, `support`, and `popart_*` already in scope. **`value_mixture_alpha` enters as `static_argnames`** for the train_step JIT — it does NOT change shape, and we don't want it to be a traced array. Confirmed compatible with the existing `static_argnames` pattern (`train_phase2.py:970-973`).

6. **(TASK_756c only)** EWMA smoothing on `oracle_values` (~15 LOC). `train_phase2.py:619-624`:
   - Replace `cum_pnl[end_idx] - cum_pnl[decision_indices]` with an EWMA over `cum_pnl[d : d+h]` deltas, weighted by `λ ** np.arange(h)[::-1]` (so most recent deltas carry highest weight). Memory: scratch buffer `(B, h)` floats, ~B·400·4 = 1 MB per batch. **Stays in RAM** — does not touch disk; computed lazily at batch-build time.
   - If `oracle_smoothing == 0.0`, fall through to the existing point-difference (byte-identical).
   - This is optional for the primary task; only ship if 756c is greenlit after primary verdict.

7. **Smoke-test scripts** (~10 LOC). Tag with the per-variant prefix. Land in `tools/run_756_value_mixture.sh`. Mirror the `tools/run_*.sh` convention from prior tasks (one file per primary variant).

8. **Logging / CSV** (~10 LOC). `train_phase2.py:1571-1601`: add `value_target_mode_active` (string) and `value_mixture_alpha_active` (float) to the per-epoch CSV writer for reproducibility post-hoc. The training trace must record α even if α=0 or α=1 (so reproducibility checks can verify a `mixture α=1.0` run matches a pure `oracle` run).

9. **SPEC file** (~30 LOC). Land at `experiments/756_value_mixture/SPEC.md` per `experiments/746_value_oracle/SPEC.md` pattern. Three sub-SPECs in the same dir if 756b/c are run: `SPEC_756b_oracle_horizon.md`, `SPEC_756c_oracle_smoothing.md`.

**Estimated total**: ~50 LOC for primary (steps 1-5, 7-8). ~15 LOC additional for 756c. ~10 LOC SPEC docs. The mechanical edits are smaller than 749/754 because both oracle and nstep batch fields already exist.

---

## 7. Smoke tests

Two-tier smoke per the TASK_749 / TASK_754 production-batch OOM rule.

### 7a. Unit-level (<60s wall, no GPU full)

```bash
cd ~/Trader && source trader_env/bin/activate
# Reproducibility check: α=1.0 mixture vs pure oracle must produce
# bit-identical value_loss for the first 5 steps.
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 5 --batch-size 32 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode mixture \
  --value-mixture-alpha 1.0 --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --seed 7 --output-dir runs/000_smoke/756_unit_a1 \
  --name 000_756_unit_alpha1 2>&1 | tee runs/000_smoke/756_unit_a1/train.log
```

Repeat with `--value-target-mode oracle` (no `--value-mixture-alpha`), output to `runs/000_smoke/756_unit_oracle`. Verify (diff on stage_b_metrics.csv `value_loss` column, first 5 rows): max abs diff ≤ 1e-6. If diff is larger, the mixture implementation has a bug in the α=1 limit and must be fixed before the production smoke.

Same check with `--value-mixture-alpha 0.0` should match a pure `--value-target-mode nstep` run within 1e-6.

### 7b. Production-batch smoke (per TASK_749 OOM rule)

Full `train_step_fn` with production batch and slot count, ~5 minutes wall:

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 10 --batch-size 128 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode mixture \
  --value-mixture-alpha 0.5 --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --seed 7 --output-dir runs/000_smoke/756_prod \
  --name 000_756_prod 2>&1 | tee runs/000_smoke/756_prod/train.log
```

**Pass criteria (all must hold)**:
1. JIT compile completes ≤ 6 minutes (no second-order autodiff — should match baseline ≈ 5 min).
2. `total_loss`, `value_loss`, `policy_loss` finite for all 10 steps.
3. Peak VRAM ≤ baseline Phase-2 + 100 MB. **The OOM rule**: production batch 128 with n_slots=4 is the configuration that historically blew up on memory in TASK_749. The mixture target adds *no* tensors beyond the existing `out_tN` (which is already computed in the `nstep` branch). Headroom should be unchanged.
4. `value_loss` magnitude within 2× of a pure-oracle run's E0 step 10 value (loose bound; we don't need convergence, just sanity).

If (3) fails: the mixture branch is allocating extra activations. The most likely cause is that `out_tN` is being computed even when the `oracle` branch is taken — verify the if/elif/else structure short-circuits correctly.

---

## 8. Eval protocol

Paired 3-seed screening with **746 as the comparison baseline, NOT a1b1**.

### Rationale

We are asking "is variant X better than current oracle?", not "is variant X better than no oracle?". 746's verdict already settled the latter (yes, marginally). The question for 756 is whether the variant adds incremental Δα *on top of* 746. Pairing against 741_mint a1b1 would conflate "is the variant useful" with "is oracle useful at all", and 746 has already paid for the latter answer.

### Protocol

- **Variant runs**: 3 paired seeds (7, 13, 23) per α ∈ {0.3, 0.5, 0.7} = 9 runs total at 200 epochs.
- **Baseline runs**: re-use `runs/746_value_oracle/oracle_seed_{7,13,23}/` for the per-seed paired Δα reference. **DO NOT re-run 746** — same checkpoint, same code, same seeds means the reference α is already known.
- **Eval**: `tools/benchmark_easy.py` against the standard 6-period benchmark set.
- **Screening verdict (write to `runs/756_value_mixture/result.md`)**:
  - **PROMOTE to 5-seed extension** if any α ∈ {0.3, 0.5, 0.7} produces mean Δα ≥ +1.5pp vs 746 AND ≥ 2/3 seeds positive AND worst seed ≥ −2pp.
  - **HOLD** if mean Δα ∈ [−1, +1.5]pp on the best α. Document per-α split. If the trend across α is monotone (e.g. Δα increases with α toward 1), the data argue for "746 was near-optimal in the family" and we close the roadmap.
  - **REJECT** if all three α produce mean Δα < −1pp vs 746. Document `memory/project_value_target_roadmap_closed.md` and retire the value-target mechanism class.
- **Promotion criteria — 5-seed extension** (seeds 31, 42 added, matched against `runs/746_value_oracle/oracle_seed_{31,42}/`): mean Δα vs 746 ≥ +1pp AND no seed < −3pp. **Δα vs 746 ≥ +1pp** ⇒ absolute Δα vs a1b1 ≥ +3.23pp (746's +2.23pp + 1pp), i.e. **clears the +3pp promotion bar in absolute terms**. This is the actual promotion gate.

### Comparison-baseline note (load-bearing)

The 3-seed mean baseline for 746 vs a1b1 is +2.54pp (paired). The 5-seed estimate is +2.23pp (proxy). If 756 mixture clears +1pp paired vs 746 at 3 seeds, the *absolute* improvement over a1b1 is +3.54pp (well above the +3pp promotion bar). At the 5-seed gate, +1pp vs 746 maps to +3.23pp vs a1b1. The "vs 746" framing makes the promotion gate stricter (~1pp instead of ~3pp) but only because 746 already paid +2.23pp.

**Conditional re-baseline**: if 752_value_oracle_baseline_pair (currently queued — see `runs/746_value_oracle/result.md:42-43`) lands before 756, swap the paired baseline at s31/s42 to use the new true-paired baseline values instead of the 3-seed proxy. This converts the soft 5-seed number to a hard one.

---

## 9. Risks

**R1 (MAJOR) — Oracle horizon sweep may not find improvement.** The current `oracle_value_horizon=400` was set in TASK_719a (April 2026) but has never been re-swept. It is *implicitly* tuned by the fact that it equals the episode length, which makes it the maximum useful horizon. Both endpoints of `{200, 400, 800}` are likely worse than the center. **Mitigation**: TASK_756b is a *secondary* probe; don't lead with it. If primary mixture variant lands ≥ +1pp vs 746, the horizon sweep is moot (we already have a winner in the family). Run 756b only if mixture lands flat or negative AND the campaign needs more signal in the family before retiring.

**R2 (MODERATE) — Mixture α optimisation is a continuous sweep.** Three α points may not be enough to find the peak. If the verdict at α=0.5 is +0.5pp vs 746 and at α=0.7 is −0.2pp vs 746, the true peak might be at α=0.6 or α=0.8 — but we don't have those seeds. **Mitigation**: pre-register the 3-α screening as the budget cap. If results are *strongly* monotone in α (consistent sign), extrapolate. If results are non-monotone, refine the most-promising α with 2 additional seeds (cheap — one screen at the best α). Do NOT pursue a 5-point or 7-point sweep — diminishing returns and budget pressure from Bella-side tasks.

**R3 (MODERATE) — Smoothed oracle may attenuate the +2.23pp signal.** The per-window peakiness of oracle PnL is the *information content* — EWMA smoothing damps it. If 756c is run and produces Δα < 0 vs 746 for all λ, that's actually informative: it confirms that the signal driving +2.23pp lives in the per-window variance, not in the lower-frequency component. Document this as a positive (mechanism understanding) rather than a regression. **Mitigation**: don't run 756c unless primary fails to promote and we want one more data point before closing the family.

**R4 (MAJOR — strategic) — 746's +2.23pp may be a statistical fluke.** Extending value-target probes after a marginal-but-positive result is *doubling down*. The base-rate of "marginal positive at 3 seeds becomes flat/negative at 5+ seeds" on this stack is non-trivial: 734_flow_matching (3-seed +3.38pp held at 5 seeds), 737_no_aux_solo (single-seed +6.21pp collapsed to +3.61pp at 5 seeds), 746 itself (3-seed +2.54pp pulled to +2.23pp at 5-seed proxy). Of these, 737 was an *outlier* that reverted; 746 was *stable*. We are betting on 746's stability. **Mitigation**: pre-register the falsification condition — *if 756 also lands in [+1, +3]pp on the paired-3-seed gate, the value-target roadmap is genuinely closed*. Two consecutive runs with marginal-positive Δα in the family, with no path to the +3pp promotion bar, is sufficient evidence to retire. Document this in `runs/756_value_mixture/result.md` at write-time, even if not triggered.

**R5 (MINOR) — Static α may be suboptimal.** A schedule (e.g. high α early, anneal toward low α — "trust oracle initially, taper to bootstrap") might outperform any static α. Out of scope for this task; if 756 lands meaningfully positive, TASK_756d schedules become Tier-3.

**R6 (MINOR) — Memory hierarchy: oracle horizons.** `oracle_cum_pnl` is computed once per slot at `_make_slot` time (`train_easy.py:762`) and stays in RAM for the slot's lifetime. Different horizons read different *indices* into the same precomputed array — no recompute, no disk touch. EWMA smoothing for 756c reads `oracle_cum_pnl[d:d+h]` (already in RAM) and computes a weighted sum in numpy at batch-build time (CPU, ~1 MB scratch buffer per batch). Total memory hierarchy impact: zero new disk reads, no VRAM increase, marginal CPU/RAM cost on the producer thread.

---

## 10. Out of scope

Explicitly forbidden in this task:

- **DO NOT modify `path_solver.py`**. The oracle path computation is fixed (`mcts_trader/jax_agent/path_solver.py:39-150`). Any variant that requires a different oracle path (e.g. soft-rank oracle, dense oracle, MCTS-based oracle) is a *separate* task.
- **DO NOT add new value-head architecture**. The PopArt-normalised 401-bin value head is fixed (`mcts_trader/CLAUDE.md` — "Value head: PopArt mandatory, 401 bins over [-10, +10]"). Variants that need a different value head are separate tasks.
- **DO NOT touch MCTS search budget**. `factored_temperature`, `num_simulations`, `max_depth` — all fixed at 746/741_mint baseline values. The change is to the value target, not the search.
- **DO NOT stack** with Bella-side mechanisms (753/754). Mixture is a value-target reformulation; Bella is an encoder-side auxiliary. They can be stacked as Tier-3 follow-up *after* both individually verify, not in this task.
- **DO NOT stack** with Titans-MAG (749) yet. Titans is in flight; if Titans promotes, the right move is to baseline 756 against Titans-positive a1b1, not to stack. Document the conditional in the SPEC at queue-placement time.
- **DO NOT pre-train Bella on different paths**. Out of scope — Bella's signal is not what this task is about.
- **DO NOT introduce a schedule** on α (mixture annealing). Static α only. TASK_756d if needed.

---

## 11. Success criteria

- [ ] Code success: 7a unit smoke produces value_loss diffs ≤ 1e-6 in both reproducibility limits (α=1 matches oracle, α=0 matches nstep). 7b production smoke completes all 4 pass criteria.
- [ ] Screening success (paired 3-seed vs 746): some α produces mean Δα ≥ +1.5pp with ≥ 2/3 seeds positive AND worst seed ≥ −2pp.
- [ ] Promotion success (5-seed vs 746, optionally vs 752 once it lands): mean Δα ≥ +1pp with no seed < −3pp. This maps to ≥ +3.23pp vs a1b1 — clears the +3pp promotion bar.
- [ ] Negative-result success: if screening rejects, `runs/756_value_mixture/result.md` documents the per-α split AND `memory/project_value_target_roadmap_closed.md` is written, retiring the value-target mechanism class with the falsification pre-registration as the supporting evidence.

---

## 12. References

- `runs/746_value_oracle/result.md` — 5-seed all-positive, +2.23pp vs proxy a1b1 baseline, the foundation for this task.
- `runs/747_value_td0/eval_td0_seed_{7,13,23}/` — flat td0 counter-example (the result.md was not written; verdict was internal — see TASK_754_REVIEW backlog).
- `experiments/746_value_oracle/SPEC.md` — template for this SPEC's protocol.
- `experiments/752_value_oracle_baseline_pair/SPEC.md` — pending re-pair at s31/s42; converts 746 proxy Δα to true paired Δα.
- `tasks/TASK_730_RANK_ORACLE.md` — oracle infrastructure background.
- `tasks/TASK_749_TITANS_PIVOT.md` — sibling planning-side experiment; structural template for this spec.
- `tasks/TASK_754_BELLA_TRAJECTORY_SHAPE_DISTILL.md` — sibling encoder-side experiment; provides §7 production-smoke template.
- `mcts_trader/jax_agent/path_solver.py:39-150` — `greedy_lookahead_path`; produces the `cumulative_pnl` array this task consumes.
- `train_easy.py:595, 725, 762-766` — oracle_cum_pnl precompute and slot storage.
- `train_phase2.py:619-633` — batch-build read sites for oracle_values (621-624) and nstep_rewards (626-633).
- `train_phase2.py:1094-1126` — value-loss branch site; insertion point for the `mixture` branch.
- `train_phase2.py:2094-2100, 2406-2411` — CLI flag block + args translation.
- `phase2_config.py:160-164` — `oracle_value_horizon=400`, `nstep_horizon=20`, `value_target_mode="oracle"` defaults.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — load-bearing for the hypothesis (planning-side vs classifier-side mechanism class).
- `experiments/QUEUE.md` — register the SPEC under Priority 2 once smoke passes (assuming Titans-MAG verdict is in).

---

## Operator notes

- The mixture target re-uses both `oracle_values` and `nstep_rewards` batch fields — no precompute change, no slot-state change. The 9-run screening (3 seeds × 3 α) is implementable from a single shared baseline and three SPECs that differ only in `--value-mixture-alpha`.
- The α=1.0 limit must reproduce 746 byte-identical. Smoke 7a is the gate for this; do NOT proceed to production runs unless the reproducibility limit holds.
- Falsification commitment (§2, §9 R4): if 756 mixture lands in [+1, +3]pp absolute Δα vs a1b1, write `memory/project_value_target_roadmap_closed.md` retiring the family. Two consecutive marginal-positive runs without crossing +3pp is sufficient.
- Memory hierarchy compliance: all variants read from RAM-resident `oracle_cum_pnl`; no new disk I/O. The mixture variant adds zero VRAM (reuses `out_tN`). The smoothed variant adds ~1 MB CPU scratch per batch. Within budget.
