# TASK_739: Dense Sign-of-Forward-Return Oracle (Tier A1)

**Status**: IMPLEMENTED (2026-04-23). Ready to launch — `bash tools/run_739_dense_sign.sh`.
**Priority**: PRIORITY 2 post-736 per `lore/notes_2026-04-22.md §49 line 870`.
**Rationale**: `lore/notes_2026-04-22.md §46 line 558` (Tier A1 definition) + §49 line 870 (priority call).
**Blocks**: nothing directly; next pivot in line after TASK_737 + (optionally) TASK_740.

---

## The Problem This Addresses

The current oracle `greedy_lookahead_path(k=4, lookahead=20)` has a structural pathology:
- 93.75% HOLD labels by construction (60/64 assets get HOLD every step).
- Only 4 assets per step receive trade labels.
- BC gradient fits the HOLD marginal; per-asset directional conditional is never densely supervised.
- TASK_736 confirmed the ExIt loop can close (value_weight fix), but `dir_acc` still caps at 0.574 — consistent with the per-asset directional signal never having been densely taught.

A1 reframes the oracle target: **every asset, every step, gets a directional label** derived from the sign of forward log-return with an ε dead-zone for low-magnitude moves. HOLD becomes a learned class (not a marginal artifact), and the effective label count jumps from ~4 → 64 per step.

The existing `DirectionHead` (TASK_727) already has this shape per-asset — it has been trained against a weaker signal all along. A1 makes the main BC target match what the direction head already predicts.

---

## The Change

### Target definition
For each `(t, asset)`:
```
r_fwd = log(p_{t+20} / p_t)
label = BUY  if r_fwd > +ε
      = SELL if r_fwd < -ε
      = HOLD otherwise
```
ε is tuned to produce trade density in the 3–10% range per asset (matching the TASK_736 model's empirical trade density of ~70% — will need recalibration).

### Oracle integration
Replace (or append to) the top-k trade mask in `mcts_trader/jax_agent/path_solver.py`:
- Add `dense_sign_oracle(prices, lookahead, epsilon)` returning `(B, T, n_assets)` in {BUY, SELL, HOLD}.
- `train_phase2.py` replaces the 4-of-64 selection mask with per-asset cross-entropy against this dense target.
- The oracle PnL calculation (`path_solver.py`) is unaffected — that still uses the top-k path for ExIt value estimation. Only the *BC training target* changes.

### CLI flag
```
--oracle-mode {topk_path, dense_sign, hybrid}
--sign-epsilon 1e-4     # magnitude threshold for dead-zone
```

`hybrid` (stretch) mixes both: top-k drives the value signal, dense-sign drives BC.

---

## Files To Modify (estimated)

| File | Change | LoC |
|---|---|---|
| `mcts_trader/jax_agent/path_solver.py` | New `dense_sign_oracle` function. Pure JAX. Reuses existing forward-price windowing. | ~40 |
| `train_phase2.py` | Route BC target through new oracle when `--oracle-mode=dense_sign`. Preserve `--oracle-mode=topk_path` as default. | ~30 |
| `phase2_config.py` | `oracle_mode: str = "topk_path"`, `sign_epsilon: float = 1e-4`. | ~5 |
| Possibly `easy_config.py` / `train_easy.py` | If we want Phase 1 to also use the dense oracle (recommended — match student and teacher targets). | ~40 |

Rough total: **~100-120 LoC + 2 hr training run for a first check.**

---

## Success Criteria

Compare against TASK_736 baseline (dir_acc 0.574, Alpha +3.87%, 6/6 beats-random).

### E20 — early signal
- `dir_acc > 0.60` → oracle density is doing its job. Run to E100.
- `dir_acc 0.58–0.60` → marginal. Continue but flag for re-evaluation.
- `dir_acc ≤ 0.57` → dense oracle didn't break the ceiling on its own. Consider stacking with B1 (drop BC) or pivoting to TASK_740 (Bella distillation).

### Benchmark
- Bal.Acc > 40% AND Alpha > +5% → **first genuine ceiling-break in project history**. Ship-candidate.
- Bal.Acc 30–40%, Alpha +3–5% → improvement. Stack with TASK_740 or Tier B1.
- No improvement → ceiling is upstream of label density. Pivot to Tier D (richer inputs) or Path γ (TASK_722).

---

## Why Now (Ordering Rationale)

1. **Cheapest ceiling-break attempt remaining.** ~100 LoC, 2 hr training. A/B against existing benchmark baseline is clean.
2. **Diagnostic value:** if dense sign oracle doesn't help, it strongly suggests the 0.57 ceiling is information-bound (60-step OHLCV simply doesn't carry enough signal), which elevates TASK_740 / Tier D2 urgency.
3. **Sequencing:** should run *after* TASK_737 so the contribution of existing aux heads is known — otherwise results are ambiguous. Can run concurrently with (or before) TASK_740 if the 737 decision matrix permits.
4. **Low code risk:** doesn't touch encoder, MCTS, value head, or rank head. Pure BC target change.

---

## Known Risks

- **ε calibration**: too low → everything becomes BUY/SELL → model learns a two-class problem → HOLD disappears. Too high → labels stay mostly HOLD → we reinvent the sparse oracle. Calibrate ε so per-asset trade density matches desired range (target: 5–15% per asset).
- **Label noise at 20-step horizon**: at crypto-minute scale, 20-step forward return has high variance; sign flips frequently. Dense labels may be noisier than TASK_727 direction_head's target and fight it. Mitigation: match the direction_head horizon exactly so they train in the same direction.
- **BC collapse on HOLD**: if HOLD is still the majority class (~85-95% depending on ε), class-balanced BC from TASK_726 will need re-tuning of the trade-weight coefficient.
- **Interaction with rank_head**: rank_head predicts ordering; dense-sign predicts direction. They're complementary but may push the shared encoder in conflicting ways on high-variance assets. Monitor per-head loss trajectories in first 20 epochs.

---

## Out of Scope (explicit non-goals)

- **Value head / MCTS**: unchanged. TASK_736's value-weight fix stays.
- **Encoder architecture**: unchanged.
- **Oracle for value targets**: still uses top-k path + n-step returns. Dense-sign is BC-only.
- **Multi-horizon consensus (A3)** or **tx-cost-adjusted (A2)**: separate tasks if A1 shows promise.

---

## Dependencies

- TASK_736 baseline (value-weight 0.5 on).
- TASK_737 suite complete, so we know which aux heads are load-bearing before stacking.
- 32-ch Phase 1 checkpoint (`jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`) for warmstart. If TASK_737 legacy_27ch wins, re-baseline on 27-ch.

---

## Implementation notes (2026-04-23)

- Oracle: `mcts_trader/jax_agent/path_solver.py::dense_sign_oracle` (+ helper
  `dense_sign_labels_to_policy`). Operates on log-returns (matches existing
  convention); per-step `r_fwd = sum(returns[t : t+lookahead])`; boundary
  forces HOLD for last `lookahead-1` rows.
- CLI: `--oracle-mode {topk_path,dense_sign,hybrid}`, `--sign-epsilon <float>`.
  `hybrid` raises `NotImplementedError` (reserved for follow-up).
- Config: `phase2_config.Phase2Config.oracle_mode='topk_path'` (default, byte-identical to pre-739) and `sign_epsilon=1e-4`.
- Injection point: `train_easy.SegmentPool._make_slot` — replaces `policy_np`
  only; `oracle_selections` / `oracle_cum_pnl` (value + path) untouched.
- Launch: `tools/run_739_dense_sign.sh` (1 run + 1 benchmark → `task_739_result.md`).
