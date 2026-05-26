# TASK_784 — Smoothed-Oracle Value Target (EWMA on oracle_cum_pnl deltas)

**Status**: DRAFT — small code change + 3-λ sweep.
**Created**: 2026-05-25.
**Class**: Planning-side (value target). Successor cell to TASK_756c §4.3.
**Anti-correlation risk**: LOW (planning-side, value-target reformulation; same class
as 746 and TASK_777 mixture). Does NOT touch policy/classifier output.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs `runs/760_train_mcts_sims/`
a1b1 sims=32 baselines under eval-MCTS sims=16 joint, **no seed < −5pp**.
**Mechanism diagnostic gate**: STD across seeds at the best-λ cell must be
< 746's 5-seed STD 2.28pp. If it isn't, smoothing didn't reduce variance and
the mechanism is null on its own terms.
**Estimated wall**: ~1 h code + ~12-14 h experiment (9 cells × ~2.5 h, 2-wide).

## Hypothesis

746_value_oracle's 5-seed paired Δα is **+0.72pp** with **STD 2.28pp** across
seeds (`runs/746_value_oracle/result.md:160-184`). Per-seed dispersion ranges
from s7 **−3.99pp** to s23 **+8.69pp** — the variance signature dominates the
mean signal.

Hypothesis: this seed-fragility is driven by **per-window peakiness** of the
raw oracle PnL trajectory. EWMA over the per-step deltas of `oracle_cum_pnl`
dampens the peakiness without changing the cumulative reachable-return level
the value target encodes. Specifically:

- `delta[t] = oracle_cum_pnl[t+1] − oracle_cum_pnl[t]` — the per-step PnL of
  the path-solver greedy selection at step t.
- Raw `remaining[d] = oracle_cum_pnl[d+H] − oracle_cum_pnl[d]` =
  `sum(delta[d : d+H])`.
- Smoothed: replace `delta[t]` with `smoothed_delta[t] = (1−λ)·delta[t] +
  λ·smoothed_delta[t−1]`. The cumulative path drifts in the same direction
  but the per-window magnitudes are smoothed.

The remaining-PnL gather then operates on `smoothed_cum_pnl = oracle_cum_pnl[0]
+ cumsum(smoothed_delta)`. By construction, λ=0 reproduces the raw path
bit-identical (smoothed_delta == delta).

**Likely-null** per `lore/ship_path_recommendations_2026-05-25.md` §10 (ranked
EV "Low"). Listed for completeness because the code change is small (~15 LOC)
and the falsification cost is tightly bounded. The value-target mechanism class
is the only class with **any** positive paired signal on this stack (746 +0.72pp,
TASK_777 mixture in flight); a near-zero-cost variance probe is justified.

**Sister task**: TASK_777 (mixture α·oracle + (1−α)·nstep) is the primary
variance-reduction probe (paired against 746 baseline). TASK_784 (this) is the
**orthogonal** probe — it attacks per-window peakiness directly on the oracle
signal itself, not by blending in a different signal.

## Verified code state (worktree HEAD 79430e3)

### Oracle precompute / read site

`train_phase2.py:619-624` (per ship-path memo) — current code (verified
2026-05-25, lines have drifted to `:640-665`):

```python
            # Oracle remaining PnL (numpy)
            oracle_cum_pnl = slot['oracle_cum_pnl']
            horizon = self.config.oracle_value_horizon
            end_idx = np.minimum(decision_indices + horizon, len(oracle_cum_pnl) - 1)
            remaining = (oracle_cum_pnl[end_idx] - oracle_cum_pnl[decision_indices]).astype(np.float32)
```

This is the read site the EWMA hooks into. `oracle_cum_pnl` is the slot-resident
RAM array precomputed once at `_make_slot` time (`segment_pool.py:513`,
`train_easy.py:725`); shape `(T-1,)` per slot with `T ≈ 400`.

### Value-loss dispatch (downstream consumer)

`train_phase2.py:1257-1297` — the `value_target_mode == "oracle"` branch
consumes `oracle_values` from the batch dict:

```python
            if value_target_mode == "oracle":
                # TASK_719a: Oracle remaining PnL as value target (same as Phase 1)
                oracle_scaled = oracle_values * value_scale
                # TASK_769: F2 fix — gate the pre-PopArt clip behind value_clip_mode.
                if value_clip_mode == "pre_popart":
                    oracle_scaled = jnp.clip(oracle_scaled, -10.0, 10.0)
                td_target_norm = (oracle_scaled - popart_mean) / jnp.maximum(popart_std, 1e-8)
                td_target_norm = jnp.clip(td_target_norm, -10.0, 10.0)
                td_target_raw = oracle_scaled
```

Smoothing is applied **upstream** of this branch (in the numpy precompute
site, before the array enters the batch dict). The dispatch code itself does
NOT need to change — `oracle_values` is the smoothed-derived `remaining` when
λ > 0, and the raw `remaining` when λ = 0.

### Config field set

`phase2_config.py:206-220` — current state has `oracle_value_horizon: int =
400`, `nstep_horizon: int = 20`, `value_target_mode: str = "oracle"`,
`value_clip_mode: str = "pre_popart"`. A new field `oracle_smoothing: float
= 0.0` is added between `nstep_horizon` and `value_target_mode`.

## Code-change scope (~15 LOC, flag-gated, default OFF, canon bit-identical)

### Phase2Config field (`phase2_config.py`)

Add new field with `0.0` default after `nstep_horizon`. Default makes the
EWMA branch a no-op and the oracle precompute byte-identical to pre-TASK_784.

### CLI flag (`train_phase2_cli.py`)

Add `--oracle-smoothing FLOAT` next to the existing `--value-target-mode` /
`--td-discount` flags. Default `None` → preserve config default (0.0).
Validate `0.0 ≤ λ < 1.0` (λ = 1 is degenerate — smoothed_delta becomes
constant and remaining → 0).

### Argument translation (`train_phase2.py:main`)

Override `config.oracle_smoothing` only when CLI flag is provided; log the
active value at startup when λ > 0 (zero-cost canon path stays silent).

### EWMA at oracle precompute (`train_phase2.py:640-665`)

The EWMA recursion runs in numpy at batch-build time. The per-slot
`oracle_cum_pnl` is ~T = 400 floats — a tight Python loop over deltas is
<<1ms per slot, well below the batch-build wall (~tens of ms). No JAX/JIT
involvement — keeps the change off the hot-path JIT trace.

Recursion (canonical EWMA on first differences):
```
delta[t]          = oracle_cum_pnl[t+1] - oracle_cum_pnl[t]
smoothed_delta[0] = delta[0]
smoothed_delta[t] = (1-λ)·delta[t] + λ·smoothed_delta[t-1]
smoothed_cum_pnl  = oracle_cum_pnl[0] + cumsum(smoothed_delta)
remaining[d]      = smoothed_cum_pnl[d+H] - smoothed_cum_pnl[d]
```

Bit-identity guarantee: when `oracle_smoothing == 0.0` the if-branch is
skipped entirely (`cum_pnl_eff is oracle_cum_pnl`) and the subtraction-gather
uses the identical numpy array — byte-identical to the pre-784 path.

### Fee_penalty_bps F1 fix interaction

The F1 fix at `train_phase2.py:653-664` uses `len(oracle_cum_pnl) - 1` and
`slot['oracle_selections']`. Both are **independent** of smoothing —
selections are unaffected by the cum_pnl shape transform. With λ > 0 and
fee_penalty_bps > 0 (off-canon corner), the smoothed `remaining` is
adjusted by the unchanged `oracle_changes * extra_bps` correction. The
two mechanisms compose orthogonally. (Default `fee_penalty_bps = 0.0` →
F1 branch is skipped, no interaction.)

### n-step rewards (unaffected)

`train_phase2.py:667-692` reads `oracle_cum_pnl` directly for the n-step
discounted sum. Smoothing applies **only to the oracle remaining-PnL gather**,
not to per-step nstep rewards. Rationale: nstep already gets its variance
reduction from the discount factor (γ^N decay); smoothing it again would
double-shape the target. Keep the cell hypothesis tight — change only the
oracle value target.

## Variants — 3 λ values × 3 seeds = 9 cells

| cell | oracle_smoothing | seed | output dir |
|---|---:|---:|---|
| s784_l01_s7  | 0.1 | 7  | `runs/784_smoothed_oracle/l01_seed_7`  |
| s784_l01_s13 | 0.1 | 13 | `runs/784_smoothed_oracle/l01_seed_13` |
| s784_l01_s23 | 0.1 | 23 | `runs/784_smoothed_oracle/l01_seed_23` |
| s784_l03_s7  | 0.3 | 7  | `runs/784_smoothed_oracle/l03_seed_7`  |
| s784_l03_s13 | 0.3 | 13 | `runs/784_smoothed_oracle/l03_seed_13` |
| s784_l03_s23 | 0.3 | 23 | `runs/784_smoothed_oracle/l03_seed_23` |
| s784_l05_s7  | 0.5 | 7  | `runs/784_smoothed_oracle/l05_seed_7`  |
| s784_l05_s13 | 0.5 | 13 | `runs/784_smoothed_oracle/l05_seed_13` |
| s784_l05_s23 | 0.5 | 23 | `runs/784_smoothed_oracle/l05_seed_23` |

λ-grid rationale: 0.1 (mild damping, ~10× horizon attenuation),
0.3 (moderate), 0.5 (strong damping). λ = 0 reproduces 746 exactly (use
`runs/746_value_oracle/oracle_seed_{7,13,23}/` as the smoothing=0 reference
when reading the paired Δα). We do NOT sweep λ > 0.5: the analytical EWMA
horizon `≈ 1/(1−λ)` reaches 2 at λ=0.5; pushing further attenuates the
signal more than the noise and risks collapsing into a trivial low-pass.

## Promotion gate

- **Paired Δα ≥ +3pp** vs `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}`
  (canon a1b1 sims=32 baselines) under `tools/benchmark_easy.py --eval-mcts-sims 16
  --eval-mcts-decode joint`.
- **No seed < −5pp** at the best-λ cell.
- **Mechanism diagnostic gate** (additional): STD across the three seeds at
  the best-λ cell must be < 746's 5-seed STD 2.28pp. If it isn't,
  smoothing didn't reduce variance — the mechanism is null on its own terms
  even before considering the mean.

If all 9 cells fail the gate, write `lore/notes_YYYY-MM-DD.md` closing the
"oracle-side variance reduction" sub-class together with TASK_777's verdict.

## Cost

- **Code**: ~15 LOC across 3 files (already shipped this commit). ~30 min
  smoke + bit-identity verification.
- **Experiment**: 9 cells × ~2.5 h Stage-B = ~22.5 h serial, **~12-14 h wall
  at 2-wide on server** (TASK_742 rule ≤2 concurrent `train_phase2.py` per
  host). Plus ~30 min eval per cell on GPU 3 (overlapped).
- **Total wall**: ~12-14 h on server.

## Anti-correlation check

LOW. Same mechanism class as 746 and TASK_777 mixture: changes **what the
model believes is reachable** (the value target), not **how confidently it
predicts the next bar** (classifier output). The
`project_classifier_alpha_anti_correlation.md` finding (738/740/741 → "loss
DOWN, Bal.Acc UP → Alpha collapse") does NOT apply: smoothing the oracle
value target should leave Bal.Acc broadly unchanged (the BC/aux/rank gradients
are unaffected). If a cell shows ΔBal.Acc > 0 AND Δα < 0 anyway, abort and
treat as evidence of a previously-unmodeled coupling pathway between the
value head and the classifier output.

## References

- `lore/ship_path_recommendations_2026-05-25.md` §10 — primary spec source.
- `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.3 — smoothed-oracle
  design (TASK_756c stub; this task is the executable form).
- `runs/746_value_oracle/result.md` — +0.72pp paired baseline with STD
  2.28pp; the variance-reduction target.
- `runs/760_train_mcts_sims/` — canon a1b1 sims=32 paired baselines for
  eval-MCTS sims=16 joint scorecard.
- `train_phase2.py:640-665` (post-edit), `:1257-1297` — oracle precompute +
  value-loss dispatch.
- `phase2_config.py:206-225` (post-edit) — `oracle_smoothing` field.
- `train_phase2_cli.py:252-272` (post-edit) — `--oracle-smoothing` flag.
- `experiments/784_smoothed_oracle/SPEC.md` — launch SPEC.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` —
  campaign context.

## Implementation log (2026-05-25)

- Worktree: `agent-ad1e54ad52f9aeb0f`, branch
  `worktree-agent-ad1e54ad52f9aeb0f`.
- Files touched: `phase2_config.py` (+17 LOC field), `train_phase2_cli.py`
  (+9 LOC flag), `train_phase2.py` (+38 LOC EWMA precompute + arg translation).
- EWMA placement: oracle precompute at `train_phase2.py:640-665`
  (recommended in the task brief and preserves the value-loss dispatch
  bit-identical when smoothing applied to `oracle_values` upstream).
- Smoke: static Python trace + numpy bit-identity verified at λ=0
  (`remaining` array byte-identical; `cum_pnl_eff is oracle_cum_pnl`).
  λ > 0 verified to reduce per-window std monotonically with λ (test:
  spike-heavy synthetic cum_pnl, std 28.2 → 25.8 as λ 0 → 0.9).
