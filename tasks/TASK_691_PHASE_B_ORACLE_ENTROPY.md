# TASK_691: Phase B — Oracle Entropy (Train→Inference Gap)

**Status**: DONE — code implemented, deployed, ran 790 epochs. Currently disabled in config (`oracle_entropy: False`) pending TASK_692.
**Priority**: — (superseded by TASK_692)
**Depends on**: TASK_689 Phase A (DONE), temporal leak fix (DONE)

## What Was Implemented

All code merged and working. No bugs in the implementation itself.

### Code

- `corrupt_oracle_selections()` at `train_easy.py:135` — trade drops + phantom trades via Bernoulli masking
- Wired into `_generate_episode_data()` (line ~588) and `_generate_batch_episodes()` (line ~812)
- Progressive warmup schedule via `_get_entropy_schedule()`: clean epochs 0-50, ramp over 150 epochs
- Config fields in `easy_config.py:76-82`: `oracle_entropy`, `oracle_drop_rate` (0.15), `oracle_phantom_rate` (0.05), `entropy_warmup_epochs` (50)

### Bug found and fixed during deployment

**Val entropy bug**: `val_generator._current_epoch` was never updated (stayed at 0) → entropy warmup never expired on validation → val always saw clean oracle state → inflated val metrics. Fixed at `train_easy.py:1581` — sets `val_generator._current_epoch = epoch` before `validate()`.

### Training run

Ran overnight epoch 430→1219 (~790 epochs, ~10 hours). No crashes, no instability.

| Metric | Pre-entropy (Phase A) | Post-entropy (691 active) |
|--------|----------------------|--------------------------|
| Train agreement | 90% | 83.6% (plateaued ~epoch 800) |
| Val agreement | 90% (inflated) | 83.6% (after val bug fix, converged to match train) |
| Loss | ~3.0 | 2.81 |

Train agreement dropped as expected (noise makes task harder). Train and val converged after val bug fix — generalization maintained.

### Current state

Config set to `oracle_entropy: False` (line 79) — disabled when TASK_692 config fields were added. Code remains in codebase, can be re-enabled if needed.

## Relationship to TASK_692

TASK_691 corrupts oracle selections with random noise before state channel computation. TASK_692 replaces oracle state entirely with model's own state via autoregressive rollout. These are different approaches to the same train-inference gap. TASK_692 is the more fundamental fix — see `tasks/TASK_692_FIX_MEMORY_THAT_CLAUDE_FUCKED_UP.md` for full analysis and 3-model review provenance.

## Files Modified

| File | Change |
|------|--------|
| `train_easy.py:135-149` | `corrupt_oracle_selections()` function |
| `train_easy.py:~588` | Wired into `_generate_episode_data()` |
| `train_easy.py:~812` | Wired into `_generate_batch_episodes()` |
| `train_easy.py:1581` | Val entropy bug fix (`val_generator._current_epoch = epoch`) |
| `easy_config.py:76-82` | Config fields (currently disabled) |

## Completion Criteria

- [x] `corrupt_oracle_selections()` implemented and tested
- [x] Noise schedule (warmup → ramp → stable) working
- [x] Config flags added (`oracle_entropy`, rates, warmup)
- [x] Wired into both episode generators
- [x] Training shows expected agreement drop (90% → 83.6%)
- [x] No train-val gap explosion (converged after val bug fix)

---

## Original Design (reference)

### Two noise modes

**1. Trade drops** (oracle says trade → force hold):
- Simulates: model fails to enter a position
- Probability: `p_drop` ramps to 0.15

**2. Phantom trades** (oracle says hold → force trade):
- Simulates: model enters a position it shouldn't
- Probability: `q_phantom` ramps to 0.05

### Data flow

```
oracle_selections → corrupt_oracle_selections → noisy_selections
noisy_selections  → compute_target_position → build_state_channels → state[t-1] → obs
oracle_selections → oracle_policy → policy_targets (UNCHANGED — targets stay clean)
```

---

*Implemented: 2026-02-21*
*Deployed: 2026-02-21 (epoch 430+)*
*Disabled: 2026-02-22 (TASK_692 config added)*
