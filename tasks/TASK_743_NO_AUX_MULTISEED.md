# TASK_743: Multi-seed Validation of `no_aux_solo` (Rank-Only Stage-B Config)

**Status**: PROPOSED (2026-05-13). Direct follow-up to TASK_737 completion.
**Priority**: HIGHEST. Today's TASK_737 result has `no_aux_solo` at AVG Alpha +6.21% — beats the TASK_736 reference by ~2.3 pp. Single seed; needs replication before treating this as the new baseline. Single-seed variance on this setup is ~3-6 pp (see a1b1_repro vs TASK_736 ref drift of 5.9 pp same config).

**Depends on**: `--seed` CLI flag in `train_phase2.py` (added 2026-05-13).

---

## The Question

Does the rank-only Stage-B config (`--rank-loss mse --rank-weight 1.0 --no-aux-price-head`) reliably outperform the TASK_736 baseline (rank ON + aux ON) across multiple seeds, or was today's +6.21% Alpha a lucky draw?

Specifically:
1. Is AVG Alpha for no_aux_solo > 0 across 3-5 seeds? (i.e., consistently profitable vs buy-and-hold)
2. Is it AVG Alpha > a1b1_repro's AVG across the same seed pairs? (i.e., the head-interaction story holds)
3. What's the standard deviation? Tight (<2 pp) → solid; wide (>5 pp) → unreliable.

---

## The Change

**Zero code change** required (after `--seed` CLI flag landing). All variation is at the command line.

---

## The Runs

Identical config to the existing `no_aux_solo` overnight run, varying only `--seed`. Reference seed is 42 (default, the no_aux_solo result we already have).

| Run tag | Seed | Output dir |
|---|---|---|
| `no_aux_s42` | 42 (already done — `runs/737_ablations/no_aux_solo/`) | re-tag |
| `no_aux_s7`  | 7  | `runs/743_no_aux_multiseed/seed_7/` |
| `no_aux_s13` | 13 | `runs/743_no_aux_multiseed/seed_13/` |
| `no_aux_s23` | 23 | `runs/743_no_aux_multiseed/seed_23/` |
| `no_aux_s31` | 31 | `runs/743_no_aux_multiseed/seed_31/` |

Five seeds total (including the already-run 42). Four new runs.

Optional companion: matching a1b1 seeds for a paired comparison.

| Run tag | Seed | Output dir |
|---|---|---|
| `a1b1_s7`, etc. | 7, 13, 23, 31 | `runs/743_no_aux_multiseed/a1b1_seed_<n>/` |

This gives a clean a1b1 vs no_aux_solo paired comparison across 4 seeds, sharing the data partitions.

---

## Launch

Use `experiments/743_no_aux_multiseed/SPEC.md` as the operator contract. The
SPEC handles batching (2 concurrent per TASK_742 rule), staggering, and
benchmark eval rollup.

---

## Success criteria

- [ ] 4 new no_aux_solo seeds complete 200 epochs cleanly (no TASK_742 recurrence)
- [ ] All 4 seeds + the existing seed=42 are benchmark-eval'd
- [ ] AVG Alpha across 5 seeds reported with min/max/mean/std
- [ ] If a1b1 companion seeds are run: paired t-test or simple delta-vs-paired-seed
- [ ] Conclusion: is `no_aux_solo` the new baseline, or is the +6.21% a single-seed outlier?

---

## Result template

```
| Seed | Bal.Acc | Alpha | ModelPnL | Beats-random |
|---|---|---|---|---|
| 42 (orig) | 0.266 | +6.21% | -0.5% | 6/6 |
| 7   | … | … | … | … |
| 13  | … | … | … | … |
| 23  | … | … | … | … |
| 31  | … | … | … | … |
| AVG | … | … | … | … |
| STD | — | … | … | — |
```

If `a1b1` companion seeds were run:

```
| Seed | no_aux α | a1b1 α | Δ | sign |
|---|---|---|---|---|
| 7  | … | … | … | … |
| 13 | … | … | … | … |
| 23 | … | … | … | … |
| 31 | … | … | … | … |
| AVG Δ | … | | | |
```

---

## Notes

- `--seed` was added to `train_phase2.py` argparse on 2026-05-13 (commit
  TBD). Without it, every run uses `easy_config.py:19 seed=42`.
- Pool sampling, MCTS rng, optimizer init, and prefetch all consume from
  `config.seed`. A different seed value gives a different segment-sampling
  order, MCTS expansion order, head-init noise, etc. — full re-trial.
- Stage A is not re-run; all Phase-1 → Phase-2 warmstart loads from the
  same `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` regardless
  of seed.
