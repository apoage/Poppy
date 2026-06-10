# TASK_804 — Tokenised-trader verdict: five-arm real-vs-random-vs-raw experiment

**Status**: 📋 SPEC (2026-06-09). **Blocked on [[TASK_800]] + [[TASK_801]]; arms 4-5 on [[TASK_802]]
(+ optionally [[TASK_803]] Phase 2).** This is the [[TASK_796]] item-11 downstream made concrete and
the branch-C Alpha verdict — supersedes the gating sketch in [[TASK_794]].

> **NOT a standard-model hand-off.** This task is experiment orchestration on the shared server
> (queue packing, baseline retraining, verdict calls) — it stays with the main thread / a strong
> model. 800-803 are the hand-off-grade implementation tasks; this file is their consumer.

## The question

> Does the token middle-layer move deployment Alpha — and if so, does its value live on the
> perception side (front-end) or the planning side (dynamics grounding)?

## Arms

| # | Arm | Tokens | Mechanisms | What it isolates |
|---|-----|--------|------------|------------------|
| 1 | `raw` | none | canon a1b1 | paired baseline (retrained, NOT reused — `lore/notes_2026-05-20.md` §151) |
| 2 | `real_fe` | real codebook | [[TASK_801]] front-end | tokens as input help? |
| 3 | `rand_fe` | frequency-matched random codebook | [[TASK_801]] front-end | capacity/embedding placebo |
| 4 | `real_dyn` | real codebook | front-end + [[TASK_802]] dynamics (+ [[TASK_803]] P2 if margin shown) | planning-side value |
| 5 | `rand_dyn` | random codebook | same as 4 | placebo for arm 4 |

- **Random codebook** = permutation of id assignment preserving the build frequency multiset —
  reuse the A8 null machinery (`eval_battery.py:498-516`), seed-fixed, identical embedding arch
  and E. Swapping real→random must touch only the id mapping ([[TASK_801]] E5).
- **Language**: the [[TASK_800]] SERVABLE winner (expected L1 and/or a window-rate member; L2/L5
  only if servable). One language per run — no multilingual ([[TASK_793]] stays deferred).

## Protocol (the canon, no shortcuts)

- **Seeds**: paired 7/13/23; extend per the QUEUE ladder (3-seed mean ≥ +1pp AND no seed < −3pp →
  add 31/42).
- **Training**: Phase-2 Stage-A→Stage-B warm-start from the canonical Phase-1 imprint, identical
  configs across arms except the flags; `--seed` CLI (added 2026-05-13).
- **Eval**: `tools/benchmark_easy.py`, **eval-MCTS sims=16 joint**, canonical benchmark periods,
  PopArt state required.
- **Promote** (branch-C green): arm 4 − arm 1 paired mean Δα **≥ +3pp, no seed < −5pp**, AND
  arm 4 > arm 5 (real beats random — the non-circular control). Per [[TASK_794]]: a clean Alpha
  tie PLUS a measured throughput/noise-filtering win counts as a win — but only with the
  real>random separation intact.
- **Diagnose** (the factorization payoff): arms 2-3 null but 4 > 5 with margin → token value is
  planning-side (the standing hypothesis since TASK_761) → next iteration goes deeper on dynamics
  grounding (and/or hosts it in the [[TASK_788]] revived ExIt loop). Arms 2 ≈ 3 ≈ 4 ≈ 5 ≈ 1 →
  branch C front-end is null like everything since 738; write the result.md and regroup.
- **Kill**: any real arm < −3pp mean, or random ≥ real consistently → tokens are dead weight on
  this stack; full stop, result.md, no rescue sweeps.
- Always: `result.md` mandatory; trade-rate, Bal.Acc, and per-seed Δα reported per arm (the
  anti-correlation diagnostics, [[project_classifier_alpha_anti_correlation]]).

## Run plan (gpu-server)

- 5 arms × 3 seeds = 15 Stage-B runs + evals. **≤2 concurrent `train_phase2.py` per host**
  (TASK_742 rule); seed-7 log-staleness watchdog ([[project_seed7_stageb_wedge]]). GPU 1/2 train
  2-wide, GPU 3 evals; GPU 0 untouchable.
- One self-advancing orchestrator in **tmux** ([[feedback_server_launches_in_tmux]]); `safe_run`
  RAM cap; PREALLOCATE default + `MEM_FRACTION=0.7` per process
  ([[feedback_server_vram_envelope]]).
- Token timelines pre-encoded once per arm to RAM (CPU, before the loop); zero tokenization
  in-loop.
- Sequencing: arms 1-3 can launch as soon as [[TASK_801]] passes acceptance; arms 4-5 follow
  [[TASK_802]]. Do not wait for [[TASK_803]] Phase 1 to start arms 1-3, but fold its verdict into
  arms 4-5 config if it lands in time.

## Order of work across the 800-series

```
800 timeline+audit (CPU, workstation OK)
 ├─ 801 front-end (code) ──────────► 804 arms 1-3 (server)
 ├─ 802 dynamics head (code) ─────► 804 arms 4-5 (server)
 └─ 803 P1 eval-only gate (server, existing ckpts — cheap early signal, parallel)
```

If 804 promotes: graduate to checkpoint-lineage training + the full evaluation cadence (back to
the canonical training/eval loop), then consider the [[TASK_788]] ExIt host. If it nulls: branch-C
trader verdict is written and the panel work stands on its own as measurement infrastructure.
