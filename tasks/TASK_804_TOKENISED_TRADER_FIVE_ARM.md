# TASK_804 — Tokenised-trader verdict: five-arm real-vs-random-vs-raw experiment

**Status**: ✅ **VERDICT — NULL (no promote), 2026-06-19.** Branch C moves no deployment Alpha on
perception OR planning. real_fe Δα −0.12pp, real_dyn Δα +0.12pp vs raw (both ≪ the +3pp promote bar).
Full scorecard + reasoning in the **RESULT** section at the bottom. This is the [[TASK_796]] item-11
downstream made concrete and the branch-C Alpha verdict — supersedes the gating sketch in [[TASK_794]].
Verdict commit `1ec61ad`; lore `notes_2026-06-19.md` §211–212; [[project_candle_language]] updated.

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
| 2 | `real_fe` | real tokens | [[TASK_801]] front-end | tokens as input help? |
| 3 | `shift_fe` | time-decoupled real tokens (cyclic shift) | [[TASK_801]] front-end | capacity/structured-side-channel placebo |
| 4 | `real_dyn` | real tokens | front-end + [[TASK_802]] dynamics (+ [[TASK_803]] P2 if margin shown) | planning-side value |
| 5 | `shift_dyn` | time-decoupled tokens | same as 4 | placebo for arm 4 |

- **Null arm REDEFINED 2026-06-12 (supersedes "random codebook")**: a global bijective id
  permutation applied consistently to train+eval is information-theoretically invisible to a
  from-scratch embedding (rows are i.i.d.-initialized; the relabeled dataset is isomorphic) —
  Δ(real, permuted) ≡ 0 by construction, so that control cannot fire the kill rule meaningfully.
  Encoding with a permuted vocab is no better: BPE leaf ids bypass the vocab and merge-rank order
  changes segmentation (B2 measured 16/3329 divergence) — a *different tokenizer*, not a null.
  **The operative null is a per-asset CYCLIC SHIFT of the real timeline**: `rolled[i] =
  real[(i − Δ_a) mod N]` (token_id and valid roll together, pre-join), Δ_a seeded per (asset,
  seed 804), large, persisted + sha'd in `experiments/804/`. Preserves frequency multiset,
  run-lengths, and grammar; destroys only alignment with the concurrent price window. real >
  shift ⇒ aligned token content carries signal beyond capacity/structure. Same hot path, no
  model branch; derived cache keyed by its own sha.
- **Language ([[TASK_800]] verdict, 2026-06-10)**: **L1 BPE primary** (SERVABLE-WITH-LAG-1, served
  with the 2-bar final-only delay; the only language with a measured grammar gap, [[TASK_799]]).
  **L8 Haar reserve** (SERVABLE exact, but no grammar probe yet). L2/L5 were not built (memo only).
  One language per run — no multilingual ([[TASK_793]] stays deferred). Main thread may override
  primary if Step-1 champion probes change the grammar picture.
- **Canonical frozen L1 vocab (PINNED 2026-06-11)**:
  `runs/799_grammar/L1_RESULT_vocab.json` (server copy; schema `candle_lang.l1_bpe.tokenizer` v1,
  sha256 `c247c907…6ebce`, codebook 1376 = 864 leaf + 512 merges). Rationale: it is the vocab the
  [[TASK_799]] grammar PASS was measured on, the only *persisted* frozen artifact (the [[TASK_800]]
  audit re-fit in-process and never saved its vocab), and the one [[TASK_803]]-P1 already serves.
  Copy it into the experiment dir and record the sha in each arm's config.
  **✅ Lag re-certification DONE 2026-06-11 — and it OVERTURNED LAG-1**: frozen vocab re-certs as
  LAG-2 @ p99 (hist {0:960, 1:22, 2:11, 3:3, 4:3, 6:1}, max 6, uncensored). Decision (reviewer):
  **`LANGUAGE_INPUT_LAG["L1"] = 3`** (p99-firm convention; max_lag+1 rule retired as
  sample-unstable). Residual ~0.7% measured + symmetric; within this experiment train/eval skew is
  exactly 0 (C3). B2/C2/C3 must be re-run under lag 3 before arms launch.
- **Arm-launch gates (all three, in any order)**: (1) B2/C2/C3 green under lag 3; (2) pool-level
  per-asset timeline cache (C4-at-scale: RAM cache keyed by (asset_id, vocab_sha), annotate each
  asset once per pool, optional one-time disk artifact keyed by vocab sha — sanctioned
  load-once-to-RAM pattern); (3) TASK_805 server-state reconciliation, then deploy from a known
  commit.

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

## Deployment ladder after promotion (gates, strictly sequential — no shortcuts to live)

1. Positive trading on historical data across **all regimes** (all 17 SegmentPool regimes,
   not just canonical benchmark periods).
2. Positive on **new/unseen historical data** (out-of-sample periods).
3. **Paper trading** (live feed, simulated execution). Hard prerequisite at this gate: the
   **incremental/streaming token annotator** (live path is bar-at-a-time; windowed
   `annotate_ohlcv` does not serve it). Build it before this gate, not before live.
4. **Live trading** (`real_trade_jax.py`).

---

## RESULT — branch C is NULL on both surfaces (2026-06-19)

Ran 5 arms × 3 seeds (7/13/23), L1-BPE frozen vocab (`runs/799_grammar/L1_RESULT_vocab.json`, content
vocab_sha `d6a718dd…`), cyclic-shift null (`shift_null_spec.json`, SHIFT_SEED 804, sha `b55f18f6…`),
warm-start a1b1 Phase-1 imprint, eval-MCTS sims=16 joint. Source: `runs/804_token_arms/eval_*/benchmark.json`
→ `summary.avg_alpha`.

| arm | s7 | s13 | s23 | mean | Δα vs raw |
|-----|------|------|------|------|-----------|
| `raw` (canon a1b1)        | 0.0674 | 0.0573 | 0.0671 | 0.0640 | — |
| `real_fe` (perception)    | 0.0641 | 0.0558 | 0.0684 | 0.0628 | **−0.12pp** |
| `shift_fe` (perception null) | 0.0768 | 0.0605 | 0.0704 | 0.0692 | +0.53pp |
| `real_dyn` (planning, +TASK_802) | 0.0641 | 0.0655 | 0.0658 | 0.0651 | **+0.12pp** |
| `shift_dyn` (planning null)  | 0.0708 | 0.0580 | n/a | — | — |

Per-seed `real_dyn` vs `shift_dyn`: s7 shift-wins (0.0708 > 0.0641); s13 real-wins (0.0655 > 0.0580,
+0.75pp); s23 control unrecoverable. Per-seed `real_dyn` Δα vs raw: s7 −0.33, s13 +0.82, s23 −0.14pp
(mean +0.12pp, no seed < −5pp).

**Verdict (per the Protocol rules above):** the promote rule is arm4−arm1 ≥ +3pp AND arm4 > arm5 —
`real_dyn`'s **+0.12pp** misses the +3pp bar by an order of magnitude → **NO PROMOTE**. Both real arms sit
on top of canon (`real_fe` −0.12pp, `real_dyn` +0.12pp); the shuffled controls do as well or better
(`shift_fe` +0.53pp; `shift_dyn` beats `real_dyn` in s7). This is the spec's "arms 2≈3≈4≈5≈1" outcome:
branch C is null like everything since [[TASK_738]] — write the result, regroup. The candle-language is
real as **structure** (grammar + cross-asset transfer, [[TASK_799]]/[[TASK_800]]) but null as a **trader**
mechanism on perception AND planning.

**Honest flutter (do not chase):** `real_dyn` (+0.12pp) edges `real_fe` (−0.12pp) by ~0.24pp and beat its
shift control in seed-13 — but it lost in seed-7, the seed-23 control is missing, and the whole spread is
inside the project's ≤5pp-single-seed-is-noise rule. Null-with-a-flutter, not weak-positive.

**Caveat — 14/15 cells.** `shift_dyn_seed23` never completed: 8 [[project_seed7_stageb_wedge]] failures
(3 on the 2-wide partition runner, 5 on a dedicated solo retry runner — frozen-log wedges + one clean exit
at epoch 0). The per-run watchdog + systemd-scope reap held (no orphans/GPU leak), but the Stage-B hang is
undiagnosed and is now the top reliability risk for any future paired campaign. The missing cell only
completes the seed-23 real-vs-shift direction; it cannot change the locked no-promote.

**Runners/artifacts:** `experiments/804/orchestrate_token_arms.py` (3-arm), `experiments/804/dyn_solo.py`
(arms 4-5, GPU-2 self-contained), plus single-cell solo retry variants. TASK_802 dynamics code `09d134d`;
disk-cache fix `e02953e`; scope-reap fix `78ccb70`; verdict commit `1ec61ad`. What stands: the tokenizer +
grammar/lag cert, the **cyclic-shift null** (reusable content-preserving control), and the TASK_802
NextTokenHead (validated-but-null architectural option).
