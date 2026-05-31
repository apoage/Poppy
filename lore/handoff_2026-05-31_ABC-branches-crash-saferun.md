# Handoff — 2026-05-31 (eve) — A/B/C experiment branches launched, workstation crash diagnosed, safe_run envelope built

**Sequence**: #7
**Previous**: `lore/handoff_2026-05-30_786-oom-fix-tmux-orchestration.md` (#6)
**Branch**: `master` (tip `47689ee`)
**Working dir**: `/home/lukas/Trader`

## What happened this session
Closed the 780/775 cycle (both null: 780 collapses on all 3 seeds — degenerate 50%-agree fixed
point; 775 oracle-mask paired Δα −0.66pp). Pivoted to three new experiment branches grounded in a
codebase survey: the supervised side is exhausted because **the trading objective was never in the
loss**, and the real **Poppy ExIt loop was never turned on**.

- **A — PnL-in-loss** (`exp/A-pnl-in-loss`, TASK_787): flag-gated over-trade penalty in Stage-B
  loss. Bit-identical when OFF (smoke verified). **RUNNING on server**: w=0.1 chain (G2: s7/s13/s23
  + a w=0.0 recipe-reproduction cell + evals) and w=0.2 arm (G1: s7/s13/s23 + evals), both paired
  vs 760 a1b1, eval-MCTS sims=16 joint. Verdicts → `runs/787_pnl_in_loss/result.md` (+ `_w020.md`),
  flags `/tmp/787_A_done.flag`, `/tmp/787_w020_done.flag`.
- **B — Poppy ExIt revival** (`exp/B-poppy-exit-loop`, TASK_788): the dormant `train_jax_fast.py`
  loop. **STABILITY CONFIRMED** — ran 100/100 epochs cold-start synthetic on the server WITHOUT
  collapsing (loss flat ~12.5, unlike the 780 bridge). De-risks B. Real version (warm-start a1b1 +
  regret-replay + sims-scaling) needs a pkl→Orbax adapter (a1b1 ckpt is a Foxglove pickle; loop
  wants Orbax dir) — not built. `runs/788_poppy_exit/coldstart_synth/train.log`.
- **C — Temporal token middle-layer** (TASK_789): researched + designed. Verdict: NOT raw-value
  quantization (Chronos walked it back) → **RevIN→patch→FSQ→Δ-IRIS GPT→stochastic-MCTS-over-tokens**
  (= Vector-Quantized-Models-for-Planning, the MuZero-with-VQ blueprint). Operator decisions LOCKED:
  **late mixing** (per-asset tokens; cross-asset attention at the decision head, over token
  embeddings — preserves the load-bearing cross-sectional signal, TASK_738 +18pp); **frozen vocab
  first**. C0 viability gate (does token edge beat a returns-shuffled control OOS) is the kill-switch.

## The crash (and the fix)
Workstation **fully crashed ~22:11** when the main agent launched a C0 build sub-agent doing heavy
JAX on the workstation's **single display-driving 3090**. Diagnosed: **GPU driver hard-hang, NOT
OOM** (128G swap unused, no oom-killer log) **and NOT CPU** (no MCE/thermal/panic; kernel log
dead-stopped). Server experiments untouched (separate machine, tmux). Two durable fixes shipped:
- **`tools/safe_run.sh`** (47689ee): every job in a `systemd-run --user` cgroup with `MemoryMax`/
  `MemorySwapMax=0` → runaway OOM-killed in its own cgroup, host survives (verified). `--cpu` keeps
  compute off the display GPU. Works on workstation + server. **Use it to wrap future launches.**
- Memory `feedback_no_gpu_jobs_on_workstation`: GPU compute → server only; CPU-only probes OK on
  workstation; encapsulate via safe_run.
C0 harness salvaged to master (`adf8c15`, `tools/tokenizer_c0/` — incl. a numpy/CPU fallback; the
dead agent's worktree is harness-locked, sweep it in TASK_785).

## State / git
```
47689ee tools/safe_run.sh: hard RAM-cap job envelope        <- master tip
adf8c15 TASK_789 C0 (salvage): FSQ tokenizer + gate harness (partial)
baf8527 Handoff #6 (TASK_786 780 OOM fix)
```
Working tree still has the big uncommitted mess (685 untracked / 867 deletions) — TASK_785 DRAFT
specs the cleanup; NOT done. Server tip `4879982` (deploy carrying A's flag + live Poppy tree).

## Next session — resume rubric
1. **Read server verdicts** (likely done by morning): `runs/787_pnl_in_loss/result.md` (w0.1:
   paired Δα + Δtrade-rate + ΔBal.Acc vs 760; **first check the w000_s7 reproduction row — must
   match 760 within ~1.5pp or pairing is confounded**) and `result_w020.md` (w0.2 dose-response).
   780A sanity gate `runs/780_oracle_as_mcts/sanity_gate_s*.csv` (expected FAIL — 780 dead).
   Flags: `/tmp/787_A_done.flag`, `/tmp/787_w020_done.flag`, `/tmp/780A_chain_done.flag`.
2. **A interpretation**: does the over-trade penalty raise Δα while lowering Δtrade-rate (breaking
   the anti-correlation)? That's the whole hypothesis. Dose-response across w=0.1/0.2.
3. **Resume C0** — CPU-only on the SERVER (survives workstation-off), wrapped in
   `tools/safe_run.sh --cpu --mem 16G`. NEVER on the workstation GPU. Finish the numpy gate, run
   RQ0 (token edge vs shuffled control, purge+embargo OOS), decode fidelity, codebook health.
4. **If A or B promising**: A → expand seeds/sweep; B → build pkl→Orbax adapter for warm-start.
5. Standing offers: execute **TASK_785** (commit the tree, 8G deploy bloat, dead worktree);
   **C1** (Δ-IRIS GPT) only after C0 proceeds.
6. `lore/notes_2026-05-30 §172` (786 + tmux-orchestrator) STILL unwritten; add §173 for this session
   (A/B/C + crash + safe_run) — Bloom-relevant (the crash + encapsulation is a good lab-notebook beat).
