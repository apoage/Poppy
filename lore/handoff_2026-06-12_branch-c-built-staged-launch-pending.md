# Handoff 2026-06-12 — branch-C tokenised trader: fully built + staged, launch is the next pass

**Previous:** [lore/handoff_2026-06-10_task800-signed-chain-unparks.md](handoff_2026-06-10_task800-signed-chain-unparks.md) (seq 5 → **6**)
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session focus:** Drove branch-C from the TASK_800 sign-off through 801 / 801b (Fork A) / the lag
re-cert / 803-P1 / TASK_805 / the full TASK_804 build-out (B1–B4 + B2.5) to **fully-built-and-staged**.
Deployed to the server (9c84315). **B3 cache pre-build + the orchestrator launch are deliberately a
fresh pass.** Full narrative arc: `lore/notes_2026-06-12.md` §196–202.

## Confirmed compaction rubric

```
KEEP IN FULL:
- STATE: server training deployed at 9c84315 (= finalized cac679e), idle. The FRESH PASS = B3 cache
  pre-build THEN orchestrator launch. B3 blocker (correctness): shift_null_spec per-asset offsets must
  cover the FULL foxglove arm asset universe (symbol-keyed, 59–64/period → DB ids), NOT the vocab's 9
  build_assets — wrong universe silently breaks the shift_fe control (TASK_740 class). B3 exit check =
  two cache shas (real vocab_sha vs shift derived_sha) distinct + deterministic.
- SERVING LAG: LANGUAGE_INPUT_LAG["L1"]=3 (p99-firm; re-cert OVERTURNED LAG-1→3 on the frozen
  512-merge vocab). input_lag_for propagates it; join + orchestrator inherit, no rework.
- CONTROL REDEFINITION: rand_fe (π-permutation) was a NON-NULL ("the control that couldn't fail",
  relabel-with-fresh-embedding ⇒ guaranteed to match treatment) → REPLACED by shift_fe (cyclic-shift
  null: roll real timeline pre-join, agreement≈Σp²). π archived dead-end. shift_fe EVERYWHERE.
- ORCHESTRATOR: real-mode SELF-ADVANCES (Popen under one tmux — the emit→spawn fix). Arms
  raw/real_fe/shift_fe × seeds 7/13/23, seed-major, 2-wide GPU 1/2, eval GPU 3, GPU 0 untouchable;
  warm-start jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl; COMMON_BASE finalized (no
  --mcts-sims ⇒ default 32 = a1b1); seed-7 watchdog + stop/extend encoded. dry-run 10/10.
- LAUNCH-PASS CHECKLIST: B3 (asset universe → build_shift_null --assets → real disk artifact in tmux →
  verify 2 shas) → orchestrator tmux real-mode → WATCH raw_s7 first spawn (Popen fires, GPU1 engages,
  log populates, watchdog quiet) → walk away → dispatch TASK_802 builder once seed-7 triple survives.

COMPRESS + POINT:
- The whole gate trail (801 model E1-E5 · 801b Fork A R1-R4 + forward-fill R2 + pool-cache C1-C5 · lag
  re-cert · TASK_805 reconciliation · B1-B4+B2.5 builds · the B2 reversal) → lore/notes_2026-06-12.md §196-202.
- 803-P1 CLOSED NO-MARGIN (gate inert on a1b1: model HOLDs 98.5%) → runs/803_gate/RESULT.md + TASK_803.

DROP:
- The ~1600-file dirty tree (TASK_785 _old purge + months-old standing M deltas = de facto canon,
  ship-as-is per the operator) — never git add -A.
- Per-build sub-agent dispatch/audit back-and-forth — keep only the commit SHAs + verdicts.
- Deploy mechanics deliberation (surgical prep, clean -fd hazard) — resolved; manifest discipline lives
  in runs/deploys/. ssh quoting fumbles, timeouts, per-poll progress checks.

VERBATIM ANCHORS:
- Build SHAs: ed623d3(B1) · 852802e+3eaa715(B4 spawn) · 99d4cf1(B2) · 125100d(B2.5) · e5ee455+cac679e(B4
  finalize/test) · 0c283d8(lag3) · 2e037cc(lore). Deploy: server 9c84315 (this session), pred 57d05ac;
  manifests runs/deploys/2026-06-{11,12}_deploy.md.
- LANGUAGE_INPUT_LAG["L1"]=3 (token_window_slice.py:39) · SHIFT_SEED=804 · vocab L1_RESULT sha
  c247c907…6ebce, cb 1376 · imprint xs32 sha cd5616a5…
- Promote: arm4−arm1 ≥+3pp, no seed<−5pp, AND arm4>arm5. Kill: any real arm mean<−3pp OR null≥real.

UNCERTAIN:
- The foxglove asset union (B3 blocker): resolve from the benchmark's symbol_to_id vs the DB asset
  table vs per-period union — NOT settled this session; the launch pass must nail it.
- Whether the real disk artifact has a standalone prebuild trigger — build_shift_null has no
  --cache-dir; no standalone entry found; a small prebuild script may be needed in the launch pass.
```

## Git state
```
2e037cc lore: branch-C pre-launch state (gate trail 801->804, the control-that-couldnt-fail reversal); TASK_805 spec
cac679e TASK_804-B4: sync dry-run test to the shift_fe rename (null_ge_real) — 10/10 green
e5ee455 TASK_804-B4: finalize build_train_cmd with 760 a1b1 COMMON_BASE (no --mcts-sims => default 32)
4a449df candle_lang: slice_token_window docstring lag 2->3 (p99-firm)
125100d TASK_804-B2.5: cyclic-shift null for shift_fe (roll real timeline pre-join, per-asset offsets seed 804)
```
Working tree: the large pre-existing dirty set (TASK_785 `_old/` purge + ~64 months-old standing M
deltas = de facto canon shipped by every deploy) is **known noise — never `git add -A`, only named
files.** All this session's durable artifacts are committed through `2e037cc`. Server `training` is at
`9c84315` (hash-verified ≡ workstation finalized code).

## Next action
**Fresh pass = the launch.** (1) B3: resolve the foxglove arm asset universe (symbols→DB ids, union),
`build_shift_null --assets <them> --seed 804`, build the real disk artifact in server tmux, verify the
two cache shas distinct+deterministic. (2) Orchestrator up in tmux real-mode; **watch raw_s7's first
spawn** before walking away. (3) Dispatch the TASK_802 dynamics builder once seed-7's paired triple
survives. On workstation boot: `udisksctl mount -b /dev/sdb1` (storage disk doesn't auto-mount → git).
