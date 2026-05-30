# Handoff — 2026-05-30 (eve) — TASK_786 ships (780 OOM real fix), 3 tmux orchestrators own the pipeline

**Sequence**: #6
**Previous**: `lore/handoff_2026-05-30_s7-wedge-780-unblock.md` (#5, 18:33Z — predates this session)
**Branch**: `master`
**Working dir**: `/home/lukas/Trader`
**Session focus**: Came in on `/status` ("inspect codebase + recent effort, draft way forward"),
overtook as workstation main agent. Found BOTH flagship bets stuck: 780 Phase-A OOM'd all 3
seeds (never trained 1 epoch), 775 had a hung s7 + all 3 evals failed (launched before ckpts
existed). Operator ran Phase 0 (kill hang, relaunch 775, boot 780 with `--bella` DROPPED).
**Discovered the `--bella` drop did NOT fix 780** — it re-OOM'd all 3 seeds at the identical
11.6 GiB. Built + shipped **TASK_786** (the real OOM fix), deployed, and **780 trained past E0
for the first time ever WITH `--bella` restored**. Built 3 self-advancing tmux orchestrators
that own the entire remaining pipeline. Workstation going down → all live state is server-side.

## Confirmed rubric

```
KEEP IN FULL:
- TASK_786 IS the real 780 OOM fix. The 11.6 GiB (12457304672 B) allocation was the single
  batched initial_inference on obs_flat=(B*(K+1),C,64,64)=(1536,...) at batch_size=256 — NOT
  the multipass tensor (~0.5 GiB). Dropping --bella only freed generator VRAM to fit the
  UNCHANGED peak; it re-OOM'd identically. Fix: bridge_trainer._encode_obs_chunked chunks the
  B*T encoder forward over its leading axis (lax.map + jax.checkpoint remat), peak → encode_chunk
  rows. Math-identical: smoke worst rel Δ=2.2e-05. Commit b8560dc. Deployed server tip 7575fac.
- 780 NOW TRAINS past E0 WITH --bella + chunked (MEM_FRACTION=0.9, GPU3 ~22 GB). First time.
- *** WATCH: 780A_s7 LOSS DIVERGED E2→E11 (Loss 8.3 → 151.7). *** Phase-A may FAIL its sanity
  gate. If all 3 seeds diverge, the bridge trainer needs a STABILITY fix (grad_clip / LR /
  value_scale / lower n_passes), not a memory fix. 786 unblocked the OOM; stability is separate.
- 3 self-advancing tmux sessions on gpu-server (all survive workstation-off):
  * 780A_s7   — seed-7 Phase-A training (E11 @ snapshot)
  * 780A_orch — waits s7 PROCESS-EXIT → run s13 → s23 → sanity gate → /tmp/780A_chain_done.flag
  * 775_eval  — waits 775p2 final.pkl (or best.pkl stale>45m = wedge) → benchmark s7/s13/s23 →
                paired Δα vs 760 a1b1 → writes runs/775_oracle_mask/result.md → /tmp/775eval_done.flag
- STALE-FLAG BUG (fixed): done-flags persist across OOM/crash (trailing `echo *_DONE` runs even
  when python OOMs). First 780A_orch gated on /tmp/780A_s7_done.flag (stale from Phase-0 OOM) →
  fell through → briefly launched s13 on G3 beside s7. Caught, killed strays, s7 unharmed.
  Serial chains now gate on PROCESS-EXIT (pgrep -f "name <tag>"), never flags.
- OPERATOR RULE: all server training launches go in TMUX (not screen) so they survive
  workstation-off + are resumable. Saved as memory feedback_server_launches_in_tmux.md.
- ARCHITECTURE: tmux orchestrators replace the server agent for DETERMINISTIC chains (run cells
  → gate → run cells → compute verdict → write result). Server agent's only residual value =
  off-hours JUDGMENT on unscripted next-experiment decisions. Directive tells it to STAND DOWN
  on 780+775. Operator leaning toward dropping the server agent.

COMPRESS + POINT:
- Full code of 786 → bridge_trainer.py:_encode_obs_chunked + easy_config oracle_distill_encode_chunk
  + train_easy --oracle-distill-encode-chunk + tools/smoke_786_encode_chunk.py + tasks/TASK_786_*.md
- Server directive (full queue + stand-down) → runs/DIRECTIVE_2026-05-30_786.md (on server)
- 775 eval orchestrator → /tmp/775_eval_orch.sh ; 780A orchestrator → /tmp/780A_orch_bella.sh ;
  s7 launcher → /tmp/launch_780A_s7.sh (all on server)

DROP:
- Phase-0 forensics (_OOM_780A-default_s*, _wedged_E135_phase2) — stashed, not needed.
- Per-check GPU HBM snapshots — only the 3 sessions + divergence-watch matter.

VERBATIM ANCHORS:
- Master tip: b8560dc (TASK_786). Server training tip: 7575fac (deploy of b8560dc + ws working tree).
- 786 commit b8560dc: 5 files (bridge_trainer.py, easy_config.py, train_easy.py,
  tools/smoke_786_encode_chunk.py, tasks/TASK_786_BRIDGE_ENCODE_CHUNK.md). encode_chunk default 384,
  sentinel 0 = legacy single-shot.
- 780A launch flags: --oracle-mcts-distill --bella --epochs 200 --batch-size 256 --steps-per-epoch
  200 --encoder-version 3 --embed-dim 64 --encoder-heads 8 ; MEM_FRACTION=0.9 ; encode_chunk 384.
- 775 eval = benchmark_easy.py --checkpoint <775 best> --db trading_data.db --json <out> 
  --eval-mcts-sims 16 --eval-mcts-decode joint ; comparator runs/760_train_mcts_sims/
  eval_baseline_paired_seed_{7,13,23}/benchmark.json ; alpha key summary.avg_alpha + per-period model_alpha.
- Snapshot ~21:38Z: tmux {775_eval,780A_orch,780A_s7}; GPUs G1=775p2_s7(E145) G2=775p2_s23(E145)
  G3=780A_s7(E11, loss DIVERGED); G0 VBox-reserved.
- 775p2 done ETA ~22:30Z; 780A_s7 done ETA ~00:07Z; full 780A chain ~05:45Z.
- Result files to read on return: runs/775_oracle_mask/result.md ; runs/780_oracle_as_mcts/
  sanity_gate_s{7,13,23}.csv ; flags /tmp/775eval_done.flag /tmp/780A_chain_done.flag.

UNCERTAIN:
- Does 780 Phase-A diverge on ALL seeds? s7 looks bad (loss explosion). Sanity-gate verdict pending.
  If yes → 786 fixed OOM but 780 still doesn't work; next is bridge-trainer stability, not Phase-B.
- 775p2 s7 wedge a 3rd time? 775_eval has a best.pkl-stale>45m watchdog to proceed anyway.
- 780 Phase-B config if sanity gate passes — NOT scripted; needs a decision next session.
- lore/notes_2026-05-30 §172 (786 + tmux-orchestrator pattern + bella-drop-wasn't-fix +
  architecture beat) NOT yet written — TODO next session (Bloom-relevant).
```

## Git state at handoff

```
b8560dc TASK_786: bridge_trainer memory-bounded encoder forward (780 OOM fix)   <- master tip (mine)
a73d727 TASK_780 buffer-init branch: route prefill to _refill_buffer_bridge when oracle-mcts-distill
c55861c Lore + tasks cleanup: scrub impl-time bias, audit + cleanup task
```
Working tree still dirty with the usual `_old/` deletion drift (pre-existing, NOT mine — committed
only my 5 TASK_786 files explicitly). Server tip `7575fac`.

## Next action (next workstation session)

1. Read `runs/775_oracle_mask/result.md` (paired Δα vs 760; gate +3pp/no seed<−5pp) and
   `runs/780_oracle_as_mcts/sanity_gate_s{7,13,23}.csv` (≥2/3 ≥ canon−3pp). Check the
   `/tmp/*_done.flag`s to know which chains finished.
2. **If 780A loss diverged on all seeds** (s7 already did): 786 fixed the OOM but the bridge
   trainer is unstable — debug stability (grad_clip 5.0 is current; try lower LR, lower
   value_scale, or n_passes warmup) BEFORE any Phase-B. The OOM win still stands and is publishable.
3. If 780A sanity gate PASSES: decide + launch Phase-B Stage-B paired 3-seed (NOT scripted).
4. Write `lore/notes_2026-05-30.md §172` (786 + bella-drop-wasn't-fix + tmux-orchestrator-vs-
   server-agent + the stale-flag bug). Bloom-relevant.
5. Fold s7-wedge into memory if it reproduced a 3rd time (check 775_eval log).
6. tmux sessions persist on server: `tmux attach -t 780A_orch | 775_eval | 780A_s7` to inspect.
   Server directive: `runs/DIRECTIVE_2026-05-30_786.md`.
```
