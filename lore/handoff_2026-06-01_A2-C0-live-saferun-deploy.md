# Handoff — 2026-06-01 — A2 (realized-PnL) + C0 (token gate) live under safe_run; A null, 780/775 closed

**Sequence**: #8
**Previous**: `lore/handoff_2026-05-31_ABC-branches-crash-saferun.md` (#7)
**Branch**: `master` (tip `936ce31`)
**Working dir**: `/home/lukas/Trader`

## Session focus
Read the overnight verdicts (A w0.1, 780A, 775), found A's penalty too weak, killed a wedged
w0.2 arm, then under the operator's "advance all three branches + cap 64G/run + don't OOM the
server" directives: built **A2** (realized-PnL value term) and launched it + **C0** (token
viability gate) on the server, both hard-capped at 64 GB via `safe_run`.

## Confirmed rubric
```
KEEP IN FULL:
- TWO experiments live on gpu-server, both 64G-capped (safe_run cgroup, verified MemoryMax=64GiB):
  790_A2 (realized-PnL value, G1, s7 SOLO then s13/s23 2-wide) and 789_C0 (token gate, CPU).
  Verdicts pending: runs/790_realized_pnl/result.md, runs/789_token_c0/RESULT.md.
- A (w=0.1 over-trade penalty) VERDICT: mean Δα +0.35pp (FAIL) but a CLEAN null — penalty cut
  trade-rate on all 3 seeds WITHOUT the anti-correlation collapse (Ba up + α DOWN). The lever is
  too WEAK, not harmful. Pairing VALIDATED (w000_s7 reproduction = −0.37pp vs 760). → motivated A2.
- A2 = realized-PnL value term: expected per-asset position from policy_logits × next-step realized
  return (future_log_returns[:,0]), MSE onto |r_next|. LOSS-SIDE ONLY (no new param leaf), compile-
  time gated by realized_value_weight (default 0) → bit-identical OFF. I APPLIED IT MYSELF because
  both build sub-agents were WRITE-BLOCKED at the harness level (Edit/Write/Bash denied in worktrees).
- 780 DEAD on all 3 seeds (bridge collapses to constant 0.5 bal-acc — sanity gate's +50pp delta is
  misleading). 775 NULL −0.66pp. Both closed.
- B (Poppy ExIt) stable (100/100). Real run needs a pkl→Orbax warm-start adapter (NOT built) → G3.
- CRASH CAUSE (from #7): GPU-hang on the workstation's display 3090, NOT OOM/CPU. NEVER run GPU on
  the workstation. safe_run (47689ee) is the envelope; --cpu keeps compute off the display GPU.
- DEPLOY HYGIENE DEBT: git-push deploy is BROKEN — an earlier direct `scp` of safe_run.sh left an
  untracked file on the server that fails the `training` working-tree checkout. Using direct scp to
  ship A2/C0. Reconcile under TASK_785.
- C0 harness needed live fixes (untested salvage): DB_PATH was workstation-absolute (sed to server
  path), --out defaulted to a dead worktree (override), python2 misfire (must `source trader_env`
  before safe_run), and a `timestamp IN (?,...)` SQLite variable-limit bug (now fetch-once + index).

COMPRESS + POINT:
- The crash + safe_run + A/B/C narrative arc → lore/notes_2026-05-31 §173 + handoff #7.
- 780/775 closure detail + the 746/734 "all-regimes" re-debunk → notes §173.

DROP:
- scp retry churn (exit 144), the python2 misfire, the 780A sanity-CSV dump, per-check GPU snapshots.
- The w=0.2 wedge forensics beyond "seed-7 Stage-B wedge, 3rd time, killed" (in memory now).

VERBATIM ANCHORS:
- master tip 936ce31 (A2); safe_run 47689ee; C0 salvage adf8c15; handoff #7 9abf7da.
- server tmux: 790_A2, 789_C0. flags: /tmp/790_a2_done.flag. cap: safe_run --mem 64G --gpu N.
- A2 flag: --realized-value-weight (config realized_value_weight, default 0.0).
- pairing comparator: runs/760_train_mcts_sims/eval_baseline_paired_seed_{7,13,23}/benchmark.json;
  gate +3pp / no seed < −5pp; eval-MCTS sims=16 joint.
- C0: tools/tokenizer_c0/run_gate_numpy.py (CPU, numpy, --out runs/789_token_c0/RESULT.md).

UNCERTAIN:
- Will C0 verdict PROCEED or KILL (token edge vs shuffled control)? Kill-switch for whole token branch.
- Will A2 s7 wedge (seed-7 pattern) or its realized term move α without a trade-rate explosion?
- Does the git-deploy need fixing before the next deploy, or keep scp'ing?
- Whether to build B's pkl→Orbax adapter next (fills G3) vs wait on A2/C0 verdicts first.
```

## Next action
Read `runs/789_token_c0/RESULT.md` (PROCEED/KILL) and `runs/790_realized_pnl/result.md` (paired Δα +
ΔtradeRate) when they land. If C0 PROCEEDs → scope C1 (Δ-IRIS GPT). If A2 moves α → sweep w=0.5.
Then build B's pkl→Orbax adapter for G3. Reconcile the broken git-deploy (TASK_785).
