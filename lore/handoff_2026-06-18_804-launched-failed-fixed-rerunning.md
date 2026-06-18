# Handoff 2026-06-18 — 804 token-arms: launched, failed-then-fixed (4 infra blockers), 6/9 evald, retry-runner closing the gap

**Previous:** [lore/handoff_2026-06-12_branch-c-built-staged-launch-pending.md](handoff_2026-06-12_branch-c-built-staged-launch-pending.md) (seq 6 → **7**)
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session focus:** Fired the staged TASK_804 launch; it failed 4× through a chain of infra blockers
(20-min watchdog too tight → 47-min cold-annotate wall → orphan-kill cascade → the TRUE root cause: a
dropped `token_frontend_cache_dir` dict key). All fixed; relaunched clean (tokenize 2852s→26s). Campaign
completed 6/9 (3 sporadic wedges); raw_seed7 is intrinsically cursed. A robust retry-runner is now
closing the gap. Result is leaning **null** (shift ≥ real_fe). Full narrative: `lore/notes_2026-06-17.md`
§203–208 (esp. §208 launch outcome + root cause).

## Confirmed compaction rubric

```
KEEP IN FULL:
- STATE: 804 token-arms campaign. 6/9 evals on disk + VALID (read summary.avg_alpha, ~0.057-0.077; the
  orchestrator RESULT.md is EMPTY only because _record_eval reads data['alpha'] not summary.avg_alpha —
  a reader BUG, data is fine). A ROBUST RETRY runner is live: server tmux 804_solo, /tmp/solo_retry.py
  (per-run frozen-log watchdog + reap-scope + 3 retries), order real_fe_seed13 (DECISIVE, first) ->
  raw_seed23 -> raw_seed7 (cursed; gives up after 3). Watcher bvwyym0xd. ETA ~3h decisive, ~7h all.
- VERDICT LOGIC: shift(null) >= real_fe in s7 (0.0768>=0.0641) and s23 (0.0704>=0.0684). If real_fe_s13
  <= shift_s13 (0.0605) -> shift>=real 3/3 -> NULL (token frontend does not beat its own shift control).
  Per-seed Da = arm.summary.avg_alpha - raw.summary.avg_alpha.
- raw_seed7 CURSED: intrinsic raw-arm-x-seed-7 wedge (E78/E19/E39, 2-wide AND solo; raw_s13 + the
  seed-7 token arms all completed fine). Likely never completes -> expect a 2-seed (13,23) paired result.
- NEXT = assemble the scorecard: read 9x runs/804_token_arms/eval_<tag>/benchmark.json -> summary.avg_alpha,
  pair per seed, apply promote/kill. Then lore the verdict + (optional) fix _record_eval for future runs.
- FIXES (all committed + on server by SURGICAL scp, NOT full deploy): e02953e cache_dir->oracle_cfg (the
  TRUE root cause: disk-cache load+write never fired); 78ccb70 _reap_scope (kills the systemd scope a
  bash-kill orphaned -> the cascade); c564215 prebuild script; a3b8d25 QA n>=2-kill + crashed-train
  guard. Disk artifacts built+VERIFIED byte-identical, loaded at segment_pool.py:332.

COMPRESS + POINT:
- The whole 4-failure launch saga (cold-annotate wall, orphan cascade, watchdog raises, relaunches) ->
  lore/notes_2026-06-17.md §203-208 (launch outcome + root cause captured there in full).

DROP:
- The pkill -f self-match fumbles (matched my own ssh shell -> exit 255/144, twice); ssh connection
  drops; the ~dozen background-watcher re-arms + their per-poll heartbeat tables; per-poll GPU snapshots.
- The ~1600-file dirty tree (standing M deltas + _old/ purge = de facto canon; never git add -A).

VERBATIM ANCHORS:
- SHAs: e02953e(cache_dir) 78ccb70(_reap_scope) c564215(prebuild) a3b8d25(QA) 09d134d(802 head, idle)
  79edfae+b02c15b(lore §208). Predecessor server deploy: 9c84315.
- Artifacts: real timelines_d6a718dd...npz, shift timelines_b55f18f6...npz (2G each, token_cache/).
  Base vocab file sha256 c247c907...; content-canonical vocab_sha d6a718dd; SHIFT_SEED=804; 833 assets.
- 6 eval avg_alphas: raw_s13 0.0573, real_fe_s7 0.0641, shift_s7 0.0768, shift_s13 0.0605,
  real_fe_s23 0.0684, shift_s23 0.0704. Wedged: raw_seed7, real_fe_seed13, raw_seed23.
- IDs: tmux 804_solo, retry runner /tmp/solo_retry.py, watcher bvwyym0xd.

UNCERTAIN:
- Keep retrying raw_s7 vs accept 2 seeds vs add a fresh seed-31 full triple — NOT settled.
- Whether to fix the orchestrator _record_eval field bug (summary.avg_alpha) — flagged, not done.
- Whether the NULL verdict is sound on 2 seeds + the 3-seed shift-vs-real direction, or needs the
  memory's >=3-seed bar (which is for CLAIMS of improvement, not negatives).
```

## Git state
```
79edfae lore §208: TRUE root cause was dropped cache_dir dict key (e02953e); relaunch clean, 2852s->26s
e02953e TASK_804: thread token_frontend_cache_dir into SegmentPool oracle_cfg
b02c15b lore §208: 804 launch outcome — fired then failed; prebuild fix in flight
78ccb70 TASK_804-B4: reap safe_run systemd scope on watchdog kill
c564215 TASK_804: standalone prebuild_token_cache.py — pre-annotate token-timeline disk artifacts
```
Working tree: `?? experiments/804/shift_null_spec.json` (local placeholder — the REAL 833-asset spec is
server-side; do NOT commit the local one). Plus the standing ~1600-file dirty set (known noise). Server
`training` = 9c84315 + surgical scps of the 4 fixes above (NOT a full deploy).

## Next action
Wait on watcher `bvwyym0xd` (re-invokes on retry-runner ALL DONE, ~7h). Then: (1) assemble the 9-eval
scorecard from `runs/804_token_arms/eval_*/benchmark.json` summary.avg_alpha → per-seed Δα → promote/kill
verdict (expect NULL: shift ≥ real); (2) write the verdict to lore + tasks/TASK_804; (3) decide raw_s7
(accept 2-seed vs add seed-31); (4) optional: fix orchestrator `_record_eval` to read `summary.avg_alpha`.
Server stays under load until then. On workstation boot: `udisksctl mount -b /dev/sdb1`.
