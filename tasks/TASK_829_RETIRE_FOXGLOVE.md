# TASK_829 — Retire Foxglove (decouple, then delete)

**Status**: READY 2026-07-07 (DECISIONS D5). Foxglove = the stage-1 supervised oracle-imprint
path — the diagnosed disease vector (oracle-as-teacher) and superseded by the vector-action
rebuild. Its checkpoints were already pruned 2026-07-07. The CODE cannot just be deleted:
two live components import through it. Decouple first, delete second, canon-gate both.

**Size: S–M. Risk: medium** (touches the active phase-2 trainer's import path and the canonical
eval harness).

## Grounded dependencies (verified 2026-07-07)
1. `train_phase2.py:484` — `from train_easy import SegmentPool`. The real home is ALREADY
   `mcts_trader/data/segment_pool.py` (extracted TASK_772.5; train_easy merely re-exports).
   Retarget: `from mcts_trader.data.segment_pool import SegmentPool` (verify the other three
   re-exported symbols aren't also pulled via train_easy anywhere: grep `from train_easy import`).
2. `tools/benchmark_easy.py:41` — `from easy_config import EasyConfig, get_large_config,
   get_medium_config, get_small_config`. benchmark_easy is the CANONICAL scorecard — it stays.
   Either keep `easy_config.py` (rename-in-place is fine; it is config, not Foxglove logic) or
   move it under `mcts_trader/`; pick the smaller diff. Do NOT touch scorecard semantics.
3. Bella is independent (`python -m diff_teacher.train`) — no Foxglove coupling. Untouched.

## Delete after decoupling
`train_easy.py`, `easy_trainer.py`, `train_easy_loop.py`, `run_dual_easy.sh`,
`bridge_trainer.py` (runs alongside easy_trainer), `merge_checkpoints.py` (easy-mode weight
averaging), `tools/orchestration/` e-dojo scripts (`e-dojo_v1.py`, `dual_dojo.py`,
`reset_edojo.sh` — check the dir for anything non-Foxglove before removing wholesale),
`tools/DSe/` (e-dojo dashboard). If `easy_config.py` was kept for benchmark_easy, it survives.

## Gates (the flow-verifier's test cases)
- `python -m trader selftest` green (factory baseline).
- `python -c "import train_phase2"` and `python -c "from mcts_trader.data.segment_pool import
  SegmentPool"` succeed; `grep -rn "from train_easy" --include='*.py' .` → zero hits outside
  deleted files.
- `tools/benchmark_easy.py` imports clean (`python -c "import sys; sys.path.insert(0,'tools');
  import benchmark_easy"` or equivalent) — scorecard path unbroken.
- No behavior change anywhere: this is deletion + import retargeting only. Any diff beyond
  imports/deletes is out of scope.

## Out of scope
`train_phase2.py` logic, `diff_teacher/` (Bella stays — D5), `tools/benchmark_easy.py` semantics,
`docs/easy_mode.md` (historical design record, planner lane).
