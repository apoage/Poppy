# TASK_859 — Trade-activity metadata in eval result JSONs (XS filler)

**Status**: READY 2026-07-24 (planner-seeded instrument hygiene; operator-approved with 858).
**Size: XS. Risk: low** (additive result keys only; zero scored-value changes — the 855 discipline).

## Why (856/rule-grid compile, 2026-07-24)
Compiling the rule-grid verdict required inferring hold-collapse from `corridor.hold_collapse`
and eyeballing PV — the result JSONs from `experiments/835_stage4_eval/eval_vector.py` do not
record HOW MUCH the operator trades. Tonight's 857 corridor reads and every future sweep get
read faster and more honestly if each result is self-describing about activity.

## Scope
1. Per cost tier in the result JSON, add an `activity` block:
   - `hold_pct` (fraction of asset-steps with no action), `trades_per_step` (mean non-HOLD
     actions per decision), `n_trades_total`, `n_long`, `n_short`,
   - `active_assets` (count of assets with ≥1 trade over the window),
   - `top5_traded` (symbol → trade count, descending).
2. Both operators (`search` and `policy-argmax`, all decision rules) populate it from the same
   decoded action stream the replay already produces — no new compute, no extra passes.
3. Refresh `results_fixture_smoke.json` baseline (additive keys; scored values byte-identical,
   tiers-SHA unchanged — same procedure as 855 scope 2).

## Acceptance (gates)
- (i) Pre-existing keys byte-identical on the fixture smoke (SHA over the old key set); selftest
  green.
- (ii) Property checks: a 100%-HOLD run reports `hold_pct == 1.0`, `n_trades_total == 0`;
  a top-m M run reports `trades_per_step == M` (post-fix top-m always trades M).
- (iii) Small panel (correctness + regression lenses).

## Out of scope
Any change to scored metrics, corridor logic, or decision rules; plotting/reporting tooling.
