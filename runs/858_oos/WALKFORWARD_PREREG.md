# Pre-registration — imprint OOS walk-forward splits (2026-07-26, planner)

Follows LEDGER `832/E5-oos-real` (+0.898 net@21 on the single 116-decision window). D17 sequence:
this is the second benchmark leg — quarterly walk-forward splits of the same OOS span. NO gate
status regardless of outcome.

## Windows (decisions ≈ 90 each; warm = 60 panel rows BEFORE the graded quarter, so the graded
region is exactly the quarter — W1's warm overlaps the in-sample tail, acceptable: warm is
context, never graded)
- **W1 (Q4-2025)**: `--start 2025-08-06 --end 2025-12-31` → graded decisions ≈ 2025-10-05..12-31
- **W2 (Q1-2026)**: `--start 2025-11-02 --end 2026-03-31` → graded decisions ≈ 2026-01-01..03-31

## Operator (identical to the 832/E5-oos-real read)
policy-argmax replay of `runs/832_e5/ckpt_imprint_full`, 853 fixed-frame panel
(`frame_provenance_imprint.json`), 848 on-window teacher floor, weight-rebalance semantics,
tiers fee0 + fee21.

## Read of record (pre-registered)
Per window: net@21 + on-window floor + delta. Interpretation grid:
- BOTH windows positive-delta AND same net sign as the full read → consistency PASS.
- One window carries everything (the other ≈ 0 or negative net with weak delta) → the full-window
  +0.898 is CONCENTRATED — flag, route to regime analysis before anything else.
- Any window with delta < 0 → walk-forward FAIL flag on the imprint.
~90 decisions/window ⇒ CIs are wide; this grades CONSISTENCY, not significance.
