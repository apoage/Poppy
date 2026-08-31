# verdict contamination audit

> 2026-07-05 forensic audit of experiments 734→819 vs the three mid-campaign bugs. CPU probes 806→819 all CLEAN. Stage-B code-independent of the reward-wiring fix. Residual risk = eval-operator Bug B: 740/743/758 have unreconciled re-score data; 739/741 never re-scored; stale result.md files on 757/758/759/809/810.

*Migrated from local agent memory 2026-08-31 (`project_verdict_contamination_audit.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Full matrix delivered 2026-07-05 (operator-prompted: "notes are biased, bugs were patched, sus
something is missing"). Verified against git history + import graphs, not doc prose.

**CLEAN (conclusions stand):**
- All standalone CPU probes: 806, 808, 813, 816, 819 (zero imports of any affected module) —
  the cost-floor / trend-CTA conclusions are uncontaminated. (809/810 clean of the three bugs but
  each has its OWN stale result.md — verdicts overturned in lore only: 809 annualization artifact,
  810 turnover artifact.)
- Bug A (Bella precompute): contaminates nothing (fixed same day; no scored run on buggy data).
- Bug C (reward wiring): contaminates only 812, which is self-reconciled by TASK_818.
  `train_phase2.py` (Stage-B) provably independent by import graph — BUT its own hand-rolled
  reward loss is pure CE, no MSE anchor, no PopArt = the SAME fragility shape 818 fixed, and it
  was never audited. Open question, not proven bug.
- Post-recal experiments (760, 762, 763, 767, 775, 779, 780, 804): born on canonical operator.
- Explicitly reconciled on 2026-05-20: 734 (null holds), 746 (demoted to −1.20pp), 749
  (−3.11 → −0.33pp), 757 (−5.80 → −0.52pp **null**, but only in analysis.md — result.md stale).

**SUSPECT (Bug B exposure, never reconciled):**
- **758 bella_augment −16.54pp** — the "third strike" that closed the Bella class
  ([encoder_bella_class_closed](encoder_bella_class_closed.md), now caveated). Re-score data exists
  (`eval_mcts_ba_seed_*`); spot-check → recovers toward null. HIGHEST-VALUE re-score, no
  retraining needed.
- **740** (−1.92pp) and **743** (multi-seed a1b1 reference; also carries an arithmetic error:
  file +4.02% vs +3.61% quoted everywhere incl. old memory) — re-score data exists unused.
- 739 (−2.63pp), 741 (−6.98pp) — never re-scored, within plausible flip range; would need actual
  eval runs on surviving checkpoints.
- 769: claimed 3-seed result not locally reproducible (only seed_7 on workstation).

**Doc-hygiene lesson:** reconciliations lived in `runs/recommendation_2026-05-20/analysis.md` and
lore, while the canonical result.md files kept the stale verdicts — downstream memory then cited
the stale numbers. Rule: any verdict change must be APPENDED to the experiment's own result.md.

2026-07-05 RESOLVED: reconciliation complete, corrections appended to all six result.mds.
Canonical-operator recal: 740 −0.69pp / 743 +0.24pp / **758 +0.13pp** — all NULL; 758's
catastrophe was an argmax artifact (s13 −37.93 → +1.88). Bella class recharacterized 0-for-3
null. 743's correct figure is +4.02% (memory's old +3.61% was the error; fixed). Remaining open:
739/741 re-eval (server-side checkpoints), Stage-B reward-head audit (in flight 07-05). Related: [oracle_design_intent](oracle_design_intent.md), [collapse_fixed_predictability_wall](collapse_fixed_predictability_wall.md).
