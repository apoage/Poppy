# imprint oos artifact

> Imprint lane: real in-window learning (+0.9914, instrument-independent as of 2026-08-08) and NO OOS edge (875, 1/5 folds); the +0.898 OOS anchors withdrawn

*Migrated from local agent memory 2026-08-31 (`project_imprint_oos_artifact.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**LEDGER 861/ref-rebase + 861/search-teacher-imprint + 861/ref-oos-rebase (2026-08-03), read
with the 2026-08-05 attribution correction:**

1. **In-window, the 832 imprint recipe is REAL**: 3 fresh seeds median **+0.9914**, spread 0.041
   (the old −8.93…+12.74 seed-noise band does not reproduce), DEC-2 regime form PASS 3/3.
2. **The +0.898 OOS / WF-pass anchors are WITHDRAWN as bars** — they do not reproduce under the
   new measurement stack (clean-trained seeds: oos_full median +0.064, wf_q4 −1.767, wf_q1
   −1.041; OLD ckpt through the same battery negative in-window too). **But the CAUSE is
   UNISOLATED** — four semantics changed at once (alias union · causal frozen-cell mask · UTC
   windows · frame pin/source), and the old ckpt's negative 2021–2023 regime legs cannot be
   explained by the 2025 interior gap at all (no temporal overlap). "Proven fabricated" was an
   overstatement I made and withdrew.
3. **Operator correction (architectural, load-bearing):** forward-fill is REQUIRED — the panel
   must be dense for the JAX kernels — and **a stale price is a valid price while nothing
   trades**. So the fill is not fabrication; the real defects were the **zero price** (recurring
   bug, fixed by D30 opt (i)) and the **ingest identity split** (fixed by 868, real data
   recovered). The frozen-cell MASK is a modeling choice that can UNDER-credit a tradeable
   stale cell — testing it is the pending 4-leg isolation ladder (LEDGER row carries the design).
4. **Search-teacher imprint FAIL 0/3** (median +0.692 vs reference +0.991) — robust to the mask
   question because both arms ran identical semantics. Teacher imitability, not quality, is what
   hard-target BC rewards; joins the parked 857/860 distill lane.

5. **SETTLED 2026-08-06 on n=5 — LEDGER `875/wf-generalization`: NO OOS EDGE, 1/5, the
   pre-registered "none" band.** 5 expanding-window folds × 3 seeds, test = the 6 unseen months
   after each cutoff. Fold medians vs their own teacher floors: +1.4957/+3.8987 ·
   +0.3995/+1.4301 · +0.3108/+1.3952 · +0.4121/+0.2635 (the only clear — and only because the
   FLOOR collapsed) · **−2.9566/+0.4244** on 2025-H2, a 33% median drawdown on the half-year
   closest to deployment. **The shape, not the tally, is the finding:** the recipe's OOS median
   is flat (+0.31..+0.41 on three folds) while the floor swings 15×, so it does not track window
   difficulty — it clears only when the baseline has a bad half-year. OOS seed spread 1.0–2.0 vs
   0.041 in-window ⇒ a single-seed OOS number on this recipe is uninterpretable. Caveat: fold 5
   overlaps the known sparse span for 3 of its 6 months, so its MAGNITUDE is confounded; folds
   1–3 are outside it and all fail the floor, so the verdict does not rest on fold 5.
   The mask question in (3) is also answered — mask EXONERATED (5/7 windows bit-identical).

6. **INSTRUMENT-INDEPENDENT 2026-08-08 — LEDGER `882/decisive-legA`.** TASK_882 briefly put every
   number above in doubt: two harnesses claiming the same argmax replay of the same weights
   disagreed by 1.6 net@21. Cause found and then *measured out*: a missing
   `--panel-frame-from-checkpoint` routed `eval_vector` onto today's coverage-ranked universe
   while the encoder is positionally universe-locked. Pinned correctly, the two agree to
   **Δ 0.000214** (+0.9915919 eval_vector vs +0.9913783 imprint.py), PV within 0.060%, residual =
   **3 decision cells out of 107,072** (float32 argmax tie-breaks). 3 seeds +0.99159/+0.99355/
   +0.95212. So **+0.9914 is RESTORED and no longer rests on one harness.** Cross-harness
   comparison is legitimate **only when the run is pinned** — `eval_vector` still fails OPEN on a
   missing pin (TASK_883). Untouched by this: the OLD `runs/832_e5/ckpt_imprint_full` anomaly
   (`861/ref-oos-rebase`) is an EPOCH question, not an instrument one.

**Program state**: the imprint lane has REAL in-window learning (+0.9914, now instrument-independent) and NO out-of-sample
edge, now on n=5 pre-registered folds rather than n=2 quarters. That is a measured verdict, not
a suspicion — but it is scoped to hard-target BC on the 832 recipe, NOT to the self-play lane
(TASK_879 rung 1) which has never been run on the fixed framework. Do NOT carry the July-2026
"imprint = validated frontrunner" framing forward, and do NOT restate the collapse as proven
contamination. **Consequence worth thinking about before phase 2: "imprint → warm-start" means
warm-starting from a policy with no OOS edge.**
Related: [vector_env_cash_wall](vector_env_cash_wall.md), [deployment_gate_ladder](deployment_gate_ladder.md).
