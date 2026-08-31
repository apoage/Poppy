# 763 767 closed null

> TASK_763 entropy reg and TASK_767 limit-orders-disable both CLOSED NULL on server 2026-05-23/25. Reshapes ship-path combination candidates.

*Migrated from local agent memory 2026-08-31 (`project_763_767_closed_null.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**TASK_763 entropy reg** — CLOSED NULL on 2026-05-25 (server `runs/763_entropy_reg/result.md`).
- Best Δα +0.55pp at ec02 cell. Standard gate (+3pp) FAIL.
- **Anti-correlation pathway never fired**: best cell shows ΔBal.Acc = +0.19pp (wrong direction — classifier sharpened instead of softened). The expected (ΔBa < 0, Δα > 0) combination did not occur at any sweep point.
- High-coef cells (ec05, ec10): Bal.Acc trends negative but Alpha also degrades.
- Caveat: ec02_s13 trained to E69 only (rank-head deadlock at epoch boundary).

**TASK_767 limit-orders-disable** — CLOSED NULL on 2026-05-23 (server `runs/767_action_space/result.md`).
- Mean Δα −0.65pp at 3-seed. Mechanism gate FAIL (mean executed-trade count INCREASED +98, opposite of intent).
- Seed 7 trade count exploded **+448**; seed 23 +22; only seed 13 dropped (−174). Per-seed behavior incoherent — seed noise dominates intervention.
- Implication: removing limit actions doesn't reliably reduce trade count.

**Why:** the campaign's two most-recently-queued mechanisms both failed. Entropy didn't trigger the anti-correlation lever; limit-orders-removal didn't constrain trade selection. Both findings narrow the surface for combination experiments built on top of them.

**How to apply:**
- **TASK_776** (rs050 + entropy): entropy null alone + anti-correlation never fires → combination at HIGH risk of null. Reorder behind 775/777/779.
- **TASK_779** (rs050 + 767): 767 null alone + seed-incoherent → combination at HIGH risk. Demote from "cheapest probe = best first launch" framing.
- **TASK_781** (return-weighted BC + 767 + rs050): 767 leg null → 2 of 3 legs are bull-blindness attacks, only rs050 has positive paired evidence. Restructure as "return-weighted BC + rs050" (drop 767 leg) OR keep stack as-is with explicit "this is the combination experiment, individual nulls don't predict joint null per project_classifier_alpha_anti_correlation logic".
- **TASK_783** (BC-balanced + entropy joint): blocked dependency was 776; if 776 is also null, 783's sweep is unlikely to clear +3pp. Hold.
- **Reshaped priority order**: TASK_777 mixture-value (independent of 763/767 nulls, builds on 746 +0.72pp), TASK_775 oracle-mask Phase-1 (independent), TASK_782 encoder-unfreeze (independent), TASK_784 smoothed-oracle (independent). Combinations (776, 779, 781, 783) downgraded to "still worth running for combination-effect evidence but no longer the lead".

**Reference:** server `runs/763_entropy_reg/result.md`, `runs/767_action_space/result.md`, `runs/HANDOFF_2026-05-26.md`. Workstation lore `lore/ship_path_recommendations_2026-05-25.md` was written assuming 763/767 status unresolved — those §2/§5/§7 EV estimates need adjustment downward.
