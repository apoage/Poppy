# encoder bella class closed

> Entire Bella mechanism class (encoder distill, shape-distill, data augment) failed at paired 3-seed 2026-05-19. Class closed. Pivot to planning-side per TASK_759.

*Migrated from local agent memory 2026-08-31 (`project_encoder_bella_class_closed.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**RECHARACTERIZED (2026-07-05 reconciliation, canonical eval-MCTS sims=16 operator):** the class
stays CLOSED for promotion but the table below is superseded — under the canonical operator the
three strikes are 740 **−0.69pp**, 757 **−0.52pp**, 758 **+0.13pp**: **0-for-3 NULL, not
regressions**. 758's −16.54pp / s13 −37.93pp "catastrophe" was an argmax-decode measurement
artifact (s13 recovers to **+1.88pp**, the best seed, under joint MCTS). "Bella actively harms"
is REFUTED; "Bella does not add alpha" stands (best +0.13pp ≪ +3pp gate). The s13-catastrophe
methodology lesson below is likewise void. Corrections appended to each result.md; see
[verdict_contamination_audit](verdict_contamination_audit.md).

Bella as a mechanism class is **closed** as of 2026-05-19. Three attempts on three different surfaces, all negative at paired 3-seed:

| Attempt | Surface | Paired mean Δα | Seeds positive |
|---|---|---:|---:|
| 740 bella_distill | encoder per-asset MSE | −1.92pp | 0/3 |
| 757 bella_shape | encoder permutation-invariant (14-float shape descriptor) | −5.80pp | 0/3 |
| 758 bella_augment | data slot substitution (synthetic OHLCV) | **−16.54pp** (s13 −37.93pp catastrophe) | 0/3 |

**Why this matters:** TASK_754 (shape-distill) was specifically designed to eliminate the column-mismatch hypothesis that killed 740. It still failed. TASK_753 (augmentation) was designed to be encoder-agnostic — it failed catastrophically with a single-seed −37.93pp collapse. Three independent surfaces, three negatives.

**Why:** Encoder-side improvements (better representation per Bella's target) are confirmed anti-correlated with Alpha on this stack per [classifier_alpha_anti_correlation](classifier_alpha_anti_correlation.md). Data-side substitution shifts training distribution off-deployment regime.

**How to apply:**
- Don't propose new Bella variants without a fundamentally different ansatz (e.g., Bella as world-model rollouts inside MCTS, not as supervision or data).
- The pivot is to planning-side mechanisms — see TASK_759 (`tasks/TASK_759_PLANNING_PIVOT.md`) for the active spec. Two probes: MCTS rollouts sweep (untouched surface) + value-target small probe (reopen TASK_756 at reduced scope).
- 758's s13 catastrophe (−37.93pp) is a methodology lesson: synthetic-data augmentation can destabilize training spectacularly when oracle-PnL ratio diverges from deployment regime. Log oracle-PnL ratio before promoting any synthetic-augmentation mechanism in future.

**Related**: [classifier_alpha_anti_correlation](classifier_alpha_anti_correlation.md), [bella_column_mismatch](bella_column_mismatch.md) (resolved by TASK_745 spot-check — confirms 740's loss is genuine mechanism failure, not bug).
