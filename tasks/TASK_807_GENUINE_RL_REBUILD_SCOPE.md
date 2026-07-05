# TASK_807 — Genuine-RL rebuild scope (pre-666 viability + plan)

**Status**: DRAFT — operator-approved direction (2026-06-20). Read-only investigation + plan. Does NOT
retrain, does NOT change code, does NOT decide to retrain. It produces the decision input; the
operator combines this plan with the TASK_806 signal verdict to decide go/no-go.

**The question** (one sentence):

> If we undo TASK_666 and return Poppy to genuine MCTS-in-training RL (`train_jax_fast.py`), what
> does that actually entail — was the pre-666 RL genuinely promising or already stuck, does the base
> still run, and what would it need to converge that it didn't have in January?

## Why this, now

Four parallel code inspections (MCTS, TRM, tokenization, Bella) and a git timeline check converged on
one root cause, now **verified from git (not lore)**:

- **2025-12-03 → 2026-02-02**: genuine MuZero RL Poppy = `train_jax_fast.py` (1,620 lines,
  self-play + MCTS-in-training + expert buffer + differentiable market physics). Oracle-free. This is
  the planning-based trader the operator remembers as "promising."
- **2026-02-02/04 — TASK_666 "Order 666"**: the pivot. Stated goal: *"Working trading bot in 4 weeks,
  not 4 months."* Method: gut the RL trainer, replace with **supervised oracle distillation** ("The
  Cheat") + inference-only MCTS ("The Thought"). The oracle (`path_solver.greedy_lookahead_path`) is a
  future-peeking teacher — a training aid, never a deployable strategy.
- **Since 666**: `train_jax_fast.py` has **zero commits** (4.5 months dormant). All ~100 campaign
  experiments (runs/734–784, Bella, tokenization) were built on the 666 oracle-distillation spine →
  **0 promotions**, because nothing in that loss is a realized-trading gradient; the network is a
  classifier that anti-correlates with trading alpha (`[[project_classifier_alpha_anti_correlation]]`).

Operator decision (2026-06-20): **drop the oracle, willing to retrain from base.** This task scopes
that rebuild honestly. The lore claim "ExIt dormant since ~May 2025" was wrong on month and year —
real date is **Feb 2026, by 666's hand**. All dates in this task must come from git, not lore.

## Scope guardrails (read + plan only)

- **No retraining.** No `train_jax_fast.py` training run, no checkpoint writes, no Phase-2 runs.
- **No code changes.** Don't modify `train_jax_fast.py` or anything else. A *viability smoke* (Leg B)
  is permitted only CPU-only on the workstation, or on a server GPU — **never the workstation display
  GPU** (that hard-locks the box; lesson burned 2026-06-20).
- Writes only: `runs/807_rl_rebuild_scope/` (the plan doc). Do not touch other `runs/`, checkpoints,
  or training code.
- Runs **in parallel with TASK_806** (independent). Does not block on it. The *retrain decision* waits
  for 806 + this plan together.

## Method — three legs

### Leg A — Pre-666 verdict: was genuine RL promising or already stuck? (THE LOAD-BEARING LEG)
Read the genuine-RL-era lore: `lore/notes_2025-12-*.md`, `lore/2025-12-31_session_notes.md`,
`lore/notes_2026-01-*.md` (esp. 01-13 → 01-28), `lore/2026-01-06_to_10_session_notes.md`,
`lore/2026-01-04_information_loss_audit.md`, and any handoff/STATUS from Dec 2025 → Feb 2 2026. Stop at
the 666 commits (2026-02-02).

Answer, each backed by a specific cite + any recovered metric (alpha/Sharpe/agreement/convergence):
1. Did `train_jax_fast.py` genuine RL ever produce **positive out-of-sample alpha** or clear learning
   signal? What was the best result achieved, on what data (real vs synthetic), at what date?
2. The **failure mode** that motivated 666 — name it precisely. Was it: non-convergence, instability
   (loss spikes/divergence), reward sparsity, MCTS too slow to sample enough, exploration collapse,
   capacity, or something else? Quote the lore's own words.
3. The operator's "promising models" memory — locate the specific runs/checkpoints/results if they
   survive (they are NOT in `runs/`, which is all May 2026+). State what they actually were. If
   unrecoverable, say so plainly — do not infer.
4. Was there a concrete moment/finding that triggered 666, or was it a fatigue/speed call?

**Verdict for Leg A:** retracing-a-lead (pre-666 showed real promise worth completing) vs
retracing-a-wall (pre-666 was also failing, and going back inherits an unsolved problem). This single
label shapes the whole rebuild.

### Leg B — `train_jax_fast.py` viability (does the base still run post-666?)
The file is syntactically intact (1,620 lines, `ast.parse` clean). But 4.5 months of refactors hit the
shared modules it imports. Check, with evidence:
1. **Import check**: does `python -c "import train_jax_fast"` (or the module's own import block)
   resolve? Enumerate every broken/drifted dependency (it imports `mctx_search`, `self_play`,
   `expert_buffer`, `replay_buffer`, `prediction.physics_core`, `adaptive_noise/fees/burn`,
   `schedules`, `evaluator`, `redo`, `exploration.sac_entropy`, `safe_checkpoint` — any may have
   moved/renamed/changed signatures since Feb).
2. **Map the genuine-RL loop precisely**: where does MCTS drive *learning* (visit-count policy targets,
   bootstrapped/n-step value targets)? Is there ANY oracle in this path, or is it pure RL? What is the
   reward signal (realized PnL? step reward?)? Where is the loss? Cite line ranges.
3. **Optional tiny smoke** (CPU-only or server GPU, never workstation display GPU): one-step import +
   forward, NOT a training run. If it runs, great; if it doesn't, the bit-rot list IS the finding.

**Verdict for Leg B:** runs-as-is / needs-N-days-repair / deeply-bit-rotted. With the specific repair
list either way.

### Leg C — Rebuild scope (what does genuine RL need to converge that it lacked in January?)
Synthesize A + B into a concrete, falsifiable rebuild plan:
1. **The specific problem to solve** = the Leg-A failure mode, restated against today's codebase. Not
   generic "RL is hard" — the named, cited mechanism that made 666 seem necessary.
2. **What's changed since January that helps**: the mature eval harness (`benchmark_easy.py`
   eval-MCTS scorecard), the data pipeline, encoder maturity, 4 months of documented dead-ends (what
   NOT to try). Be specific about what's now available that wasn't.
3. **The minimal genuine-RL loop worth retraining**: which files, what the first milestone is, what
   stays from the current (richer) codebase vs. what comes from `train_jax_fast.py` as blueprint. The
   operator's framing: re-introduce genuine-RL *training* into the current codebase — NOT a
   time-machine `git revert` (you keep the encoder/eval/data/tokenization-tooling infra).
4. **A falsifiable convergence gate**: a cheap, early signal that genuine RL is actually learning to
   trade (not just fitting) — so we can kill it in days, not months, if it's retracing the wall. State
   the metric, the threshold, and the checkpoint.
5. **Honest cost + risk**: wall-time estimate, what can go wrong, and the explicit condition under
   which the rebuild should be abandoned (tie it to 806: if 806 = NULL, abandon before starting).

## Deliverables (what the operator/auditor verdicts against)

1. `runs/807_rl_rebuild_scope/RESULT.md`:
   - **Leg A verdict** (lead vs wall) + the pre-666 best result with cite, + the named failure mode
     with cite, + the fate of the "promising models" (found/unrecoverable).
   - **Leg B verdict** (runs/repair/rotted) + the enumerated dependency/bit-rot list with line cites.
   - **Leg C plan**: the problem-to-solve, what's-now-available, the minimal loop, the falsifiable
     convergence gate, cost/risk/abandon-condition.
   - A one-paragraph **go/no-go recommendation** that explicitly says "deferred to operator, pending
     806" — this task does NOT greenlight a retrain.
2. All dates from git (`git log --format="%ci ..."`), not lore. Flag any lore date that contradicts
   git.

## Acceptance checklist (audited separately — be self-evident)

- [ ] Leg A "promising vs stuck" backed by ≥1 concrete pre-666 metric + cite; "promising models"
      located or declared unrecoverable (no vibes).
- [ ] Leg A failure mode is a named mechanism with a lore quote, not "RL was hard."
- [ ] Leg B is a real import/dep check with an enumerated list, not "looks fine."
- [ ] Leg C plan names the specific Jan failure mode against today's code, and states a falsifiable
      convergence gate (metric + threshold + kill-condition).
- [ ] No code changed, no retrain run, no checkpoint written. Any smoke was CPU/server, not
      workstation display GPU.
- [ ] All dates cite git, not lore; any lore/git contradiction flagged.
- [ ] The doc explicitly defers the retrain go/no-go to "operator + 806 result" — does not
      self-authorize a rebuild.

## Risk (how this could mislead)

- **Thin pre-666 lore**: early-project notes may be sparse/anecdotal. The "promising models" may be
  memory, not artifact. Mitigation: report what's recoverable; label inference as inference.
- **Deep bit-rot**: `train_jax_fast.py`'s shared deps may have drifted far. "Needs significant
  repair" is itself a valid, important finding — don't paper over it.
- **Over-commitment**: this is a scoping task. The temptation is to start fixing/running. Don't —
  scope only; the operator decides after 806.
- **Confirmation bias toward the operator's "go back" lean**: the auditor will check that Leg A
  honestly reports a wall if that's what the evidence shows, not just the lead the operator hopes for.

## References

- `train_jax_fast.py` — the genuine-RL base (oracle-free, MCTS-in-training). Blueprint for the rebuild.
- `tasks/TASK_666_EASY_MODE_BRANCH.md`, `TASK_666_IMPLEMENTATION_PLAN.md`, `TASK_666_DEPENDENCY_MAP.md`
  — the Feb-2026 pivot ("4 weeks not 4 months" → supervised oracle distillation).
- Verified timeline (git): RL era 2025-12-03 → 2026-02-02; 666 pivot 2026-02-02/04; `train_jax_fast.py`
  untouched since.
- `lore/notes_2025-12-*.md`, `lore/notes_2026-01-*.md` — the genuine-RL-era record (Leg A source).
- `tasks/TASK_806_PREDICTABILITY_PROBE.md` — the signal gate; 806 = NULL ⇒ abandon rebuild before
  starting.
- Four inspection findings (MCTS / TRM / tokenization / Bella) — converge on the supervised-oracle-spine
  diagnosis; MCTS + TRM agents confirmed `train_jax_fast.py` is the dormant genuine-RL path.
- `[[project_classifier_alpha_anti_correlation]]`, `[[project_single_mechanism_exhausted]]` — the
  failure-pattern memories the rebuild is meant to escape.
