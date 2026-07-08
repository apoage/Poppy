> **STATUS (2026-07-08): factory half SHIPPED as `task832-linear-imprint`.**
> `experiments/832_linear_imprint/` — buffer-injection BC runner: 830-book teacher trajectories
> walked ON the TRAINING panel (PIN b verified — no off-by-one vs the self-play obs rule), one-hot
> marginals through `JAXReplayBuffer(vector_action=True)` (the shipped factored CE IS the BC loss),
> ExIt half = the existing `--load-checkpoint --reset-optimizer --clear-buffer` flags (anchors
> verified). Smoke: CE 2.60→0.299 decreasing, teacher agreement 0.873. Medium panel: 1 MAJOR
> (imprint entropy_coef=0.001 leaked into the ExIt half via `trainer_metadata.json` — CLI override
> ineffective) — FIXED (`--ckpt-entropy-coef` default canon 0.01, proven with an adversarial-constructor
> restore probe) + re-verified. **Remainder (scheduled ≤1-GPU-day run, DEC-2):** the imprint-sanity
> verdict, the post-ExIt corridor position, and the ε=+0.10 PASS with the per-window consistency leg.
> Operator: verify the `Restored entropy_coef: 0.01` log line at ExIt launch (SPEC PIN c).

# TASK_832 — Linear-book imprint: can search beat its own (valid) teacher? (E5, R2)

**Status**: GATED on 828 Stage 3 banner + TASK_831 E1 PASS.
**Pathway**: `docs/pathway.md` rung R2. Hypotheses: H5 (teacher validity) + H1 (the smallest
possible "reasoning > rule" test).
**Size: M. Risk: medium-high** (touches Stage-3 training config, flag-gated). Budget: ≤1 GPU-day
imprint + short ExIt; full runs escalate to R3 only on PASS.

## Purpose
The oracle-as-teacher was the diagnosed disease (anti-correlation, bull-blindness). The 819
linear book is the VALID teacher: it actually makes money, and its trajectories are expressible
as `action_vec` sequences (exactly what TASK_830 replays). Two questions, pre-registered:
1. **Imprint sanity**: BC on linear-book vector trajectories → does the imprinted policy
   reproduce ~floor performance in-env? (If BC on a working teacher also degrades, that
   implicates the BC/representation path itself — a finding, per H5's falsification clause.)
2. **The test**: from the imprint, run the Stage-3 ExIt loop (search-improved targets). Does
   even a SHORT improvement phase move performance ABOVE the teacher (floor +ε, pre-register ε)
   on held slices, ≥3 seeds? That is reasoning adding value over the rule — the first time the
   campaign would ever have measured it directly.

## Deliverables
- `experiments/832_linear_imprint/SPEC.md` + runner; imprint dataset generated ON-THE-FLY from
  the 830 replay path (no npz pre-generation — placement rules). Flag-gated additions only;
  canon and `--vector-action` defaults untouched.
- LEDGER lines: imprint corridor position; post-ExIt corridor position; per-seed.

## Interpretation guard
PASS → strongest possible evidence for launching R3 at scale (warm-started variant ranked
first). FAIL on (2) with (1) green and E1/E3 green → H1 weakens materially; planner records
it and pivots hypothesis rather than scaling training (pathway kill logic).
