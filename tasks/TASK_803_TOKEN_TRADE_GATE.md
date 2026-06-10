# TASK_803 — Token-derived trade gate: attack the trade-rate explosion directly

**Status**: 📋 SPEC (2026-06-09). **Phase 1 blocked only on [[TASK_800]]** (no trader code changes,
no training). Phase 2 joins [[TASK_804]] arms 4-5.

## Why

Every Alpha collapse in the scar tissue has the same signature: **trade-rate explosion with
over-confident wrong trades** (737/741/738/757/758 — [[project_classifier_alpha_anti_correlation]],
`tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 1). The token layer gives a per-bar, causal,
host-computable "informativeness" measure for free: the [[TASK_800]] `codelength[t]` channel (bits
under the build bigram) plus mega-token / flat-doji run detection ([[TASK_794]] already wants null-run
skipping for throughput — same machinery, second use). This is the one mechanism aimed at the
failure mode itself rather than at representation quality.

## Phase 1 — eval-only gate on EXISTING checkpoints (cheap, fast signal)

No training. Modify only `tools/benchmark_easy.py` (decision logic, `:601-688` / MCTS decode
`:406-429`), behind `--token-gate <mode>` default off:

- **Gate rule v1**: during quiet runs — current token is a mega/flat token, or trailing-k mean
  `codelength[t]` below threshold θ — force HOLD (suppress non-HOLD actions before position
  stepping). **Lag convention ([[TASK_800]])**: the gate reads tokens as a decision-time input, so
  L1 streams use the 2-bar final-only delay from the [[TASK_801]] serving-lag convention
  (L8 unshifted). Same slicing layer, same rule.
- Sweep θ over a small grid on the **canonical scorecard periods** with the **existing 760 a1b1
  checkpoints**, eval-MCTS sims=16 joint, all 3 seeds.
- Report per θ: Δα (paired vs ungated same checkpoint), Δtrade-rate, ΔSharpe.
- **Read**: if HOLD-gating quiet bars is Δα-positive on existing checkpoints, the over-trading
  margin is real and token-detectable — strong motivation for Phase 2 and a result in its own
  right. If flat-to-negative, the gate idea dies cheaply without touching the trainer.
- Mind the safety rails (`benchmark_easy.py:945-967`) — the gate must not trip the cheat-checkpoint
  refusals.

## Phase 2 — learned variant (only if Phase 1 shows margin)

- `codelength[t]` enters as **one float channel** alongside the [[TASK_801]] token embedding
  (same zero-init `token_proj` path — widen E by 1, no separate plumbing), letting the policy
  learn its own quietness prior instead of a hard θ.
- Optional: action-prior bias in the search root (penalize non-HOLD logits by
  `λ·max(0, θ−codelength)`) — eval-time, no training, composable with any checkpoint.

## Acceptance

- Phase 1: a θ-sweep table in `runs/803_gate/RESULT.md` with paired Δα/Δtrade-rate per seed ×
  period; verdict MARGIN-EXISTS / NO-MARGIN.
- Phase 2 (if run): E1/E2 no-op checks as in [[TASK_801]]; gate channel reaches the optimizer
  ('train' label).

## Discipline

- Phase 1 is GPU eval — **server only**, tmux + safe_run, single verified-idle GPU, ≤2-wide rule
  ([[feedback_server_launches_in_tmux]], [[feedback_server_vram_envelope]]).
- Timeline pre-encoded to RAM once per eval run; zero tokenization in the eval loop.
- Commit + report; **never `deploy.sh`** ([[feedback_sub_agent_no_deploy]]).
