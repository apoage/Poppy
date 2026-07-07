# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-07-08** (pathway wired — `docs/pathway.md`; rung verdicts gate GPU spend, code may land
ahead).

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 1) | M | vector env step `step_vector(state, action_vec)` in `mcts_trader/env/jax_trading_env.py`, flag-gated `--vector-action`, canon scalar path bit-identical; equivalence test = vector action == equivalent scalar-step sequence. Risk: high (touches the canon env). |
| 2 | `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 2) | L | Sampled-MuZero search over K sampled action vectors from the factored policy head (`muzero_model.py:256`). Needs Stage 1 banner. Risk: high. |
| 3 | `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 3) | L | training loop + factored policy targets + action-conditional reward head (D2: the 827 substance — kill the 60× action-variance gap; E3 probe is a hard gate). Needs Stage 2 banner. CODE may land, but the TRAINING LAUNCH waits on TASK_831 E1 PASS (pathway R1 gate). Risk: high. |

## Filler — anytime, parallel-safe

- `tasks/TASK_829_RETIRE_FOXGLOVE.md` (S–M, risk medium) — decouple `train_phase2.py:484` +
  `benchmark_easy.py:41` imports, then delete the Foxglove files. Disjoint from 828 Stage 1
  (touches no env/search files); do NOT run concurrently with 828 Stage 3 (both touch
  train_phase2.py's import surface).

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `tasks/TASK_830_INSTRUMENT_CORRIDOR.md` (E0 corridor) | 828 Stage 1 banner. First in line once open — its numbers calibrate everything above |
| `tasks/TASK_831_SEARCH_CAPACITY_PROBE.md` (E1/E2 search probe) | 828 Stage 2 banner + 830 corridor recorded. Its E1 verdict gates the Stage-3 TRAINING launch |
| `tasks/TASK_832_LINEAR_IMPRINT_EXIT.md` (E5 linear imprint) | 828 Stage 3 banner + 831 E1 PASS |
| `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 4 eval) | Stages 1–3 shipped + rung verdicts green; real-data training/eval is a SCHEDULED server run, not a factory-loop job |

## Parked (D1/D4, 2026-07-07 — non-binding "interesting outview"; not factory work)

- `tasks/TASK_824_SERVER_HONEST_REMEASURE.md`, `tasks/TASK_825_CROSSASSET_TREND_CTA.md`,
  `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` (D4: OHLC-only, live within the 24GB DB),
  `tasks/TASK_823_VPS_MICROLIVE_CTA.md` (D0: rejected).

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.
