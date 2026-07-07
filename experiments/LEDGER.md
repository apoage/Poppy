# Experiment LEDGER — one line per verdict

The pathway's results record (`docs/pathway.md`). Every experiment ends here, PASS or FAIL —
nulls are results. Corridor = position in [linear floor, oracle ceiling] from TASK_830/E0.
Historical experiments 000–780 predate this ledger; their verdicts live in task banners and
the memory index (scalar-architecture scope — see MEMORY.md "Direction").

| ID | Date | H# | Question | Verdict | Corridor | Pointer |
|----|------|----|----------|---------|----------|---------|
| 830/E0 | 2026-07-08 | none (instrument) | Does the 819 book replayed through `step_vector` reproduce the `trader/` engine net@21? | **FIXTURE-SMOKE — real verdict PENDING scheduled CPU run.** floor misses ref by ±0.10 (Δ net@21 −0.33, gross −0.20; BEAR22 floor +0.23 vs ref +1.22 gross). NOT a `step_vector` defect: env reproduces an independent cash-account sim to 3–4 sig figs (finalPV to the dollar). Miss = replay-MAPPING artifact — dominated by fixed-weight 0.5/A vs 819 renormalized 1/n_present (gross 0.956→0.612) + hold-shares vs daily constant-$-rebalance (BEAR22); `step_vector` adds only +0.013 over the pure mapping. Attributed to the replay mapping, NOT 828 Stage 1; NOT tuned. Which mapping is the official floor = pending planner DECISIONS row. | fixture: [floor **+0.42**, ceil **+29.70**] net@21 (ref +0.75) | `experiments/830_corridor/SPEC.md` · `corridor.py` · `results_fixture_smoke.json` |
