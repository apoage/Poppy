# TASK_737 Ablation Summary

Reference: TASK_736 Alpha +3.87% AVG / Bal.Acc 26.5% / 6/6 beats-random

| Tag | Bal.Acc AVG | Alpha AVG | ModelPnL AVG | Beats random | Dir.Acc (final) | Policy loss (final) |
|---|---|---|---|---|---|---|
| no_aux | 25.1% | -6.95% | -13.65% | 6/6 | 0.570 | 1.5820 |
| no_rank | 23.2% | -0.92% | -7.62% | 6/6 | 0.569 | 1.5784 |
| minimal | 25.1% | +7.06% | +0.37% | 6/6 | 0.569 | 1.5819 |
| legacy_27ch | 33.7% | -17.40% | -24.10% | 6/6 | 0.570 | 1.5901 |

## Verdicts (≥2pp Alpha change = significant)

- **TASK_735 (aux_price_head)** — removing it (`no_aux`) costs −10.82pp Alpha. **Load-bearing; keep.**
- **TASK_730 (rank_head)** — removing it (`no_rank`) costs −4.79pp Alpha. **Contributes; keep.**
- **TASK_732 (cross-sectional channels, 32-ch)** — removing them (`legacy_27ch`) costs −21.27pp Alpha (largest single cost). **Strongly load-bearing; keep.**
- **minimal (both aux heads off, 32-ch)** — gains +3.19pp Alpha, first positive-PnL configuration. Inverts spec's linear decision rule: individually harmful heads are net-neutral-to-helpful as a pair. **Needs repeat runs to confirm this is not noise (6-period SE ≈ ±5-6pp).**

Bal.Acc and Alpha diverge sharply on `legacy_27ch` (highest Bal.Acc 33.7%, worst Alpha −17.40%). Bal.Acc is NOT a reliable proxy for trading skill in this regime.
