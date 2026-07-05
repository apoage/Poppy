# TASK_808 — Cost-aware OOS backtest of the proven linear signal (Step 1: first working-trader candidate)

**Status**: DRAFT — operator-approved direction (2026-06-21). The first executable step of the
`runs/HANDOFF_2026-06-21_path-to-working.md` path. **Reuses the audited TASK_806 linear path; adds
portfolio construction + realistic costs + a post-cost gate.** Does NOT retrain anything heavy — it is
a ridge + backtest, CPU/server-side.

**The question** (one sentence):

> The cross-sectional linear signal 806 proved *predictable* (clean-OOS IC 0.06–0.15) — does it survive
> realistic trading costs to become a **profitable, deployable trader** at some horizon × frequency?

## Why this, now

806 settled that the data has signal and the campaign's failure is method, not data. But 806 proved
**predictability, not post-cost alpha** — IC 0.15 pre-cost at 1-min may be entirely eaten by Kraken
fees. The campaign has **never had a working/deployed trader** (0 promotions, nothing live). This task
produces the first one *if the signal survives costs*, and either way it establishes the **measurement
bar** (post-cost Sharpe) every learned model must hereafter beat. It is the missing foundation, done by
the simplest means — exactly 666's "simplify to ship" intent, done right this time.

## Scope guardrails

- **Reuse, don't rebuild.** Inherit TASK_806's data path, `linear_xs` feature construction, and the
  audited leakage guards (G1–G7 + UNIVERSE-LOCK, already proven in `runs/806_predictability_probe/`).
  Cite the inherited guards; only re-audit what you add (portfolio + costs).
- **No oracle, no Bella, no encoder, no MCTS.** Pure linear cross-sectional. (The learned models get
  their turn in Step 2, measured against whatever this task produces.)
- **CPU / server-side.** No GPU needed (ridge + backtest). Run on gpu-server under the OOM-safety rules
  (cgroup cap, `python -u`); never the workstation display GPU.
- Writes only: `runs/808_linear_backtest/`. Don't touch 806 artifacts, training code, or checkpoints.

## Method

### Signal (inherited from 806, unchanged)
- `linear_xs`: per decision time `t`, the 64 assets × 63 lagged log-returns over the causal window
  `[t-63, t]` → ridge → predicted cross-sectional forward return `r(t,H) = log c(t+H) − log c(t)`.
- Same forward train/val/test split as 806 (train 2023-05-16→2024-06-01, val →2024-11-01,
  test 2024-11-01→2025-04-01), purge gap ≥ H, strided sampling, train-only scaling, val-only ridge α.
- **Fit ridge on train only** (clean OOS). This is the same fit 806 already proved gets IC 0.06–0.15.

### Portfolio construction (NEW — the trader)
- From the predicted cross-sectional forward returns at each rebalance, **rank** the 64 assets.
- Build two variants: **long-short** (long top-k, short bottom-k, market-neutral) and **long-only**
  (long top-k). Equal-weight or inverse-vol within leg; document the choice.
- `k` ∈ {8, 16, 32} (top/bottom decile/quintile/quartile of the 64-asset universe).

### Costs (THE CRUX — model honestly or the result is fiction)
- **Kraken taker fees** (~16 bps single-side → ~32 bps round-trip for taker; use the real schedule).
- **Slippage**: a model grounded in reality — e.g., a fixed bps per trade, or a linear-impact model
  from turnover × a per-dollar cost. State the assumption and sensitivity.
- **Turnover**: computed from the ACTUAL position changes the strategy generates (1-min long-short
  churns near-maximally — this is where the signal lives and where costs bite). Report turnover ratio
  (one-way) per cell. Do NOT assume low turnover.
- Report a **cost-sensitivity sweep** (0, half, full, 2× costs) so the margin is visible.

### Grid (pre-registered)
- `H ∈ {1, 5, 20}` bars × rebalance `∈ {1, 5, 20, 60}` min × `{long-short, long-only}` × `k ∈ {8,16,32}`.
- Pre-register before running; don't add cells post-hoc to chase Sharpe.

### Gate (post-cost, with significance)
- A cell **works** iff: post-cost annualized **Sharpe ≥ 1.0** AND positive net PnL over the full test
  window, with a **block-bootstrap 95% CI on Sharpe excluding 0** (block length ≥ max rebalance period).
- Report **pre-cost vs post-cost** Sharpe / cumulative PnL / max drawdown / turnover per cell, so the
  cost ceiling is explicit. State the bar clearly; verdict follows mechanically.

## Deliverables (audited)

1. `runs/808_linear_backtest/result.md`: the per-cell table (H × freq × variant × k: pre-cost Sharpe,
   post-cost Sharpe + CI, PnL, maxDD, turnover) + the gate verdict + the cost-assumption set +
   **if any cell passes: a deploy plan** (paper-trade wiring via a thin sibling of `real_trade_jax.py`).
2. `runs/808_linear_backtest/backtest.py`: single reproducible entry point, reusing 806's data/guard
   path. No oracle import.
3. `runs/808_linear_backtest/GUARDS.md`: which guards are inherited from 806 (cite) + what's new
   (portfolio/cost construction — confirm no look-ahead in the rebalance: positions at `t` use only
   predictions from features ≤ `t`, and execution assumes no knowledge of `t`'s close).

## Acceptance checklist (audited separately)

- [ ] Reuses 806's `linear_xs` + split + guards (cite); only portfolio/cost code is new.
- [ ] Realistic costs modeled (Kraken taker fees + slippage), with a sensitivity sweep — not zero.
- [ ] Turnover actually computed from position changes, not assumed.
- [ ] Gate is **post-cost** Sharpe ≥ 1.0 with block-bootstrap CI; pre-cost reported alongside.
- [ ] Rebalance uses no future info (positions at `t` from predictions on features ≤ `t`).
- [ ] No oracle/Bella/encoder/MCTS. Server-side, OOM-safe, no workstation GPU.
- [ ] Verdict follows the gate mechanically; if nothing passes, says so plainly (the cost ceiling is
      then the finding — and Step 2's learned model must beat whatever linear achieves post-cost).

## Risk

- **Costs eat the signal at 1-min.** Likely. The sweep's job is to find the horizon/frequency where it
  survives (longer H / lower rebalance frequency = lower turnover = costs bite less). A NULL here is
  still informative — it sets the cost ceiling and reframes Step 2.
- **Look-ahead in rebalance/execution.** The one new leak surface — positions at `t` must not use
  `close(t)`. Audit explicitly in GUARDS.md.
- **Survivorship/universe.** Inherit 806's train-window-locked 64-asset universe (already guarded).

## References

- `runs/806_predictability_probe/` (probe.py, LEAK_AUDIT.md, result.md) — the audited linear path to reuse.
- `runs/HANDOFF_2026-06-21_path-to-working.md` — Step 1 of the path (this task).
- `real_trade_jax.py` — deploy path (paper-trade wiring if a cell passes).
- `config/benchmark_periods.json`, `trading_data.db` — real data (via 806's loaders).
