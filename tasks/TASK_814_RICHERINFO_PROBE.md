# TASK_814 — Richer-information edge-vs-cost probe (news + funding, before the reasoning model)

**Status**: DRAFT — operator direction (2026-06-26). The disciplined first step toward a reasoning model:
**probe whether richer information carries edge > cost on Kraken BEFORE building a reasoner on it.**
Reuses the 806 apparatus. Kill-gated. The model is the easy/proven part; **the data + its leak-free
timestamp alignment is the real work.**

## The question (one sentence)
Do **news sentiment** and/or **funding rates** carry predictive edge **> 42 bp cost** on Kraken 1-min,
OOS, leak-free, FDR-surviving — i.e., is there ANY richer-information signal that clears the cost floor
that price-only (806→813) provably could not?

## Why this, now
- 806→813 closed **price-only** on Kraken 1-min spot: real signal, linear-optimal (IC 0.15, ~3 bp edge),
  cost-bound across prediction / planning / vol-harvesting. The binding constraint was the **data** (only
  OHLCV was consumed), not the model class.
- The one untested surface is **richer information** — news, funding, on-chain, order-book, DEX — exactly
  what a reasoning model is built to integrate. But richer signals might **also** be cost-bound. **Probe
  before you build** (the 806→813 lesson — don't construct a reasoner on an unvalidated foundation).
- **DB reality (verified):** only `ohlcv` + `fx_rates`. No news/funding/on-chain/order-book/DEX. The
  richer signals must be **sourced** with aligned timestamps.

## Scope guardrails
- **Probe, not a build.** No reasoning model / planner here — just measure incremental edge from richer
  info vs the price-only baseline. If it clears → TASK_815 (the reasoner).
- **Reuse the 806 apparatus** (linear probe, leak guards, forward target, train/val/test split, per-
  decision H-strided Sharpe, BH-FDR, gross AND net). The richer signals are *additional features*.
- **Two cheapest signals first:** news sentiment + funding. On-chain/order-book/DEX deferred (cost/
  sparsity/ETL — see Data).
- Server-side, OOM-safe (cgroup cap), never the workstation display GPU. Audit discipline.

## Method

### Phase 0a — data sourcing + leak-free timestamp alignment (the real work)
Source historical, timestamped:
- **News:** CryptoCompare News API / CryptoPanic (historical, timestamped). Classify → sentiment/event
  embedding with **FinBERT** (cheap, pre-GPT-1 tech, ~0.5 GB). Asset-specific where possible + market-
  wide macro sentiment.
- **Funding rates:** Binance / Bybit / Kraken-**perp** APIs (historical, **free**, every 8 h, timestamped).
  Note: funding is perp data — either a directional signal for spot, or flag that funding-carry on perps
  is the cleaner structural edge (defer the instrument decision to the verdict).
- **DEX data** (operator-noted, DEFER v1): on-chain DEX trades / liquidity-pool flows could add flow +
  whale-move signal, but blockchain ETL + schema are hard to mine. Park as a Phase-0c option only if
  news/funding show promise.

**Leak-free alignment (CRITICAL — the new leak surface):** every richer signal known at time `t` may only
enter the state for decisions **strictly > t**. News **publication time** = known-at (NOT the market-
impact time — that leaks). Funding **publication time** = known-at. Assert per signal, per bar.

### Phase 0b — edge-vs-cost probe (806-style, incremental)
On the audited 806 split, fit a linear probe on **combined features** and measure incremental edge:
- **price-only** (806 baseline: `linear_xs`, IC 0.15) vs **price + news** vs **price + funding** vs
  **price + news + funding**.
- Metric: incremental IC (does richer info raise IC above 0.15?), and whether the **combined per-trade
  edge > 42 bp round-trip** post-cost, OOS, leak-free, BH-FDR-surviving. Per-decision H-strided Sharpe,
  gross AND net, block-bootstrap CI.
- **The decisive comparison:** does richer info add predictability that price-only lacks, enough to clear
  the cost floor? (806's linear-optimality was about *price-only* features — it does NOT bound richer
  info.)

### Gate (kill or go)
- **GO:** ≥1 richer signal (or combo) yields **incremental edge > 42 bp** post-cost, OOS, CI-separated,
  FDR-surviving, beyond the price-only baseline → richer information clears the cost floor → **TASK_815
  (the reasoning model: news encoder + 812 planner + fusion) is warranted.**
- **KILL:** no richer signal clears → richer info is also cost-bound on Kraken spot; the reasoning model
  has no foundation either. STOP honestly.

## Deliverables (audited)
1. `runs/814_richerinfo_probe/DATA.md` — what was sourced (news/funding), coverage, timestamps, the
   leak-free alignment rule + the assert that enforces it.
2. `runs/814_richerinfo_probe/probe.py` — reproducible; reuses 806's data/guard path; richer signals as
   additional features.
3. `runs/814_richerinfo_probe/RESULT.md` — per-signal incremental IC + edge vs price-only, post-cost
  Sharpe + CI, GO/KILL, and (if GO) the signal(s) that cleared.

## Acceptance checklist (audited separately)
- [ ] Richer signals sourced with **real timestamps**; leak-free alignment asserted (known-at t → state
      for > t only). **This is the #1 audit item** — a timestamp-leak manufactures fake edge.
- [ ] Reuses 806's split + guards + per-decision H-strided Sharpe; richer signals are additive features.
- [ ] Honest 42 bp round-trip cost; gross AND net both reported (810's lesson).
- [ ] Verdict is the **incremental** edge (richer vs price-only), not the absolute — the question is
      whether richer info ADDS edge that clears cost.
- [ ] BH-FDR across the signal grid; OOS test only; leak-free.
- [ ] Server-side, OOM-safe, no workstation GPU.

## Risk
- **Timestamp leakage** — the new, dominant leak surface. News/funding known-at vs used-at; get this
  wrong and every "edge" is fake. Audit hard.
- **Richer signals may also be cost-bound** — genuinely uncertain; news/funding might be weak or
  arbitraged. The probe settles it in days.
- **Data-sourcing effort** — historical news availability/coverage; funding is free but perp. DEX is hard
  (deferred).
- **Honest prior:** richer info is the one untested surface and a reasoning model's whole rationale — but
  it's not guaranteed. State the prior in the result.

## References
- `runs/806_predictability_probe/` (probe methodology + the price-only IC 0.15 baseline you measure
  incremental edge against + leak guards).
- `tasks/TASK_813_VOLHARVEST_AGENT.md` (the detector+gate architecture the reasoner inherits).
- The exhaustive price-only close (806→813): richer info is the one surface it didn't bound.
