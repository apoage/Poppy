# TASK_917 — second-moment & event-conditional instrument pack

**Status**: PROPOSED 2026-09-05 (planner; salvage of the dead mechanism-design panel
wf_80f2d139 — full digests in `lore/harvest_6b32_appendix/mech_{A,B}.md` — plus the loops-panel
post-listing observation and the never-tested 1h–6h band named in
`lore/fable_direction_memo_2026-09-03.md`).
**Size S/M. Risk: LOW (read-only instruments, new module(s), no canon surface). CPU only,
≤3 CPU-days worst case.** Four independent instruments; each carries its own pre-registered
kill band and its own placebo. NOT TASK_915 cells (915's grid is frozen; these are new rows
by its own discipline).

## The shared thesis (one line)

Drift is measured (≈dead); second moments, event-conditional structure and the intermediate
horizon band are NOT — and OHLC ranges estimate them natively.

## P1 — variance-ratio persistence (gates every regime idea) — hours

Market-level pooled VR(30) persistence ρ̄ = corr(VR_t, VR_{t+1}) on the PIT panel, plus the
cross-sectional sd of VR(30)−1. Per-asset sd[VR(30)] = 0.16–0.46 is 5–40× the 0.02 break-even
margin ⇒ **only market-level pooling can resolve the sign** (do not report per-asset verdicts).
Break-even VR(30) ≥ 1.024. Circular-shift placebo + Bonferroni over 8 cells.
**KILL: ρ̄ < 0.10 ⇒ every regime-gating idea (gated trend/reversal, CPD switches, XS-AUTOCORR
books) is closed on this venue — record and stop carrying them.**
(Mason 2024 caveat already on record: CPD variants did not beat non-CPD models.)

## P2 — post-shock reversal-slope κ table (event-conditional family) — 2 CPU-h

Events: |z| ≥ 2.5 ∧ (range ∨ volume ≥ 2× median) on the true-daily panel. Outcome: fraction of
the shock reversed, hierarchical-Bayes slope κ per date-clustered cohort (10–15/yr expected).
**Entry at the FIRST TRADED PRINT after the event, never the shock-day last-print close** —
close-entry IS the bid-ask bounce (bias 0.04–0.16 vs the 0.08 PASS bar; Cox–Peterson 1994);
report the bounce gap per liquidity decile. Honest hedge math uses 916's borrow + taker fees
(the refuter's real round trip ≈ 111 bp with the BTC hedge).
**KILL: κ < 0.18–0.20 on 10–15 cohorts/yr (and the κ≈0.10 branch needs ~90/yr, dead at real
costs) ⇒ the event-conditional liquidity-provision family is closed.**

## P3 — post-listing event study — CPU minutes

First 60–90 days post-listing as one shared-structure regime class (273-asset PIT panel, dead
coins retained — this panel is uniquely good at this). Direct event-study IC of residual
returns vs listing age; placebo = shifted listing dates. Absent from every prior instrument
including 915's F-families. **KILL: no age-conditional IC with placebo-clean sign ⇒ closed.**

## P4 — hourly residual IC surface (closes the never-tested 1h–6h band) — ≤2 CPU-d

True hourly bars aggregated from the minute cache (ends 2026-03-31 — window to 2026-03,
DISCLOSED; do not wait for 899). Residual targets, h ∈ {1,2,4,6,12,24}h, **skip entry (open
t+1) + stale-tape screen** (open t+1 = close t on a stale tape is vacuous — screen it). Overlay
the measured break-even: required IC_1h ≈ 0.037 (HL 6h) / 0.0145 (HL 24h) vs the repo's only
prior hourly read implying ≈ 0.008; at 4h the fee line alone eats ~94% of gross (gross Sharpe
≈16 needed at the fast end). Also re-read 816's residual-TSMOM best cell (+1.53,
CI[+0.32,+2.88], under-powered lead) at daily/4h horizons with a proper CI — its first direct
test since 816. **Expected outcome (designers' own arithmetic): real IC, cost-bound — the band
closes decisively either way at ≤2 CPU-days.**

## Preconditions

None hard. P4 uses the minute cache as-is; P1–P3 use the true-daily PIT builder (819 POWERED).
Run before or parallel with 915; nothing here gates it, and nothing it finds licenses a book
build without 915's ceiling first.

## Out of scope

Any book/controller build (all five panel designers' books died on borrow+cost arithmetic —
see harvest §1; the instruments are what survived refutation). Minute-scale horizons < 1h
(897 closed them). GPU anything.
