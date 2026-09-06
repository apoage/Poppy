> **STATUS (2026-09-06): SHIPPED as `task917-instrument-pack` (`8f411d1`, factory run 54).**
> All four instruments landed. **Production numbers, not fixture proxies** — `ohlcv_daily` is
> byte-identical between the fixture and `trading_data.db` (sha256 `90d67f40…`, 499,430 rows),
> independently re-hashed. Verification: 4 lenses (cycle 1) → BLOCKER + MAJOR → fix pass (opus)
> → 2 lenses (cycle 2, opus) → 6 MAJOR on the fix's *arguments* → cycle-3 D76 closing pass,
> no verdict changed. Suite 17/17 throughout.
>
> **P2 — KILL FIRED, ROBUST. The event-conditional liquidity-provision family is CLOSED.**
> κ₃ = +0.0361, CI [−0.022,+0.094], 167 cascade cohorts (13.1/yr); threshold sweep 8→40 never
> approaches the 0.18–0.20 line. Hedge round trip **338–356 bp** at 916's verified `base_taker80`
> vs **+62.7 bp** capture. Placebo shows κ ≈ 0.036 is generic post-large-move behaviour, not
> cascade-specific.
>
> **P1 — INCONCLUSIVE. The kill does NOT fire; regime-gating is UNMEASURED, not closed.**
> Do NOT cite this row as closing gated trend/reversal, CPD switches or XS-AUTOCORR books.
> Revision 1 fired on an un-inherited 180d block; 120d cells invert at +0.106…+0.213 with MORE
> pairs (27–28 vs 18). Sweep spans ρ̄ ∈ [−0.342,+0.470]. **Decisive fact is power:** dispersion is
> indistinguishable from estimator noise at all 11 block lengths, ceiling ~0.3 |ρ̄| with SE ≥ 0.14,
> anchor power **~9–10%** to separate ρ̄=0.10 from 0 → no block length could have fired a
> defensible kill. 0/44 cells exclude the line from above either. → **D84 OPEN.**
>
> **P3 — SURVIVES, placebo-clean.** Age-conditional IC **+0.0304, NW-t +4.15**, 946 dates,
> 398-asset PIT panel, dead retained. Leakage audit refuted the listing-date-is-collection-start
> hypothesis (2/398 assets; excluding them changes the IC to 6 dp); LOO market identical; NW t
> stable 4.05–4.23 over lags 0–90; `include_dead=False` → +0.0279 / t 3.50. Operative null is
> i.i.d. per-asset: **p = 0.0050 at the design minimum, z ≈ 5.5**. An **instrument reading**, not
> a promotion candidate and not a reopened edge thread.
>
> **P4 — PREPARED, NOT RUN** (compute policy: expensive validation is scheduled). `--full`
> refuses with exit 2 before opening a DB. **Pre-`--full` BLOCKER:** its local
> `discover_universe()` omits the `source_filter="kraken_rest"` P1–P3 use, so it carries 8 real
> assets as a 10-column panel (`TWX/USD`, `HOLE/USD` admitted) — it would measure a different
> universe than P1–P3. Reconcile before any real invocation.
>
> **REMAINDERS.** D84 OPEN (P1's 0.10 threshold was inherited from `mech_B.md`'s *different*
> statistic and never calibrated to this estimator). Estimator defects recorded, NOT fixed:
> undisclosed **tiling-origin** free parameter (15.6% of origins put ρ̄ above the kill line);
> numerator/denominator sample mismatch (own-null level 0.955, so the observed 0.904 is
> **z = −0.70 from a random walk**, not a shortfall against the 1.024 breakeven);
> E[VR_block] 0.765→0.908 across the grid, so R1 demanded unanimity across eleven different
> estimands. The cycle-3 closing pass added null-construction code that was **not itself
> adversarially reviewed** — corroborated by judges' independent implementations, flagged.

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
