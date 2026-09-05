# Mechanism-design panel digest — mech_A

Source: 6 structured result objects in `/tmp/harvest_6b32/wf_80f2d139-2b0/` (panel died of rate limits; 13 agents in `_FAILURES.txt` produced nothing). Mapping: 3 designers survived (SCRR, HCB, ARC-GP), 3 refuters survived (of Valerian, of Comfrey, of SCRR). The Valerian and Comfrey designs themselves are lost; their refutations summarize them. The two `partial_*.md` files are dead designers with zero analysis (one held the dispersion-shock angle SCRR took up; the other a variance-ratio/Hurst autocorrelation-regime book — that angle remains unwritten anywhere).

Goalposts for reference: unseen years, 21 bp/side, ≥3 seeds, net@21 per-decision Sharpe ≥ +0.97 (survivor book; +0.56 PIT), no HOLD collapse, positive 2022 bear.

LATE FACT applied throughout (panel never saw it): Kraken shorting = margin rollover 0.01–0.02%/4h BTC (≈22–44%/yr), 0.02–0.04%/4h alts (≈44–88%/yr) on shorted notional; the repo record NEVER charges borrow. Note this cuts both ways: the +0.97/+0.56 bars were set by a trend book whose bear short leg also never paid borrow, so the bar itself is inflated; any re-grading should charge the bar's short leg too.

---

## 1. SCRR — Shock-Cohort Residual Reversal (DESIGNER, `result_a0f15d4037d365041.json`)

**Role**: designer. Event-gated, hold-to-horizon, beta-hedged liquidity provision on the PIT 273-asset *true daily* OHLCV panel.

**Mechanism**
- Data flow: minute table → true UTC-midnight OHLCV (open/high/low/close/volume/trade_count) via `trader/data.py` aggregator; needs a new (T,A,5) panel reader (~40 lines — both existing readers return close only). PIT universe + 819 liquidity filter + present_mask. Everything causal at close t.
- Event definition (analytic, pre-registered): |z_e| ≥ 2.5 AND (RZ ≥ 2 OR VZ ≥ 2), where z_e = market-residual/(Garman-Klass 30d range vol), RZ = log(H/L) vs 30d median, VZ = volume vs 30d median. Cascade flag: cross-sectional dispersion DZ ≥ 1.5 AND ≥ 8 triggers. (Yang-Zhang rejected: 24/7 venue has no overnight gap.) β from trailing 90d, shrunk 50% toward 1.
- Learned object: hierarchical Bayesian linear on EVENT ROWS ONLY (~5–10k rows): y = −(k0+k_R·RZ+k_V·VZ+k_D·DZ+k_S·1{e<0})·z_e + a_i, asset random effect a_i~N(0,τ²) (partial pooling over 273 assets, most with <20 events), Huber loss. Secondary: monotone LightGBM depth≤3. Target y_{i,t,h} = Σ_{s≤h} residual return / σ_i,t, h∈{1,3,5}, h=3 primary — same target family as TASK_915.
- Control law: analytic gate (NOT learned): trade iff m_i − √δ·se(m_i) − c_i > 0. One-shot sizing w ∝ (m−c)/(λhσ²). Enter close t, exit close t+h, NO re-trim. Arm A (primary): long losers, hedge with BTC short (−Σwβ). Arm B: long-loser/short-winner within cohort (margin-enabled pairs only). Caps: 2.5%/name, 25%/cohort, 100% gross; default state FLAT.
- Cost model: 2×21 bp round trip PLUS explicit short carry (12–24 bp/day alts, 6–12 bp/day BTC — Kraken fee page fetched 2026-09-05) — the only design in this set that pre-charged borrow.
- Guards: anti-HOLD = logged cohort-count with FAIL if <50% of pre-registered rate; one primary cell frozen (sha256) in SPEC; date-clustered SEs (≥40 dates/400 events per cell); circular-shift + random-sign placebos; White's Reality Check over 36-cell grid; SR0(36)=1.35 (2.5yr)/0.79 (7.5yr) printed; PIT dead coins retained.

**Odds (self-assigned)**: P(Stage 0 passes: κ_3 ≥ 0.08 date-clustered t≥3 on 2019–23) ~40–50%. P(Stage 1 book ≥+0.56 net@21 on 2024–26 hold-out) 12–18%. P(full goalposts) 6–10%; unconditional ~7% — ~2× tactics-panel families, ~half of TASK_915's 10–15%. Positioning: a special case of 915's F1×F2×F3 interaction cell; independent value = event-conditional estimator power (signal on ~1% of asset-days) + sparse hold-to-horizon control law.

**Kill experiment (cheapest decisive)**: Stage 0, no trading, ~2 CPU-h inside 915's data build: pooled reversal slope κ_h = −cov(y,z_e)/var(z_e) on event rows, h∈{1,3,5}, date-clustered SE; event-vs-non-event contrast; loser-only/winner-only; interactions; 500 circular-shift placebos. PASS = κ_3(event) ≥ 0.08 t≥3 AND contrast ≥ 0.04 t≥2 AND > 95th pct placebo AND ≥40 dates/400 events. Stage 1 (book) only on S0 PASS; ≤1 CPU-day, 2–3 implementer-days.

**Numbers worth keeping**: bar decomposition — +0.97 needs κ_3 ≈ 0.18–0.20 on ~10–15 cascade cohorts/yr OR κ ≈ 0.10 on ~90+/yr (essentially all events); +0.56 at κ0.10 × 30 cohorts. Per-event SR_pos = (μ−c)/10%: κ0.05→+0.02, 0.10→+0.08, 0.20→+0.21; cohort ×1.79 (n=12, ρ0.25). Fee drag 3.4%/yr at 16 turnover units/yr (0.044/day vs trend book 0.155/day — 3.5× fewer cost events). Required within-cohort conditional IC 0.06–0.18 = 3–8× the unconditional daily IC bar. 2.5-yr hold-out CI needs SR > ~1.24 → full-span CI is the only certifying window. Event supply 900–1400 asset-events/yr on 60–120 dates (2024–26); ~2500–4000 total for fitting. Lit: Nagel 2012 (reversal = liquidity provision, peaks in turmoil); Bianchi-Babiak-Dickerson 2022 (crypto liquidity-provision returns concentrated in illiquid pairs — also the cost risk); Bremer-Sweeney 1991 (κ~0.18 anchor, FROM MEMORY, unverified); Memon 2026 (BTC losers reverse, winners continue — sign asymmetry); Liu-Tsyvinski-Wu 2022 (weekly momentum opposes at h≥5).

**margin fact: wounds it** — SCRR pre-charged borrow at exactly the confirmed rates (12–24 bp/day alts, 6–12 BTC), so the fact validates rather than surprises; but at confirmed rates Arm B's alt-short carry (36–72 bp over h=3) exceeds the 42 bp fee round trip → Arm B is borrow-dead; Arm A survives on BTC-hedge carry (18–36 bp/cohort). Minor offset: the +0.97 bar itself never paid the trend book's short-leg borrow, so a borrow-charged SCRR faces a slightly softer effective bar.

---

## 2. SCRR refutation (REFUTER, repo-record lens, `result_a504aedfd00e90306.json`)

**Role**: refuter of SCRR. **Verdict: REFUTED** (as a buildable book; a measurement survives amended).

**Decisive argument** (three prongs):
1. **The inherited instrument bakes in bid-ask bounce as fake kappa.** `trader/data.py` daily close = close of LAST traded 1-min bar; scorer enters AT that same close (`backtest.py:219-224`). A −12.5% shock day on an illiquid alt closes on a bid-side print; the "reversal" contains +half-spread mechanically. With unmeasured alt spreads of 100–400 bp on cascade days, pure-bounce κ = 0.04–0.16 — the S0 PASS bar (0.08) sits inside the bias band, exactly in the illiquid names where Bianchi et al. locate the premium. Cox & Peterson 1994 (abstract fetched): the equity version of this exact effect = bounce + liquidity; large decliners UNDERperform at longer horizons — opposite sign of loser-longs.
2. **Hedge fee omitted → corrected round trip ≈ 111 bp** (42 + 42β + 27β borrow, β≈1), break-even κ = 0.089 on a 12.5% shock. Every "κ 0.10 on many cohorts" clause is false: K ≈ 1500 cohorts/yr needed for +0.97 (impossible; 60–120 event dates/yr exist). Only κ ≥ 0.18–0.20 on ≥15 cascade dates/yr survives.
3. **Power and scope errors**: log-return scorer makes a capped −95% loser cost 7.5% of book (not 2.4%); a −99% costs 11.5%. 2018–20 has ~no event supply (12.8–25.2 live assets) → informative span 5.5 yr, SR0(36)≈0.92. Unseen-window CI needs SR > 1.24 → row can at best return MARGINAL. `cell_metrics` cannot score the claimed windows — must inherit `898 decay_harness.score_window`; hold-shares accounting needs a new PnL path. 813's B3/B1 continuation cells are a prior the design ignored: shock-day reversal gross ≈ −1.5..−2.0 at h=1d, +0.3..0.7 at h=4d, 33–88 bp/fire — an order of magnitude below the κ 0.18–0.20 need.

**Odds after refutation**: P(goalposts) 2–4% (design said 7%). P(amended S0 passes honestly) 20–30%. P(κ ≥ 0.18–0.20 on ≥15 cascade dates/yr) 8–12%. P(Arm A ≥ +0.56 on hold-out with hedge fee + shock-day slippage) 5–8%.

**Surviving mechanism / amended kill**: Stage-0 MEASUREMENT ONLY, as a TASK_915 appendix (~2 CPU-h): κ_h after shocks with forward return measured from the FIRST TRADED PRINT of day t+1 (minute table) as PRIMARY, close_t as bounce-inclusive diagnostic, and the close→first-print gap reported per liquidity decile AS THE BOUNCE ESTIMATE; rank-residual event definition primary (kills cascade-day β error); event-vs-non-event contrast; 813 B3/B1 cited as horizon prior. Book licensed only if bounce-robust κ_3 ≥ 0.18–0.20 on ≥15 cascade dates/yr 2021–23 AND 2024–26 slope holds sign at ≥50% magnitude, hedge charged 21 bp/side.

**margin fact: irrelevant to the verdict** — the refuter already charged borrow (27β in the 111 bp corrected round trip); the confirmed rates only harden the "κ 0.10 branch dead" arithmetic.

---

## 3. HCB — Horizon-Curve Book (DESIGNER, `result_a26b05c2816c48f8e.json`)

**Role**: designer. Hourly-native multi-horizon residual forecaster + GP horizon optimiser + minute-timed execution.

**Mechanism**
- Data flow: streaming pass over 1-min table → TRUE UTC-hour OHLCV bars (hour key, not the `prices_np[::bar]` stride all prior "lower-frequency" cells used); PIT builder include_dead; hourly liquidity gate. ~17.7M asset-hours.
- Features (hourly-native, cross-sectionally ranked): F-R returns 1h–7d (market-residual via 30d hourly β); F-V Parkinson/GK/YZ vol 6/24/168h, semivariance difference, jump share, variance-ratio; F-Q dollar-volume z, trade_count z, signed volume, CLV; F-X dispersion, β, idio-vol, hour-of-day × F-R (session periodicity). ~60 features.
- Targets: market-residual, vol-standardised forward return, h ∈ {1,2,4,6,12,24,72,168}h.
- Learned objects: per horizon, ONE pooled ridge + ONE shallow LightGBM (depth≤4). Quarterly refits, purge ≥ h+24h, 7-day linear blend so refits can't force a book turn. Nothing else learned.
- Controller (closed form, Gârleanu–Pedersen 2013): x_t = (1−a/λ)x_{t−1} + (a/λ)·aim_t; aim weights per horizon 1/(1+φ_h·a/γ) — fast-decaying signals auto-down-weighted; THE HORIZON OPTIMISER IS NOT LEARNED. λ calibrated so full re-rank costs 21 bp. No-trade band 0.25σ̂; Σ|x|=1, Σx=0, no leverage; flat unreachable by construction.
- Execution: taker at the minute (within 20-min window) chosen by the audited 806 `linear_uni` 5-min forecast (IC 0.098) — minute signal used ONLY as timing, never position.
- Guards: 48 diagnostic cells graded on IC NW-t + Bonferroni + circular-shift placebo, never OOS Sharpe; ONE pre-committed composite on OOS; matched-turnover random-aim null (892); refit-blend; "median gross <0.5 or >50% flat days = FAIL"; SR0(8)=1.36, SR0(48)=1.85 printed.

**Odds (self-assigned)**: P(interior optimum in 1h–6h clears +0.97) **1–2%** — own arithmetic: IC_req(4h) ≈ 0.089–0.17 ⇒ gross Sharpe ≈ 16 at N_eff 15, cost eats 94% of gross. P(composite clears +0.97 at ANY horizon, i.e. value sits at 1–7d) ~10%; P(≥+0.56) ~22%; P(timing saves ≥1.5 bp/unit) ~35% (worth ≤ +0.1 SR); P(the IC(h)×cost curve closes the intermediate band two-sidedly) > 90% — "the real product of running it".

**Kill experiment**: pre-registered IC(h)×cost curve, ≤2 CPU-days: (1) 48 diagnostic cells with measured IC±2SE; (2) overlay measured break-even line IC_be(h) = 21·T_h/(1.25·σ̄_res(h)); (3) K1 BAND KILL: if every h≤6h cell's mean OOS IC < IC_be − 2SE, the 1–6h cross-sectional band is CLOSED at 21 bp — LEDGER row, never reopened without a cost-tier change ("expected outcome"); (4) K2 single composite OOS 2024-01→2026-03; (5) K3 paired timing comparison. OOS truncates at 2026-03 (minute coverage).

**Numbers worth keeping**: IC_be(h) = 0.24·T_h/√h — h=1h: 0.34; h=24h T=0.7: 0.034; T=0.31 (ρ=0.95): 0.015. Smoothing converts a 4h signal into a ~5-day one (needs ρ≈0.97) — no interior optimum in 1–6h exists at 21 bp; honest optimum lands h≥1–3d = TASK_915's territory. Record IC priors: 0.151 @1min, 0.141 @5min, 0.060 @20min, ~0 @4h (minute features); 816 gross +2.0–3.7 @1h → net −80..−120 @21bp. Execution timing ≤ +0.10 SR upper bound (2 bp/unit × 0.3 turns/day). GP eq. 15 verified in full by the Valerian refuter. Lit: GP 2013 (read in full); Ding-Martin 2017 (breadth cannot rescue unstable IC); Chen-Velikov 2022 (avg equity anomaly ≈ 4 bp/month after costs); Boyd et al 2017 (ℓ1-MPC with borrow cost — the right home for the short-leg carry); Kraken fee schedule fetched in full (margin fees IN ADDITION to trading fees).

**margin fact: kills it (fast band) / wounds it (whole)** — the book is market-neutral ⇒ ~50% short CONTINUOUSLY; at 22–88%/yr on shorted notional that is ≈11–44%/yr on book ≈ +0.4–1.8 Sharpe of continuous drag at σ_book ~21–25%, on top of a fast band that already needed gross Sharpe 2.7–5.3. The design itself flagged "~20%/yr comparable to the entire cost stack" in open_doubts but charged it nowhere; at the slow end (HL 3d, fees 5.9 bp/day) the borrow alone exceeds the entire fee budget.

---

## 4. ARC-GP — Anchored Residual Controller over the GP book (DESIGNER, `result_a5661f8d44281006a.json`)

**Role**: designer. Learned bounded CORRECTION to a closed-form cost-aware controller — the only design whose learned object is a sizing/timing map, not a forecaster.

**Mechanism**
- Data flow: PIT daily OHLCV panel (273 assets); forecasts are FROZEN TASK_915 G-A outputs (ridge/GBM on residual YZ-standardised returns, h∈{1,3,7,21}d) + σ̂_YZ, β̂ (90d), Σ̂ Ledoit-Wolf, decay φ_h from IC autocorrelation.
- Base controller: GP closed form + hard β-neutral projection + Σ|w|≤1. Differentiable end-to-end (linear recurrence + projection).
- Learned object δ_θ: ONE shared MLP 20→64→64→3 (~5.5k params), inputs per (i,t): μ̂ at 4 horizons, w_GP, x_{t−1}, vol ratios, dollar-volume rank, β̂, idio-vol, trailing 60d cross-sectional IC, dispersion, market vol, DeepSets date context. Three bounded, nameable channels: s∈[0,2] (forecast rescale = state-dependent shrinkage), κ∈[0.25,4] (trading-rate multiplier = learned no-trade), δw∈[−c,c] with c=0.5|w_GP|+0.002 (cross-asset netting). Identity-initialised: s=κ=1, δw=0 ⇒ epoch 0 IS the GP book (Silver et al. 2018 residual policy).
- Loss (decision-aware, no per-asset target): −√365·mean(r_net)/std(r_net) with r_net charging 21 bp on DRIFT-INCLUSIVE turnover and simple returns (fixes the scorer's two accounting gaps); + anchor penalty λ_a·[(log s)²+(log κ)²+(δw/c)²]; + β-neutrality and gross-floor penalties. BPTT through the GP recurrence; 11 six-month walk-forward folds 2021H1→2026H1, 21d purge+embargo, 3 seeds, λ_a∈{0.1,1,10} on validation only.
- Execution: daily UTC close, target weights, skip |Δw|<0.2%, taker 21 bp graded (maker arm diagnostic); graded by unmodified `cell_metrics` AND `eval_vector` (882 two-instrument discipline).
- Guards: 887 attractor unreachable (anchor is the GP identity, not zero; δ cannot flatten >half the book; flat-FAIL bands); 892 re-trim decomposition is a gate (negative maintenance component with CI<0 deletes δ); 813 multiplicity = N_test 1, SR0(N) printed (N=10→0.97, N=27→1.26); 20 circular-shift placebo trainings; ℓ1-MPC closed-form control arm — learning must beat the best closed form or be deleted.

**Odds (self-assigned)**: channel bounds: cost channel +0.05–0.10, variance +0.05–0.15, calibration (TC 0.6→0.75) +0.15–0.25 ⇒ total plausible Δ +0.15–0.35, expected +0.10–0.20. P(915 G-A passes) ~30%; P(GP book ≥+0.60 | G-A) ~50% (the ADMISSION gate — below it the row is not built); P(δ adds Δ≥+0.10 with CI>0 | admitted) 30–35%; P(goalposts) ~4%; **P(δ is the reason) ~2%**. Power admission: SE(Δ_SR) ≈ 0.17 (5.5yr)/0.25 (2.5yr) ⇒ a true Δ=+0.20 is ~1σ; NULL is the pre-registered most likely outcome and is scored FAIL.

**Kill experiment**: pure paired test, few CPU-hours + 2–3 agent-days (on top of 915 artifacts): frozen W_GP vs W_GP+δ on identical panel/forecasts/scorer; 892-calibrated block bootstrap on the daily Δ series; placebo 95th-pct gate; ℓ1-MPC comparison gate; channel ablations reported for attribution. PASS needs all of: seed-median Δ>0 CI lo>0 on ≥2/3 seeds; goalpost window ≥+0.97 (MARGINAL if ≥+0.56); Δ > placebo 95th pct; Δ > MPC; hold-positions bands; BEAR22 >0 and Δ(BEAR22) ≥ −0.1; 2026H1 > 0; turnover ≤1.5× GP's.

**Numbers worth keeping**: at exactly 915's floor (IC=0.02, N_eff=8): gross 0.65, net ≈ +0.40 at τ=0.08 — below both bars; δ would have to add +0.57. Cost drag 0.15–0.31 Sharpe at τ 0.05–0.10/day ( perverse: market-neutral book's lower σ makes the same turnover cost MORE Sharpe than the 819 book). λ calibration trick: match quadratic to linear cost at d=0.5% trade ⇒ λ ≈ 480 for an 80%-vol asset. Lit anchors: Butler-Kwon 2023 IPO vs two-stage Sharpe 0.759 vs 0.374 (GROSS of costs, commodities); Lim-Zohren-Roberts 2019 Exhibit 8: cost-in-loss rescues LSTM from −5.31 to +0.91 but only to benchmark parity (+0.86), not above; Zhang-Zohren-Roberts 2020: end-to-end gap shrinks +0.38→+0.10 as cost goes 1→10 bp; Cederburg 2020: 8/103 vol-managed strategies significant; repo's every hand-sizing tweak read ≤0 (−0.06/−0.09/−0.16).

**margin fact: kills it** — β-neutral ⇒ ~50% short continuously ⇒ +0.44–1.76 Sharpe of uncharged continuous drag at σ_book 0.25, i.e. 1.3–6× δ's entire bounded plausible increment (+0.10–0.35); even the +0.60 GP admission-gate read is borrow-inflated. The design listed this in open_doubts but no gate charges it; as a goalpost contender the learned residual cannot pay the short leg's rent.

---

## 5. Valerian refutation (REFUTER, no-free-lunch/arithmetic lens, `result_a19c9e08b5706d49f.json`)

**Role**: refuter of "Valerian" — hourly market-residual signal ladder (1h–7d lookbacks, 1h–24h targets, pooled ridge+GBM twin, GP aim-tracking calibrated to 21 bp, hourly decisions, daily-aggregated grading). Design itself lost; refutation is self-contained. **Verdict: REFUTED.**

**Decisive argument**: the band's required IC looks small only because quoted per hour. In daily units it is the largest signal ever asked of this venue: HL 6h ⇒ IC_1h 0.037 = daily-equivalent IC_24 0.065 = gross Sharpe 5.3 while paying 82% of gross in fees (34.5 of 42.3 bp/day); HL 24h ⇒ IC_24 0.052, gross Sharpe 2.7, 64% fees; HL 3d ⇒ 0.037, 1.7, 44%. The daily surface has never produced IC above its |0.059| four-draw noise band; the repo's only hourly-cadence read (816 H=60: gross +2.03, turnover 1.40, net −107 @21 bp) implies IC_1h ≈ 0.008 — one fifth of the HL-6h line. Second: hourly cadence buys ~nothing where TASK_915 already reaches — against a DAILY GP-smoothed book on the same signal the hourly GP saves ~4% IC at HL 24h and ~0% at HL 3d (the advertised 22–25% is vs a strawman raw daily re-rank). Unique territory is HL<12h, exactly where the requirement is highest AND where stale-tape bid-ask bounce (ρ1 ≈ −0.10 for a 50 bp spread alt) manufactures uncapturable reversal IC 0.05–0.1 — and the design's pre-registered guard is vacuous ("enter at open of t+1" = same print as close of t on a stale tape; 816's G8 actually entered at close of t+1, one full bar later).

**Odds after refutation**: P(G-A as written passes) 15–20%, mostly a FALSE pass via the bounce artifact. P(genuine true-skip-robust cell clears its line) 4–6% (design said 12%). Unconditional P(+0.97) 1.5–2%; P(≥+0.56) 3–4%. Most likely outcome (~60%): hourly IC 0.005–0.02, t≫3, below the line — the same "real, cost-bound" shape as 806/811/816/897.

**Surviving mechanism / kill**: hourly DIAGNOSTIC IC surface folded into TASK_915 as a second panel (~1 CPU-day): h∈{1,2,4,6,12}h, ridge only, with (i) TRUE one-bar-skip entry (close of t+1) plus first-print-≥10-min arm; (ii) stale-hour screen from trade_count; (iii) majors-only vs all split; (iv) required-IC line on MEASURED tau/TC/realised book SD; (v) Bonferroni + SR0. HL≥1d cells dropped as redundant with 915. Its FAIL composes with 915's G-A fail into the memo's pre-committed D4 repeal.

**Numbers worth keeping**: IC_24 = IC_1h·(1−ρ²⁴)/((1−ρ)√24) conversion; design's own arithmetic verified correct (IC* = (504·τ + 22.1/√N_eff)/(2130·TC)); fee shares 82%/64%/44% at HL 6h/24h/3d; 10 bp spread overrun adds 6–16 bp/day ⇒ IC* +15–45%; bounce ρ1 formula −(s²/4)/(σ²_1h+s²/2). Lit re-checks: Fischer-Krauss-Deinert 2019 = 7.1 bp/day NET = ~2.6%/yr on a 6-name L/S book (Sharpe 0.05–0.15) over 3 months — "real, cost-bound", not a precedent; Kitron-Wengrowicz 2026: 1.3 bp gross vs 5 bp cost at 15 min on Binance; Bianchi et al is a DAILY-horizon study cited for hourly structure.

**margin fact: kills it** — continuous hourly market-neutral book rolls borrow every 4h on ~half its notional: 22–44%/yr (BTC) to 44–88%/yr (alts) on the short half ⇒ 6–12 bp/day on book, vs a total cost line of 34.5 bp/day at HL 6h (adds 20–35% to an already-impossible IC requirement) and 5.9 bp/day at HL 3d (borrow alone exceeds the entire fee budget). The refuter flagged "unmodelled margin rollover" as one G-B risk; the confirmed rates make it decisive on top of the refutation.

---

## 6. Comfrey refutation (REFUTER, no-free-lunch/arithmetic lens, `result_a42177ecf79fe2dac.json`)

**Role**: refuter of "Comfrey" — contextual learner of per-asset TACTIC PnL: 8 shadow deterministic OHLC rules (trend-10/30/90 sign, 1d residual reversal, 5d xs reversal, 21d xs momentum, vol-targeted trend, constant-mix) generate supervised targets y = [p·e − c|Δp|]/σ̂; pooled LightGBM/ridge over ~30 features; β-hedged GP partial-adjustment book, turnover ≤0.06/day; claimed odds 5–7%. Design itself lost. **Verdict: REFUTED.**

**Decisive argument**: at the design's OWN admitted gate values the book cannot pay for itself. K1 admits IC 0.02 (8 bp/day at σ̄ 4%) and K1'' admits IC e-folding τ_IC = 5 d ⇒ a position's LIFETIME gross = m·σ̄·τ = 8 bp × 5 = 40 bp < its 42 bp round trip — it never breaks even with infinite patience. Generalised (cost drag 7.2/H at N_eff 10, σ̄ 3.5%): net ≥ +0.97 needs τ_IC ≥ ~250–300 d at (IC 0.02, N_eff 10); ≥55 d at (0.02, 20); ≥30 d at (0.03, 10); ≥16 d at (0.03, 20) — the 5-day gate is 3–60× too lenient. The surviving object ("slow weeks-to-quarter residual signal at IC≥0.03, breadth≥20") is exactly TASK_915's h=7/21 cell; the tactic transformation subtracts: 1–5d arms unmonetisable by construction; the cost term INSIDE the target is learnable from features alone (per-date IC 0.028 for k1, 0.05 for k4 with zero return predictability, NW t≈13 — passes K1 as bookkeeping; only a sign-flip placebo exposes it); cost charged again in the controller (double count + spurious momentum tilt ≈ +0.05 σ-units from k4 alone = 2.5× the required m); the Stage-0 stop uses an ex-post ORACLE selector that earns max(p·e,0) ≥ 0 daily and can never fire. Also: the β-hedge removes the market-trend component = the record's ONLY measured positive (BEAR22 short leg +1.095); 816 H=1440 daily market-neutral cells read gross −2.16/−3.33 — a prior against the xs arms; mirror arms are exactly anticorrelated (zero information); DoF budget n_eff·SNR ≈ 4–20 parameters ⇒ ridge is the efficient estimator and the GBM is variance-dominated; multiplicity SR0 ≈ 1.0–1.2 on 2024–26 ⇒ "+0.97" is inside the null.

**Odds after refutation**: as specified, P(K2 ≥ +0.97 by real edge) ≈ 0 (any pass is more likely the multiplicity null). Corrected narrowest version: P(goalposts) 1–2%, P(≥+0.56) 3–5% — at or below the tactics panel's 3–5%, not above it.

**Surviving mechanism / kill**: ONE pre-registered cell block inside TASK_915 (minutes on top of it): GROSS residual YZ-standardised target at h ∈ {7,21,63} (63 d to test the ≥50-day persistence the arithmetic requires), cost kept OUT of the target and charged once in the controller; features = 915's F1–F3 + sign-interacted block s_L⊗x_regime (sign of RESIDUAL L-day return, L∈{30,90}) + per-asset trailing tactic-PnL mean as one ridge column; ridge only; both placebos (sign-flip mandatory); measured N_eff from the residual correlation spectrum (closes TASK_825's parked item); τ_IC gate DERIVED from the goalpost at measured N_eff (≈55 d at N_eff 20/IC 0.02 … ≈250 d at N_eff 10/IC 0.02), not a fixed 5 days.

**Numbers worth keeping**: the persistence-vs-cost law: for a book renewed every H days, drag = 7.2/H Sharpe (N_eff 10, σ̄ 3.5%, 21 bp); required τ_IC table above. Learnable-cost-term IC ≈ 0.028–0.05 (the bookkeeping trap — any design with cost inside a learned target must add a sign-flip placebo). Attenuation of raw-sign vs residual payoff ≈ √(1−R²_mkt) ≈ 0.63 at R² 0.6. Range-ratio feature SE ≈ 0.28 (not 0.12 — ratio, denominator-dominated). 819 calibration cross-check: trend book = m 0.033, N_eff 1.4, H≈13d, TC≈1 → net +0.56 — the framework is not biased against designs; "the trend book worked because its signal outlived its holding period".

**margin fact: kills it** — the β-hedged book is ~50% short at turnover 0.06/day; at 22–88%/yr on shorted notional that is 11–44%/yr on book, dwarfing the mechanism's entire net ceiling (which was already refuted on τ_IC arithmetic alone). Independently: it also softens the +0.97 comparison bar (the trend-book reference never paid its bear-short borrow), but no re-grading rescues Comfrey.

---

# Cross-cutting synthesis

**Convergence on TASK_915.** Four of six reports' surviving mechanism is a 915 appendix/cell block: Valerian refuter (hourly IC second panel), Comfrey refuter (sign-interacted ridge cell at h∈{7,21,63} with derived τ_IC gate), SCRR refuter (amended Stage-0 kappa table + F3 event features), HCB designer (defers h≥1d to 915). ARC-GP is explicitly gated on 915 G-A/G-C passing. The panel's aggregate verdict: 915's daily IC surface is the single load-bearing next measurement; everything else is a cheap appendix or dead.

**What the refutations killed.** Valerian (hourly ladder): required IC = largest ever asked (gross Sharpe 2.7–5.3), bounce artifact unguarded, redundant with 915 at HL≥1d. Comfrey (tactic-PnL learner): lifetime-gross-below-round-trip law (40 bp < 42 bp at its own gates), cost-term-in-target bookkeeping, oracle stop that can't fire. SCRR (shock-cohort book): same-close entry bakes bounce into kappa (bias 0.04–0.16 vs PASS bar 0.08), hedge fee omission (real round trip 111 bp), log-space tail 4× worse than claimed. All three refuted=true; all three survivors are measurements, not books.

**The three surviving designs' honest self-odds**: SCRR ~7% (→2–4% after refutation), HCB ~10% for +0.97 anywhere / 1–2% for the fast band, ARC-GP ~4% (2% that learning is the reason). Every designer's own arithmetic says the most likely product is a decisive negative measurement, and the designers mostly say so.

**Margin-fact scoreboard (post-hoc)**: continuous market-neutral books (HCB, Valerian, ARC-GP, Comfrey) are killed or near-killed — 11–44%/yr drag on book (≈0.4–1.8 Sharpe) that no gate charged and no plausible learned increment covers. SCRR is merely wounded (it pre-charged borrow at the confirmed rates; Arm B dies, Arm A survives on cheap BTC-hedge carry). Refutations of SCRR and Valerian already contained the borrow at declared rates — no verdict flips, margins widen. Systemic point: the +0.97/+0.56 bars were set by a borrow-uncharged trend book, so the bars themselves need re-grading (charge the trend book's short-leg borrow) before being used to kill mostly-long designs; that re-grade can only lower the effective bar.

**Cheapest decisive next actions, in order**:
1. TASK_915 G-A daily IC surface (everything is gated on it; ~1 CPU-day).
2. SCRR amended Stage 0 inside 915's data build: κ with first-traded-print entry, bounce gap per liquidity decile (~2 CPU-h).
3. Comfrey cell block: sign-interacted ridge, gross target, h∈{7,21,63}, derived τ_IC gate, sign-flip placebo (minutes on top of 915).
4. Hourly diagnostic panel h∈{1,2,4,6,12} with true skip entry + stale-hour screen (~1 CPU-day) — closes the last unmeasured OHLC band two-sidedly; its FAIL composes into the D4/D79 repeal.
5. Zero-cost prerequisite flagged by three reports independently: write down the operator's actual Kraken fee/margin tier, and add a borrow line to the scorer (then re-grade the trend-book bar).

**Dead-agent angles**: the variance-ratio/Hurst autocorrelation-regime book (trend on persistent assets, reversal on anti-persistent, from OHLC roughness; Dao et al. 2016 λ/2·σ²·(VR−1) payoff identity) died with no analysis and remains unmeasured anywhere in the record — the only structural angle from this panel with no surviving representative.
