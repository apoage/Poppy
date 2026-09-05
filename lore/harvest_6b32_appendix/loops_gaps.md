# Gap report: raw panel reports vs `lore/fable_self_learning_loops_panel_2026-09-05.md`

Source: 15 files in `/tmp/harvest_6b32/wf_650304b3-151/`. All 15 read in full.

## 1. Roster and verdict cross-check

Explorers (5):
- `a5ec91b28724d9956` — continual/few-shot/meta (odds 1–2% goalposts, 3–5% PIT)
- `a65bc2a1435285311` — execution-only (P(change any goalposts verdict) ≈ 0)
- `abec1738bc08a93b8` — world model serving MC leaves (1–2%)
- `ae76753b16b893427` — adversarial/minimax self-play (0–1%)
- `afb15a124706c2ca4` — evolutionary/GP rule library (2–4%)

Refuters (10), all `refuted=true`:
- Execution: `a03fddbea8fd788d9` (amplifier), `abcb4871debcb315a` (repo-record)
- Continual: `a7c457aee653fdbaf` (amplifier), `a2f70b8a77a1e5859` (repo-record)
- World model: `a39629dfb17a6086d` (amplifier), `ac91572ec178de858` (repo-record)
- GP: `a6589bc00107b0381` (amplifier), `ace723fafc6b35c9a` (repo-record)
- Adversarial: `af91a35eabc5b3d49` (amplifier), `af7a3c12ca3b25d44` (repo-record)

Verdicts: no raw verdict is missing from the lore file (10/10 refuted, 5 explorers self-refuting,
"0–5%" matches every honest_odds field). Two places where the lore file states something a refuter
explicitly rebutted:
- Continual table row "negative: re-estimation = re-trims" — `a2f70` shows 892 measured frozen-policy
  weight adjustments vs matched coin flip; no parameter was refit and refit frequency was not a
  variable ("analogy presented as measurement"). The 819 TEST B2 refit loss is the correct receipt
  for that claim (below).
- Continual row "detection window for IC 0.02–0.03 exceeds regime lifetimes" — `a7c45` shows this
  rests on σ_date = 1/√N_eff ≈ 0.35; at the corrected SD (lore point 4 itself: ≈0.06–0.07) w* is
  19–100 days and a 3–6-month rolling estimator may be ABOVE the floor. The lore supplies the
  correction (point 4) but keeps the uncorrected conclusion in the table.

## 2. Salvage content in raws but absent from the lore file

Execution family:
- **Rails conflict (load-bearing)**: the mandatory terminal market cross and 15–60-min rests are
  forbidden in code (`phase2_rails.py:170-178` post-only only, "never a market order, ever";
  auto-cancel 1–2 bars; TASK_907:629-636; D65/D68/D78 no-fifth-round guard). Any real-order phase
  needs a new operator ruling, not a reuse. Absent from lore.
- **Tail coverage**: 4 majors ≈ 1.5% of book turnover (~19 sign-flip orders/day over ~250 assets,
  ~7k orders/yr); a majors-measured tier repeats the fee16/slip5 transfer error; re-scope to tail
  pairs, label by regime. Absent.
- **STEP 0 is nearly powerless**: gap-free 15–60-min episodes exist only in the last ~12.8 h of the
  4.117-day capture (≈13 independent 1-h windows/pair, one UTC-midnight slot); hour-of-day
  regression not estimable. K2's state-quintile t≥3 on 3 bp needs ~4,000 orders (~50 days) before
  Bonferroni. Absent.
- **Estimand hygiene**: 907 estimand-1 (post-fill markout) is parent exposure, not execution cost;
  only the non-fill branch (drift-to-deadline) enters. Absent.
- **Missing arm**: live trader already runs "limit, then market after 60 s"
  (`config/real_trade_config.py:20,23`) — the baseline any measurement must beat. Absent.
- **Tier arithmetic**: book trades ≈4.65×NAV/30d, taker ≤18 bp needs NAV ≥ ~$215k; two fetched
  Kraken ladders disagree; TASK_809 once assumed a maker REBATE (−2..−5 bp) vs real 4–25 bp. Absent.

Continual family:
- **Post-listing event study** (explorer open-doubt 7 + refuter surviving-claim 3): first 60–90
  days post-listing is the one regime class with shared cross-asset structure; direct event-study
  IC with shifted-listing-date placebo, CPU minutes, as a 915 F-family. NOT in lore — the single
  largest omitted salvage.
- **819 TEST B2**: walk-forward per-regime-refit ridge lost 4/4 OOS regimes (net@21 −2.34/−2.41/
  −1.60/−2.09 vs TSMOM +0.29/+2.21/−0.23/−0.28; OOS IC −0.04..+0.05). The direct in-repo
  precedent for "adaptive forecaster". Absent.
- **−0.16 qualification**: it is the 20-asset SURVIVOR universe, 2021-01→2025-09, net@8, never run
  PIT; broad universe −0.09; per-regime sign flips. Lore cites it unqualified.
- **τ test power**: 30 months → t≥2 needs ρ ≥ 0.27–0.37; at N_eff≈8 even perfect persistence reads
  t≈1; SURVIVE reachable only if IC ≥ 0.05 or N_eff ≥ 30–40; pre-register power vs measured σ_date.
  Lore has the statistic but not the power curve.
- **Oracle null**: with inter-book ρ 0.6–0.8 and yearly SE≈1.0, ex-post best reads ≈ A+0.5–0.75 by
  selection alone; "oracle ≥ +0.97" fires 30–45% by chance under null. Absent.
- **Conditional window rule**: if G-A passes and φ_h large, ridge refit window = pre-registered
  Inoue–Jin–Rossi choice inside the GP book ("the only legitimate form continual learning takes
  here"). Absent.

World model:
- **758 metric normalisation**: per-step oracle-PnL ratio 7× too easy BUT cum-PnL ratio 0.03×
  (Bella harder) for the same runs; the disqualifier is normalisation-sensitive and must be
  pre-registered. Lore carries only "7× too easy".
- **Volume premise**: the true-daily builder DOES carry volume/trade_count (`trader/data.py:147,160`);
  "volume dropped" applies only to 806/819 minute-derived caches — don't exclude volume features on a
  false premise. Minute table ends 2026-03, so minute-cache RV targets can't cover 2026H1. Absent.
- **MLP/flow-implied-mean as one extra forecaster column** in 915's grid (≤1 GPU-day) as the hedge
  for "F1–F4 miss interaction structure"; 819 CAPACITY too weak to close that doubt. Lore records
  the overclaim but not this hedge option.
- **K arithmetic correction**: leaf-noise-vs-signal needs K ≈ 1/(IC²·N_eff) ≈ 312 (not 2500);
  Grinold E ≈ IC·√N_eff·σ ≈ 20 bp/day at N_eff 8. Optimiser's curse ≈ σ·√(2 ln K_cand); 831x/D's
  K is candidate books, not rollouts. Absent.

Adversarial:
- **Null-family failure mode**: global reflection is an exact invariance of sign-rule tactics
  (PnL invariant), blockwise time-reversal preserves autocovariance, 21-day blocks preserve 7d
  reversal dependence — reversal/reflection surrogates do NOT null autocorrelation tactics; correct
  family is circular shift + per-date cross-sectional permutation. Lore picked the right family but
  doesn't record why the tempting one fails.
- **Bella skeleton has explicit drift knobs** (`skeleton/types.py:35,50` trend_direction,
  alpha_shape) — pin to zero for R4 stress use; and Bella emits ONE trading day per window (400-step
  minute training), so a daily surrogate panel is ~900 stitched windows. Absent from lore's R4 item.
- **TASK_110 record hygiene**: banner "Phase 1-4 Complete" is stale — generator update is a TODO
  (`adversarial_training.py:410-411`), checkpoint dir empty, only a 2-epoch smoke (regret 0.689→0.719);
  TASK_110 id collision with ADAPTIVE_BURN_CONTROLLER. Both refuters flag for disarm. Absent.
- **Concrete DRO**: δ_n = O(1/n) via RWPI at 95%, n ≈ 910 days on 2024-01→2026-06; compare √δ_n to
  ‖μ̂_resid‖ once the IC table exists; suggest a one-line LEDGER/DECISIONS closure citing Prop. 1.
  Absent (lore has the formula, not the instantiation).
- **830/E0-REAL oracle ceiling** +29.7 net@21 / +75.5 per-asset gross alpha — why PAIRED regret
  saturates at 1. Absent.
- **Scope doctrine**: pre-E-c LEDGER verdicts are PROVISIONAL; 758 is scalar-era, citable as history
  only. Absent.
- EW/B&H by-regime numbers (gross −0.40 FULL, −2.54 BEAR22, −2.28 BEAR18, −0.29 BULL24, −1.63
  CHOP25) — the measured 1/N limit. Absent.

GP/evolutionary:
- **In-repo E[max] observations**: 813 best-of-132 +5.81 (n_events 6) → 0/48, winner −4.08 on
  disjoint hold-out; 819 best-of-8 +0.97 failed BH q=0.05 (p 0.010) → PIT +0.56. The law observed
  twice locally; lore cites neither. Keep with the SR₀(N) rule.
- **CPCV is a no-op for parameter-free rules** (all "OOS-path" Sharpes identical = full-sample
  in-sample; 813's disjoint-span split is the stronger design). Reusable caution. Absent.
- **Hold-out coin flip**: 2026H1 σ_SR ≈ 1.42 (875's 1.02–1.99 seed spread is the in-repo
  confirmation); P(pass|0) ≈ 0.25, P(pass|+0.97) ≈ 0.50; Bailey–LdP: holdouts don't control
  overfitting. Absent.
- **Build facts**: in-tree shift nulls are PER-ASSET (`tools/candle_lang/test_shift_null.py`,
  `804/build_shift_null.py`) — the common-offset variant is new code (~100–200 lines); `bh_fdr`
  (backtest.py:57) is the only multiplicity tool in-tree; no DSR/E[max]/RC/CPCV code exists; the
  pinned PIT panel is close-only (`_close_pit_daily.npz`), so range/volume terminals need a pinned
  PIT OHLCV frame (shared with 915). Absent.
- **GP N is undefined** (correlated offspring; T < ½M(M−1) by 5 orders makes correlation-based N̂
  "pointless"); N exact only on hashed grids. DSR at N=25,000 sets SR₀ +1.47 (7.7y)/+2.58 (2.5y) —
  power ≈0.001 against a true +0.97. Absent.
- **Conditional gated grammar**: N ≤ 50 hashed IF-vol-regime-AND-trend grid allowed IF the
  VR-persistence diagnostic is nonzero AND G-A passes a gated cell; graded by IC not book Sharpe;
  readable rule = auditability concession ~30–50% of gross IR (TC 0.5–0.7). Absent.
- **H=3d reads +0.90 PIT** vs +0.56 at H=1 (POWERED robustness) — a row-2 number. Grammar must
  exclude/attribute H and hysteresis or cost savings report as edge. Absent.

## 3. Honest-odds numbers not in the lore file

- Execution: P(rule-based maker execution saves ≥5 bp/side) ≈ 50–60%; P(learned beats rule by
  ≥2–3 bp) ≈ 20–30%; realistic saving +0.03–0.05 Sharpe at the recorded maker {8,16} tier
  (vs lore's +0.06..0.12 at 21→8–12); infrastructure value contingent on the D79 pivot staying on
  Kraken spot (funding/basis = perps).
- Continual: P(kill experiment SURVIVE) ≈ 10%; P(KILL-1) 60–70%; P(KILL-3 oracle < +0.97) 55–65%.
- GP: P(archive contains a readable ≥+0.56 PIT rule | G-A PASS) ≈ 25–35%; P(loop without gates
  reports a bar-clearing rule) ≈ 1; P(finds edge 915 would miss) < 1%.
- World model: P(null-engine role helps calibration) 60–70%; P(σ-skill vs HAR-RV) 50–60% — though
  the amplifier refuter shows the null-engine item is dominated by the model-free placebo.
- Adversarial: P(P2 PASS) 5–10% (and the repo-lens refuter shows that is mis-calibrated: 2026H1
  CI half-widths ≈3 Sharpe make P2 need ≈+3 in six months).

## 4. Literature worth keeping (author-year + number relied on)

- Wood–Kessler–Roberts–Zohren 2024 (X-Trend, arXiv 2310.10500), full-text verified: Table 1 2018-23
  Long 0.48 / TSMOM 0.23 / MACD 0.27 / DMN 2.27 / X-Trend-Q 2.70 (+18.9%); Table 2 zero-shot TSMOM
  −0.26 vs X-Trend 0.47; C(i)=0 verified; CPD adds +11.3%. DMN break-even 2–3 bp/side (tactics panel).
- Cakici et al. 2025, JF 80:3771–3790: verified sentence "Rectifying this error qualitatively
  alters the findings, invalidating most conclusions"; the "little help" sentence is NOT in the
  Wiley abstract and the FST reply is unread — cite as contested, not settled.
- Yu et al. 2019 (arXiv 1901.08740), full: 7 equities + cash, single run no CIs, fees 20+50 bp;
  Table 1 generator module (DAM) adds 0.00–0.01, forecaster (IPM) supplies the lift.
- Aghapour et al. (arXiv 2507.09916v4), full: FF-10 monthly walk-forward 2010-03..2025-02,
  long-only, costs excluded, fixed seed; Table 4 EW 0.883, GenTD3 1.225 — and the ablation tell:
  short-context MarkovTD3 0.8657 sits BELOW equal-weight (the gain is long-history conditioning =
  a forecast).
- Spooner–Savani 2020 (arXiv 2003.01820), full: robust MM terminal wealth 67.0±12.0 → 65.1±6.7
  (mean FELL), footnote 3 defines "Sharpe" as a terminal-wealth ratio over 10⁵ episodes scored on
  the nominal adversary; single-asset, spread income. Blanchet–Chen–Zhou (1802.04885): §3.1
  equal-allocation limit holds under the budget constraint φᵀ1=1 — the unconstrained "flat" limit
  is an extrapolation.
- Allen–Karjalainen 1999 (JFE, full): evolved trees "effectively a 250-day MA"; −2.05%/yr OOS at
  0.25% cost, negative 9/10 periods. Bajgrowicz–Scaillet 2012 (full): FDR⁺ detects 52.6% vs
  Romano-Wolf 0.6%; <5% of rules survive one re-selection; 5–35 bp costs erase OOS. Sullivan–
  Timmermann–White 1999 (full): RC p 0.90/0.99 on OOS/S&P futures where nominal p 0.04/<0.002.
  Hudson–Urquhart 2021: MHT-corrected TA on altcoins (not BTC) — the strongest crypto
  counter-evidence, pre-2019 sample. Anghel 2021 (crypto RC). Hsu–Kuan 2005 (young markets).
  Harvey–Liu–Zhu 2016: t>3.0 hurdle (915's gate). Bailey–LdP 2014 + MinBTL (E[max] formula,
  N=46-vs-100 flips DSR; holdout warning). Alpha-GPT IC 0.0095–0.0115 → 0.0216–0.0276 at breadth
  3,000+ stocks.
- Execution lit, full reads: Nevmyvaka–Feng–Kearns 2006 (V=5k/10k shares, H=2/8 min, fees assumed
  negligible; market-var increments +7.97/+8.69/+12.85%); Ning et al. 2018 (0–12 bp of TWAP P&L,
  nil 2/9); Digital Finance 2023 Kraken study (static kernel 26 → 10.1 bp — the crypto saving is
  fee-structure, not learning; verified at snippet level only); Schnaubelt 2022 (10 BTC, abstract);
  Moallemi–Yuan 2016 (queue value ~ half-spread).
- Continual-theory: Pesaran–Timmermann 2007 (optimal window includes pre-break data);
  Smith–Wallis 2009 / Claeskens et al. 2016 (combination puzzle — prices selection > averaging);
  Inoue–Jin–Rossi 2017; Timmermann 2008; Hurst–Ooi–Pedersen 2017 (equal-weight 1/3/12m default);
  Wang–Lera 2026 (FinPFN, no-cost Sharpe 9.8 on A-shares — unverified snippet).
- Negative-result citations worth one line each: no peer-reviewed Dreamer-on-trading OOS net-cost
  Sharpe found in three searches; DeMiguel–Garlappi–Uppal 2009 (14 models never beat 1/N);
  SFAG 2026 (baseline GAN backtests >2000% ann.).

## 5. Covered — no action

Iterated-expectations identity; both attractors; Kraken ladder tiers and the fee16 inconsistency;
frozen-mid 38.9% events-vs-time; adverse-selection 0.3–0.5 → 0.04–0.05 correction; 819 CAPACITY
overclaim (nonlinear above ridge inside noise band); per-date IC null SD ≈ 1/√(N−1); MC attenuation
formula and K ≥ ~25,000; block-Rademacher/circular-shift null replacing the four-draw floor;
monthly-IC τ with overlap-aware null; multiplicity SR₀(N) rule with MinBTL N ≤ ~50; DRO shrinkage
gate μ̂ᵀφ − √δ‖φ‖ − cost > 0; 849/E5-inversion3 mechanism and the 887 2026-08-19 correction;
Bella minute-bar/survivor/7×-too-easy rebuild requirement; X-Trend C(i)=0; compute story
(no GPU needed, 1 GPU-day world-model bound).
