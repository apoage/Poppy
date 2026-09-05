# Mining gap report — subagent facts NOT in `lore/fable_direction_memo_2026-09-03.md`

Sources: final reports of the four 2026-09-03 mining subagents (transcripts under
`~/.claude/projects/-home-lukas-Trader/6b32cd7f-.../subagents/`, agents ac730f…/aceb1f…/ad2e23…/af4b3f…),
diffed against the memo's used claims. All receipts below are absent from the memo unless noted.
Two bonus items (marked †) were verified directly in primary files during harvest.

## 1. Facts that constrain the memo's own TASK_9xx design

- **Data-availability ceiling on the G-C gate.** Minute panel ends **2026-03-31**; minute bars
  Apr–Aug 2026 are **not API-fetchable** (Kraken v0 OHLC serves a rolling ~720-candle window:
  minute ≈ 12 h, hourly ≈ 30 d, daily ≈ 2 y; no v2 candles REST; deep trades-paging rejected) —
  they wait on Q2/Q3 dump zips; 899's staged catch-up has **no DB writes until the RO02 drive
  fix** (`lore/notes_2026-08-31.md:55-72`). Memo G-C tests 2024-01→2026-06 on *both* instruments;
  the `eval_vector` in-env leg cannot be fed past 2026-03-31. `pathfinder_audit:161-162` flags
  this as the binding constraint on any future OOS claim ("every 'recent' window is ≥4.5 months
  stale; verdicts age in dog years", `:105-109`).
- **Minute hole inside the test window**: near-empty 2025-03-31→2025-10-01 on **11 of 64 slots**
  (D31, `tasks/DECISIONS_ARCHIVE.md:16`); D25 ghost cluster 2025-12-08→12 (139,157 rows) sits in
  `ohlcv_quarantine`. Any minute-aggregated daily panel inherits these.
- **`ohlcv_daily` is dual-source**: `source='agg1m'` (398 assets, 2013-10-06→2025-12-12) vs
  `'kraken_rest'` (376 assets, **2024-07-15**→2026-07-04) (`trader/data.py:393`, `trader/paper.py:91`,
  `:320-323`). REST-only assets have no pre-2024-07 daily history; table covers **only 33/64**
  canonical slots (breadth floor 0.9 refuses the daily pin, `TASK_865:196`); 499,430 rows/411
  assets; per-year breadth 2013→2, 2017→13, 2018→16, 2020→50, 2021→93, 2023→229, 2025→404;
  integrity 0.0000% mismatch vs 1-min agg on 50 sampled assets (`notes_2026-07-05.md:35-39`).
- **893's caveat is load-bearing for the new featurizer.** The leading **backward** fill is real
  (29/39 assets carry future-stamped prices in leading rows; 4,904 fabricated decision cells =
  7.38%, all in 2021–2023); +0.97 doesn't move only because the trend book routes through
  `present_mask`; "any consumer reading `close_d`/`logc` WITHOUT routing through present_mask —
  a different signal, a model featurizer, an eval harness — gets no such protection"
  (LEDGER row `893/prefix-cells`; positive control moved 11/12 metrics).
- **Instrument divergences bearing on G-C's "agree within 0.1"**: `trader/` engine scores **log**
  returns (`trader/backtest.py:205-228`), vector env **simple** PV returns
  (`eval_vector.py:2871-2874`); `trader/` has **no execution lag** (pos[t] fillable at close(t),
  flagged optimistic in 819 `RESULT.md:148`); **BEAR22 differs by instrument**
  (`trader/config.py:72-78`: 2021-11-10→2022-12-31 vs scorecard `bear22w` 2021-12-15→2023-01-15).
- **816's turnover-invariance mechanism**: turnover stays **≈1.39 per rebalance at every H**
  because the cross-sectional model re-ranks the whole book each rebalance — lowering frequency
  doesn't lower per-rebalance turnover; daily cells were **gross-negative before cost**
  (full64 −2.16 / clean33 −3.33 / majors15 −0.48; `PHASE1_RESULT.md:35,51,67,80-83`; `probe.py:254`).
  A structural receipt the GP-controller design must beat, not just the minute-feature point.
- **PIT builder specifics**: `_close_pit_daily.npz` = 276 cols × **3,117 daily rows**; 273 pass
  (BTT/GST/MATIC dropped by active-window liquidity filter); rules: USD quote, no stable/fiat/gold,
  no wrapped/legacy dups, span ≥1 y, zero-return frac <0.15, ≥365 active days on own window;
  16 assets carry the 2018 bear; residual survivorship admitted (DB holds only Kraken-ever-listed)
  (`POWERED_RESULT.md:21,27`; `trader/config.py:83-86`). Naming trap: 837's `build_pit_universe`
  was verified **NON-PIT** (`TASK_837:1`) — don't reuse it for the frame re-cut.
- **TASK_901 exists** (`experiments/901_range_based_dependence_probe/`, shipped `5c9932b`): tested
  {acf_band, variance_ratio, spectral_zero} as certified range instruments — **measured negative**,
  "no certified general range instrument, direction not closed". It is a bootstrap-block-length
  instrument on strategy PnL, *not* a return-predictability test — but the memo's F2 range family
  should cite it to preempt "you missed 901".

## 2. Diagnosed failure modes sharper than the memo's summary

- **892 decomposition** (LEDGER row 892): anti-edge lives in the standing book's **maintenance
  re-trims** (Δ −0.060/−0.062/−0.050, CIs exclude 0); **entries/exits are NULL** (under-powered,
  n≈514–600); not beta (forward market return ≈ −0.045 both arms). Directly relevant: the GP book
  also lives on re-trims — turnover is exactly the suspect channel. †Also in the LEDGER row but in
  no subagent report: the **secondary search leg was never measured** (`trace.py:635` hard-codes
  `policy-argmax`; "does search add per-action EV over argmax?" is UNANSWERED; no row owns it).
- **898 flattery diagnosis**: clipping shorts de-levered gross to **43%** of baseline (27.6% in
  BEAR22); leverage-matched FULL goes **+0.321 → −1.308**; BEAR22 baseline +1.095 vs variant −2.471
  ("BEAR-positive is the short side") (`TASK_898:4-17`).
- **844 postmortem**: no detectable short preference in value/policy/credit, 4/4 ckpts; failure
  mode = **"no profitable behavioral prior"**; the ≈6L/10S short book is rollout flow mechanics
  (no-same-step-flip assumption false in weight semantics, `env:1613-1630`)
  (`notes_2026-07-14.md:49-52`; LEDGER row 844).
- **Stage-B reward-head disease → null verdicts weakened**: direct probe 743/seed_7: across-action
  reward STD 5.34e-3 vs across-latent 0.323 (60.4×); "the 738–784 null verdicts STAND but are
  weaker evidence than assumed; small real effects could be masked"; eval-MCTS absolutes are
  `decode='joint'` artifacts (1/64 cap ≈1.56%) (`stageb_reward_head_diseased.md:26-31,:20-24`).
- **836/E4-p1b caveat**: ref imprint ckpts "likely have untrained value heads (BC-only) — this
  needs a subject check first" (`pathfinder_audit:49,99-103`) — sharpens the memo's "never
  re-measured" doubt.
- **887 audit corrections**: its "alpha ordered identically" line is arithmetically wrong and the
  "no edge to protect" gloss overreaches — the env contains findable edge (trend teacher
  positive-EV at fee0 in-window), so 887 measured a learning/search failure
  (`pathfinder_audit:39-40`).
- **E5-autopsy receipt**: same weights, operator alone flips **+0.839 → −6.92** (PV 299k → 6.4k);
  885: 16× sims range moves hold 0.13pp (not search depth — the distillation loop).

## 3. Unexploited observations / never-run items the miners flagged

- **806 encoder beat linear at long horizon**: frozen canon encoder IC **0.0836/0.1348/0.1226**
  at H=1/5/20 min vs linear 0.151/0.141/0.060 — at H=20 the encoder is ~2× linear
  (`runs/806_predictability_probe/result.md`). "Encoder-beats-linear" (2/3 outcomes) was never
  followed up on any daily panel; complicates the flat "linear-optimal" reading.
- **persistence IC −0.0519 at H=1** (1-minute mean reversion) — a real measured reversal never
  featured or traded.
- **816 tsmom_residual best cell**: clean33 L30/H7 **+1.53 net@3bp CI[+0.32,+2.88]**, 0/90 FDR —
  "a real-looking UNDER-POWERED LEAD, not a flat null" (`tsmom_residual.py:37`).
- **819 lever attribution**: "more paths +0.22" was the **only** lever that helped (risk-sizing
  −0.06, lookback-ensemble −0.16) — direct experimental support for the breadth thesis the memo
  argues theoretically (`diversified_trend_shipcandidate.md:22-23`).
- **Candle-language structure is real**: causal transformer 7.61 bits/tok vs bigram 8.19 — the
  +0.58 bits gain is genuinely order-dependent (LM-on-shuffled ≈ unigram) yet never converted to
  alpha (Δα ±0.12pp vs +3pp bar) (`candle_language.md:85-98`).
- **The 907 L2 capture as an unmined signal source**: 1.45 GB / 11.76M lines / 4.117 days /
  4 pairs (BTC-ETH-SOL-XRP), schema_mismatch=0, byte-identical on both hosts. Framed strictly as
  fill-model calibration ("cannot reopen 897"); the lore miner's explicit note: that framing "is
  a pre-registration commitment about this artifact, not a ruling against ever mining the
  capture; but no row proposes doing so".
- **Never-proposed classes (grepped, zero hits)**: rough volatility, options/vol-surface,
  market-making-as-strategy. Regime machinery that *does* exist and is reusable for the
  market-timing arm: TF-IDF + k-means regimes (`tools/candle_lang/stage3_regimes.py:46`).
- **Oracle receipts**: the 50,000% oracle figure was a k=4 4× leverage path-solver bug (honest
  65–136%/ep; selective 42 bp zigzag +754%/ep, ~1024 >42bp trades/ep); **blind control**: same
  big moves traded blindly at 21 bp/side = +115% gross → −85% net; model recall ≈0.02 was a
  pre-fix trade-count proxy — true direction-aligned recall **never measured**
  (`oracle_design_intent.md:22-26`; `collapse_fixed_predictability_wall.md:16-23`).
- **831/E2**: K>32 is net-negative (K16 +5.29 → K128 +4.68) — "extra candidates buy TURNOVER TAX,
  not alpha" — relevant if search returns as controller.

## 4. Record-integrity doubts not carried into the memo

- "**Can ANY pre-868 number be trusted even directionally**" stays open — 861 B/C isolation legs
  (alias-union vs UTC) never ran (`pathfinder_audit:92-97`).
- Imprint anchor-collapse cause **UNISOLATED** — four semantics changed at once
  (`imprint_oos_artifact.md:15-17`); fold-5 magnitude confounded by the sparse span (:37-39).
- Epochs **E-e (XBT↔BTC fold, 92,639 rows recovered, TASK_877) and E-f are still PENDING**
  (`LEDGER.md:17-18`) — memo names its epoch "post-E-d" without noting the two open seams.
- Contamination-audit leftovers: 739 (−2.63pp) / 741 (−6.98pp) never re-scored, "within
  plausible flip range"; 769 not locally reproducible (`verdict_contamination_audit.md:31-33`).
- **No trustworthy realized adverse-selection number exists yet**: every quoted markout is a
  defect repro (median 4000.52 bp contaminated limit price; |bp| to 7996.97; the flagged-span
  bug moves the published 1 m mean −0.00 → **+106.63 bp** by shifting one print 5 s); B2 blockers
  include `MidQuote` carrying no observation timestamp (`MAX_MID_STALENESS_S=60`,
  `shadow_engine.py:435`) and an undeclared emission policy that disabled the 2b detector.
- 897 extras: **0/188 cells clear even at adverse-selection = 0**; window-b best **−8.425**
  (`notes_2026-08-25.md:7-11`).
- 804's `real_fe` failed its cyclic-shift control 3/3 → VERDICT NULL (the fake-execution control
  discipline itself, never cited by the memo).

## 5. Ops facts (cheap wins / blockers)

- **913**: journal stalls = `udisksd` SMART poll (5,452/5,452 SG_IO EIO, ~600 s cadence, 0.39 s
  lead); pending operator action `systemctl mask --now udisks2` (not `stop` — D-Bus reactivates;
  not `smartctl` first — reading SMART *is* the trigger). Traps: `journalctl` realtime AND
  `-o short-monotonic` are receive clocks; never decode `/proc` `starttime` against `btime`.
  Duty-cycle vs D74's 11.56% unreconciled.
- **914**: `read_capture` peak RSS **14.83 GB** on the 1.45 GB capture; demoted behind B2 round 4
  because both rows rewrite `read_capture`/`_pair_coverage` (3,749-line hand-merge risk into the
  round carrying the permanent-halt trigger).
- **Governance**: D65's second half *already* schedules the operator's fresh-direction call —
  trigger sequence B2 ships → shadow analysis → probe artifact → D65 closure → operator call with
  875/892/898/897 + **906's dispersion finding** + 907's curve on the table
  (`notes_2026-08-29.md:101-102`). D78: the operator chose patch over re-cut against
  factory+planner recommendation. The memo does not acknowledge this standing process.
- Operator preferences with receipts: **idle GPUs are a failure mode** (tension with the CPU-only
  ceiling sweep); "simplicity is a constraint" (D72); box hardware out of scope (D75);
  gpu-server shared — disturb nothing (`QUEUE.md:39-43`).
- DB: canon `trading_data.db` 24.4 GiB, six tables (assets, ohlcv, ohlcv_daily, fetch_log,
  fx_rates, ohlcv_quarantine); fixture bug `fixtures/make_fixture.py:115` queries
  `data_type='ohlcv_1'` (matches nothing) → silently falls through to the `ohlcv_daily` ranking.

## 6. Misc concrete facts worth keeping

- **806 grid**: 987,904 rows 2023-05-15→2025-03-31, top-64 frozen by train bar count; purge g=20;
  strided N_eff 217,025/43,415/10,861; hit rates 3–5 pp above base rate; R² +0.0389 at H=1 but
  **negative** at H=5/20; features 64×63 lagged minute returns = 4,032 dims (volume SELECTed,
  stored, never read).
- **CAPACITY panel detail**: built from `_close_crossregime.npz` (2,453,824×20 on a 60 s grid,
  2020-12-31→2025-08-31) → 1,705 daily rows; warm-up day 91; TRAIN <1096 / VAL <1278; the four
  shuffle draws per arm: ridge-rich −0.0087/+0.0016, MLP −0.0586, GRU −0.0182; TSMOM sign-30d
  control IC −0.0213/+0.0009; train-test IC gaps +0.15–0.20 on GBM/MLP; analytic ±0.048 ≈ 1/√426
  "bolted on rhetorically" (`cap_aggregate.py:92`).
- **Universe/frame**: 1,205 assets in DB; 64-slot frame positional (slot k = asset k,
  `eval_vector.py:789-795`); selection = coverage-rank the *minute* table, never `ohlcv_daily`;
  OOS leg = 116 decisions, 62/64 slots, with a **~5-week embargo gap** after train end
  (2025-09-01 → 2025-10-05); 875's f5 = 33% median drawdown.
- **Corridor BEAR22**: floor −0.143 / ref +1.049 / ceil +27.061 (830/E0-REAL); +0.97 vs +0.751
  are different windows/panels, not point-comparable (`pathway.md:145`).
- **Env holes**: price validity not enforced in `step_vector` (guard only in
  `compute_action_mask`, `:2127`); fee tiers are separate rollouts (fees change trajectories);
  legacy `vectorbt_env.py` fee 26 bp non-canonical; training curriculum fees 5/13/26 bp.
- **Encoder**: `n_channels` default **32** (5 XS channels at 27–31 need ≥32, inactive in
  self-play's 27); diagnostic **cheat channel 32** must never ship in a checkpoint
  (`jax_encoder.py:61-64`); the 3-D OHLCV branch is reachable only from `train_phase2.py`
  (`:720-726`); real ATR/true-range code exists dead (`observation_encoder.py:110`) or live-only
  (`encoder/indicators.py:43`) — reusable wiring for F2.
- **Train/serve mismatch (unflagged anywhere in the memo)**: the live path
  (`real_trade_jax.py:40,250,443` → `encoder/state_fast.py`) feeds **real** Kraken high/low/volume
  into channels 1/2/3/8-11/24-26 — seven channels shift distribution between close-only training
  and deployment; that live encoder is marked DEPRECATED (`mcts_trader/CLAUDE.md:28`).
- 893 (support for the +0.97 bar): fabricated-prefix check Δ = exactly 0.000000 on all 12 metrics
  — with the circularity caveat in §1 above.
