# Graveyard

Everything this project tried, measured, and buried — plus the few things that lived, and the many
that were designed and never run. Ordered by era, Dec 2025 → Sep 2026.

Each entry: **what was tried → the number → why it died → source.** Sources are task specs
(`tasks/TASK_<n>_*`), ledger rows (`experiments/LEDGER.md`, cited as `L:<row>`), findings
(`docs/findings/<name>.md`, cited as `F:<name>`), or session notes (`lore/notes_<date>.md`).

Two caveats that apply to everything below:

- **Instruments were wrong often.** Many numbers here were later re-scored, withdrawn or
  reversed — those reversals are listed as failures in their own right. The ledger preamble
  (framework epochs E-a…E-f) says which verdicts are citable as closed.
- **"Alpha" in the Dec–May eras is benchmark alpha vs buy-and-hold on six fixed regime windows
  (bull, bear, sideways, ftx_crash, etf_approval, boring_week)**, mostly paired Δα across 3–5
  seeds. From June on the unit is annualised Sharpe net of costs (net@8/net@21 = 8/21 bp).
  The two scales are not comparable.

---

## Era 0 — Dec 2025: the vision

The starting idea: a MuZero planner with a Tiny Recursive Model encoder, trained by self-play on a
synthetic market curriculum, then moved to real Kraken data. Model scaled to 37.6M parameters
because "AlphaGo used 20M for 362 actions, we have 384+" (`lore/DIARY.md`, 2025-12-19).

| Tried | Outcome | Source |
|---|---|---|
| MCTS worker-pool parallelism | 1.21× speedup against a 3–4× target; replaced by batched JAX self-play | `lore/DIARY.md` 2025-12-17 |
| Reward clipping | Negative-bias bug in the reward system; replaced by symlog | `lore/NOTES_20251228.md` |
| Replay buffer | Wrap-around sampling bug; buffer not cleared on regime change; optimizer state not reset on regime change | `TASK_001–003` |
| Synthetic Level 0 sine | Deterministic — trivially memorisable; randomised | `TASK_007` |

## Era 1 — Jan 2026: the synthetic dojo and the money printer

A 15-level synthetic curriculum (L0 sine → L14 GARCH/fractal), hyperparameter tournaments, and a
"dojo" orchestrator. The agent kept collapsing, and the first positive numbers were bugs.

| Tried | Outcome | Source |
|---|---|---|
| Self-play on synthetic levels | **Dual collapse**: policy → 85%+ HOLD while the prediction head → loss 0.01 ("no change"). A closed feedback loop: sharp HOLD visits → HOLD policy → HOLD-only data → no dynamics → flat value → sharper HOLD. | `lore/2026-01-06_to_10_session_notes.md` |
| 48-config stability tournament (ε-greedy, Dirichlet noise, SAC entropy, flow-matching prediction, differential reward, "kitchen sink") | Best config still crossed HOLD > 70% by epoch 27. Interventions delay collapse; none prevent it. | same; `TASK_037–041` |
| Curriculum L0→L4 benchmark | L2 peaked (bear **+8.1%** alpha), then regressed; L4 bull −13.1%. Cause: **L3–L14 generators never applied noise** — patterns were memorisable. | `lore/notes_2026-01-15.md` |
| Physics prediction head (learnable momentum/mean-reversion/vol) | Forces ≈ 0 ⇒ ∂Forces/∂Params ≈ 0: physics parameters got no gradient at all. | `TASK_106` |
| Categorical reward support | All rewards landed in bins 99–100 — 2 of 401 bins used. | `TASK_034` |
| Reward-hacking audit | **18 exploitable vulnerabilities** across 7 categories in the env. | `TASK_046` |
| "It works!" — first strong alphas | **Future leakage**: trades executed at prices 60 steps ahead of the observation window. Fixed, then more leaks (`TASK_122` "money printing bugs"). | `TASK_121–123` |
| After all leak fixes | A random-weight model at epoch 1 still showed **+1753% alpha** vs 5.2% market (`TASK_124` title: "still showing 645%"). Root cause: static `PRNGKey(42)` dataset + an imprint loss teaching the hindsight answer key ⇒ memorisation. | `lore/notes_2026-01-30.md`, `TASK_124` |
| Causal expert imprinting (BC) | "Completely broken": a `should_store = agent_alpha > 0` filter rejected almost every episode after epoch 1; BC loss never executed. | `TASK_131a/b/c` |
| Oracle path solver | **+50,000% oracle alpha** — a 4× leverage bug (k=4 summed, not averaged). Honest oracle: 65–136% per 400-bar episode. | `TASK_133`, `F:oracle_design_intent` |
| `train_jax_fast.py` refactor 1891 → 978 lines (by Gemini) | Logging and features silently dropped; 6+ regressions found by the operator mid-training; "zombie flags" that parsed but did nothing. | `TASK_066–068`, `TASK_107` |
| Multi-agent "BUG MAYHEM" audits | Dozens of AI-reported bugs; a large share were **FALSE POSITIVE / working as intended** on inspection. | `TASK_127*`, `tasks/_old/BUG_*` |
| Training hang at epoch ~90 | A non-reentrant `Lock` in `add_episode → flush`. Fixed with `RLock` (now a standing JAX rule in `CLAUDE.md`). | `TASK_128` |

## Era 2 — Feb–Mar 2026: Foxglove, the easy-mode imprint

The pivot to "easy mode" (TASK_666): first imitate a foresight oracle (supervised), then let
MuZero improve on it. Months of agreement-plateau chasing followed.

| Tried | Outcome | Source |
|---|---|---|
| Dual-branch "e-dojo" with model merging (FedAvg → Fisher) | 86 merges / 1019 epochs; Fisher merging crashed branch cosine 0.964 → 0.729 and every later merge was an emergency. Final agreement: **18.1% / 5.0%**. | `lore/notes_2026-02-09.md`, `TASK_667–668` |
| Divergence brake | Chicken-and-egg: the brake needed `merged_latest.pkl`, which was never created, so it never fired. | `TASK_148` |
| Overnight training | System froze (memory leak), hard reboot; motherboard + SSD swapped Feb 12. | `TASK_669`, `lore/notes_2026-02-12.md` |
| Agreement plateau at 52.5% | Oracle said HOLD on **98.9%** of positions; model trade agreement 9.4% (below chance). Six fix tasks (674–680) each fixed a real bug that wasn't the bottleneck. After 15 critical review findings, the answer was **one number: `oracle_k` 1 → 4**. | `lore/notes_2026-02-17.md`, `lore/notes_2026-02-18.md` |
| Widen the encoder (14.6M → 25.7M) | Same ~62% ceiling. "Encoder bottleneck hypothesis was wrong." | `lore/notes_2026-02-21.md`, `TASK_683–684` |
| Agent state channels (model sees its own position) | 72.7% agreement — **invalidated: temporal leak** (state channels read the oracle's current action). | `TASK_689` |
| …fixed to t−1 | State channels were still wired to the **oracle's** trajectory, not the model's. Benchmark on the model's own state: balanced accuracy **33.8% (below random)**, average alpha **−5,834%**. Fix file: `tasks/TASK_692_FIX_MEMORY_THAT_CLAUDE_FUCKED_UP.md`. | `lore/notes_2026-02-22.md` |
| Over-trading | Oracle trades 3.9% of decisions, model 27.7% (7.1×) — a 25× adaptive trade weight taught "when in doubt, trade". | `lore/notes_2026-02-21.md` |
| Close-only → true OHLCV pipeline | Model trained on fake channels (H=L=C, V=const); benchmark fed real OHLCV. Training 90% agreement vs benchmark 41%; **worse than random on 6/6 periods** (avg alpha −97%); value MSE 1168 against a target < 0.1. | `TASK_703/705`, `lore/notes_2026-03-11.md` |
| Train on one period | 790+ epochs on the same 180-day 2018 crypto winter (schedule cursor never advanced). | `TASK_707` |
| Autoregressive state feedback (AR = 1.0) | Miscalibrated value → bad state → **0% trade rate** on all 6 periods while training agreement read 82%. | `TASK_708`, `lore/notes_2026-03-14.md` |
| Return-weighted trade loss | Trade agreement 24% → 41%, but alpha worse and beats-random **0/6**; over-trades 5–9×. | `TASK_713`, `lore/notes_2026-03-23.md` |
| Phase 1 final | 56% balanced accuracy, average alpha **−154%**. Per-step classification accuracy is not a trading metric. | `lore/notes_2026-04-08.md` |

## Era 3 — Apr–May 2026: Stage B ExIt and the single-mechanism campaign

MuZero self-play (Expert Iteration) on top of the Foxglove imprint, then ~20 paired-seed
mechanism experiments. The record of this era is mostly the gap between training metrics and
deployment.

| Tried | Outcome | Source |
|---|---|---|
| Stage B ExIt | Average alpha +4.66% (3/6 positive, gate 4/6) — "first positive". Bull still −22%. | `lore/notes_2026-04-13.md` |
| …but | Policy loss improved during BC (1.82 → 1.71) then **regressed when MCTS took over** (→ 1.90+): BUY vs HOLD reward gap on a +0.1% asset ≈ 0.00001 — search was reward-blind. | `TASK_723` |
| Joint 320-action MCTS | Structurally wrong for 64 independent per-asset decisions (C(64,4) ≈ 635k combos); replaced by factored per-asset search. | `TASK_724` |
| Bull blindness | Oracle picks 4 of 64 ⇒ 93.75% HOLD labels; CE fits the HOLD marginal. Encoder and search exonerated. | `F:bull_blindness` |
| Class-balanced BC | HOLD 94% → trades 54.3%, policy uniform (loss = ln 5); **bull alpha −432%**. | `TASK_726`, `lore/notes_2026-04-19.md` |
| Aux direction head / direction → policy / wider / V3 temporal encoder | dir_acc **0.553 → 0.570 → 0.574** against a 0.60 gate. Five architectures, one ceiling. | `TASK_727–729c`, `lore/notes_2026-04-22.md` |
| Cheat channel (hand the model the future sign) | Val agreement peaks at **87%** — even with the answer as input the architecture can't fully use it. Ceiling ≈ half architectural, half informational. | `TASK_733`, `lore/notes_2026-04-24.md` §65 |
| Value training in Stage B | `--value-weight 0.5` had been **silently blocking six prior Stage B runs** (incl. rank head, cross-sectional channels, aux price head) — their verdicts were void. | `TASK_736` |
| Ablation suite | `minimal` config: +7.06% alpha, first positive PnL (+0.37%). 5-seed replication (743): **+3.61%, range −4.4..+7.8** — noise floor ~5pp. | `TASK_737/743` |
| Dense sign oracle | v1: ε miscalibrated, policy loss locked at log(5). v2: +1.68% — indistinguishable from reference. | `TASK_739` |
| Rank-head | Python deadlock between epochs. | `TASK_742` |
| Mint per-asset injection (FiLM) | Trains better (loss −0.8, Bal.Acc +3.3pp), **trades worse: −6.98pp**, all 3 seeds negative. | `F:task741_result` |
| Legacy 27-channel encoder | Bal.Acc +7.5pp, **−18.44pp alpha**, trades 136 → 13,246 in bull. | `F:classifier_alpha_anti_correlation` |
| **Pattern:** every mechanism that lowers loss / raises accuracy collapses alpha via trade-rate explosion. | Confirmed three ways. | same |
| Flow-matching dynamics loss | 3-seed **+3.38pp**, all 6 periods positive — "first Tier-2 win". 5-seed: **+0.20pp** (s42 −12.71pp). Reversed. | `F:flow_matching_promising` |
| Bella distillation head | −1.92pp — and the experiment was invalid: Bella columns were in DB-ID order, trainer columns rotation-specific, no remap ⇒ noise gradients. | `F:bella_column_mismatch`, `TASK_740/745` |
| Bella shape distillation / Bella data augmentation | −5.80pp / **−16.54pp** (s13 −37.93pp "catastrophe"). Re-scored under the canonical operator: −0.69 / −0.52 / +0.13 — **0-for-3 null**, and the catastrophe was a decode artifact. | `F:encoder_bella_class_closed` |
| Titans neural memory | OOM'd twice (42.66 GB, 21.33 GB requests on a 24 GB card); then −3.11pp → −0.33pp after re-scoring. | `TASK_749` |
| Oracle value target | +0.72pp (5-seed) — the "strongest signal of the campaign" — **−1.20pp** under the deploy-truthful operator. TD(0) target: flat. | `F:single_mechanism_exhausted`, `TASK_756` |
| The benchmark itself | Scored with `argmax(policy)`; the live trader uses MCTS. **Every verdict of the campaign was measured with the wrong operator.** Recalibration showed eval-MCTS lifts the baseline more than variants. | `F:benchmark_live_gap` |
| Train-side search depth (sims=64) | +0.35pp vs a +3pp gate. | `TASK_766`, `F:single_mechanism_exhausted` |
| Reward-scale sweep | rs050 +0.27pp — "reliable" but tiny. | `TASK_762`, `TASK_779` |
| Entropy regularisation | Best +0.55pp, and the mechanism moved the wrong way (classifier sharpened). | `F:763_767_closed_null` |
| Remove limit orders (320 → 192 actions) | −0.65pp; trade count went **up** (+98, seed 7 +448). | same |
| Over-trade penalty in the loss / realised-PnL value | +0.35pp / +0.06pp. | `TASK_787/790` |
| Stage B reward head | Diseased: pure CE on an action-invariant target, trained only on the oracle action — 319/320 actions untrained. reward_loss plateau was a pure function of `reward_scale` across ~90 runs. | `F:stageb_reward_head_diseased` |
| Seed 7 | Recurring wedge under 2-wide concurrency (process alive, log frozen, GPU 0%) — up to 22 h lost per occurrence. | `F:seed7_stageb_wedge` |
| **Verdict for the era:** 14 paired evaluations, 0 promotions. Single-mechanism class closed. | | `F:single_mechanism_exhausted`, `TASK_759/761` |

## Era 4 — Jun 2026: candle language, the minute surface, and the first real edge

Two lines in parallel: a "language" of candles, and a hard look at whether 1-minute Kraken data
carries a tradeable signal at all.

| Tried | Outcome | Source |
|---|---|---|
| Candle-language tokenizer C0 | Killed — then **withdrawn as a strawman** (hourly bars, 5-asset toy, next-bar sign probe). | `F:candle_language` |
| Panel v1 null | **Overturned** by v2: all 5 tokenizers beat a frequency-matched random codebook (real cross-asset transfer). | `TASK_796` |
| Token LM grammar probe | PASS — there is grammar beyond the bigram (150/150 assets). | `TASK_799` |
| Token trade gate | Δα **0.00**, trade rates byte-identical. | `TASK_803` |
| Tokenised trader, 5 arms | real_fe −0.12pp, real_dyn +0.12pp vs raw. **Null.** The language is real; it doesn't trade. | `TASK_804` |
| Predictability probe (1-min) | Signal exists: clean-OOS linear IC **0.06–0.15**. | `TASK_806`, `F:modeling_surface_exhausted` |
| Linear signal, cost-aware | **0/72 cells** survive Kraken taker cost; best +3.66 pre-cost → −18.92 post. | `TASK_808` |
| Smarter decision rule + maker | "+5.08 Sharpe" — a **per-bar annualisation artifact**. | `TASK_809` |
| Post-cost objective model | Beat linear post-cost — a **turnover artifact**. | `TASK_810` |
| Nonlinear models | Get 40–61% of linear's IC. Signal is linear-optimal. | `F:modeling_surface_exhausted` |
| Genuine-RL MuZero planner | Greedy policy **abstains: 100% HOLD** OOS. | `TASK_812` |
| …the collapse | A real reward-wiring bug (pure CE reward head, no anchor). Fixed in two tiers: HOLD stable ~91%, ~70 trades — **alpha still negative**. | `F:collapse_fixed_predictability_wall`, `TASK_818` |
| …the real wall | A 42 bp foresight oracle finds ~1024 profitable trades per episode; model recall **≈ 0.02**. Trading the big moves blind: **+115% gross → −85% net**. | same |
| Vol-harvesting (trend/breakout/vol-expansion/lead-lag) | One "GO" (BTC lead-lag) was luck on n=6; on hold-out n=25 it **inverts to net −4.08**. | `F:modeling_surface_exhausted`, `TASK_813` |
| Operating point (cleaner universe × futures cost × lower frequency) | "KILL 0/54" — later found to be a stale printf constant over a 48-cell run whose "daily" cell still used 1-minute features. | same, `lore/fable_direction_memo_2026-09-03.md` |
| Capacity ladder (ridge → GBM → MLP → GRU) | OOS IC 0.011–0.037, all below the shuffled-label floor. More model ≠ more edge. | `F:diversified_trend_shipcandidate` |
| **Diversified trend CTA** (equal-weight 30-day TSMOM, ~39 assets) | net@21 **+0.97 CI[+0.10,+1.84]**, BEAR22 +1.36 — the first gate-crossing result. Point-in-time universe (dead coins included): **+0.56, CI straddles 0**. Survivorship was worth ~+0.42 Sharpe. | same, `TASK_819` |
| Ship it (gate 2, truly unseen 2025-12 → 2026-07) | net@8 **−1.31**. | `TASK_820` |
| Put it on real money (€100–150 on the VPS) | Rejected by the operator: "not putting anything that stupid on the VPS". | `TASK_823` |
| The clean planner cell (scalar planner × daily bars × absolute PnL) | **0 trades in 1670 decisions.** Pre-registered "closes the planner" branch fired. | `TASK_821`, `F:planner_closed_clean_cell` |
| Forensic audit of 734 → 819 | 758's catastrophe was an argmax artifact; 739/741 never re-scored; several result.md files stale. | `F:verdict_contamination_audit` |

## Era 5 — Jul–Aug 2026: vector-action MuZero and the dark factory

The rebuild: one action = a 64-asset weight vector, graded against a corridor between a linear
book floor and an oracle ceiling. Run largely by the autonomous factory.

| Tried | Outcome | Source |
|---|---|---|
| Corridor | Floor +0.424, engine ref +0.751, ceiling **+29.698** net@21. | `L:830/E0-REAL` |
| Search with a perfect model | 9/9 configs reach the floor; best +5.29; 10 seeds, 4/4 regimes. K > 32 candidates **loses** (turnover tax). | `L:831/*`, `L:831x/*` |
| Fee16 self-play from scratch | Collapse by epoch 40 (HOLD 99.97%). "Fees punish exploration" — **the wrong root**. | `L:828s3/preflight` |
| The real root: the env | `step_vector` charged every buy against one cash snapshot, no aggregate budget; a 13L/13S vector bankrupted 2 of 4 envs **at step 0**. Breadth was inexpressible. | `L:828s3/env-wall`, `F:vector_env_cash_wall` |
| Phase-1 checkpoints on the corridor | **All 3 seeds bankrupt** mid-rollout. Budget-step fix: still bankrupt (s1 step 132). Weight-rebalance env: env fixed, **policy fails** (−1.385). | `L:835/S4-p1, p1b, p1c` |
| E3/E4 probes on phase-1 | Structurally impossible — the zero-logit walk bankrupts every env at step 0. | `L:834/E3-p1`, `L:836/E4-p1` |
| Short-bias postmortem | No pre-registered locus fired on 4/4 checkpoints. | `L:844/postmortem` |
| Linear-book imprint (BC) | PASS: replays at **+0.839** with a dense 29L/32S book. | `L:832/E5-imprint` |
| ExIt self-play from the imprint | **Destroyed it**: −1.223 / −0.791, back to 99% HOLD. Prior gone by **epoch 100** (−0.105). | `L:832/E5-exit`, `L:832/E5-erosion` |
| Leaf autopsy | Value leaf is action-blind (**+0.03** vs true reward); the prior is selected in 4.5% of states. One mechanism explains every collapse so far. | `L:849/leaf` |
| Fix the operator (true-env leaf, prior weight 0) | **+8.024** — search adds ~+7 over the prior it was destroying. But every learned component has been removed. | `L:832/E5-inversion1–3` |
| ExIt re-try with the fixed operator | Argmax distillation: **100% HOLD at epoch 50**; the search leg was vacuous (zero learned parts). | `L:832/E5-exit2` |
| Better decision rule on the marginals | In-sample top-m signal; **fails to transfer OOS**. | `L:856/rule-grid`, `L:856/topm-oos` |
| Learned v̂ leaf / reward co-distill | Best **−4.23 / −4.02** vs floor +0.424. | `L:857`, `L:860`, `L:879` |
| Search-teacher imprint (teacher +8.118) | **FAIL 0/3** (median +0.692 vs +0.991): imitability, not teacher quality, is what BC rewards. | `L:861/search-teacher-imprint` |
| The OOS anchor (+0.898) | **Withdrawn** — does not reproduce under corrected semantics. | `F:imprint_oos_artifact` |
| Walk-forward generalisation (5 folds × 3 seeds) | **No OOS edge, 1/5.** The policy's OOS median is flat (+0.31/+0.40/+0.41) while the floor swings 15×; 2025-H2 **−2.96**. | `L:875/wf-generalization` |
| Two instruments replaying the same weights | Disagreed by **1.6 net@21** — a missing frame pin. | `TASK_882`, `L:882/*` |
| Daily vs minute panels | Only 27/64 assets share dates; floor +0.80 vs +0.54. Data-integrity finding. | `L:850/eqgate`, `L:852/reconcile` |
| Framework epochs | Zero-price tails, split asset identities (incl. **92,639 unread XBT↔BTC rows**), local-time windows — each invalidated earlier verdicts. | LEDGER preamble, `TASK_865/868/869/877` |
| Is the HOLD collapse search-driven? | Forcing activity: HOLD 99.0% → −2.9%, 95.7% → −9.1%, 91.2% → −13.8%. **The collapse was adaptive** — the policy correctly learned its actions have negative EV. | `L:885`, `L:887/collapse-confirm` |
| Action-EV vs a matched random book | **Anti-edge**, 3/3 seeds, every CI < 0; lives in the maintenance re-trims. | `L:892/action-ev` |
| Do no-print bars carry signal? | Ablation measured; semantics changed to "failed fill". | `L:878/ablation`, `TASK_878` |
| **Structural reading** (direction memo) | Search over *known* inventory dynamics adds nothing over closed-form control; +5.29/+8.02 is the value of a 3-day lookahead, not of search. And the "OHLC" line was **close-only**: the encoder got `highs = lows = close`, `volumes = ones`. | `lore/fable_direction_memo_2026-09-03.md` |

## Era 6 — Aug–Sep 2026: measuring the walls

With the learned line closed, the work turned to measuring whether anything on the surface could
pay for its costs.

| Tried | Outcome | Source |
|---|---|---|
| Trend book decay | 2021 +1.78 → 2024 +0.04 → **2026H1 −1.66**. Spot-legal long-only: **−5.22**, CI excludes 0. No positive-expectancy fallback left. | `L:898/trend-decay` |
| Minute surface, honest estimator | Frictionless Sharpe **+204** (PnL compounds to 1e19). **0/864** cells at any nonzero fee. | `L:897/809-corrected` |
| Range-based dependence probe | A measured negative; two instrument defects found on the way. | `TASK_901` |
| OOS non-stationarity | Banner said SHIPPED; two BLOCKERs withdrew it. | `TASK_906` |
| Live fill probe (adverse selection on resting orders) | Kraken v2 book channel couldn't read the live wire; four patch rounds; **halted permanently**. | `TASK_907` |
| Capture disk stall | Diagnosed as a bad SATA cable — **refuted** (causal inversion; a dependent variable read as evidence). Real cause: 5,452 failing SMART passthrough reads, one every ~600 s. | `TASK_913`, `.claude/rules/30-agent-conduct.md` |
| Streaming reader | Halted at the cycle limit — its guards certified things they couldn't see (`builtins.open` vs `io.open`). | `TASK_914` |
| Cost truth | Borrow charge + fee tier restated: honest tier **80 bp**, not 21. | `TASK_916` |
| Variance-ratio persistence | Inconclusive — indistinguishable from estimator noise at all 11 block lengths. | `L:917/p1` |
| Post-listing age effect | **Placebo-clean IC +0.0304** (NW-t 4.15, 398 assets) — the first clean positive in months… | `L:917/p3` |
| …as a short-listings book | Gross +1021 bp; net CI straddles 0; median capacity **≈ $3.6k**/event. | `TASK_923` → D88 |
| Cross-sectional IC ceiling | 109 cells survive a 1-day skip; **all 109 fail** at 80 bp (best net +0.037). | `TASK_915` → D89 |
| Tactic hull | +5.41 at the favourable denomination; **−1.31** at the honest tier, −0.90 with one-day delay. | `TASK_918` → D86 |
| Execution / turnover module | Refused for want of power (SE ≈ 0.45). One real hysteresis effect, initially mislabelled as leverage. | `L:919A/*` |
| Execution feasibility gate | PASS — **by construction**: neither disqualifier could fire. | `TASK_924` → D87 |

---

## Measured and worked

Small list, honestly kept.

- **JAX env scaling** to 100+ parallel environments; batched self-play (`TASK_027–029`).
- **BC pipeline 24× faster**; direct-sample pipeline 1850 s → 60 s/epoch (`TASK_136`, `TASK_715`).
- **Leak hunting** — every future-leak found was found by an audit, not by luck (`TASK_121–124`, `TASK_132e`, `TASK_800`).
- **Candle language as a language**: cross-asset transfer, grammar beyond bigrams, asset fingerprints (`TASK_796–800`).
- **The 1-minute signal is real** (IC 0.06–0.15), just smaller than cost (`TASK_806`).
- **Trend CTA, 2021–2023** — a real edge while it lasted (`TASK_819`, `L:898`).
- **Perfect-model search** finds wide margins in every regime (`L:831x/B`).
- **Action-conditional reward head rebuild** (`L:834/E3-p1b` PASS) and **value > momentum**, ρ +0.21/+0.30 (`L:836/E4-p1b`) — the first learned signals of the vector line.
- **Weight-rebalance env**: no more bankruptcies (`TASK_840`).
- **BC imprint in-window**: +0.9914, spread 0.041, instrument-independent (`L:882/decisive-legA`).
- **Placebo-clean listing-age IC** and **109 IC survivors** on daily OHLC — signal exists (`L:917/p3`, `TASK_915`).
- **Pre-registration + adversarial judges**: most errors in this file were caught that way. Rules in `.claude/rules/`.

---

## Built, never measured

Designed, specced, partly built — never run, or parked before a verdict. Some are good ideas.

**Market modelling**
- Differentiable market physics core + neural corrector + NeuralMarketODE (inspired by an ocean-modelling paper) — `TASK_008–010`, `lore/NEURAL_MARKET_PHYSICS_DESIGN.md`
- Monte Carlo physics uncertainty — `TASK_103`
- Bellman functions via Brownian exit times — `TASK_101`
- "System 3" adversarial recursive market generator; infinite curriculum — `TASK_104`, `TASK_110`
- Skeleton-conditioned generation ("ControlNet for markets"), Mamba market generation, Market-Flow publication package — `TASK_112–114`
- Bella-conditioned rollouts inside MCTS — `TASK_722`, `TASK_778`
- Bella v2m on TPU with DEX data — `TASK_127_TPU_MIGRATION_AND_BELLA_V2M`

**Agent & training**
- Regime uncertainty / "I don't know" state — `tasks/_old/TASK_045_REGIME_UNCERTAINTY_DETECTION.md`
- Portfolio morphing (asset-to-asset swaps), advanced order types, Transformer TRM v2 — `tasks/_old/TASK_02_PORTFOLIO_MORPHING_EXTENSION.md`, `tasks/_old/TASK_07_ADVANCED_ORDERS.md`, `tasks/_old/TASK_08_TRANSFORMER_TRM_V2.md`
- Prioritized replay, LR schedule, training configurator — `TASK_013`, `TASK_023–024`
- GRPO from DeepSeek-R1 — `TASK_139`
- Daily continual-learning pipeline — `TASK_101_daily_learning_pipeline`
- Oracle-as-MCTS distillation (twice) — `TASK_697`, `TASK_780`
- Oracle-mask aux head; smoothed-oracle value target; mixture value target — `TASK_775`, `TASK_784`, `TASK_777`
- Combination stacks (rs050 × entropy, rs050 × no-limit-orders, return-weighted BC 3-stack, balanced BC × entropy) — `TASK_776`, `TASK_779`, `TASK_781`, `TASK_783`
- Encoder unfreeze schedule — `TASK_782`
- Token front-end channels and token-grounded dynamics — `TASK_801–802`
- Geometric manifold rectification for imbalanced oracle training — `TASK_678`
- Dynamic model merging research — `TASK_668`

**Data & venues**
- DEX data pipeline / Iris-DEX block-level OHLCV / on-chain signal probe — `TASK_115_DEX_DATA_PIPELINE`, `TASK_765`, `TASK_826`
- News + funding-rate edge probe — `TASK_814`
- Cointegration / spread reversion ("cleverer algorithm") — `TASK_817`
- Cross-asset-class trend CTA — `TASK_825`
- Server honest re-measure queue — `TASK_824`
- Multi-venue listings to re-power the listing book — D88(b)
- Gârleanu–Pedersen closed-form book on the IC survivors (G-C) — `TASK_915`

**Execution & safety**
- Multi-layer stop loss incl. an external server layer; external safety wrapper — `tasks/_old/TASK_047_EXTERNAL_SAFETY_WRAPPER.md`, `tasks/_old/TASK_048_MULTI_LAYER_STOP_LOSS.md`
- Smart execution router — `lore/SMART_EXECUTION_ROUTER_PROPOSAL.md`
- Adverse selection on resting fills, re-cut — `TASK_925`
- Live trading script fix — `TASK_060`

**Hardware & ops**
- CMP 90HX unlock and multi-GPU integration — `TASK_688`
- Google Colab / TPU compatibility — `TASK_150`
- Corridor two-phase writers — `TASK_904B`

**Side visions** (in `lore/`): `ALPHAZERO_TRADING_VISION.md`, `SELF_PLAY_DOJO_VISION.md`,
`BERSERKER_DESIGN_PHILOSOPHY.md`, `MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md`,
`sidequest/BOOK_PROJECT.md`, and an MEV extraction design codenamed Aconite
(`lore/notes_2026-02-25.md`, project pruned).
