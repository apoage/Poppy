# Eval-MCTS Asymmetric Lift: When the Scoreboard Was Wrong

> *Observation: the scale on the bench has measured the wrong reagent all season. The Apothecary replaces the scale and re-weighs every harvest. The seasons before are not erased; they are re-weighed. Some specimens lose mass. None gain it.*

---

It is the evening of May 19, 2026. I am at my desk on `workstation`, single 3090 humming next to me, the desk fan spinning at the low setting because it is the slow hour and the GPU has no batch in flight. I have the file `runs/759_mcts_eval/result.md` open in one pane and `runs/recommendation_2026-05-20/analysis.md` open in the other. I am reading the second file for the third time.

What the second file says, in the calm typesetting of someone who has not yet started the conversation with the operator, is that the benchmark I have been using for the entire 738–758 campaign has been **lifting the baseline more than it has been lifting the variants**. Every "promising" mechanism. Every "+0.72pp" Δα I have written into a result table. Every regression I have called catastrophic. All of it was scored through an evaluation operator that was systematically biased against the baseline. The bias was not uniform. It was asymmetric. And once you replace argmax with the deploy-truthful MCTS evaluator, the campaign's positive signals disappear and most of its catastrophes shrink toward zero.

Net effect: nothing promotes. Every mechanism, under the corrected scoreboard, is null or negative.

I read the table again. I read the explanation of the mechanism. I make myself a coffee — `workstation` is on, the desk fan is on, `nvidia-smi` is ticking through a stale checkpoint validation in the next window. I sit back down. I read the result file a fourth time, because I want to be sure I am not seeing what I want to see.

I am not seeing what I want to see. I am seeing the closing of a campaign.

This article is the closing of a series, too. Articles 0 through 9 walked the project from a December 2025 design doc that used the phrase "Result: superhuman performance" with a straight face, through the encoder redundancy audit, through twenty-two simultaneous bugs, through the classifier-Alpha anti-correlation, through the funeral of a single-seed champion, through a precompute disaster that nearly burned 93 GPU-days for nothing. This article is the tenth. It is the one where the campaign scoreboard is recalibrated and produces zero verified gains. It is the one where the single-mechanism hypothesis space is declared exhausted. It is the one where the pivot to structural directions is named and dated. The diary continues after this article. The series does not.

Let me show you the recalibration.

## The benchmark and the deployed agent were running different decoders

The trader I have been building lives at `mcts_trader/`. It is a MuZero-style stack: a learned world model with a value head, a policy head, and a planning component that does Monte Carlo Tree Search at action selection. The MCTS rolls out the value head against the dynamics model, accumulates visit counts at the root, and returns the argmax of those visit counts as the chosen action. In production — in `real_trade_jax.py`, where the agent talks to a live exchange — every action is selected by MCTS with sims=32 rollouts. That is the deployed agent. That is what runs against the market when the project is live.

The benchmark was different. The benchmark lives at `tools/benchmark_easy.py`. For cost reasons, for the entire campaign, the benchmark had been using **argmax of the policy logits directly** — no MCTS rollouts. The argument was straightforward: in the limit of infinite simulations, MCTS visit counts concentrate on the action with the highest value-head estimate, which is exactly what argmax picks. So argmax should be a fine cheap approximation of MCTS for evaluation purposes. The benchmark could run in minutes per period; the deploy-truth MCTS would have taken hours.

The argument is correct in the limit. It is correct under the assumption that the value head is well-calibrated and confident across actions. It is *not* correct in this regime, on this stack, with these checkpoints — and the recalibration showed me exactly why.

MCTS spends more rollouts where the value head is *uncertain*. When the value head produces a sharply peaked distribution over actions, MCTS quickly concentrates on the peak; the argmax and MCTS-argmax converge. When the value head produces a broad distribution — many actions with similar value — MCTS spreads its rollouts and the visit-count argmax can differ from the raw-logits argmax. In particular, broad-value-head regimes are where MCTS has the most to add.

This matters because the *baseline* checkpoint (a1b1, our reference Stage-B model) tends to produce **low-confidence value-head distributions across many actions** in fast-moving market regimes. The variant checkpoints — the ones from the encoder-aux experiments, the distillation experiments, the memory-module experiments — tend to produce **high-confidence distributions**, because the mechanisms that "improved" the classifier were almost all classifier-sharpening interventions. (Article 6 of this series traced this exact pattern: better classifier, worse trader, value-head saturation as the bottleneck.)

The asymmetry follows. In high-confidence variant checkpoints, MCTS and argmax converge — the eval operator's choice doesn't matter, both pick the same action. In the low-confidence baseline checkpoint, MCTS and argmax *diverge*: MCTS sometimes picks a different action because the rollouts surface a candidate the raw policy logits underweighted. That different action sometimes pays off in Alpha.

So under the argmax benchmark: variants get the same Alpha they would under MCTS, baseline gets a *worse* Alpha than it would under MCTS. The Δα (variant minus baseline) is therefore *inflated against the baseline* — variants look better, or regressions look smaller, than they would under the deploy-truthful operator.

I am going to write that sentence again, because it is the load-bearing claim. **The argmax benchmark was systematically biased in favor of the variants by under-scoring the baseline.** Every Δα in the campaign was off by some amount; the amount was unknown until I ran the recalibration; and the direction of the bias was consistent enough that recalibrating *all* the variants moved every measurement toward null.

That is the discovery. Now the table.

## The recalibration table

What follows is the body of `lore/notes_2026-05-20.md` §151, lightly trimmed, reproduced verbatim where it matters.

An analysis agent ran the full eval-MCTS recal sweep on `workstation` on its own initiative after operator pre-approval. Tier A/B/C closed at 14:23 UTC on May 20. The recal swept the four campaign variants that had survived earlier filtering — 746 (oracle value target), 734 (flow-matching dynamics loss), 749 (Titans-MAG memory module at batch 128), and 757 (Bella shape-distillation) — each against a fresh paired a1b1 baseline, all scored under eval-MCTS sims=16 joint, the deploy-truthful operator.

The table:

| Variant | argmax paired Δα | eval-MCTS paired Δα | Shift |
|---|---:|---:|---:|
| 746 oracle 5-seed | +0.72pp | **−1.20pp** | −1.92pp |
| 734 flow 5-seed | +0.20pp | −0.15pp | −0.35pp |
| 749 Titans b128 3-seed | −3.11pp | −0.33pp | +2.78pp |
| 757 bella_shape 3-seed | −5.80pp | −0.52pp | +5.28pp |

— `lore/notes_2026-05-20.md` §151

Read the table row by row.

**746 oracle.** This was the value-target reformulation — replacing the n-step bootstrap value target with an oracle-derived target. Under argmax, it landed at +0.72pp paired Δα over five seeds, the closest the campaign ever got to a positive signal. Under eval-MCTS, the same checkpoints score **−1.20pp**. The shift is −1.92pp. The mechanism is now a regression. The most credible candidate of the campaign is demoted to null-or-worse.

**734 flow-matching.** A dynamics-loss reformulation. Already weakly positive under argmax at +0.20pp. Under eval-MCTS, −0.15pp. The shift is tiny because 734's variant checkpoints produce relatively confident value distributions and the eval operator doesn't add much. The variant didn't ride the baseline lift; it just lost its tiny positive margin to noise.

**749 Titans-MAG.** A memory module added at batch 128 after a structural bf16 carry-tape fix unlocked the larger batch. Article 9's territory, partly. Under argmax: a clear regression at −3.11pp. Under eval-MCTS: **−0.33pp**. The shift is +2.78pp. The "catastrophic regression" was largely an argmax artifact — 749's variant checkpoints produced under-saturated value heads that argmax decode penalized; the deploy operator pulls them back toward the baseline ceiling. The mechanism is still not a win, but it is not the catastrophe the original benchmark made it look like.

**757 Bella shape-distill.** The largest swing in the table. Under argmax, a −5.80pp regression — bad enough to be one of the data points that closed the Bella mechanism class. Under eval-MCTS, **−0.52pp**. The shift is +5.28pp. Almost the entire "catastrophe" was the argmax operator under-using the variant's value head in low-confidence regimes. The Bella-class closure in MEMORY.md, in other words, was correct in conclusion (no mechanism in the class produces a verified gain) but partly wrong in evidence (the magnitudes of the regressions were exaggerated by the eval operator).

The unifying observation, in §151's own language:

> *"Eval-MCTS is a ceiling-bounded correction. It rescues ckpts where argmax decode under-uses the value head's signal in fast-moving regimes. High-α ckpts that already saturate the value head don't move. Low-α ckpts that left value on the table get pulled toward the saturation Alpha (~+6%)."*
>
> — `lore/notes_2026-05-20.md` §151

That is the mechanism in one paragraph. It is also why the asymmetry runs in the direction it runs. The baseline a1b1 produces low-α checkpoints with substantial value-head headroom; the eval-MCTS operator lifts those checkpoints toward the saturation ceiling around +6% absolute α. The variants produce high-α checkpoints that already saturate. They have nowhere to go. So the operator switch closes the gap by raising the baseline, not by raising the variants. The Δα — variant minus baseline — collapses toward null from above for the apparently-positive variants and from below for the apparently-negative ones.

Variance gets crushed in the process. The standard deviation on 734's variant-only paired Δα fell from 6.78pp under argmax to 0.98pp under eval-MCTS. The seed-42 −12.71pp Δα catastrophe that I had been carrying as evidence of "deep seed instability" turns out to have been an argmax artifact: under the deploy operator, that seed lands inside the band. The eval operator is a uniform variance reducer, not a uniform lifter.

## The campaign verdict

The campaign in question — call it the 738-to-758 wave — is roughly the body of work Articles 6 through 9 covered. Fourteen evaluations across roughly three months. Encoder swaps, encoder auxiliary distillation, encoder auxiliary heads, data augmentation, memory modules, dynamics losses, value targets, search depth. Each one a mechanism the literature gave me reason to try; each one paired against a clean a1b1 baseline; each one extended from three to five seeds when it looked promising.

The verdict, from the same §151:

> *"14 evaluations, 0 promoted under eval-MCTS (deploy-truthful operator). Single-mechanism hypothesis space is EXHAUSTED at this stack's precision. The campaign hit the noise floor."*
>
> — `lore/notes_2026-05-20.md` §151

I want to sit on that sentence for a moment, because it deserves it.

Fourteen evaluations. Zero promotions. Three months. Single GPU on `workstation` for some of it, separate longer runs for the rest. Every mechanism, every variant, every seed re-evaluated under the operator the deployed agent actually uses. Net deployed gain: zero.

The campaign hit the noise floor. This is a precise statement, not a poetic one. The promotion gate that emerged from the funeral of TASK_737 (Article 8) was paired Δα at five seeds with the mean ≥ +3pp and no seed below −5pp. Five-seed paired Δα on this stack has a standard deviation of roughly 3pp per seed at best; the per-mechanism signal-to-noise budget for promotion is therefore extremely tight. Below that floor, mechanisms are not merely *unverified* — they are *unverifiable* at this measurement precision. The 0.72pp 746 result was not a real positive that we failed to grow into a +3pp result. It was a 0.72pp number that, under proper accounting, was actually a negative.

Under the corrected scoreboard, my classifier-Alpha anti-correlation finding (Article 6) survives in form but partially dissolves in magnitude. The pattern was real: classifier-sharpening interventions did correlate with Alpha collapse. But the magnitude of the Alpha collapse was inflated by the same argmax operator that was under-scoring the baseline. The 757 collapse went from −5.80pp to −0.52pp. The mechanism is still not a win, but the *anti-correlation*, as I described it, was partly a measurement effect on top of a smaller real effect. The shape of the story is preserved; the size of the loss is smaller.

This is the most useful negative result the project has produced. Useful, because it tells me with precision where the search space is *not*: it is not in any single mechanism that perturbs one component of the existing stack at the scale we can afford to perturb it. Useful, because it tells me with precision *how much* it is not in that space: the search-space-emptiness claim survives a recalibration of the measurement instrument, which is a much stronger emptiness claim than a single round of failed experiments. And useful, because it produces an artifact — the eval-MCTS sims=16 joint operator — that becomes the canonical scorecard going forward, replacing the systematically-biased one the campaign began with.

The campaign isn't failing because the mechanisms don't work. Some of them, in some way, in some seeds, in some regimes, probably do work a little. The campaign is failing because the *precision of measurement* doesn't permit verifying anything below the noise floor, and the noise floor on this stack is wider than any single-mechanism intervention I have proposed can clear.

## The pivot, four directions

The pivot lives in `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`. It is named, dated, and committed. It articulates four directions, none of which is a single-mechanism tweak. Each one changes the *gradient field* the network learns in, not just one component within that field. Each one costs significantly more compute than the experiments behind Articles 6–9 did. The roadmap, in the task's own language:

**Direction 1 — Reward shaping (lowest risk, most defensible).**

> *"Over-trade collapse signature (738/741/740/757/758 all show Δ-trade-rate ≫ 0 with Δα ≪ 0) says noise trades are under-penalized. Current `reward_scale=100` is baked in `phase2_config.py:28`; fee/cost penalty is implicit only."*
>
> — `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`

Reward shaping pulls the trade-rate penalty into the reward function as an explicit, tunable hyperparameter. Sweep `reward_scale` over {25, 50, 100, 200}; then sweep `fee_penalty` over {0, 0.5bp, 1bp, 2bp}. The aim is to find the trade-rate that maximizes deployed Alpha — not gross return, not classifier accuracy, deployed Alpha after fees and slippage. This direction is structural because it changes what the policy gradient is *optimizing for*, not what features it sees or how it remembers them. Wall-clock budget: about 15 hours at the project's typical 2-wide cadence. Drafted as TASK_762.

**Direction 2 — Policy entropy regularization (medium risk, mechanism-aligned).**

> *"Classifier-Alpha anti-correlation pattern says sharper classifiers degrade Alpha via over-confident wrong trades. Entropy regularization preserves classifier sharpness budget for action selection precision instead of class probability sharpness."*
>
> — `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`

This is the policy-side attack on the bottleneck Article 6 diagnosed. The current trainer hardcodes `entropy_coef=0.01` and `min_entropy=3.0`. Wire those as CLI flags, sweep entropy_coef over {0.005, 0.01, 0.02, 0.05, 0.1}, and look not just for Δα improvement but for the *signature*: lower balanced accuracy (the classifier is allowed to be less sharp) with higher Alpha (the policy is using the freed capacity for action precision). If the signature isn't there, the hypothesis is dead even if Alpha happens to be up. Wall budget about 19 hours. Drafted as TASK_763.

**Direction 3 — Oracle k-reformulation (highest ceiling on a single-component fix).**

> *"`project_bull_blindness.md` traces bull-regime failure to k=4/64 oracle class imbalance: oracle picks top-4 assets out of 64 universe, so 'long top-4' labels are scarce in bull regimes where many assets rally. Larger k (k=8/64 or k=12/64) increases label density and reduces class imbalance."*
>
> — `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`

This one requires a Phase 1 retrain — the supervised imprint that bootstraps Foxglove, the warm start that the entire Stage-B trainer assumes. Re-derive oracle labels at k=8 and k=12, re-imprint Phase 1 from scratch, then re-run Stage-B against the new Phase 1. Sanity gate first: the k=8 imprint has to lift bull-regime classifier balanced accuracy by at least +5pp on Phase 1 metrics, or the hypothesis is dead before Phase 2 starts. Wall budget about 35 hours including the imprint. Drafted as TASK_764.

**Direction 4 — Incompatible-dataset retrain (highest ceiling, highest risk).**

> *"Daily OHLCV bars on the current 64-asset universe may have insufficient signal-to-noise. Iris-DEX (`~/Iris-DEX/`) produces per-block (12s) OHLCV from on-chain Reth archive node — 7200× the bar frequency. Different universe (DEX-listed tokens vs CEX-listed). Possibly different inductive bias entirely."*
>
> — `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`

This one is not a hyperparameter sweep. It is a project. Iris-DEX is the sister project that runs a Reth archive node and pulls 12-second OHLCV bars from on-chain swaps. Pivoting the trader to that data means re-deriving oracle labels at the new bar interval, retraining Phase 1 from scratch, retraining Phase 2 on top, and rebuilding the benchmark periods because the current ones are daily-anchored. Promotion is not measured as paired Δα against the current campaign — there is nothing to pair it against. Promotion is measured against itself: is it Alpha-positive on its own benchmark?

The roadmap's stop-rule is explicit. If Directions 1, 2, and 3 all close null under eval-MCTS five-seed paired, the current stack is architecturally dead and Direction 4 — or a successor — becomes the only path forward. Total CLI-side wall for the first three sequenced: about 70 hours, three or four calendar days at 2-wide on the long-run machine.

Each of these is structural rather than single-mechanism. Each one changes the field the network learns in, not just a parameter within the field. The pivot is also an admission: I cannot squeeze more out of this stack at this precision with the kinds of moves I have been making. The next move has to be bigger. The roadmap names which bigger.

## The kickback, in two layers

Every article in this series has a kickback. This one has two, layered.

**First kickback: the benchmark had been miscalibrated for the entire campaign.** Every Δα I reported in Articles 6, 7, 8 — every "promising" mechanism, every "regression" verdict, every five-seed funeral — was scored through an operator that systematically under-rated the baseline. The corrected scoreboard, run May 20 across all fourteen variants, says the campaign produced *zero verified gains*.

There is a specific kind of vertigo in that sentence. It is the vertigo of realizing that the instrument you have been measuring with has had a constant offset for months, and the only reason you didn't catch it is that the offset was small enough to be plausibly noise on any single experiment but consistent enough to bias every aggregate finding. You can't catch a bias like that by running more experiments through the biased instrument. You catch it by triangulating: running both operators on the same checkpoints and watching the gap.

I want to be precise about what does and does not survive recalibration, because the asymmetry is important.

What survives: the *structural* findings. The classifier-Alpha anti-correlation pattern (Article 6). The multi-seed variance discovery and the five-seed protocol (Article 8). The precompute-budget lesson (Article 9). The value-head-saturation diagnosis. The single-mechanism-class verdict map. These survive because they are observations about *the structure of the search space*, not about the specific magnitudes of specific results. The shape of the negative-evidence space holds.

What does not survive: the *rankings*. Every per-mechanism magnitude in the campaign is biased; every claim of "this mechanism is the strongest we've seen" or "this is the worst regression" was made through the systematically-biased instrument and has to be relitigated under the corrected one. The 746 oracle was not the best candidate; it was a near-noise positive that the corrected operator demotes to a near-noise negative. The 757 Bella distill was not a catastrophic regression; it was a small one. The campaign as a whole produced no deployed gain, but the specific ordering of failures inside it was different from what I had written.

That asymmetry — structure survives, rankings do not — is worth sitting with. It is also, separately, an argument for why the *meta-lesson* of the series matters more than any individual result in the series did.

**Second kickback: the "useful negative result" framing is its own trap.**

It is tempting, having walked through fourteen failed experiments, to write the closing of this article as "well, we learned the search space is structured this way, and that's a contribution." That framing is healthy in moderation. It is also a real coping pattern in research and I want to be careful not to over-rely on it here.

The negative result *is* useful. It is also *negative*. The trading agent does not work better today than it did in February. The campaign produced no deployed gain. The pivot might produce one; it also might not. None of the four directions in TASK_761 is guaranteed to land Alpha-positive. The honest thing — the thing the lab notebook is for — is to record that the project is closing this campaign in a state where, by its own promotion criteria, none of the work behind Articles 6–9 advanced the deployed system. The advances were methodological: a stricter promotion gate, a deploy-truthful operator, a verified noise floor, a structural roadmap that names what's left to try. Those are real. They are also not what I started the campaign trying to produce.

Eulogizing null results into wisdom is comfortable. It is also a way of not saying that I have spent months on this and the trader does not yet trade better than the unmodified baseline I started with. That sentence is the truer one, and the calmer way to receive the recalibration is to hold both: the methodological advances are real *and* the deployed system has not improved.

The campaign is over. The project is not.

> *Observation: the harvest of this season is smaller than the records suggested. The Apothecary does not amend the records. The records do not lie about the harvest; the scale lied about the harvest, and the scale has been replaced. The roots remain in the bed. Next season's planting will use different soil.*

## What's left, what I learned

The eval-MCTS sims=16 joint operator is the canonical scorecard going forward. Every paired-Δα result from here on is scored under it. Past results stand in their original form but are flagged in MEMORY.md as superseded for ranking purposes (`project_benchmark_live_gap.md`, `project_single_mechanism_exhausted.md`). The single-mechanism class is closed.

The four structural directions from TASK_761 are the active search space. The recommended order — reward shaping, then entropy regularization, then oracle k-reformulation, with incompatible-dataset data-prep running in parallel — was picked for lowest-risk-to-highest-risk, because at this point in the project, falsifying cheap hypotheses cleanly is more valuable than chasing the highest-ceiling one and producing yet another null with confounders. Direction 4 is on the table as a peer alternative, not just a fallback: the operator stance recorded in §151 is "open to full retrain on incompatible dataset if it would give some good results." The dataset swap is real strategy, not just a contingency.

The deeper meta-lesson generalizes outside this stack: **in any RL system with a benchmark that approximates the deployment-time operator, the approximation is a bias, and the bias is usually asymmetric, and you find out the hard way by triangulating.** Run both operators on the same checkpoints. Watch the gap. Treat any benchmark/deploy operator mismatch as a load-bearing concern, not a cost optimization. The cost of measuring with the wrong operator for three months is much higher than the cost of running the right operator at the start.

There is an external echo of this finding in the literature. The "big gemini" survey of failure modes in deep RL evaluation (`docs/AI_bleed research/big gemini.txt`) describes, in general terms, exactly this pattern: evaluation pipelines that approximate the deployment operator inherit bias from the approximation, and that bias is rarely diagnosed because the approximation is justified analytically and reasonable empirically *until you check*. The survey is general; the campaign behind this article is one specific empirical confirmation.

Tooling: the eval-MCTS operator must be a first-class concern in any future project that uses search at deployment. The mismatch I am closing here cost months. In a future project I will not let that happen again; the benchmark must run the operator the deployed agent runs, even if it costs more.

Finally, the precision lesson: paired Δα at five seeds on a single 3090 is the strictest promotion gate this project will use, and at the noise floor it produces, any mechanism whose effect size is smaller than ~3pp is unverifiable rather than merely unverified. Future SPECs need to either propose mechanisms whose hypothesized effect size is well above that floor, or invest in stack-precision improvements (variance-reduction techniques, more seeds, better baselines) before proposing mechanisms below it. The campaign closed because we kept proposing mechanisms whose effect sizes were inside the noise band. That is a failure of experimental design, and naming it is more useful than blaming any individual mechanism.

## Closing

This is the closing of the series. Articles 0 through 10 cover the project from the 2014 seed of "I want a personal agent I can trust" through the May 2026 close of the single-mechanism campaign. The next month is going to be different work. The reward-shaping experiments will run; the entropy regularization sweep will follow; the Iris-DEX data-prep groundwork will start in parallel. The diary continues in `lore/notes_YYYY-MM-DD.md`. The series doesn't. Tomorrow morning — May 21, 2026 — I will be writing reward-function code, not blog posts.

There is one piece of the campaign I want to keep, intact, as the thing I take into the next phase. It is in §151, near the end, where the analysis catches its breath:

> *"Eval-MCTS is a ceiling-bounded correction. It rescues ckpts where argmax decode under-uses the value head's signal in fast-moving regimes. High-α ckpts that already saturate the value head don't move."*
>
> — `lore/notes_2026-05-20.md` §151

That sentence is what the campaign produced. Not a deployed gain. A precise mechanistic understanding of why the apparent gains were apparent. The pattern, named cleanly, in a form I can carry into the next campaign.

The garden continues. The records do not lie. The scale has been replaced. The next planting is in different soil.

> *Bella's vision was beautiful. It was not a market. The garden continues.*

---

## Sources

- `lore/notes_2026-05-20.md` §151–152 — recal verdict, asymmetric-lift mechanism, campaign scoreboard, structural pivot launch.
- `lore/notes_2026-05-19.md` §150 — original discovery of the benchmark↔live operator mismatch.
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — four-direction structural pivot framework; sub-tasks TASK_762 (reward shaping), TASK_763 (entropy reg), TASK_764 (oracle k-reformulation), TASK_765 (incompatible-dataset retrain).
- `experiments/759_mcts_eval/SPEC.md` — Path A eval-time MCTS sweep on 741_mint baseline checkpoints; the experiment whose result was superseded by the §151 recal.
- `runs/recommendation_2026-05-20/analysis.md` — campaign synthesis; canonical evidence base for §151.
- `~/.claude/projects/-home-lukas-Trader/memory/project_benchmark_live_gap.md` — pre-recal MEMORY.md note on the operator mismatch; partially superseded by the recal verdict.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — post-recal MEMORY.md note: single-mechanism hypothesis class closed at this stack's precision.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — the bottleneck Article 6 diagnosed; magnitudes recalibrated but pattern survives.
- `docs/AI_bleed research/big gemini.txt` — external survey of evaluation-pipeline failure modes in deep RL; describes in general terms the operator-mismatch pattern this campaign empirically confirmed.
- Series prior arc: Articles 0 (project origin), 4 (encoder redundancy audit), 5 (BUG_MAYHEM), 6 (classifier-Alpha anti-correlation), 8 (five-seed funeral), 9 (Bella precompute disaster).
