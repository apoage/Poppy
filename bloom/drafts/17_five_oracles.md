# The Five Oracles: How a Supervised Target Keeps Outgrowing Itself

> *Observation: the gardener's measuring stick has grown five marks this year. Each spring a new notch is cut higher than the last. The old notches remain — crossed through, not erased — because the seedling needs to remember what was once thought tall. The seedling, for its part, has stopped growing.*

---

It is the morning of May 22, 2026. I am at my desk on `workstation`, the single 3090 idling at forty-one degrees, the desk fan on the low setting, an editor open to `mcts_trader/jax_agent/path_solver.py`. The file is 943 lines long. Roughly half of those lines are oracles that the codebase no longer calls. They are kept because each one was the canonical answer to a question that the project later decided was the wrong question, and removing them would also remove the lineage of how the question changed.

The oracle, in this project, is the answer to *"what should the agent have done?"* It is the supervised target for Foxglove's Phase-1 imprint — the labels the policy head is trained to imitate before MCTS self-play takes over. A simple object on paper: a function from a (T, n_assets) array of log-returns to a (T, n_actions) tensor of policy targets. In practice, the most contested object in the codebase. Five distinct definitions over six months. None of them, by itself, has produced a deployment-Alpha gain. All of them are still in the file.

This article walks the lineage. It is the seventeenth in the series, and the third one to circle the same observation from a different angle: the bottleneck of this stack is not the encoder, not the planner, not the memory module. It is the supervised target. The encoder can be sharpened endlessly and the planner can be tuned endlessly, but the imprint is upper-bounded by what the oracle defines as correct. If the oracle is mediocre, no downstream improvement saves you. The campaign verdict in Article 10 — fourteen evaluations, zero promotions — pointed straight back here. The first of Article 10's four pivot directions is *retrain Phase 1 with a richer oracle*. TASK_761 named that direction. TASK_764 is the spec. It is sitting in the queue as of this writing, unrun.

So: five oracles, a sixth on deck, none yet sufficient. Let me show you the marks on the stick.

## Oracle 1: The optimal path

The first oracle was a piece of dynamic programming, and it was clean. TASK_119, dated late January 2026, asked the simplest possible version of the question: given the full price series of an episode and a budget of k positions, compute the trade sequence that maximizes captured Alpha. Hindsight optimization — the oracle knows the future, the agent does not.

The spec was direct:

> *"TASK_118 gives us: max_alpha ≈ Σ top_k(|returns|) — a ceiling. This task gives us: the actual path to get close to that ceiling. TASK_118: 'You could have made $1000.' TASK_119: 'Here's the exact trades to make $950.'"*
> — `tasks/TASK_119_OPTIMAL_PATH_SOLVER.md`

The implementation that landed lives at `mcts_trader/jax_agent/path_solver.py` and is named `greedy_lookahead_path`. It is a JAX `lax.scan` over T timesteps; at each step it looks ahead `lookahead=20` steps, scores the cumulative absolute return of each asset over that horizon, applies a transaction-cost threshold, and selects the top-k by score. The selections are converted to per-asset actions (buy on positive expected return, sell on negative) via a function called `optimal_path_to_policy`. The benchmark numbers came back at 24ms for T=1000 — well under the 100ms-per-episode budget that the imprint loop needed.

It was, in the moment, beautiful. The math was tight, the implementation fit in 50 lines, the agent had a teacher. Two weeks of training data later, the teacher turned out to be wrong in three ways, each one a different category of "should."

The first wrongness was numerical, and it killed training for a week. TASK_133 — dated February 1 — flagged that the oracle was emitting log-returns of 50,000% per episode. With k=4 assets and 200-step episodes, that worked out to roughly 6% per minute per asset, which would imply that Bitcoin was moving 60× per hour. The agent, faced with this teacher, gave up — its alpha was nowhere near the oracle's, so the imitation loss collapsed to "always hold." The root cause was a single arithmetic mistake at `path_solver.py:113`:

```python
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask)
```

Summing absolute returns across the k=4 selected assets without dividing by k. The oracle was reporting 4× leverage per step — and that 4× compounded over 200 steps into the 10× total inflation that produced the 50,000% number. The fix was one line. The fix verified to the gram: oracle alpha fell from ~5000% to 1241%, a 4× reduction exactly as predicted.

I record TASK_133 here not because the bug was deep — it was a portfolio-weighting slip that any code review would have caught — but because it was the first time the project had to ask: *what does the oracle's number even mean?* Before TASK_133 the oracle's alpha was a number we treated as ground truth. After TASK_133 the oracle's alpha was a derivation, with assumptions, that could itself be wrong. The teacher became a teacher with audits. The supervised target acquired its first epistemic kickback.

That was Oracle 1. The second category of wrongness — that the teacher is *stateless*, that it tells you what to hold rather than what to *do* — needed a different oracle.

## Oracle 2: The state-aware path

TASK_141, dated February 8. Training had plateaued at 21% policy-agreement after 103 Fisher-merge cycles. The pattern in the metrics was: the agent traded constantly, the oracle wanted it to trade only sometimes, the agent could not stop trading. The root cause turned out to be two interlocking bugs in how the oracle's path was being converted to per-asset actions.

The first bug: `greedy_lookahead_path` selected exactly k=4 assets every single timestep, by construction. The oracle never said "hold cash"; the oracle never said "trade only two assets right now"; the oracle was constitutionally incapable of staying flat. The agent dutifully learned this. *Trade four things every minute.*

The second bug was deeper, and named the entire task. The oracle was *stateless* — it emitted a top-4 selection at each step but did not know the agent's current holdings. The conversion logic `optimal_path_to_policy` treated each step independently:

> *"BUG: selected + positive return → BUY (even if already long!). BUG: not selected → HOLD (doesn't close existing position!)."*
> — `tasks/TASK_141_STATE_AWARE_ORACLE.md`

If the oracle said "asset 7 is selected" and the agent was already long asset 7, the target was `BUY` — the agent learned to keep buying what it already held. If the oracle said "asset 7 is not in the top-4 this step" and the agent was long asset 7, the target was `HOLD` — the agent learned to never close a position. The trading actions had been written as if the world were stateless. The world is not stateless.

The fix added a `prev_selections` argument to `optimal_path_to_policy` and a transition table:

```
prev=0, new=1, ret>0 → BUY (open long)
prev=0, new=1, ret<0 → SELL (open short)
prev=1, new=1        → HOLD (maintain position)
prev=1, new=0        → SELL (close long)
prev=-1, new=0       → BUY (close short)
prev=0, new=0        → HOLD (stay flat)
```

And a profitability threshold — `min_alpha_threshold=0.002`, two times transaction cost — so the oracle could select *fewer than k* assets and let the agent sit in cash. After the fix, the oracle's HOLD ratio went from 0% to roughly 20% on standalone tests; the agent's plateaued policy-agreement broke through 21% on the next training run.

Oracle 2 corrected the *grammar of action*. Oracle 1 had treated actions as positions; Oracle 2 treated actions as transitions on positions. This is not a refinement — it is a different definition of "should." Oracle 1 said: *in steady state, this is what you should hold.* Oracle 2 said: *given what you are currently holding, here is the trade you should execute.* The supervised target moved one layer of abstraction up.

It also moved variance up. State-aware targets are noisier — each per-step label now depends on the previous label, and small differences in prior state propagate forward. The agreement signal got cleaner *and* the gradient got harder to fit. Both, simultaneously.

## Oracle 3: The multipass ensemble

TASK_673, dated February 15. After Oracle 2 stabilized, training plateaued again — this time at 56% balanced agreement, and the failure mode was different. The oracle picked exactly one asset per step (k=1, the production setting), but in any given minute there were typically five to ten assets with near-identical forward returns. The oracle had to choose one of them, and the "one" was effectively a coin flip among ties. The agent, trying to imitate that exact pick, was penalized for choosing any of the *other* equally-good assets. The signal-to-noise ratio of the gradient went to hell precisely where the market was most ambiguous — the cross-sectional ambiguity got rewritten as a one-hot label, and the one-hot was wrong half the time.

The fix was a soft-target ensemble. Run the oracle N times with Gaussian noise on the input returns; each pass produces a different (still-correct, still state-aware) one-hot policy; average the N one-hot policies into a single soft distribution. The implementation, `multipass_oracle_policy`, lives in `path_solver.py` lines 222–344. The spec captured the algorithm cleanly:

> *"Multiple valid paths are captured — model learns 'which assets are tradeable' not 'which exact asset.' Gradient signal is proportional to quality — near-optimal picks get partial credit. Ambiguous markets produce ambiguous targets — when no asset is clearly best, the soft target reflects that uncertainty, teaching the model confidence calibration."*
> — `tasks/TASK_673_MULTIPASS_ORACLE.md`

Three things to like about this. One, it preserved the state-aware logic per pass — each of the N passes ran the open/close/maintain transition table from Oracle 2 independently, so the averaging never broke directional correctness. Two, the loss function did not change at all: `softmax_cross_entropy(logits, soft_target)` is `KL(soft_target || softmax(logits))` up to a constant, which is exactly the right loss for a soft target. Three, the compute cost was negligible — double-vmap over (N=8 episodes) × (n_passes=10) produced eighty oracle solves in one JIT-compiled call, costing about 15ms total against an oracle budget of 100ms.

Multipass moved the ceiling from 56% to roughly 60% balanced agreement. It did not move the deployment Alpha. The supervised signal was now *more honest about its own uncertainty*, which is a virtue, but the bottleneck was no longer the oracle's confidence — the bottleneck was downstream, in how the policy head consumed it. (That bottleneck would not be named for two more months, in the form of the classifier-Alpha anti-correlation — Article 6 of this series.)

Oracle 3 corrected the *epistemics of the target*. Oracles 1 and 2 had pretended the right answer was sharp; Oracle 3 admitted that, often, the right answer is a distribution over similar answers. The supervised target moved from hard labels to soft labels. It is the version of the oracle that the production pipeline still uses by default, with `oracle_multipass=True` and `n_passes=10`.

## Oracle 4: The rank target

TASK_730, dated late April. The pivot direction is in the title: *Cross-Sectional Rank Oracle*. The diagnosis came out of the V3 encoder gate failure (E50 at dir_acc=0.570) and a literature pointer to arXiv:2012.07149 on cross-sectional ranking. The framing was severe:

> *"Current oracle greedy_lookahead_path(k=4, lookahead=20) produces 93.75% HOLD labels by construction (60/64 assets get HOLD every step). Only 4 assets/step receive trade labels. BC gradient fits the HOLD marginal; per-asset BUY/SELL conditional never learned. Mode-collapse textbook case: KL-regularized RL with sparse labels → majority-class collapse."*
> — `tasks/TASK_730_RANK_ORACLE.md`

The reframe was: stop labeling four assets as "trade" and sixty as "hold." Instead, label *all sixty-four* assets with their forward-return rank — a normalized score in [0, 1] saying "out of the universe right now, this asset is at the 73rd percentile of expected return." Every asset gets a label every step. The supervisory signal is now dense by construction, and the target is no longer a sparse classification but a dense regression on cross-sectional position.

The implementation was lean. The rank target is computed inline in the loss function:

```python
ranks = jnp.argsort(jnp.argsort(fut_ret, axis=-1), axis=-1) / (n_assets - 1)
```

A `RankHead` was added to the model — a small MLP over per-asset features producing a scalar score per asset — and the loss was MSE between predicted scores and normalized ranks. The rank loss runs alongside the existing BC loss; both signals are blended.

Oracle 4 is structurally different from Oracles 1–3. Oracles 1–3 were path-solvers: they computed a trade sequence and emitted action targets. Oracle 4 is not a path-solver at all. It does not say "trade this asset"; it says "this asset is at this percentile." The action decision is left to the model. The teacher is no longer demonstrating a policy — it is demonstrating a feature.

This is also the oracle that Article 6 of this series circled around, although Article 6 framed the finding in terms of the classifier-Alpha anti-correlation rather than the rank target itself. The rank head improved classifier metrics measurably — better Balanced Accuracy, better directional accuracy on held-out windows. It did not improve deployment Alpha. The 738 legacy_27ch result was the harshest version of the same pattern: Bal.Acc up by +7.48pp, Δα down by −18.44pp, trade count 50× over-baseline. *Better classifier, worse trader.* The rank oracle was producing a model that knew, with surprising precision, which assets were going to outperform, and that knowledge was being converted into trading behavior that lost money.

Oracle 4 corrected the *density of the signal*. The target went from sparse-and-categorical to dense-and-ordinal. The fix was real on its own terms. It also surfaced the deeper problem: the supervisory richness wasn't the bottleneck either.

## Oracle 5: The mask-prediction head

TASK_695, dated February 22 — earlier than TASK_730 in calendar time, but later in the lineage as a target reformulation, because it does something none of the others do. It exposes a second signal *from the same oracle*. The reference paper is Klissarov et al. 2025, RL²F, summarized in the task:

> *"Training a student to predict the teacher's feedback (not just follow it) lets the student internalize the teacher's reasoning. At inference the student can self-correct without a teacher — the 'autodidact' result. Surprisingly, the autodidact outperforms the teacher-student setup."*
> — `tasks/TASK_695_ORACLE_MASK_PREDICTION_HEAD.md`

The map onto Foxglove is direct. The oracle is a teacher with privileged information (the future 20 steps of returns). The model is the student. Currently the student is trained to *imitate* the oracle (policy loss) and to *predict the value* of the oracle's path (value loss). What it is not trained to do is predict *what the oracle sees* — the selection mask itself, the (batch, n_assets) binary tensor of "which assets did the oracle pick this step?"

The architecture is a small auxiliary head branching off the shared TRM representation, a two-layer MLP, sigmoid output per asset, binary cross-entropy against the oracle's selection mask:

```python
oracle_mask_loss = optax.sigmoid_binary_cross_entropy(
    oracle_mask_logits, batch['oracle_mask']
).mean()
total_loss = policy_loss + value_weight * value_loss
            + oracle_mask_weight * oracle_mask_loss
```

The mask head is auxiliary — `oracle_mask_weight = 0.1` by default — because its job is to shape the encoder's representation, not to dominate the policy gradient. At inference, the head provides an internalized version of the oracle: the model can ask itself "would a privileged observer pick this asset here?" and use the answer as a confidence signal during MCTS. That is the Phase-3 promise. The Phase-1 reality, in the ablations that ran, was that adding the mask head pushed Bal.Acc up by a measurable amount and did not move Alpha — the exact pattern we have now seen four times in this article. The encoder learned the oracle's perception more accurately; the trader didn't trade better for it.

Oracle 5 corrected the *width of supervision*. Oracles 1–4 emitted one signal per step (an action, or a rank, or a soft policy). Oracle 5 emits two: an action target *and* a perception target, the second one being "what would a future-knowing observer see right now?" The teacher gave the student its eyes as well as its hands.

## (Oracle 6: Distillation of the search itself)

TASK_697, dated February 23, status *on hold*. This is the most ambitious version — and the most architecturally disruptive — so the article touches it briefly rather than walking it.

The idea is to stop computing the oracle as a path solver entirely and instead frame it as the *limit case of MCTS itself*. The multipass oracle from TASK_673 already runs N noise-perturbed rollouts and aggregates them; that is structurally a degenerate MCTS, with each pass acting as one simulation and the averaged policy acting as the visit-count distribution. TASK_697 takes this seriously: train the Phase-1 model on MCTS-shaped targets — visit-count distributions, backed-up Q-values, per-step reward targets, dynamics consistency — using the oracle multipass as the source. Phase 1 becomes a search-shaped pre-training; Phase 2's real MCTS bootstraps a network whose representations *already speak search*, instead of being shaped for classification and re-learning during self-play.

It is on hold because it is not a target reformulation — it is a trainer reformulation. It requires replacing `easy_trainer.py`'s loss with the full MuZero loss (policy CE on visit distributions, categorical value, reward, consistency), training the dynamics model end-to-end against oracle trajectories, restructuring the buffer to store (observations, actions, rewards, visit_distributions, values, dones) instead of (observations, oracle_targets). It is two weeks of work as estimated. It is also the cleanest possible answer to the question this article keeps circling: *what should the agent have done?* — namely, *what would a search with privileged information have searched.*

It is mark six on the measuring stick. It has not been cut yet.

---

> *The Apothecary sets aside the measuring stick. Five marks have been made. The sixth notch is sketched in pencil and not yet cut. The seedling is the same height it was after the third mark. The Apothecary records both facts in the same column of the same ledger.*

---

## The kickback, in three layers

There is no neutral oracle here, and the reformulations were not arbitrary. Each one was an honest attempt to fix a specific failure mode of the previous one, and each one succeeded at the failure mode it was attacking. Oracle 1's arithmetic bug was fixed by Oracle 1's own audit. Oracle 1's hyperactive-agent failure was fixed by Oracle 2's state-awareness and threshold. Oracle 2's coin-flip ambiguity in cross-sectional ties was fixed by Oracle 3's multipass softening. Oracle 3's still-sparse trade labels were fixed by Oracle 4's dense rank target. Oracle 4's lack of internalized teacher perception was fixed by Oracle 5's mask-prediction head. Each one was a local improvement. Each one is still in the codebase. Each one, in isolation, did the thing it set out to do.

What none of them did was move the deployment Alpha. This is the first layer of the kickback. Articles 8 through 10 of this series walked the slow, multi-month closure of the single-mechanism hypothesis space. The campaign verdict — fourteen evaluations, zero promotions under eval-MCTS deploy-truth — was the operational closure. The oracle reformulations are *inside* that closure. Oracle 4's rank head, Oracle 5's mask head, were among the mechanisms in the campaign. They produced the same pattern as everything else: classifier metrics up, Alpha flat or down. The supervisory signal got richer; the trading agent did not get better at trading.

The second layer is structural, and it is the one that TASK_761 named explicitly: *the oracle is the upper bound on what imprinting can teach*. This sounds obvious when stated. It is less obvious when you are inside the encoder-tuning loop, watching directional accuracy climb from 0.55 to 0.57 to 0.59 and wondering why deployment Alpha refuses to track. The encoder cannot extract structure the oracle did not encode. The planner cannot search a value landscape that the value head, trained against the oracle's PnL, did not learn to map. If the oracle defines "should have" as "top-4 by 20-step lookahead absolute return," then the agent's ceiling — even with a perfect encoder and a perfect planner — is exactly the alpha that strategy captures. No amount of downstream sharpening clears that ceiling. The first of TASK_761's four pivot directions is "Phase 1 retrain with richer oracle (k=8 or k=12)" precisely because the campaign verdict pointed here. Direction 3 of `TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` is the oracle k-reformulation. The hypothesis is in `tasks/TASK_764_ORACLE_K_REFORMULATION.md`:

> *"Increasing k (k=8/64 or k=12/64) increases label density and reduces class imbalance, which should improve bull-regime classifier Bal.Acc AND give the planner access to a richer action space (more assets pre-labeled as profitable longs)."*
> — `tasks/TASK_764_ORACLE_K_REFORMULATION.md`

`project_bull_blindness.md` — in my MEMORY notes — traces the bull-regime failure mode to exactly the class imbalance Oracle 1 baked in: with k=4 of 64, "long top-4" labels are rare in broad bull regimes where many assets rally simultaneously, so the encoder learns the wrong class distribution and the agent fails to participate in rallies. The bull-blindness has been documented since January. It has never been directly tested by retraining Phase 1 at higher k. TASK_764 is that test. It is the sixth mark on the stick.

The third layer is the deepest, and it is the one I have come around to slowly over the last six months. *The definition of "what the agent should have done" is contested, and every contest leaks into the design.* Oracle 1's "should" was a path-solver's "should": the optimal sequence of trades under known returns. Oracle 2's "should" was a transition's "should": given current state, what is the correct action. Oracle 3's "should" was an ensemble's "should": across noisy perturbations of the world, what is the consensus pick. Oracle 4's "should" was a feature's "should": where on the cross-sectional ranking does this asset sit. Oracle 5's "should" was a perception's "should": what would a privileged observer pick, regardless of whether the agent acts on it. Oracle 6's "should" is a search's "should": what would a tree-search with privileged information visit and value.

There is no neutral choice among these. Each one encodes a different theory of the agent. Each one prescribes a different ceiling. And here is the part I want to be careful to claim rather than perform: the choice is not just technical. The choice is a commitment to a particular shape of teacher, and every shape of teacher produces a particular shape of student. The Foxglove imprint is not pre-training in the language-model sense, where you are shaping a general-purpose representation that downstream tasks can re-purpose. The Foxglove imprint is *constitutive*. It defines what the model thinks the world rewards. Change the oracle and you do not just adjust a label distribution — you change what the trader is trying to be.

This is why TASK_764 is also the most disruptive single change in the queue. Its risk section is blunt:

> *"Phase 1 retrain breaks the imprint contract — every prior experiment used the same Phase 1 ckpt. Re-imprinting means every prior paired-Δα number ceases to be directly comparable. This is the most disruptive single-component change. Save it for after the CLI-side mechanism class is fully exhausted."*
> — `tasks/TASK_764_ORACLE_K_REFORMULATION.md`

Re-imprinting at a new k makes the prior twelve months of paired-Δα measurements incommensurable with the new measurements. The codebase has a contract — *all experiments use the same Phase 1 imprint* — that is the only reason the noise floor on per-experiment differences is as tight as it is. Touching the imprint breaks that contract. And the experiment is, properly, the *last* one in the queue rather than the first, because every cheaper experiment has to run first to rule out smaller perturbations.

This is the final shape of the kickback. The reformulation that the campaign verdict most clearly points to is also the reformulation that breaks the measurement contract that produced the verdict. Running it costs the project its own prior precision. There is no way to test the richer-oracle hypothesis without also rebuilding the scoreboard.

## What's left

The current production oracle, as of this writing on May 22, 2026, is Oracle 3 — multipass state-aware path with k=4, lookahead=20, n_passes=10, transaction-cost threshold of 0.001. The rank head from Oracle 4 has been ablated and is not currently in the canonical Stage-B stack. The mask head from Oracle 5 was an auxiliary experiment that did not promote. Oracle 6 is on hold. The line of code that selects the oracle is gated by `getattr(self.config, 'oracle_multipass', False)` and the default is `True`. The rest of the lineage is preserved, commented out only in the sense that it is reachable through other config flags and historical task notes, but not on the default path. The file is 943 lines, half of them lineage, and removing any of it would also remove the lineage.

TASK_764 will raise k. The sanity gate is bull-regime Bal.Acc ≥ +5pp; the promotion gate is paired Δα ≥ +3pp under eval-MCTS sims=16 joint, no seed below −5pp. The Phase-1 retrain is approximately 35 hours of wall on a single GPU, and `workstation` does not have the headroom to run it without displacing other work for a day and a half. The decision to launch it has not been taken. Two cheaper experiments — TASK_762 reward shaping, TASK_763 entropy regularization — are queued ahead because their results, regardless of sign, will either close part of the search space or sharpen the case for the more expensive re-imprint. If both close null, TASK_764 runs. If either of them produces a verified Alpha gain, TASK_764 may not.

The right oracle for this stack is still unknown. The five reformulations did not converge on it. They walked five different cuts at "what the agent should have done" and each cut revealed a property of the question that the prior cut had missed: stateless vs state-aware, sharp vs soft, sparse vs dense, single-signal vs multi-signal, path-shaped vs search-shaped. Each of the cuts was the right cut against the failure mode it was attacking. None of the cuts, individually, told the model how to make money. The campaign verdict suggests that no further single cut, of the same kind, will. The richer-oracle direction — TASK_764's k-reformulation — is the project's bet that the next mark on the stick has to be cut at a different scale entirely. Not a new transition rule, not a new ensembling scheme, not a new auxiliary head. A different theory of what the agent should be imitating in the first place: more assets labeled positive in bull regimes, more density in the supervisory signal, a teacher who concedes that "the top four" was an opinion, not a fact.

The Apothecary's measuring stick will be re-notched if TASK_764 runs. The old notches will not be erased. They will be crossed through, kept for the lineage, and the new mark will be cut higher up the stick. The seedling will either grow this season or it will not. Either result is data.

> *The Apothecary tends the measuring stick before the seedling. It is the more honest object.*

---

## Sources

- `tasks/TASK_119_OPTIMAL_PATH_SOLVER.md` — Oracle 1, optimal hindsight path
- `tasks/TASK_133_ORACLE_ALPHA_BUG.md` — the 50,000% Alpha portfolio-weighting bug
- `tasks/TASK_141_STATE_AWARE_ORACLE.md` — Oracle 2, state-aware oracle with profitability threshold
- `tasks/TASK_157_ORACLE_K_AND_ADAPTIVE_WEIGHT.md` — the trade-density / adaptive-weight tradeoff
- `tasks/TASK_673_MULTIPASS_ORACLE.md` — Oracle 3, multipass soft-target ensemble
- `tasks/TASK_681_ORACLE_TRADE_DENSITY_CEILING.md` — the k=1 / 1.56% density ceiling analysis
- `tasks/TASK_695_ORACLE_MASK_PREDICTION_HEAD.md` — Oracle 5, auxiliary mask head (RL²F-inspired)
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` — Oracle 6 (on hold), MCTS-shaped target distillation
- `tasks/TASK_730_RANK_ORACLE.md` — Oracle 4, cross-sectional rank target
- `tasks/TASK_764_ORACLE_K_REFORMULATION.md` — the pending k-reformulation, Direction 3 of TASK_761
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — campaign-verdict pivot directions
- `mcts_trader/jax_agent/path_solver.py` — the 943-line file holding all five oracles
- `project_bull_blindness.md` (MEMORY) — bull-regime failure mode, dated January 2026
