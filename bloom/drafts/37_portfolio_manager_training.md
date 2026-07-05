# The Gardener of Gardens

*On the layer above the berserker, the design doc that named the arbitrator, and the awkward truth that the layer above only matters if the layer below already works.*

> *Observation: the bed of poppies has been thinned. The bed of foxglove has been topped. The bed of yarrow runs to seed on its own schedule. A gardener tends each bed. None of them walk between the beds. Someone has to walk between the beds. The gardener-of-gardeners decides which bed gets the water this week, which gets the shade cloth, which gets the second pass with the hoe. She does not pull a single weed. That is not her job.*

---

## Where article 3 left off

Article 3 ended on a question I had been ducking for six months. *At what point in the project's evolution does the berserker get to grow opinions?* The answer I gave there was the careful one: never on its own, sometimes in ensemble. A single model with strategic awareness is the failure mode the design was built to avoid. An ensemble of berserkers, each scoped narrowly, with a higher-layer arbitrator deciding capital allocation across them, can in principle carry strategic intelligence without putting it inside any single model.

That sentence was easy to write because it pointed somewhere else. The somewhere else was a document I already had on the shelf, dated earlier than article 3 but never quite acted on: `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md`. Twenty-nine hundred words on what the arbitrator is, how it would be trained, what action space it would speak. The thing that comes after the berserker. The thing I gestured at when I called the strategy layer "the project after this one."

I am writing this on 2026-05-22. `workstation` is on; the desk fan is on; `nvidia-smi` ticks through a checkpoint validation in the next window, GPU 0 at 71°C, 1×3090, no second card any more — the CMP 90HX got pulled on 2026-05-17 for reasons that belong in a different article. The repository state is clean. The portfolio manager design doc is not implemented. It has never been implemented. It sat on the shelf through fourteen single-mechanism experiments, the entire Bella campaign, the eval-MCTS recalibration, and the TASK_761 structural pivot. It will probably sit there a while longer. This article is about *why it has to sit there*, even though the architecture it describes is — I am increasingly sure — the right one.

The shape of the piece: walk the design doc, walk the math, walk the training plan, then say plainly what the preconditions are and why none of them are met yet.

---

## What the design proposes

The doc opens on a sentence that, re-read in 2026-05, lands differently than it did at the time of writing:

> "This is NOT a trading agent. This is a **PORTFOLIO MANAGER**."
>
> — from `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md`

The capitalization is the doc's. The contrast is load-bearing. Poppy, as deployed and as discussed in articles 3 through 32, is a *per-asset trader* — given an asset and a window of state, output a directional bet. The portfolio manager is the layer that decides which assets are being looked at this minute, how much capital sits in each, what risk caps apply, when to rotate. The doc lays out the change of frame in five bullets at the top:

> - Manage 64 assets simultaneously with complex correlation awareness
> - Trade cross-pairs directly (BTC/ETH, not BTC→USD→ETH)
> - Develop inhuman tactics (like AlphaGo's move 37)
> - Face adversarial opponents (other models trying to break it)
> - Break fast, document, fix, iterate
>
> — from `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md`

The first four are the agenda. The fifth is the engineering posture. We'll come back to whether the first four are reachable from where the project sits right now; for the moment I want to walk the design as if the preconditions were met, because the design is the interesting part.

### The agent hierarchy

The portfolio manager sits *on top of* the per-asset trader. In the doc's intended endgame there are multiple per-asset traders — call them N berserkers, each scoped to a narrow asset universe or a particular strategy style — and the manager allocates capital and attention across them. The doc never says "berserker" but the architecture is recognizably the ensemble article 3 sketched: each lower-layer agent is fast, opaque, short-horizon, with no opinions about strategy; the manager carries the strategic context and assigns work.

The thing this gets you, structurally, is that strategic intelligence lives in the layer that is *designed* to carry it. Article 3's concern — that a model which knows it has a portfolio can develop opinions about portfolios — is addressed by making the model-with-portfolio-opinions a different model, with a different state space, a different action space, and a different loss. Capability is not removed; it is *moved*.

### State space at the manager level

The doc's manager doesn't see candles. It sees aggregates. The state at the manager level is:

- Per-berserker performance estimates (recent realized return, recent realized volatility, recent realized drawdown).
- Per-asset aggregate exposures (current weight in the portfolio, sign and magnitude of open position).
- A regime estimate — the doc is loose here, but the implied signal is "where are we in the correlation/volatility surface right now."
- The portfolio's own meta-statistics: cash buffer, leverage utilization, time since last rebalance.

Notably *absent* from the manager's state: raw OHLCV. The lower-layer berserker consumes price; the manager consumes berserker-output. This is not just a frugality choice. It is the architectural commitment that the manager's job is *coordination*, not perception. If the manager wanted to see prices it would be a bigger trader, not a different layer.

### Action space at the manager level

The doc is most concrete about action space, because action space is where the cross-pair-trading and multi-action-per-step proposals live. Three options are sketched. The headline one:

> Option 1: Cross-Pair Trading Matrix
>
> ```python
> action_space = {
>     'source_asset': Discrete(64),    # Which asset to sell
>     'target_asset': Discrete(64),    # Which asset to buy
>     'size': Box(0, 1),               # Position size
>     'urgency': Discrete(4)           # 0=limit, 1=market, 2=aggressive, 3=maniac
> }
> ```
>
> — from `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md`

The `urgency` enum is the doc in a thimble: 0 for a limit order, 3 for a "maniac" execution that accepts 10× slippage. The labels are crude on purpose; the underlying observation is sober — execution intent is part of the policy, not a downstream concern. A manager that wants to rotate cleanly out of an asset uses urgency 0 and waits. A manager that has seen a regime break uses urgency 3 and pays. The doc proposes evolving from this explicit form (Option 1) toward a *target portfolio vector* form (Option 3) where the manager outputs desired weights and the environment computes the minimum-trade path to reach them. That's the cleaner abstraction. The first form is the trainable one.

There is also the *multi-action-per-step* extension, which the doc treats as a separate axis: the manager can emit up to ten sub-actions in a single timestep, allowing complete portfolio rotation inside a minute instead of across ten minutes. This pairs naturally with the `MULTI_ACTION_DESIGN.md` work on action-point allocation and with `SMART_EXECUTION_ROUTER_PROPOSAL.md`, which proposes a translation layer between the manager's intent and atomic exchange orders, specifically to deal with the case where a `SELL A` fills but the following `BUY B` doesn't and the portfolio is left in an unintended state.

### Training: a curriculum

The doc proposes seven levels, indexed 0 through 6. Levels 0–3 are the existing per-asset curriculum: trend surfing, lag hunting, dip catching, fee awareness — the things a berserker should be able to do before it gets capital. Levels 4–6 are new and are what the manager is for:

- **Level 4 — Portfolio Rebalancer.** Sixty-four assets, random correlation structure, opportunity rotates every fifty steps. Pass criteria: Sharpe > 2.0, turnover > 50%, correlation with top asset < 0.8. The intent is to force the manager to *exit winners* and *enter new opportunities* — to learn that capital is finite and attention is finite.
- **Level 5 — Adversarial Market.** Two models train against each other. A *protagonist* manager and an *antagonist* market manipulator. The antagonist can shift prices within a budget; its loss is the protagonist's positive reward. The doc draws the explicit AlphaGo analogy and proposes alternating freezes to prevent the antagonist from running away.
- **Level 6 — Chaos Mode.** Flash crashes, delistings, correlation flips, fee spikes. Pass criteria are framed in survival language — bankruptcy rate zero, max drawdown < 30%, "even staying flat is a win in chaos." This is the level that mirrors 2020 and 2022.

The curriculum makes a specific structural claim: that the manager-level skills (rotation, hedging, robustness to manipulation, tail-risk awareness) are not learnable inside the per-asset trader. They require a state space the per-asset trader doesn't see, and an action space the per-asset trader can't speak. That claim feels correct to me even now, after the campaign turned out the way it did. The manager's job exists; it just doesn't yet have a competent layer underneath it.

---

## The math

It's worth writing the joint structure down explicitly, because the architectural claim is easier to see when the conditioning is on the page.

Let there be N per-asset berserkers, indexed i = 1 … N. Each has its own policy:

```
π_i(a_i | s_i, h_i)
```

where `s_i` is the per-asset state (the OHLCV window, the candle features, the local order-book stats), `h_i` is the per-asset hidden state from the recurrent core, and `a_i` is the per-asset action — in the legacy form, sell/hold/buy; in the cross-pair form, the (source, target, size, urgency) tuple scoped to this asset.

The portfolio manager has its own policy:

```
π_PM(w, m | S)
```

where `S` is the *aggregate* state — the per-berserker performance estimates, the per-asset exposures, the regime estimate, the cash buffer — and the output is `(w, m)`: a capital allocation vector `w ∈ Δ^N` (the N-simplex; weights sum to one) and a set of meta-decisions `m` (which berserkers are active, what leverage cap applies, how much cash to hold in reserve).

The joint action at time t factors as:

```
a_joint(t) = π_PM(w, m | S(t)) ⊗ concat_i π_i(a_i | s_i(t), h_i(t))
```

where `⊗` is the operator that scales each berserker's emitted trade by its allocation weight w_i and masks it by activation under m. The crucial property is that the joint policy is *not* a single distribution over a combinatorial joint action space. It is a *product of conditionally independent factors*, with the manager controlling the coupling.

This factorization is the entire engineering point. A single model trying to learn a joint policy over (asset choice × allocation × per-asset action × urgency × meta-decisions) faces an action-space explosion that the doc itself flags — Option 1 alone is "64 × 64 × continuous × 4 = ~16K discrete combinations + continuous." A factored policy of the form above sees each layer's local action space, and lets the coupling live in the slow, low-frequency, high-context manager.

The loss structure follows the factorization. The per-asset loss is what it always was — Poppy's MuZero value/policy/reward loss against per-asset rollouts. The manager's loss is on *portfolio-level* return:

```
L_PM = − E_τ [ Σ_t γ^t · r_portfolio(t) ]
        + λ_risk · risk_penalty(τ)
        + λ_turnover · turnover_penalty(τ)
```

where `r_portfolio(t)` is the realized PnL of the entire portfolio at step t, and the trajectories τ are sampled with the per-berserker policies *fixed*. The risk and turnover terms are not optional — without them the manager has no reason to limit drawdown or trade-rate, and the level-4/5/6 pass criteria are written in those exact terms.

If you try to train both layers jointly from the start, credit assignment goes sideways. The manager's policy gradient on `r_portfolio(t)` has to flow through the berserkers' simultaneous policy gradients, and the berserkers' simultaneous policy gradients have to flow through the manager's allocation choice that selected them. Without the factorization-with-freeze, the manager and the berserkers chase each other around a non-stationary objective. The doc never spells this out as a theorem, but it spells it out as a sequence: per-berserker training first, then manager on top, *then* optional joint fine-tune. That sequence is the correct one.

---

## The training plan

The doc proposes a three-phase build. Or rather, a two-phase build with a hopeful third phase.

**Phase 1 — Per-berserker imprint.** This is the existing pipeline. Foxglove (supervised imprint, `train_easy.py`) and Poppy (MuZero+TRM, `train_jax_fast.py`) on per-asset data through levels 0–3. The output is N berserker checkpoints, each known-good on per-asset metrics: positive Alpha on benchmark, the published Sharpe and trade-rate envelopes, the multi-seed robustness that articles 7, 8, and 10 catalogued the search for.

The doc's phase-1 acceptance criterion is implicit in its level-0 instruction:

> "Expected: Passes in < 500 epochs. If not, architecture is broken - STOP and debug."
>
> — from `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md`

That sentence is the *door*. The portfolio manager build doesn't open until the per-asset trader has walked through it. We'll come back to whether the door is open.

**Phase 2 — Portfolio manager training, berserkers frozen.** With phase-1 checkpoints in hand, the per-berserker policies are *fixed* — their weights stop updating, they emit per-asset actions on demand, they otherwise behave as oracles-of-themselves. The manager is trained on top, with its own MCTS or policy-gradient loop, on the level-4/5/6 curriculum. Its loss is on portfolio-level metrics; its inputs are the aggregate state plus the berserkers' outputs; its action space is the (allocation, meta) tuple.

The frozen-berserker constraint is the whole reason phase 2 is *trainable*. Without it the manager faces a moving target — the berserkers' policies drift under their own training, so the manager's allocation logic is fitting to a distribution that no longer exists by the time the next gradient step lands. With the freeze, the manager's training problem is well-defined: given fixed downstream policies of known competence, learn the coupling. The credit assignment is *separate from* the per-asset trader's credit assignment. That separation is the entire engineering value of the layer.

The frozen-berserker phase also makes the failure modes auditable. If the manager fails to learn rotation, it's the manager's problem — not a per-asset competence collapse. If it fails to handle adversarial perturbation, it's the manager's robustness — not the per-asset perception. The factorization gives you bisectable failure surfaces, which the year of articles 5 through 14 should make me appreciate by reflex.

**Phase 3 — Optional joint fine-tune.** Only if phase 2 succeeds. Unfreeze the berserkers; let small gradients flow from `L_PM` back through them; co-adapt. The doc does not commit to this phase, and reading it now I think the doc is right not to: phase 3 is where you cash in the structural separation in exchange for a few percent of joint-optimization Alpha, and the case for doing that should be evidence-driven, not pre-committed.

The doc closes the training-plan section with a one-liner that aged well:

> "Speed > Perfection. Break things, learn, fix."
>
> — from `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md`

The aging is bittersweet. Speed-over-perfection is correct as an engineering ethos. It is incorrect as a defense against the more fundamental question of *what you are being speedy about*. The doc was speedy about building a manager on top of a trader I had not yet built. Speed in the wrong direction is still in the wrong direction.

---

> *The Apothecary stands at the head of the row. The poppies are pale this year. She has prepared, on the bench behind her, a more careful plan: which beds will be watered first, which last, which will be turned over and re-sown. She reads her plan again. She returns it to the drawer. She walks back to the row of poppies. They are still pale. The plan is sound. The plan is for next year.*

---

## The kickback

Three layers to this one, deeper each time.

### Layer one: the doc was written before the verdict

The portfolio manager design doc is dated earlier than the campaign that taught me what the per-asset trader is actually worth. By May 2026 — through the work documented in articles 6 (the classifier–Alpha anti-correlation), 8 (the five-seed funeral of TASK_737), 10 (the eval-MCTS recalibration that demoted the +3.66pp lift to a baseline-only effect), and 22 (the year-of-negative-results retrospective) — the per-asset trader sits at noise-floor performance on benchmark. Not catastrophic, not blowing up, not net-negative; *at the noise floor*. Single-mechanism improvements get crushed under multi-seed validation. Paired Δα is the truth and paired Δα is small.

The portfolio manager design assumed the per-asset trader was good enough to warrant a manager above it. The phase-1 acceptance criterion was specifically: pass level 0 in under 500 epochs, or stop and debug. By the time TASK_761 was drafted on 2026-05-20, the per-asset trader had not in a stable, deployable way passed *any* of the level-0-through-3 criteria in the strict multi-seed sense. It passes them on average. It passes them with seed variance large enough that you can pick a winning seed and a losing seed from the same configuration. Mean Alpha is positive on some configurations and not others, depending on the eval operator. The phase-1 door is not open.

If you train a portfolio manager on top of a noise-floor trader, you do not get a strong portfolio. You get an arbitrator allocating capital across N noisy slot machines. The manager will learn *something* — probably something like "spread thin and don't lever up" — but the something it learns will be a function of the slot machines' noise structure, not of any underlying skill, because there is no underlying skill to pick up on yet.

The doc's preconditions are not yet met. This is the cleanest framing of the kickback. The design isn't wrong. The dependency graph is.

### Layer two: the structural insight survives the timing issue

The interesting part is that admitting layer one does not retire the design.

Article 3's berserker frame implied that the strategy layer must be *elsewhere* — that a per-asset trader with strategic intelligence is the failure mode the architecture is built to avoid. That implication leaves an unfilled blank: *where elsewhere?* The portfolio manager doc names where. The architecture is `π_PM(w, m | S) · concat_i π_i(a_i | s_i, h_i)` with a frozen lower layer during manager training. That is, as far as I can tell, the structurally correct answer.

So the artifact splits cleanly. The *architecture* of the doc is keepable; it is, in fact, the only architecture I have that consistently respects the article-3 constraint. The *implementation* of the doc is gated. Gated on what? On the per-asset trader earning its place — passing levels 0–3 in the multi-seed sense, with stable Alpha and bounded seed variance, under the eval-MCTS operator that article 10 promoted to canonical. That's a specific and measurable bar. The portfolio manager build doesn't begin the day it has a champion; the portfolio manager build begins the day there is a credible component to be the bottom of its hierarchy.

This framing is useful because it gives me a concrete answer to the question "what would change my mind about the shelf-time of this doc?" The answer is: a paired-Δα win on the per-asset trader, replicated across ≥3 seeds, under eval-MCTS sims=16 joint scoring, on a level-0-through-3 benchmark. If TASK_761's four directions deliver that, the doc comes off the shelf. If they don't, the doc stays on the shelf. The condition is not vague.

### Layer three: ensemble-of-narrow > one-broad

Now the deeper observation, and the one I want to be careful about because it's easy to overclaim.

Every credible proposal I have for the next campaign implicitly assumes the right shape is *multiple specialized components with a coordinator*, not *one big model that does everything*. TASK_761's four structural directions read this way — richer oracle, reward shaping, entropy regularization, incompatible-dataset retrain are all *separable* surgeries, each one a candidate component in a hypothetical lineup, not a single monolithic redesign. Article 23's hindsight-optimal lineage exercise produced a tree of specialized branches, not a single deeper trunk. This portfolio manager doc, read against article 3, is exactly the ensemble-with-arbitrator shape. The shape keeps showing up.

Article 0's modular-AGI framing was *right about the shape* and *premature about the substrate*. The shape is: multiple narrow components, each scoped to a problem it can plausibly learn, with a coordinator above them that carries the strategic context. The substrate isn't here yet — not for me, not on a single 3090 in 2026-05. The narrow components themselves are still embarrassing on multi-seed validation. The coordinator can't be trained on top of components that aren't credible. So the shape is the answer and the schedule is the question, and I had been confusing the two.

I want to be careful here. *Ensemble-of-narrow > one-broad* is not a theorem. It's a strong prior, supported by the structural argument from article 3, by the credit-assignment argument from the math section above, by the dependency-graph argument from this section's layer one. It is not supported by an empirical win in this project; this project has not yet built the ensemble. The claim is "the shape that keeps appearing in good design proposals is the ensemble shape." The claim is not "the ensemble has been shown to work here." Article 22 is the corpus of evidence that I have not yet shown anything to work here. I would like to keep my priors about shape distinct from my evidence about results.

---

## What's left

The portfolio manager design doc is on the shelf at `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md`. It is 2964 words. It proposes seven levels, an action space, a curriculum, a two-phase training plan, an adversarial sub-system, and an evolving execution-intent vocabulary. It is, on the architecture, sound. It is, on the implementation timing, premature.

What I am going to do with it, concretely: nothing yet. TASK_761 runs its course. The four structural directions get evaluated under eval-MCTS sims=16 joint scoring, paired-Δα over three or more seeds, on the canonical benchmark. If any of them lands a per-asset trader that passes the level-0-through-3 envelope cleanly, the portfolio manager doc comes off the shelf and phase 1 of the build begins — which, in the doc's framing, means re-running the per-asset trader on level 0 with the new mechanism and confirming that it converges under 500 epochs in the strict sense. Then phase 2: manager training with the new per-asset policies frozen, on level-4 data first. Then, only if phase 2 demonstrates manager skill, phase 3 joint fine-tune.

What I have learned writing this article, and would not have known if I hadn't sat down to walk the doc carefully:

The portfolio manager design is the *first* document in this project that names the coordinator layer in concrete operational terms. The state space, the action space, the loss factorization, the freeze protocol, the curriculum — all named, all specified, all implementable on the workstation the day the lower layer is ready. Article 3's "the arbitrator is where the strategic awareness lives" was a sentence. This doc is the build plan that backs the sentence. The two artifacts together — article 3's structural argument plus PORTFOLIO_MANAGER_TRAINING.md's operational specification — are the closest thing I have to a roadmap for what comes after Poppy.

I had been embarrassed about the portfolio manager doc for most of 2026. Not because it was wrong, but because it was *unbuilt* — the way a beautifully detailed blueprint for a second floor reads when the foundation is still being tested for load. The embarrassment was the wrong feeling. The right feeling is patience. The doc is ready when the floor under it is ready. The floor under it is what TASK_761 is for.

If the four structural directions all collapse under multi-seed — if 2026 ends with the per-asset trader still at noise floor and no credible mechanism in hand — the portfolio manager doc continues to wait. That is also a defensible outcome, in the sense that the doc is not made worse by waiting; manager-level architectures do not depreciate. What depreciates is the patience of the person doing the work, and that is a different problem from the design.

The doc is right. The floor is not ready. Both sentences are true at once.

---

## Closing

Article 3 ended on the line *the arbitrator is where the strategic awareness lives. Right now that arbitrator is me, drinking coffee, looking at recommendation docs on Wednesday mornings.* The portfolio manager doc is the proposal for what eventually sits in that chair. The chair is empty. The chair has a very detailed instruction manual taped to its underside. The instruction manual specifies that the chair only gets occupied once the row of poppies in the bed below it grows reliably enough to merit a gardener walking the path above. The poppies are pale this year.

> *The plan is sound. The plan is for next year.*

---

## Sources

- `~/Trader/lore/PORTFOLIO_MANAGER_TRAINING.md` — primary source. All block-quoted passages are verbatim, dated to the late-2025 design phase.
- `~/Trader/lore/SMART_EXECUTION_ROUTER_PROPOSAL.md` — the intermediate execution-translation layer between manager intent and atomic exchange orders; sister proposal to the multi-action work.
- `~/Trader/lore/MULTI_ACTION_DESIGN.md`, `~/Trader/lore/MULTI_ACTION_IMPLEMENTATION.md` — the action-point allocation mechanism that the manager's cross-pair action space depends on.
- `~/Trader/bloom/drafts/03_berserker_philosophy.md` — article 3, the berserker frame and the explicit "the arbitrator is where the strategic awareness lives" claim that this article is the build-out of.
- `~/Trader/bloom/drafts/17_five_oracles.md`, `~/Trader/bloom/drafts/18_teaching_simulator_to_hurt.md` — the oracle and microstructure-realism articles that bracket this one in the structural-argument thread.
- `~/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the four structural directions whose verdicts gate whether the portfolio manager build opens.
- `~/Trader/lore/notes_2026-05-20.md` §151 — campaign-exhaustion verdict that frames why the per-asset trader is the precondition the portfolio manager doc is waiting on.
- `~/Trader/bloom/drafts/22_year_of_negative_results.md` — the corpus of multi-seed evidence that the per-asset trader sits at noise floor as of 2026-05.
