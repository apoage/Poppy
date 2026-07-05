# Physics Core: A Differentiable Market I Could Differentiate

*On hybrid physics-neural prediction, the prettiest code I ever wrote, and why it never connected to a training run.*

> *Observation: the specimen called Poppy dreams of futures. Most of those dreams are noise. I have been told a tincture exists that bounds her visions to the shape of a market — that prevents her hallucinations from drifting into prices that no market ever quoted. I prepare the tincture. I label the vials α, β, σ, γ. I will administer it when she is ready. I write the date on the shelf, and I wait.*

---

## I. The seduction of trustworthy long-horizon prediction

In December 2025 my desktop wallpaper was a printout of a paper I had read three times. The paper was about ocean models. It had no business being on the desktop of someone building a crypto trading agent.

The paper was **NeuralOGCM** (Wu et al., arXiv:2512.11525v1) — a hybrid model where the *known physics of the ocean* (advection, Coriolis, diffusion) ran in parallel with a neural corrector that learned the residual dynamics. The physics had learnable coefficients. The corrector was a transformer. Both were trained end-to-end. The selling point was the part that lit me up: the rollouts didn't diverge. Two hundred steps and the model was still producing ocean.

This was the thing I was actually missing. Not capacity. Not data. *Stability*. Pure neural world-models, when you ask them to imagine futures by feeding their own outputs back as inputs, accumulate error like a drunk accumulates poor decisions — gradually, then all at once. At MCTS planning depth, that drift becomes a planning catastrophe: the search tree branches out into prices the asset has never seen, the value head evaluates fantasy states, and the policy starts trading against ghosts. I had been hand-waving this problem all autumn under the phrase *"rollout hallucinations"*. Reading the NeuralOGCM paper, I felt the kind of clean professional joy that, in retrospect, should have been a warning.

If oceans, why not markets. Markets have laws too — weak laws, contested laws, but laws. Things revert to means. Things carry momentum. Things correlate. Things diffuse under volatility. Each of those is a one-line ODE. Each of those has a parameter. *Make the parameters learnable.* Add a neural corrector for the dirty residual — order flow, sentiment, the part the textbooks won't write down — and you have a world-model that can't drift into impossible territory, because the impossible territory has been priced out of the parameter space.

I opened a fresh file. I called it `physics_core.py`. I had not yet written a single line of code, and I already knew what I would put in the abstract.

## II. The bestiary of learnable parameters

The dream had four creatures.

**α — momentum.** Prices in motion stay in motion. In one line:

```
dp/dt = α · velocity
```

Make α a `nn.Parameter`. Start it at 0.8. Let gradient descent argue with it.

**β — mean reversion.** Prices return to a moving average. In one line:

```
dp/dt = -β · (price - moving_average)
```

Start β at 0.2. Soft-plus it to keep it positive. Let the model decide how aggressively the rubber band pulls.

**σ — per-asset volatility.** Brownian noise, properly scaled:

```
dp/dt = σ · ε · sqrt(dt),   ε ~ N(0,1)
```

A learnable σ *per asset*. So Bitcoin can be a different beast from a stablecoin and the network doesn't need to discover that fact through ten thousand gradient steps — it's a coordinate in the parameter space, and it slides to wherever the loss tells it to slide.

**γ — cross-asset correlation.** A learnable [n_assets × n_assets] coupling matrix:

```
dp_i/dt = Σ_j γ_ij · (p_j - p_i)
```

Correlated assets pull each other. This is the line that should have made me suspicious. A 64×64 dense correlation matrix is 4096 parameters of pure overparameterization, eagerly fitting noise. The fix arrived a few days later as `LowRankCorrelation` — factor decomposition Γ = L Lᵀ / k, eight factors for 64 assets, 512 parameters instead of 4096, naturally positive semi-definite. The structure (latent risk factors) was a side gift of the constraint. I quote my own December note on this with a wince at how proud I was:

> *"Low-Rank Correlation: Factor decomposition `Γ = L @ L.T / k`. 8 factors for 64 assets = 512 params (not 4096). Naturally positive semi-definite. Captures market factors (sectors, risk factors)."*
> — `lore/notes_2025-12-30.md`

The four tendencies sum:

```
G_physics = G_momentum + G_reversion + G_volatility + G_correlation
```

Add a neural corrector — encoder → cross-asset attention → decoder — that outputs `G_neural` on the same shape. Step the whole thing with a Forward Euler integrator (later upgraded to RK2 midpoint for a sniff more accuracy):

```
price_{t+Δt} = price_t + Δt · (G_physics + G_neural)
```

That's the model. That's the entire mathematical content of the dream. What I loved about it, what made me lose two evenings to it before writing a line of training code, was how *the inductive bias was right there in the type signature*. The physics couldn't predict a price that broke conservation laws because the conservation laws were the parameters. The neural corrector could only nudge the physics, not replace it — its outputs were tendencies, not states, so the worst it could do at any single step was add a wiggle. Compound a thousand wiggles autoregressively and yes, you still get drift, but the drift was bounded by a known mechanism, not by whatever the corrector hallucinated.

This was supposed to be the foundation underneath MCTS. Search depth 5, branches 3, 243 leaves — every one of those leaves a *plausible future*, not a fantasy. Then the value head evaluates real-shaped states, the policy plans against a world that obeys at least the laws it learned, and the agent is, at last, allowed to think.

## III. What I built

Across two days at the end of December I wrote three files. They live, still, at `mcts_trader/jax_agent/prediction/`.

**`physics_core.py`** holds `DifferentiableMarketPhysics`, the Flax module that owns α, β, σ, γ. It also holds two pieces of furniture I'm still fond of: `LowRankCorrelation` (the factor-decomposed coupling matrix) and `GARCHVolatility` — a small recurrence that lets per-asset σ become time-varying. The GARCH update is the standard one,

```
σ²_t = ω + α_g · r²_{t-1} + β_g · σ²_{t-1}
```

— a learnable ω, α_g, β_g per asset, capturing volatility clustering. Crypto markets have *clusters*. The post-LUNA week is not the pre-LUNA week. A static σ would have averaged across regimes and produced a corrector that was always wrong by the same amount in opposite directions. The GARCH head meant the variance term could *breathe*.

**`neural_corrector.py`** holds `NeuralMarketCorrector`. The architecture is the one piece of the design I'd defend without qualification today: assets-as-tokens cross-asset attention. Most market models, when they want to capture cross-asset structure, flatten the asset dimension into the batch and lose the relational signal. By treating assets as tokens of a sequence (with learned positional embeddings for their identities) and running multi-head self-attention across them, the corrector could learn dynamic, asymmetric coupling — *the BTC token attends strongly to the ETH token; the stablecoin token attends to no one* — and update those relationships as the rollout progressed. From the source:

> *"CrossAssetAttention: Multi-head attention treating assets as tokens. Instead of treating the batch as the sequence, we treat assets as tokens. This allows the model to learn cross-asset dependencies dynamically."*
> — `mcts_trader/jax_agent/prediction/neural_corrector.py`

The corrector inherited a `SimpleNeuralCorrector` MLP variant for lightweight ablation, but the transformer was the production path. All operations in log-space — the entire file's docstring closes on the line *"All operations in LOG-SPACE for numerical stability,"* a discipline I had learned the hard way two weeks earlier on a different module.

**`market_ode.py`** is the integrator. It owns the physics core and the corrector, holds the timestep `dt`, and runs the ODE step. It exposes `_euler_step`, `_rk2_step`, and — the one I was most proud of — `rollout()`, which uses `jax.lax.scan` to unroll arbitrary multi-step trajectories under JIT. It also holds `MarketImpact`, a tiny action-conditioning head so the predicted trajectory could be conditioned on the agent's hypothetical action (`asset_idx = action // 3`, `action_type = action % 3`). This was the piece that promised to let MCTS evaluate "what happens if I buy ETH right now" inside the world-model, not against a frozen replay snippet.

The corrector application, lightly trimmed:

```python
def compute_tendency(self, state_t, features_t, noise, action=None):
    G_phys = self.physics_core(state_t)
    G_neural = self.neural_corrector(state_t['log_prices'], features_t)
    G_impact = self.market_impact(action) if action is not None else 0.0
    return G_phys + G_neural + G_impact

def _rk2_step(self, state_t, features_t, noise, action):
    k1 = self.compute_tendency(state_t, features_t, noise, action)
    mid_state = self._advance(state_t, 0.5 * self.dt * k1)
    k2 = self.compute_tendency(mid_state, features_t, noise, action)
    return self._advance(state_t, self.dt * k2)
```

Three files. About 45,000 characters of code. The kind of code that, when you re-read it cold, still reads cleanly — variables named what they are, comments where they have to be, no clever tricks, the math directly visible in the function bodies.

On December 30 I ran the integration test. The terminal scrolled, and at the bottom of the scroll were the seven lines I had been writing the code for the whole week:

```
Total parameters: 276,354
Forward pass (JIT): 0.121 ms per call
G_phys magnitude: 0.120823
G_neural magnitude: 0.031191
G_impact magnitude: 0.000003
Trajectory shape: (4, 11, 64)
All tests passed!
```

Forward pass: **0.12 ms**. Ten-step rollout: **1.2 ms**. The target for the forward pass had been `<1 ms`; I was eight times under. The target for the rollout had been `<10 ms`; I was eight times under. The physics components were sane (`G_phys` an order of magnitude bigger than `G_neural`, which was an order of magnitude bigger than `G_impact` — the corrector was making small, careful corrections, exactly as designed). The rollout shape was `(4, 11, 64)` — four batches, eleven steps, 64 assets — the trajectories were *tensorial*, JIT-able, and ready to plug into anything.

I closed the laptop. I told myself I would integrate it into MuZero training the next week. I labelled the vials and put them on the shelf, and I went to bed at peace.

## IV. The kickback

It is now May 2026. Five months later.

The physics core has never been integrated into a training run.

I want to be precise about what that sentence does and does not say, because the truth is more interesting than "I forgot." The code is not dead in the strict sense. `physics_core.py`, `neural_corrector.py`, `market_ode.py`, and a fourth file added a few weeks later — `tree_search.py`, the physics-aware MCTS surrogate — all still exist, still pass their imports, still benchmark at the numbers above. There is even an integration point in the main MuZero training entry: `train_jax_fast.py` imports `DifferentiableMarketPhysics` at line 57, and around line 1442 it pulls the learned α, β, GARCH parameters out of the params tree and logs them to the DS24 dashboard. Tree search has a CLI flag — `--use-tree-search`, with `--tree-depth`, `--tree-branches`, `--risk-lambda` companions — wired into `mcts_trader/training/tree_search_setup.py`.

The flag's default is `False`. I have searched the entire experiments directory, every committed run config, every dojo orchestrator, and every spec under `~/Trader/experiments/`. **No experiment has ever set `--use-tree-search`.** The relevant entry in my own notes from January is unambiguous:

> *"Low-priority flags (`--flow-matching`, `--use-tree-search`, `--adaptive-calibration`, `--early-graduation`) are NOT used by dojo_v4."*
> — `lore/notes_2026-01-24.md`

And the campaign that actually consumed every GPU-hour of the next four months wasn't run through `train_jax_fast.py` at all. It was run through `train_easy.py` — Foxglove, the supervised Phase 1 imprint, the stage where Poppy doesn't yet plan, she just learns to imitate an oracle. Foxglove's trainer initializes models with `use_prediction_head=False, # No dynamics in Phase 1`. The whole physics machinery is on the wrong side of the Phase 0 / Phase 1 line. I built a beautiful organ for a body that was still learning to crawl.

So: why didn't I plug it in? Three reasons, none of them dignified.

**The integration cost was non-trivial and the timing was bad.** To wire the physics core into the active training loop meant editing the MuZero training loop in `train_jax_fast.py` at the very moment that loop was the only thing producing checkpoints I trusted. The replay buffer needed an auxiliary path for `next_log_prices`, `moving_avg`, `noise`. The loss aggregation needed a `λ_pred` annealing schedule. The MCTS search itself needed surgery to thread `params['price_predictor']` through to the tree-search reward adjuster. Each of those is a 50-line change. Together, they were a multi-day refactor of code that, at the time, *was producing learning curves I didn't want to invalidate*.

**The hardest problem — efficient backprop through MCTS-rollout-via-physics-core — I had not actually solved.** Running the physics core during planning is one thing; differentiating through a depth-5 search tree, 243 leaves, each leaf the output of a 5-step ODE rollout with its own noise sample, while the encoder upstream is also being trained, was a problem I had sketched on a napkin and not implemented. The napkin sketches involved straight-through estimators, partial path differentiation, gradient checkpointing — three things I had used individually but never composed. I'd told myself I would solve it "when needed." It was never quite needed enough to justify the week it would have cost.

**Status-quo gravity.** This is the worst of the three reasons and the most honest. The training loop *worked*. It produced losses that went down, checkpoints that benchmarked, dashboards that filled with curves. Every day I didn't integrate the physics core, the training loop produced more imprint progress that I would have had to invalidate the moment I changed the world-model. The cost of switching grew weekly, and the perceived benefit — which was always "future MCTS stability that I am not currently testing for" — stayed abstract. It is exactly the kind of decision a sober reviewer would have flagged, and exactly the kind I made anyway. "Tomorrow" is the most expensive word in research.

And then the deeper kickback — the one that, when it finally arrived in May, made the whole physics-core question retroactively beside the point.

**The bottleneck wasn't where I had built the cure.** Across April and May, three independent experimental setups — TASK_737, TASK_741, TASK_738 — showed the same pattern: improving the encoder's classification quality (loss ↓, Bal.Acc ↑) made the *trader* worse (Alpha ↓), because the policy became more confident, traded more aggressively, and let fees eat the edge. The encoder was not the bottleneck. By extension, neither was the world-model that fed it. The note that closed this question is in `lore/notes_2026-05-15.md` §110, and the operative line is:

> *"The bottleneck is downstream of the encoder."*

The hallucination problem I'd designed the physics core to solve — the drift in rollouts beyond a few steps — was real, but it was not the load-bearing failure. The load-bearing failure was in the policy head, in value-head saturation, in the entire planning-side machinery downstream. Even a perfectly stable 200-step rollout would have fed the same policy a different fantasy and gotten the same over-trading collapse out the other side.

I built a beautiful solution to a problem that turned out, when measured, not to be the bottleneck.

---

> *The Apothecary keeps the vials. They are still well-stoppered. The labels — α, β, σ, γ — have not faded. The tincture would still bound the visions if she chose to administer it. But the patient's complaint, when finally diagnosed, was not in the eye. It was in the hand that held the trade.*

---

## V. What's left, what I learned

What's left, after the inventory, is more than I expected.

The differentiable-parameter *idea* turned out to be the load-bearing insight, just not on the path I thought. **Bella**, the market generator built later, partly inherits the spirit: a parameterized synthetic-market process whose coefficients can be tuned, scheduled, even (in a later variant) learned. The lineage from `DifferentiableMarketPhysics` to Bella's per-regime parameter pool is direct. The pieces I'd reuse without hesitation are `LowRankCorrelation` (the factor-decomposed coupling is still the cleanest way I know to keep n×n correlation matrices honest), `GARCHVolatility` (variance has to breathe), the assets-as-tokens cross-asset attention pattern, and the log-space discipline. The dead code on disk is, in fact, an inventory of solved sub-problems waiting to be re-used.

The lessons are the kind that arrive slowly and stay.

*"It works in isolation" is not the same as "it earns its place in the production pipeline."* A passing unit test, a clean benchmark, a satisfying parameter count — none of those are integration. The thing that integrates is the willingness to invalidate weeks of upstream progress to slot the new module into the actual loop, on the actual data, with the actual loss. I underweighted that cost. The next time I write a beautiful self-contained module, I will write the integration plan before I write the module.

*Confirm the bottleneck before writing the cure.* This is the larger lesson, and the one that paid for itself many times over the following months. I designed the physics core against a hypothesis ("rollout hallucinations are killing planning") I had never instrumented. By May the instrumentation existed — paired-Δα over multiple seeds, eval-MCTS vs argmax decomposition, trade-rate diagnostics — and the hypothesis was falsified. The work-order, in retrospect: build the instrument first, confirm the diagnosis second, write the cure third. I had done them in the reverse order on instinct.

The standing question I have not answered is: **would a physics-grounded oracle do better than the current `path_solver.py` at Phase 1 imprint?** The oracle in Foxglove is a pure deterministic path-solver — given a window of future prices, it back-solves the optimal trade sequence and that sequence becomes the supervision target. A physics-grounded oracle would, instead, simulate forward under the learned market dynamics and back-solve under those simulations, producing supervision targets that are physically constrained rather than oracular. That's TASK_761 territory — the structural-pivot roadmap I'm writing now, after the campaign's single-mechanism hypothesis space closed. The physics core may yet earn its place, but on the imprint side of the line, not the planning side. The vials are still on the shelf.

## VI. Closing

The prettiest code I ever wrote sits in three files in a directory called `prediction/`, importable, tested, benchmarked, and unused. It is not a failure. It is a deferred bet — a piece of bench-stable craft, kept clean against the chance that the diagnosis changes again. In a project this long, that turns out to be most of what one does. You build the tinctures. You label the vials. You wait for the patient.

The patient, this time, had a different illness.

---

## Sources

- `lore/NEURAL_MARKET_PHYSICS_DESIGN.md` — full design document, Dec 19, 2025.
- `lore/NEURAL_MARKET_PHYSICS_SUMMARY.md` — quick-reference cheat sheet.
- `lore/notes_2025-12-30.md` — implementation completion log, "All tests passed!" benchmark.
- `lore/notes_2026-01-24.md` — dojo_v4 explicitly excludes `--use-tree-search`.
- `lore/notes_2026-05-15.md` §110 — "the bottleneck is downstream of the encoder."
- `mcts_trader/jax_agent/prediction/physics_core.py`, `neural_corrector.py`, `market_ode.py`, `tree_search.py` — the implementation, still resident.
- `mcts_trader/training/tree_search_setup.py`, `mcts_trader/training/config.py` (line 267: `use_tree_search: bool = False`) — the wiring, with the default.
- `train_jax_fast.py` line 57 (import) and line 1442+ (diagnostic logging only) — the partial integration.
- `easy_trainer.py` line 52: `use_prediction_head=False, # No dynamics in Phase 1` — the explicit exclusion on the active training path.
- Wu et al., **NeuralOGCM: Differentiable Ocean Modeling with Learnable Physics**, arXiv:2512.11525v1 — the paper that started it.
