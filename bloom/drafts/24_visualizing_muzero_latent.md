# Visualizing a MuZero Latent Space

*On the gap between drawing a latent state and understanding one. On two long research reports commissioned to enumerate the techniques, and the dashboard that ended up implementing about a third of them. On why a beautiful PCA plot still failed to predict the most consequential bug of the campaign — and why I am keeping the dashboard anyway.*

> *Observation: the roots of Poppy are dark and densely braided beneath the soil. I cut a small window in the side of the pot — three centimetres square, no more — and seal it with glass. From the window I draw what I can see. The drawing is faithful to what is visible. Most of the root system is still hidden in the dark earth on the other side of the pot. I record this asymmetry, in pencil, in the margin of the page.*

---

## What's open on the desk

`workstation` is on, the desk fan is on, and on the left monitor the DS24 dashboard is open in Firefox at `http://localhost:8051`. The Metrics tab is active. A scrolling chart of reward-per-episode is updating every few seconds because behind me, in a separate tmux window, `train_phase2.py` is running a sims=64 paired-seed sweep on the 3090. The GPU is at 67°C and 318W; `nvidia-smi` says memory is pinned at 19.4 GB of the 24 it has. It is the afternoon of 2026-05-22 and I am writing this article in the same Python environment the training loop is running in.

I am writing it because earlier this week, going through the `docs/aires/` directory for an unrelated reason, I rediscovered a pair of long research reports I commissioned through a deep-research agent at some point earlier in the campaign — I can no longer be precise about when, the file dates say spring but the content suggests they were running in batches. The two reports together are about a hundred pages of synthesis on a single topic: *how to visualize the latent space of a MuZero-style agent built in JAX.* One report is the mathematics side — Riemannian geometry, vector fields, Lyapunov exponents, NeRFs and Gaussian splatting as ways of looking at a manifold. The other is the engineering side — which heads to log from, which projections to compute, which dashboard panels actually pay rent.

I opened them because I wanted a refresher on streamplot conventions. I ended up reading both, end-to-end, for the first time. And I noticed something I want to put on the page in this article, because it is the kind of observation the diary is for: *the dashboard I actually built (DS24) implements maybe thirty percent of what those two reports describe.* The other seventy percent was either too expensive to run online, or had unclear diagnostic value, or, in a few notable cases, was a technique I read about, tried to imagine in service of debugging the actual campaign, and could not picture catching anything Poppy was doing wrong.

This article is a walk through both halves of that gap. The techniques the reports describe. The techniques DS24 implements. And the deeper question — the kickback at the bottom of the page — of whether visualizing a latent space was ever going to deliver the diagnostic breakthroughs I had been hoping it would. (Spoiler: the most consequential bug of the campaign, the *better-classifier-worse-trader* anti-correlation of Article 6, was found by aggregating metrics across runs, not by squinting at any single latent projection.)

---

## What you have to visualize, when you visualize MuZero

A quick frame, because the rest of the article depends on it.

MuZero, as a class of agents, is unusual in that its world model is *learned*. There is no environment simulator written down in code that the planner consults during search; instead, three networks are trained jointly:

- An **encoder** `h: o → s` that takes an observation `o` (in my case, a window of multi-asset market state) and produces a latent state `s`.
- A **dynamics function** `g: (s, a) → s'` that, given a latent state and an action, predicts the next latent state. There is no decoder back to observation space. The dynamics function operates entirely in latent coordinates.
- A **prediction function** `f: s → (π, v)` that, given a latent state, outputs a policy prior `π` and a value estimate `v`.

The planner — MCTS with a PUCT-style selection rule — composes these. From a root state `s_0 = h(o)`, it expands trees of imagined trajectories by repeatedly applying `g`, scoring leaves with `f`, and backing up values. The action it ultimately picks is a function of root-visit counts.

The thing that is interesting and also difficult about MuZero is that nothing in this stack constrains the *meaning* of the latent state. The latent `s` is whatever the gradients found to be useful. It does not, in general, correspond to "the state of the market"; it corresponds to "whatever encoding the value head most easily reads from, while the dynamics function can roll it forward without too much error." The latent can be 64-dimensional, 256-dimensional, sometimes more. It lives in a space whose geometry is induced entirely by training pressure.

To debug such a system — to know whether the encoder is *learning anything*, whether the dynamics are *consistent*, whether the value head is *calibrated*, whether the policy is *exploring* — you have to *see* the latent. And visualization of high-dimensional learned latents is itself an open research problem. There is no canonical answer. There is a family of approximate answers, each with trade-offs, and you assemble a dashboard out of them.

This is the problem the two reports in `docs/aires/` are about.

---

## The two reports

The first report is titled *Geometric Deep Learning and Latent Dynamics: A Comprehensive Framework for Visualization in JAX*. It runs about fifty pages with sixty embedded figures, and its register is academic-survey: differential geometry first, then the JAX implementation tooling (Diffrax, Equinox, Optax), then projection and rendering techniques borrowed from computer graphics. Its opening claim is the framing one:

> "The epistemology of machine learning is undergoing a profound transformation, shifting from a discrete, algebraic perspective toward a continuous, geometric understanding of information processing. […] As the field matures, the imperative is no longer merely to classify static points but to understand dynamic flows — the trajectories of information as it evolves through the 'depth' of a network or the 'time' of a dynamical system."
>
> — *Geometric Deep Learning and Latent Dynamics*, §1, `docs/aires/`

The second report is titled *Visualization Architectures for JAX-Based MuZero Agents*. It is shorter, denser, and more applied. Its first section frames MuZero's dynamics network as a discretized continuous flow — the residual stream of the predictor as a forward-Euler step of an underlying ODE — and asks how to *observe* that flow in JAX:

> "If the Transformer layers approximate a continuous flow, then the token representations should converge to a unique solution of an ODE as the number of layers increases. Visualizing this convergence, and the deviations introduced by the discrete attention heads, is essential for diagnosing stability and expressivity."
>
> — *Visualization Architectures for JAX-Based MuZero Agents*, §1.1, `docs/aires/`

Between them the reports cover, by my count, somewhere north of forty distinct visualization techniques. Some are foundational (PCA of latent rollouts, UMAP of value-head outputs). Some are exotic (Gaussian splatting of latent density, persistent-homology Betti numbers, Lyapunov-exponent maps of the chaotic regime). Some are aspirational (real-time NeRF rendering of the latent manifold, animated streamline plots of the learned vector field, fractal-dimension overlays on PCA projections).

The reports do not tell you which subset to actually build. That is, on reflection, the part I most needed and the part neither report could provide — *you have to triage the techniques against your specific campaign's diagnostic surface, and that triage is local to the problem you are trying to debug.* I will get to my triage further down. First, the techniques.

---

## Eight techniques worth knowing

What follows is the working subset I came out of both reports able to describe and defend. Each entry: the math in one or two lines, what the technique is *for*, and whether it caught anything in this project. Where it caught nothing, I say so plainly — that asymmetry is the most useful thing this article can give a future reader.

### 1. Latent-trajectory PCA over rollout

The simplest viable technique. Run the encoder on a batch of observations to get a batch of `s_0`. Apply the dynamics function `g(·, a)` repeatedly for `k` steps along some action policy (greedy under the learned `π`, or uniform-random, or a fixed scripted policy). You now have an `(N, k, d)` tensor of latent states — `N` rollouts, `k` steps each, `d`-dimensional latents.

Flatten to `(N · k, d)`, compute the top-2 principal components, project. You get a 2D scatter where each rollout is a polyline of `k` points connected in time. Color the polylines by some semantic variable — episode index, regime label, realized return.

The math is one line:

```
s_flat = stack([h(o_i)] ∪ [g(s_{i,t-1}, a_{i,t-1}) for t in 1..k])
U, S, V = svd(s_flat - mean(s_flat))
proj = s_flat @ V[:, :2]
```

What it is for: *checking whether the latent has structure at all.* If the projected polylines are a uniform blob, the dynamics function is moving latents around randomly and the planner is reading noise. If the polylines form coherent flows — most trajectories drifting in similar directions in a given regime — the dynamics has learned something.

What it caught in this project: it caught the failure mode where, early in Phase 2, the dynamics function had collapsed to near-identity. Every rollout was a tight cluster, almost a single point — `g(s, a) ≈ s` for all `a`. The PCA scatter looked like a constellation of dots, not flows. That was diagnostic. Fixing it took weeks; *seeing* it took five minutes of plotting.

### 2. UMAP of value-head outputs across states

Take a large sample of states (encoder outputs across many observations), pass each through the value head, color by predicted value. Embed the *latents* (not the values) into 2D via UMAP. You now see whether the latent space is *value-organized* — whether states the value head considers high-value cluster together in latent coordinates.

UMAP cost function (approximately):

```
L = Σ_ij [ p_ij log(p_ij / q_ij) + (1 - p_ij) log((1 - p_ij) / (1 - q_ij)) ]
```

where `p_ij` is the fuzzy nearest-neighbor probability in latent space and `q_ij` is the same in the 2D embedding. The optimizer tries to make the embeddings agree with the latent-space neighborhoods.

What it is for: validating that *similar latents predict similar values*. If high-value and low-value states are interleaved across the UMAP, the latent is not value-aligned and the value head is doing the entire job of discrimination by itself — which suggests the encoder is under-utilized.

What it caught: on the canonical 39-channel Stage-B baseline, the UMAP is moderately value-organized — clear high-value and low-value modes, with a messy boundary. On TASK_738 (the legacy-27ch ablation that produced the catastrophic over-trading collapse of Article 6), the UMAP was *more sharply value-clustered* than baseline. That sounds like an improvement, and on the dashboard it looked like one. It was, in retrospect, the visual fingerprint of the bug — the encoder had learned a sharper directional projection, which the policy then over-bet on. The picture was beautiful; the deployment Alpha was catastrophic. *I had the diagnostic image. I did not have the diagnostic.*

### 3. Policy-entropy heatmap by state

For each sampled state `s`, compute the entropy of the policy prior `H(π(·|s)) = -Σ_a π(a|s) log π(a|s)`. Project the states into 2D (via the same UMAP from above, or via PCA), then color by entropy.

What it is for: detecting regions of state-space where the policy is *over-confident*. A region of low entropy means the policy has crystallized around a single action there. If that region is large or covers semantically diverse states, you have policy collapse — the agent has decided one action fits everything and stopped exploring.

What it caught: this is one of the panels DS24 actually has, in the Reasoning Dynamics tab, and on long Phase-2 runs it caught the gradual entropy collapse during epochs 40–80 that ended up requiring an entropy floor in the Stage-B loss. The heatmap turned cold (low entropy) in a wide swath of state-space around epoch 60, and stayed cold; once we added the entropy floor, the cold patch loosened. That is a clean save — Article 6's anti-correlation is *not* the only over-confidence pathology, and this one the visualization did help me catch.

### 4. MCTS visit-count histograms

For each sampled state, run MCTS rooted there with the production simulation budget. Plot the histogram of visit counts across actions. Repeat across many sampled states; aggregate.

A healthy histogram in a multi-action stack should look multimodal — most states should distribute visits across two or three competitive actions. Pathological histograms are *spiky* (one action gets nearly all the visits, even when value differences are small) or *uniform* (the search is just exploring randomly because no value signal is gradient-flowing through it).

The math is mostly bookkeeping. The diagnostic is in the shape.

What it caught: this is a panel in the Planning Analysis tab. On the May-2026 over-trading variants (TASK_741 Mint xs, TASK_738 legacy_27ch) the visit-count histograms were *aggressively spiky* — the search was concentrating on a single action at every root, with the second-ranked action barely getting any visits. The dashboard showed this; I noticed it; I told myself it meant "MCTS is confident," which was true and also the wrong conclusion. Confidence at the root, given a sharper value surface, is exactly the mechanism Article 6 walked through. *The picture was correctly drawn; I drew the wrong inference from it.*

### 5. Saliency maps for the encoder

Take a state, take its encoder output, take the gradient of some scalar function of that output (e.g. the norm of the latent, or the value prediction `v(s)`) with respect to the input observation. The magnitude of that gradient, per input channel and per timestep, is a saliency map — it says which parts of the observation the encoder is *reading from* to produce this latent.

```
saliency_ij = | ∂||h(o)|| / ∂o_ij |
```

What it is for: catching channel under-utilization or channel dominance. If the encoder is mostly reading from two of the 39 input channels and ignoring the rest, you have a *de-facto bottlenecked encoder* whose nominal capacity is a lie. Similarly, if the cross-sectional channels (27–38) are weighted near zero, the encoder is effectively running on the legacy 27-channel input and TASK_732's whole engineering effort was wasted.

What it caught: nothing in this project, because *I never implemented this panel in DS24.* The technique is on the page in the engineering report — explicit, with code examples — and I wrote it down in `VISUALIZATION_RESEARCH_AND_PLAN.md` as a P3 priority and never came back to it. In retrospect this is one of the gaps I most regret, because a saliency map of the cross-sectional channels would have been a cheap and direct read on the question TASK_738 was eventually asked the hard way.

### 6. Latent dynamics consistency

This is the one technique that is somewhat unique to MuZero. The dynamics function `g(s, a)` and the encoder `h(o')` are supposed to produce the same latent for the same future observation:

```
g(h(o_t), a_t) ≈ h(o_{t+1})
```

If they don't, the world model is incoherent — the planner is rolling forward through a fiction that the encoder, given the real next observation, would not agree with. The residual

```
r_t = || g(h(o_t), a_t) - h(o_{t+1}) ||_2
```

is the diagnostic. Plot `r_t` over training. Plot its histogram across `(o_t, a_t)` pairs. Plot it conditioned on action — if `r` is small for the most-common action and huge for the rare ones, the dynamics has overfit to a narrow behavioral distribution.

What it caught: an annoying period in early Phase-2 where the consistency residual stayed flat regardless of action. That meant `g` had become approximately action-invariant — `g(s, a) ≈ g(s, a')` for all `a, a'`. The dynamics function was, effectively, a no-op layer. We fixed that by upweighting the consistency loss term and the residual histogram developed action-conditional structure within twenty epochs. Cheap to compute, easy to read, paid for itself.

### 7. Value-head temporal calibration

Predicted value `v(s_t)` vs. realized discounted return-from-`t`. Scatter, fit a line, check the slope and intercept. Should be the 45° line. A flat slope means the value head is saturated (outputs are insensitive to state); a steep slope means it is over-shooting; an intercept offset means it has a systematic optimism or pessimism.

What it caught: in May 2026, this panel was how I noticed that the value head on the over-trading variants had developed a wide-range output distribution — `v(s)` was producing more extreme positive and negative values per state than baseline, with the same realized-return support. The scatter was *steeper than 45°*, and that steepening was the visual signature of the sharper-value-surface mechanism Article 6 unpacked mathematically. So this panel *did* catch the structural failure, in a sense — but only after I knew what I was looking for. Until the multi-seed Δα table told me there was an anti-correlation, the steepening scatter just looked like "value head is learning better."

### 8. Action-coverage matrix

Across all rollouts in a held-out evaluation, count how many times each action was taken. Normalize per regime. Plot as a heatmap with regimes on one axis and actions on the other.

This is the cheapest visualization in the entire reporcsuite — it is a 2D `numpy.histogram2d` and a colorbar — and it is also the one DS24 has had since day one, in the Metrics tab as "Action distribution." It catches mode collapse instantly. If the bear-regime row is 99% "short" and the bull-regime row is 99% "long," the trader has learned regime-conditioned mode collapse — *which sounds correct on its face* and is in fact a pathology, because the right answer in many bull regimes is "long with some hedges," not "100% long."

What it caught: TASK_738's over-trading collapse, in a way that should have been a louder alarm than I let it be. The action-coverage row for the bull period under TASK_738 was *non-collapsed* — the action distribution was nearly uniform across long/short/hold. That looks like healthy exploration. It is, on the second pass, the exact opposite — uniform action coverage *given* a high trade rate means the trader is *flipping* positions, not exploring. The visualization had the data. I had not encoded the corollary that "uniform actions + high trade rate = churn" into how I was reading the heatmap.

---

## What DS24 actually does

DS24 lives at `tools/DS24/`. It is a Flask server, a vanilla HTML/JS frontend, Chart.js for plotting, and a CSV+JSON-lines logging path that the training loop writes into via `metrics_logger.py`. It runs on `localhost:8051`. It has eight tabs.

Here is the mapping from those tabs to the techniques above:

- **Metrics**: action distribution (technique 8), value-calibration scatter (technique 7), policy entropy over time, loss components, reward-over-time. About half of these correspond to techniques in the reports; the other half are training-side metrics with no latent-space interpretation.
- **Reasoning Dynamics**: TRM cycle deltas (a project-specific diagnostic for the recursive predictor — not in either report), policy crystallization over H-cycles, latent magnitude over training. The latent-magnitude trace is a weak proxy for technique 1 (PCA over rollouts) — it tells you the latent has *some* structure but not what.
- **Physics Model**: G_momentum / G_reversion / G_volatility / G_correlation breakdown for the Neural ODE dynamics regularizer. This is project-specific instrumentation, not in the reports, but conceptually it is in the same family as technique 6 (dynamics consistency) — it tries to give the dynamics function physically-interpretable components and visualize each.
- **Planning Analysis**: MCTS visit entropy, simulation-ROI, horizon-accuracy. Technique 4 (visit histograms) lives here. Technique 7 (temporal calibration) is partially here as horizon-error charts.
- **Component Interactions**: gradient norms, loss-contribution stacked area, TRM↔Physics correlation. Diagnostic-but-not-visualization-of-latent.
- **Analysis**, **Checkpoints**, **Benchmarks**, **Logs**: operational, not interpretive.

Counting strictly, of the eight latent-visualization techniques I described above, DS24 implements three with full fidelity (4 MCTS histograms, 7 value calibration, 8 action coverage), two as partial proxies (1 latent magnitude as PCA-substitute, 3 policy entropy without the spatial heatmap), and three not at all (2 UMAP-of-values, 5 saliency, 6 explicit consistency residual). Of the forty-plus techniques in the two reports, the share DS24 implements is closer to thirty percent — the eight above plus a handful of MuZero-specific items the reports describe (reward calibration, prediction-vs-actual scatter for the dynamics head, gradient-flow histograms) that DS24 also has.

The remaining seventy percent — the streamline plots of the learned vector field, the Lyapunov-exponent maps, the persistence-homology Betti numbers, the divergence overlays, the NeRF-rendered latent density, the symplectic-area-preservation diagnostics — exists in DS24 only as items in `VISUALIZATION_RESEARCH_AND_PLAN.md` under future-work headings. None of it shipped. The reason is the same in every case: the implementation cost was nontrivial, the additional diagnostic *yield* was unclear, and the dashboard was already useful enough that the marginal panel was hard to justify against the marginal experiment.

---

## The Apothecary, briefly

> *I lift the glass from the side of the pot and the dark earth presses against it. I draw what I see. The light is unequal: the roots near the glass are bright and clearly veined; the roots two centimetres behind them are silhouettes; the roots at the centre of the pot are absence. I label the drawing and date it. I do not pretend the drawing is the plant.*

---

## The kickback, in three layers

The first layer is the smallest and the most foundational, and it is the one I want to put on the page before I go anywhere fancier: *visualizing a latent space is not the same as understanding it.* A PCA plot tells you the latent has structure. It does not tell you whether the structure is the *right* structure. A UMAP says high-value states cluster. It does not say the clustering is what causes the policy to bet well. A visit-count histogram says the search is confident. It does not say the confidence is earned. Each of these visualizations is a faithful projection of one slice of the latent, and a faithful projection is also, definitionally, a *partial* one. The map is not the territory; the projection is not the latent. The discipline of latent visualization is the discipline of remembering which question your picture answers and which it does not.

The second layer is the one I keep coming back to in the diary, because it lands the work-versus-yield calculus in a specific number. The two research reports together describe more than forty visualization techniques. DS24 implements about a third of them. *The other two-thirds were deferred because the implementation cost was high and the diagnostic yield was unclear.* That is not a failure of ambition; it is a triage. A Lyapunov-exponent map of the latent dynamics would have been beautiful to ship and almost certainly would not have changed any decision I made in May. A NeRF-rendered density of the latent manifold would have been a publication-grade visual and would have answered no question the action-coverage heatmap couldn't answer more cheaply. The reports are aspirational documents — they enumerate what is *possible*, not what is *necessary*. The triage from possible to necessary is the engineering, and the engineering is local to the campaign.

I think this is a real point about commissioned research, by the way, and not specific to visualization. A research report on a topic gives you the *space* of techniques. It does not give you the *triage*. The triage requires knowing where the specific bug is going to land and what visual signature it will have, and that knowledge is the thing you don't have yet, by construction — if you had it, you wouldn't need to debug.

The third layer is the deepest, and it is the one I owe most plainly because it is the part the project actually proved. *Visualization was supposed to help diagnose the campaign's failures. It did help — somewhat. But the actual diagnostic breakthrough of the campaign — the better-classifier-worse-trader anti-correlation that Article 6 unpacks — came not from any visualization but from aggregating multi-seed deployment Alpha across runs and noticing that the column of training-side Bal.Acc and the column of paired Δα had opposite signs in three independent experiments.* The pattern was global, across runs. The visualizations were local, per run. No single picture of TASK_738's latent space, no matter how artful the projection, would have given me the anti-correlation — because the anti-correlation only exists in the comparison between the variant's deployment behavior and the baseline's, and you need both runs' aggregated returns side by side, in a table, not in a scatter plot, to see it.

That is a hard lesson, and I want to be careful with it. It is not that the dashboard was useless. The entropy-collapse catch of technique 3 was real. The dynamics-consistency catch of technique 6 was real. The action-distribution sanity-check is what made me first notice the trade-count explosions even before I had Alpha numbers. The dashboard did its job. *It just did its job in service of single-run diagnostics, and the most important bug of the campaign was a property of the comparison between runs.* The right tool for that comparison is a paired-seed result table — text, columns, hard numbers — and the right tool for the visualization was, in the end, not a visualization. It was a sort.

---

## What is left, what carries forward

DS24 is still the daily-driver dashboard. I open it every morning. The Metrics tab, specifically, is the first thing I look at when a run finishes — action distribution, loss components, calibration. The Reasoning Dynamics tab gets glanced at during long runs to confirm the entropy floor is doing its job. The Planning Analysis tab is the second thing I look at when a benchmark number surprises me — visit histograms tell me whether the policy concentrated or spread, and that is information I cannot get from the result-card table.

The Reasoning, Physics, Planning, and Interactions tabs are technical debt of a specific kind. They are not technical debt in the "code is bad" sense; the code is fine. They are technical debt in the sense that each tab represents a hypothesis about *which* diagnostic surface is going to matter, and the project has, in the year since they were designed, drifted away from some of those hypotheses. The Physics Model tab's G_momentum/G_reversion/G_volatility decomposition was load-bearing when the Neural ODE regularizer was the central research question. After the structural pivot of `TASK_761`, the dynamics regularizer is no longer where the action is, and the tab is read maybe once a week. I keep it because tearing out shipping panels has a cost too. But if I were designing DS24 from scratch *today*, knowing what I know after Article 6, the Physics Model tab would be smaller and the multi-run comparison tab — which doesn't exist — would be the headline.

The two research reports in `docs/aires/` are still on disk. I do not regret commissioning them. I do regret reading them as if they were a build-list rather than a menu. *Both reports are best read as enumeration of the design space, not as priorities*; the priorities have to come from the project. If a future campaign — a Season 2, in `_OUTLINE.md` terms — runs into a failure mode whose visual signature might be the divergence of a learned vector field, or the dimension of an attractor in latent space, or the rate of energy drift under the dynamics, then the unimplemented two-thirds of the reports is exactly the menu I want to be holding. Until then, the menu sits on disk and the dashboard runs on `localhost:8051`.

There is also the question of what the next visualization-shaped diagnostic in this campaign would even *be*, and I think the honest answer is: probably a cross-run comparison panel. Something that takes three completed runs — variant, baseline-paired-1, baseline-paired-2 — and renders their per-period Δα, per-regime trade-rate, per-action visit distribution, and Bal.Acc-vs-Alpha scatter side by side. Not a latent visualization at all. A meta-experiment visualization. The next picture I want to draw is not of the latent space; it is of the *comparison between latent-space-bearing experiments*, because that is where the campaign has actually lived for the past month and where the next set of decisions is going to be made.

The Apothecary cut a window in the side of the pot and drew the visible roots faithfully. The drawing was useful. The drawing was also, in the most consequential way, *incomplete*: most of the root system was on the other side of the pot, in the dark, where the window did not reach. The discipline is to know that, and to keep drawing anyway, and to know which questions the drawing cannot answer so that you go looking for those answers somewhere else.

> *The drawing on the page is faithful and incomplete. The plant grows where the drawing does not see. I record this without complaint, and I sharpen the pencil.*

---

## Sources

- `docs/aires/Geometric Deep Learning and Latent Dynamics: A Comprehensive Framework for Visualization in JAX.html` — full deep-research synthesis, ~50pp + ~60 figures, on differential-geometry-flavored latent visualization in JAX. Quoted §1 (introduction).
- `docs/aires/Visualization Architectures for JAX-Based MuZero Agents.html` — engineering-side companion report, ~40pp + ~50 figures, MuZero-specific diagnostic panels for JAX implementations. Quoted §1.1 (transformer-as-discretized-flow).
- `tools/DS24/` — the shipping training dashboard. `README.md`, `VISUALIZATION_RESEARCH_AND_PLAN.md`, `index.html`, `dashboard.js`, `metrics_logger.py`, `training_hooks.py`, `server.py`. Eight tabs as enumerated. Localhost:8051.
- `tools/DSe/` — the e-dojo companion dashboard (Foxglove-specific), `localhost:8052`. Out of scope for this article but lives in the same family.
- Article 6 of this series (`06_classifier_alpha_anticorrelation.md`) — the *better-classifier-worse-trader* anti-correlation finding referenced throughout. The bug visualization could not catch.
- `lore/notes_2026-05-15.md` §110 — the canonical pointer for the anti-correlation pattern; the moment the cross-run table did what no single dashboard panel had done.
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the pivot away from encoder-side mechanisms that reshaped what DS24's tabs would optimally diagnose.
