> *Belladonna: the eye-medicine. A pupil dilated past its native aperture sees more of the room, and more of the dark. The dose that grants this vision is two drops; three drops blinds. I weigh the powder on a scale that reads to the milligram, because the difference between sight and blindness is the difference between a measurement and an estimate.*

---

**by Lukas — May 22, 2026**

# Bella: A Diffusion Model for Markets That Don't Exist

*On the architecture of a market generator that learned to dream plausible price histories, the math that made the dreaming possible, and what happened when the dreams were given to a trader who had to bet real numbers on them.*

---

## The question Bella was supposed to answer

It is the afternoon of May 22, 2026. `workstation` is on. The desk fan is on. In the right pane `nvidia-smi` is showing one EVGA RTX 3090 at idle — 8 W, fans off, memory empty — because the last training job finished about ninety minutes ago and the next one is still being drafted. In the left pane `vim` is open on `experiments/762/SPEC.md`. The SPEC is for a reward-shaping sweep, the first of the four directions in the TASK_761 structural-pivot roadmap. None of the four directions is Bella-based, and that is the reason I am writing this article today rather than yesterday: Bella as a *training-data source* has been formally retired, the mechanism class is closed, and the postmortem can finally be written without it sounding like a verdict in progress.

The question Bella was supposed to answer is a small one with a long shadow. **Where does the training data come from when you have only a few years of real market history?** The agent — Poppy, the MuZero+TRM trader in `mcts_trader/` — needs to see *regimes*. Bull, bear, sideways, volatile, calm; thin liquidity, fat liquidity; correlated crashes, uncorrelated grinds; the rally that takes thirty minutes and the rally that takes thirty days. The real corpus, on the minute-bar scale that the agent acts on, has maybe four to six years of usable history per asset. That covers a small finite number of each regime. A trader that only ever sees those exact episodes overfits to those exact episodes. The standard remedy in RL is a synthetic environment that can produce arbitrarily many *plausible* episodes from *plausible* distributions of regime, while preserving the structural rules of the real one — fat tails, vol clustering, cross-asset correlation, the cumulative-return arithmetic that makes a price series a price series.

That synthetic-environment generator is Bella. Belladonna in the apothecary frame. The vision of unseen markets, titrated carefully enough that the agent's pupil dilates without going blind. Article 9 in this series covered the *operational disaster* of Bella's precompute — the SPEC that read "30–60 minutes" and the workload that read 93 GPU-days. This article is the one I should have written first. **What Bella actually is, mathematically and architecturally.** And then, at the end, what happened when she was finally deployed.

I want to land the structural lesson at the top of this piece, because the article is going to spend several thousand words on math and design and then arrive at a result you can see coming: **a good generative model of X is not the same as a useful teacher for an agent acting on X.** The two qualities — generative fidelity and downstream utility — are, on this stack, orthogonal. The math worked. The samples were beautiful. The samples did not help. I'll be specific about what *help* and *not help* mean numerically, and I'll be specific about why the orthogonality is the lesson rather than the *severity* of the result. But the shape of the kickback is settled before I start, and the math below is presented with that knowledge.

## What a diffusion model is, in five paragraphs

A diffusion model is a probabilistic generator that learns to invert a process of gradual corruption. You take real data, you spend many small steps adding Gaussian noise to it until it becomes pure noise, and then you train a network to undo that process one small step at a time. At inference you start from pure noise and run the network's denoising chain forward — pure noise becomes plausible data. The training objective is, in its most popular form, *denoising score matching*: predict the noise that was added at step t, conditioned on the noisy input at step t and on t itself.

Formally, the *forward* process is a Markov chain that takes a clean data sample $x_0$ and produces a sequence $x_1, x_2, \dots, x_T$ where each step is a tiny Gaussian perturbation. With a noise schedule $\{\beta_t\}_{t=1}^{T}$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\big).
$$

A useful identity is that the marginal at step $t$ is closed-form: $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$, where $\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$ and $\epsilon \sim \mathcal{N}(0, I)$. You don't have to simulate the chain step by step at training time — you sample $t$, sample $\epsilon$, build $x_t$ in one line, and ask the model to recover $\epsilon$ from $x_t$ and $t$.

The *reverse* process is what the network learns. It approximates $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta)$ with the mean parameterised through an $\epsilon$-predictor: $\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right)$. The training loss collapses to a single MSE:

$$
\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{x_0,\, t,\, \epsilon}\Big[ \big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2 \Big].
$$

That is the canonical DDPM objective. It is equivalent (up to a per-step weighting that you can either keep or strip) to the *denoising score-matching* objective from Hyvärinen and Vincent — the score is $\nabla_{x_t} \log p_t(x_t) = -\epsilon / \sqrt{1-\bar\alpha_t}$, so an $\epsilon$-predictor is a score predictor in disguise. At sampling time you start at $x_T \sim \mathcal{N}(0, I)$ and iterate $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$ with $z \sim \mathcal{N}(0, I)$, for $t = T, T-1, \dots, 1$. That gives you $x_0$, a sample from the data distribution.

A more modern and frankly better-behaved variant is **flow matching**, which Bella ended up using. Instead of learning a noise-prediction MSE under a stochastic forward process, flow matching learns a *velocity field* $v_\theta(x_t, t)$ along a deterministic straight-line path between noise and data: $x_t = (1-t)x_0 + t\,x_1$ for $t \in [0, 1]$, with $x_0 \sim \mathcal{N}(0, I)$ and $x_1$ a real datapoint. The target velocity is just $\partial_t x_t = x_1 - x_0$, and the loss is:

$$
\mathcal{L}_{\text{OT-CFM}}(\theta) = \mathbb{E}_{t,\, x_0,\, x_1}\Big[\big\|\,v_\theta((1-t)x_0 + t\,x_1,\, t) - (x_1 - x_0)\,\big\|^2 \Big].
$$

At inference, you integrate the learned vector field with an ODE solver from $t=0$ to $t=1$. The path is straight, the steps are bounded, and Euler with as few as 1–10 steps gives competitive samples; sampling is one to two orders of magnitude cheaper than DDPM. Bella's `conv_flow_matching.py` and `mamba/mamba_flow.py` both implement this objective, with `sigma_min = 1e-4` as the canonical regularization (the OT-CFM paper's recommendation). Optimal Transport CFM additionally pairs $x_0$ and $x_1$ such that the squared transport cost is minimized within the batch, which makes the learned paths even straighter and the few-step sampling even more accurate.

## What is different about market data

Image diffusion is the canonical success story for these models, and image diffusion is *forgiving* in ways market data is not. A photograph is stationary on the pixel grid — the same statistical assumptions hold for the top-left and the bottom-right pixel. Pixels have locality: neighbours in (x, y) are nearly always neighbours in value, which is why convolutional backbones work. The data manifold is high-dimensional but has approximately uniform texture. Each image is independent of every other image at training time.

Market data violates all of this.

- **Non-stationarity.** The distribution of returns on a 2017 BTC minute bar is not the distribution of returns on a 2022 BTC minute bar. Volatility is a function of time. Mean returns are a function of time. The *correlations* between assets are a function of time. The conditional distributions an honest model would have to learn are not fixed; they drift, sometimes slowly, sometimes catastrophically (a regime shift on Mar 12 2020 looks like a different probability law showing up in the same dataset).
- **Regime shifts.** A market spends weeks or months in one regime — rally, sideways, calm — and then transitions, often abruptly, to another. Within a regime the local statistics are stable; across the transition they aren't. A generator that doesn't *know* which regime it's drawing from will produce samples that interpolate across regimes, and interpolation across regimes is exactly what doesn't happen in nature.
- **Multi-asset correlation structure.** Bella generates 64 assets jointly. The correlations are not pixel-grid locality; they are graph-structured (BTC-correlated assets cluster, alt-correlated assets cluster, the cross-cluster correlation moves) and time-varying. There's no convolutional kernel that respects this graph; the model has to learn the graph implicitly.
- **No pixel-grid locality.** Time is one axis (causal), asset is another (permutation-equivariant up to a learnable ordering), feature is a third (OHLCV: only five channels, but they have hard arithmetic constraints — `low ≤ open, close ≤ high`, `volume ≥ 0`). A 2D convolution over (time, asset) has no semantic warrant. A 1D causal convolution over time, applied per-asset with cross-asset mixing through other mechanisms, is closer to right.
- **Heavy tails and clustering.** Returns are leptokurtic; volatility clusters (GARCH); the data has stylized facts that any honest generator must reproduce or it produces samples a trader can identify as fake.

These five constraints jointly forced four design decisions that I'll walk through in the next section. The short summary is: **latent-space diffusion** (not pixel-space) for compute reasons; **conv vs Mamba backbones** for sequence modeling; **skeleton conditioning** for controllable regime structure; and **regime-conditional sampling** at inference. These are the four pieces that distinguish a Bella from a generic time-series DDPM.

## Bella's four design choices

**Latent-space, not pixel-space.** The training window is `(window_size=400, n_assets=64, 5)` OHLCV — 400 minutes by 64 assets by 5 features, 128,000 floats per sample. Running a diffusion process over that raw space is expensive: the network has to denoise at full resolution, and the noise schedule has to be tuned to a space whose effective dimensionality is far smaller than 128,000 (the data lives on a much lower-dimensional manifold). The latent-diffusion remedy, popularised by Stable Diffusion, is to first train a **VAE encoder** that compresses each window to a small latent (Bella uses `latent_dim=256` in fp16), and then run the flow-matching / diffusion process *in the latent space*. The generative model only has to learn the structure of the latent manifold; the VAE's decoder reconstructs OHLCV when you need it. The cost is one extra training stage (the VAE has to converge first); the benefit is roughly two orders of magnitude in compute and memory at the diffusion stage. Bella's two-phase trainer (`diff_teacher/teacher.py`'s `DiffusionTeacher`) implements exactly this: phase one trains `MarketEncoder` (VAE) on OHLCV reconstruction with a KL prior; phase two freezes the encoder and trains the flow-matching network on the latent codes.

**Conv vs Mamba backbones.** The architecture decision that ate the most design time. Both variants are real and shipped — see `diff_teacher/conv_flow_matching.py` for the conv path, `diff_teacher/mamba/mamba_flow.py` for the Mamba path. Quoting `diff_teacher/CLAUDE.md` verbatim:

> *Two model families:*
> *Conv (default): 302M params, 1536d, 16L. Dilated causal WaveNet-style. O(n) memory → fits 1440 in 6GB VRAM*
> *Transformer: 22–100M, 1024d, 12L. O(n²) attention. Rarely used (expensive)*

I want to be honest about this paragraph before I write the rest of it, because the order in which I list the three variants — conv default, Mamba hybrid, transformer "rarely used (expensive)" — is the *outcome* of the decision, not the order in which the decision happened. The original aspiration was transformers. Transformers are what the field has been training for sequence modeling since 2017; they have the attention pattern you actually want for cross-asset structure; the diffusion-transformer (DiT) family had a couple of strong 2023–2024 papers and was the obvious starting point. I wanted a transformer.

The 24 GB VRAM ceiling of a single 3090 said no. Attention is O(n²) in sequence length. At a 1440-step generation window, even a modest 8-head transformer with `d_head=64` materialises a `1440 × 1440 × 8 × 64` attention-score tensor per layer — roughly 2 GB in fp16 per layer just for the QK matrix, before activations, before the value-side projection, before backward-pass storage, before optimizer state. Stack sixteen layers and the attention-score tensors alone are 33 GB — already over budget on a card that has to *also* hold the model parameters, the gradient accumulators, the Adam moments, the activation checkpoints, and ideally a non-trivial batch. The single-3090 constraint and the O(n²) attention cost compose multiplicatively into "no transformer at the sequence lengths we actually need." We tried. The 22–100M-parameter transformer variant in `diff_teacher/transformer_flow_matching.py` exists because someone ran the experiment, watched it OOM at every interesting batch size, and shelved the file with the *"Rarely used (expensive)"* note. The note is being generous. *Couldn't afford* is more accurate.

The conv variant won the default slot, then, not because conv is theoretically right but because conv is *what fit*. 6 GB VRAM for a 1440-step generation window meant Bella inference could share the single 3090 with the agent's training process without OOMing the card. The dilated causal stack — WaveNet's original receptive-field trick, exponentially-growing dilation per layer — gives the conv model a 1440-step receptive field with O(n) memory, which is the only memory complexity the budget allows at this sequence length. The Mamba/Jamba hybrid (`HybridMambaFlow` in `mamba/mamba_flow.py`) is the more theoretically interesting variant *and the one that came closest to being elegant rather than forced*: bidirectional Mamba blocks with a sparse attention layer every 8 blocks (1:7 attn:Mamba ratio, Jamba pattern). The selective state-space mechanism — input-dependent A, B, C matrices instead of fixed ones — is the right inductive bias for non-stationary sequence data, because the model can choose moment by moment whether to attend to recent input (large $\delta_t$) or trust accumulated state (small $\delta_t$). Functionally this is *an adaptive Kalman filter*, which is what you actually want for price series. Mamba's O(n) recurrence makes it fit where transformers don't; the 1:7 sparse-attention insertion lets the model still do *some* full-sequence reasoning at the places where Mamba's state-space compression can't carry the load.

The honest framing of the whole architecture, then, is: **the design ceiling was 24 GB, the design floor was a 1440-step generation window, and every architectural choice in this article exists in the gap between those two numbers.** Latent-space diffusion (not pixel-space) for compute. Conv default for O(n) memory. Mamba for the same reason with a better inductive bias. The transformer never made it past the OOM. None of this is what the literature would have called the principled choice; all of it is what the budget allowed. The Mamba variant exists and trains; in practice it has been used for ablations and for downstream experiments where the longer effective context mattered. The conv variant is the workhorse. Both are downstream of the same constraint.

**Skeleton conditioning.** The single most important design choice for Bella as a *teacher*. A raw diffusion model samples from the joint distribution of all market regimes mixed together; Bella has to be able to sample *conditional on a chosen regime*. The mechanism is the `MarketSkeleton` (defined in `diff_teacher/skeleton/types.py`), which is a multi-modal control input that tells the generator what kind of market it's drawing. Its components are:

- `trend_direction: (n_timesteps, n_assets)` in $\{-1, 0, +1\}$ — per-asset trend per moment.
- `vol_regime: (n_timesteps,)` normalised to $[0, 1]$ — global volatility curve.
- `correlation_clusters: (n_assets,)` — static cluster IDs (which assets move together).
- `alpha_shape: (n_timesteps,)` — cumulative opportunity curve.
- `regime_sequence: (n_segments,)` — categorical labels from `{crash, rally, sideways, volatile, calm}` plus one unconditional pseudo-regime.
- `events: List[Event]` — sparse discrete shocks (timestamp, type, magnitude).

The conditioning enters the network through three orthogonal injection paths, chosen for the type of signal:

1. **FiLM** (feature-wise linear modulation) for global, slow signals — regime label and vol regime. $y = \gamma(z) x + \beta(z)$ where $z$ is the skeleton embedding. Cost O(1) per token.
2. **AdaLN** for the diffusion timestep $t$ and the regime conditioning combined — this is the DiT-style adaptive layer norm that stable-trains by initialising the gate to near zero. Quoting the gotcha from `diff_teacher/CLAUDE.md`: *"AdaLN gate init — `gate = sigmoid(x - 5.0)` so gates start near zero. Critical for stability (LOW-003)."*
3. **Cross-attention** for sparse events (earnings, FOMC) — the events are a short list of `(time, type, magnitude)` tokens; cross-attention lets the backbone attend to them at the relevant timesteps.

The `SkeletonConditioner` in `diff_teacher/skeleton/encoder.py` is the module that fuses these signals into the `(gamma, beta, h0)` tuple that flows through every Mamba/conv block. The skeleton is constructed from real data at training time (the `extract_skeleton` function in `skeleton/extraction.py` runs HMM regime detection on log-returns, rolling means/stds for trend and vol, and a correlation-matrix clustering step for the static clusters), so the model is being trained to predict velocity *conditioned on the skeleton it was extracted from*. At sampling time you can supply *any* skeleton — including ones that didn't occur historically — and the model will produce a trajectory that satisfies them.

**Regime-conditional sampling.** This is the inference-time payoff of skeleton conditioning. To generate a 1440-step rally, you build a skeleton with `regime_sequence = [rally]`, set `vol_regime` to a low-curvature ramp, set `trend_direction` to mostly +1 across the dominant correlation cluster, and sample. The flow-matching network integrates the velocity field conditioned on that skeleton from $t=0$ to $t=1$, and the VAE decoder turns the latent trajectory back into OHLCV. The Euler solver at 10 steps is the default (`generation/generator.py`'s `MarketGenerator.generate`); the Dopri5 adaptive solver is available for the cases where the conditioning is unusual enough that Euler's straight-line approximation falters. The whole pipeline — VAE encode (training only), flow-matching velocity field, ODE solve, VAE decode — is JAX-end-to-end, traced under `nn.remat()` for layerwise gradient checkpointing, and runs at roughly 150 single-window forward passes per second on the 3090 when Bella has the card to herself.

## What the bones let you ask for

I want to slow down here, because if the prior sections read like an architecture spec then the next paragraph reads like a list of party tricks, and the difference between the two is what makes Bella the slightly grim part of this garden.

A skeleton is six tensors. The HMM extractor in `skeleton/extraction.py` will read those six tensors off any historical window — give it a real 1440-step segment from August 2023 and it returns the trend direction per asset, the vol regime curve, the correlation clusters, the alpha shape, the regime label, and the event list that *actually happened*. That is the training-time use: skeleton extracted from real data, model trained to reproduce real data *conditioned on* its own skeleton, so that at inference the model has learned the conditional map from "this skeleton" to "trajectories that satisfy this skeleton."

The inference-time use is where the herbal name starts to earn itself.

You can keep the skeleton from one window and swap one tensor. Take the August 2023 skeleton, leave the vol regime and correlation clusters and events alone, and flip `regime_sequence` from `[sideways]` to `[crash]`. Sample. What comes back is a 1440-step OHLCV trajectory whose volatility curve, asset clusters, and event timing match August 2023 *exactly* — but whose price path is a crash that did not occur on that calendar. The stylized-facts evaluator in `evaluation/fidelity.py` passes it. The controllability evaluator confirms it lands in the requested regime under HMM re-detection. A trader looking at the trace in a blind test would not flag it as synthetic above chance — the article gets to that claim later when it talks about what Bella *did* succeed at; the claim is real here.

You can do this with any axis. Flip `trend_direction` per asset and Bella will draw you a market where the BTC cluster fell and the ETH cluster rallied during the same hour. Stretch `alpha_shape` and Bella will give you a 1440-step window where there was more cumulative opportunity than any real period ever offered, distributed across assets according to a real correlation matrix. Add a synthetic `Event(timestamp=720, type='shock', magnitude=−0.05)` and Bella will draw the cascade that would have followed a 5% mid-window shock — propagated through the cluster graph, decaying along the correlation structure she learned from real cascades. The events list is the most pointed of these because the events list is where you tell Bella to *break things on purpose*: insert a flash crash at minute 412, insert a liquidity vacuum at minute 900, insert a regime flip with no warning at minute 600.

The grim part is that Bella does not know any of this is hypothetical. The flow-matching network has learned a velocity field, and the velocity field is conditioned on the skeleton; if you give it a skeleton no real market produced, the field still integrates, and the output still passes fidelity. There is no "this is impossible" exception case in the generator. The architecture exists *to* extrapolate; the extrapolation is the design intent. The training corpus is a few years of minute bars; the inference space is *every plausible market that the skeleton vocabulary can describe*, which is unboundedly larger.

This is, narrowly, an enormously useful capability. Stress testing the trader against a 2008-magnitude drawdown that never happened on minute bars in our dataset is one paste of a skeleton away. Counterfactual histories — *what would my trader have done if the FTX collapse had come a quarter earlier* — are one event-list edit away. Adversarial generation against the agent — the AlphaZero-self-play idea from Article 1 that quietly died as a system but never died as an aspiration — is a closed loop with Bella as the adversary's substrate. None of these uses landed in this campaign (Bella as a *training-data source* didn't help the trader, per the closing kickback below), but the existence of the capability is independent of whether we used it well. The tool is loaded; the safety on the tool is the user.

The grim is also the herbal name reminding you it is doing its job. Belladonna dilates the pupil correctly *and* poisons the patient at a slightly higher dose, and the same plant does both. Bella generates plausible markets correctly *and* generates implausible-but-fidelity-passing markets correctly, and the same network does both. The skeleton conditioning is the dose control. The dose control is the part of this article I want to flag for whoever lifts this codebase later: keep the skeleton extractor and the skeleton injector separated, keep an audit log of which skeletons get sampled with, keep the dose written on the bottle. The mechanism is too good at its job to use without the chart.

## The architecture, by directory

The `diff_teacher/` tree is laid out to make the two-phase training and the conditioning machinery explicit. The canonical reference is `diff_teacher/CLAUDE.md`; I quote the layout verbatim here because the structure is load-bearing:

> *`skeleton/` — `types.py` (regime/event/skeleton defs), `encoder.py` (TrendEncoder, VolRegimeEncoder, ClusterEncoder, `SkeletonConditioner`), `extraction.py`, `builder.py`*
> *`mamba/` — `ssm.py` (selective scan sequential/parallel), `mamba_block.py` (per-step input/output projections), `bimamba.py` (bidirectional + SelfAttention), `mamba_flow.py` (`HybridMambaFlow`, 1:7 attn:Mamba ratio)*
> *`evaluation/` — `stylized_facts.py`, `fidelity.py`, `controllability.py`, `report.py`*
> *`generation/` — `generator.py` (`MarketGenerator.from_checkpoint`, `.generate`), `streaming.py`. No `VRAMGenerator`*

Three pieces of root-level code are doing the heavy lifting:

- **`teacher.py`** holds the `DiffusionTeacher` class, which is the two-phase trainer. Phase one runs the VAE on OHLCV windows with a reconstruction loss plus KL prior; phase two freezes the VAE and trains the flow-matching network. The two phases share the dataloader (`MarketDataLoader`, which is *not* thread-safe per `HIGH-015`; the simple dict cache breaks under concurrent access). The trainer state is a Flax `TrainState`, which has the lovely property of producing double-nested checkpoint dicts — `{params: {params: {...}}}` — that `MarketGenerator.from_checkpoint` unwraps automatically.
- **`encoder.py`** is the VAE encoder. **Distinct from** `skeleton/encoder.py`, which is the conditioning module. Both are used; both are called "encoder"; this naming overlap has caught me in code review more than once and I have not yet renamed it because the rename touches every other file in the subtree. The VAE encoder maps `(400, 64, 5) → latent_dim=256` per window. The skeleton encoder maps `MarketSkeleton → (gamma, beta, h0)` per layer per step.
- **`conv_flow_matching.py`** and **`mamba/mamba_flow.py`** are the two velocity-field networks. They take a noisy latent $x_t$, the timestep $t$, and the skeleton conditioning, and predict the velocity $v_\theta(x_t, t, \text{skel})$. The conv variant is the dilated causal WaveNet stack with FiLM/AdaLN at every layer; the Mamba variant is the Jamba-style hybrid with bidirectional Mamba blocks and sparse attention.

The training loop is straightforward by JAX standards. For each batch: sample a window from `MarketDataLoader`, encode it through the VAE to a latent $x_1$, extract its skeleton from the underlying prices, sample $t \in [0, 1]$ and $x_0 \sim \mathcal{N}(0, I)$, build the OT-CFM target $v^* = x_1 - x_0$ and the interpolated latent $x_t = (1-t)x_0 + t x_1$, forward through the velocity network conditioned on the skeleton, MSE-loss the prediction against $v^*$, backprop, Adam step. The RNG handling is a subtree-specific gotcha:

> *JAX RNG — use `jax.random.fold_in()` per step, NOT `split()`. Direct split breaks seedability (HIGH-009 fix, ubiquitous).*

That fix is the kind of thing you find by accident, fix in one place, and then have to fix everywhere because it had been silently breaking the reproducibility of every Bella experiment for several weeks. The other large memory-pitfall in the Mamba variant is the parallel associative scan, which materialises a `(batch, seq, d_inner, d_state)` 4D tensor — ~18 GB by itself — and so is gated behind `use_parallel=False` by default, with sequential scan as the safe path (`CRIT-004`).

The training entry point — quoted verbatim because it is the one-line invocation that has stayed stable across the entire Bella arc:

```bash
python -m diff_teacher.train --db-path trading_data.db --epochs 100
```

The Bella checkpoint we ended up using for all downstream experiments lives at `checkpoints/diff_teacher/final/generator`. It is the converged Phase-1 generator after about 100 epochs on the full minute-bar corpus, validation samples passing the stylized-facts evaluator in `diff_teacher/evaluation/fidelity.py` (heavy tails reproduced, volatility-clustering autocorrelation reproduced, cross-asset correlation matrix reproduced to within 5% Frobenius), and skeleton-controllability tests in `evaluation/controllability.py` passing (regime-conditional samples land in the requested regime label under HMM re-detection >90% of the time).

By the standards of the generative-modeling literature this is a good model. The samples look like markets. They have the right statistics. The conditioning works. If I had set out to build a market generator as a publication artifact — Bella as a research contribution in her own right — this would have been the end of the story and I'd be writing a paper, not an article in a series about negative results.

---

> *The Apothecary holds the dried belladonna leaf to the light. The dose is correct to the milligram. The chart of doses is correct to the row. The cultivar — Atropa belladonna, three years from seed — is unimpeachable. The Apothecary administers two drops to the eye of the specimen, watches the pupil dilate to the requested aperture, and records the observation. The next day the specimen is unable to read text at arm's length. The eye-medicine produced exactly the dilation that was requested. It also produced a patient who cannot work.*

---

## The kickback, in three failures

Causality kicks back at the end of every chapter in this series, and this one's kickback is the cleanest I have written so far. **The architecture works. The math is correct. The generated markets are visually plausible. They do not help the downstream trader.** Three independent attempts to wire Bella into Poppy's training pipeline, three independent failures, three independent mechanisms. Let me put them on one table.

| Task | Mechanism | Paired Δα (3-seed mean) | Verdict |
|---|---|---|---|
| **TASK_740** | Per-cell prediction distillation — auxiliary head on the agent's encoder regresses Bella's per-asset per-horizon predictions | **−1.92pp** | Soft regression, 0/3 positive seeds |
| **TASK_757 (754)** | Trajectory-shape distillation — auxiliary head regresses the first 4 moments of Bella's predicted return distribution per window (coarser, denser gradient) | **−5.80pp** | Clearer regression |
| **TASK_758** | Bella as data augmentation — sample synthetic regimes from Bella and mix them into the training corpus | **−16.54pp** (one seed at **−37.93pp**) | Catastrophic |

Zero for three across the three distinct deployment routes I tried. Each route was designed to address the failure mode of the previous one. TASK_740's per-cell target was hypothesised to be noise-dominated (each individual cell of Bella's prediction is a noisy estimate of a small log-return; variance dominates mean), and so TASK_757/754 coarsened the target to four moments of the predicted distribution per window — a permutation-invariant, denser-gradient signal designed to extract the regime-level information Bella *can* express reliably while discarding the per-cell noise she cannot. The hypothesis was that the coarser signal would move the mechanism out of the classifier-Alpha anti-correlation class (Article 6) and produce Δα ≥ 0 at the 3-seed gate. The result: Δα went from −1.92pp to **−5.80pp**. The coarser target made things worse, not better. TASK_758 then bypassed distillation entirely and used Bella as an *augmentation source* — generate fresh regimes, mix them into the training corpus, let the agent train on the augmented data directly. That collapsed to −16.54pp average, with one seed at −37.93pp, the worst single-seed result of the entire May campaign.

A second, separate Bella failure mode surfaced in the middle of this — the **column-mismatch bug** documented in `memory/project_bella_column_mismatch.md`. Bella's outputs were indexed in database-ID order; the trainer's distillation loss indexed them in rotation-specific asset order; there was no remapping. This meant the per-asset MSE gradient was, for several weeks, gradient-on-noise. TASK_745 was the spot-check task that confirmed the mismatch and the Option A per-rotation remap that fixed it. After the fix, TASK_740 was re-evaluated under corrected semantics. The Δα was still **−1.92pp**. The mechanism wasn't bugged. It was just bad.

The verdict in the lore is unambiguous:

> *Entire Bella mechanism class CLOSED 2026-05-19. 740 encoder (−1.92pp) + 757 shape-distill (−5.80pp) + 758 data augment (−16.54pp, s13 catastrophe −37.93pp) = 0-for-3 across all surfaces. Pivot to planning-side via TASK_759.*
> — `memory/project_encoder_bella_class_closed.md`

I want to be precise about what failed and what didn't.

**What didn't fail.** The generator is good. The fidelity evaluator passes. The controllability evaluator passes. A trader looking at one of Bella's regime-conditional 1440-step OHLCV traces would not, in a blind test, identify it as synthetic above chance. The math — VAE, OT-CFM, BiMamba, FiLM/AdaLN/cross-attn — is implemented correctly; the gotchas (`fold_in` not `split`, sequential scan not parallel, AdaLN-zero init) are caught; the training loss converges; the checkpoint loads. If you measure Bella by the standards of a generative model — fidelity, controllability, training stability, inference speed — she earns her grade.

**What did fail.** Every attempt to use Bella's outputs as a training signal for the downstream trader. Per-cell. Per-window. As data augmentation. Three independent surfaces, three independent regressions, in some cases catastrophic. The structural lesson — the one I'm going to land as the closing claim of this article — is the orthogonality of generative quality and downstream utility for an agent acting on the generated data. **A good generative model of X is not the same thing as a useful teacher for an agent acting on X.** The two are correlated in domains where the model's marginal distribution is what the agent needs to know (image classifiers training on synthetic images, language models training on synthetic text where the *content* is the signal). They are not correlated, on this stack, in a domain where the agent's job is to *exploit small predictable structure in a near-random sequence* — because the synthetic generator's failure modes (the residual entropy in its samples, the smoothing of regime transitions, the slight misalignment of tail behaviour) are exactly the structure the trader is trying to extract.

Said differently: Bella's samples are *plausible markets in their large-scale features*. They are *not plausible markets in the fine-grained predictability structure that an agent with millisecond-grade compute can hope to exploit*. A trader training on Bella's samples is training to exploit Bella's residual noise patterns, not the market's. Deployed on the real market, the agent has trained the wrong reflexes.

I do not think this is the only possible result for a market generator. I think it is the result *we got from the particular generator we built, for the particular downstream task we cared about, on the particular stack we used*. A larger Bella with a richer conditioning vocabulary, deployed against a different agent — a policy that doesn't try to micro-time and instead trades on slower regime signals — might earn its compute. But the question of *whether to invest in that larger Bella* has to be answered after the existence proof has produced its evidence, and the evidence we have is that on the agent we have, on the stack we have, three independent deployment surfaces produced three independent negative results, and one of them was catastrophic. The mechanism class is closed.

## What's left

Bella is still in the repo. `diff_teacher/` is still committed; `MarketGenerator.from_checkpoint` still loads; `python -m diff_teacher.train --db-path trading_data.db --epochs 100` is still the training command. The architecture survived. The math survived. The checkpoint at `checkpoints/diff_teacher/final/generator` is still on disk on this workstation, occupying 1.7 GB of `storage disk` because the symlinked git objects directory needed the space and the checkpoint qualified as data, not source. Bella's role demoted: from *training-data source for the downstream agent* to *diagnostic tool for visualization*. When I need a regime exemplar for a SPEC document — "this is what the kind of market I'm testing against looks like" — I sample one from Bella, plot it, and paste the plot. That role doesn't require Bella to *help the trader*. It only requires her to *produce plausible-looking samples on demand*, which she does, and which she will continue to do.

The four directions in the TASK_761 structural-pivot roadmap — reward shaping, policy entropy regularization, oracle k reformulation, incompatible-dataset retrain — are listed in `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` and none of them is Bella-based. The pivot away from generator-as-teacher is not phrased as a defeat of generative methods; it is phrased as a structural reorientation away from *single-mechanism encoder-side improvements* toward *planning-side and reward-side reformulations*. Bella was a single mechanism in the encoder-side class. So were 738, 739, 741, and 749. The class closed for the *class's* reason, not specifically for Bella's. The Bella result was confirming evidence for a structural diagnosis that the next campaign is built around.

The architecture itself — skeleton-conditioned flow matching on a Mamba/conv backbone with VAE-latent diffusion — is the thing I'd preserve if someone asked me which piece of `diff_teacher/` to lift into a different project. The skeleton design is the bit I'm proudest of, because the multi-modal injection scheme (FiLM for global slow signals, AdaLN for timestep+regime, cross-attention for sparse events) is a clean instance of *matching the conditioning mechanism to the signal's structure*. The OT-CFM choice over DDPM was the right one for compute budget; the conv default over Mamba was the right one for the 6 GB VRAM constraint; the two-phase VAE-then-diffusion training was the right one for the latent-space efficiency. None of those design choices is the reason Bella didn't help the trader. They are all individually correct, and they are all individually preserved.

The deeper thing I take from the Bella arc is a slow update on what *generative models can and can't be used for in RL settings*. The literature on world models (PlaNet, Dreamer, MuZero's own learned dynamics, Sora-for-robotics) is full of confident claims that a generative model of the environment can serve as a planning surrogate. The claim is *sometimes* true. It was not true here. The conditions under which it is true — large stationary environments with smooth physics, an agent whose decisions are at a coarser temporal grain than the generator's failure modes, downstream tasks where the agent's exploitable structure is *contained within* the generator's reliable structure — are conditions that high-frequency market trading happens to violate on all three axes. That's not a defeat of world models; it's a constraint on where they earn their keep.

I am writing this paragraph at 16:42 on May 22, 2026. `workstation` is on. The desk fan is on. The 3090 has spun up — the experiments/762 SPEC is finalised, the scale arithmetic is on the page, the run is launched, and the card is drawing 312 W as the first epoch warms. Bella's checkpoint is not loaded. None of the four pivot directions will load her. That's the right call. That doesn't mean she was the wrong project.

---

> *Bella's vision was beautiful. The chart of doses, in the end, was not the chart of cures.*

---

## Sources

- `~/Trader/diff_teacher/CLAUDE.md` — canonical architecture summary for the Bella subtree. Quoted directly for the conv/transformer model-family description, the subdirectory layout, the JAX-RNG gotcha, the AdaLN gate init, and the parallel-scan memory hazard.
- `~/Trader/docs/UNIFIED_MARKET_GENERATION_ARCHITECTURE.md` — the original research synthesis. Source for the Skeleton-Mamba-Flow architecture diagram, the FiLM/AdaLN/Cross-Attention conditioning table, the OT-CFM loss formulation, and the Mamba selective-scan financial interpretation.
- `~/Trader/docs/market_diffusion_architecture.md` — companion research document for the diffusion-side architecture choices.
- `~/Trader/docs/Market Generation Research Plan.md` — the longer-form research plan that motivated TASK_108.
- `~/Trader/tasks/TASK_108_LATENT_DIFFUSION_MARKET_GENERATOR.md` — the original SPEC that committed Bella to a single-3090 deployment budget. The "≤6 GB inference VRAM" constraint shaped the conv-default decision.
- `~/Trader/tasks/TASK_111_CONV_DIFFUSION_TRAINING.md`, `TASK_112_SKELETON_CONDITIONED_GENERATION.md`, `TASK_113_MAMBA_MARKET_GENERATION.md`, `TASK_114_MARKET_FLOW_PUBLICATION.md` — the SPECs for the four sub-arcs of Bella's construction: conv training, skeleton conditioning, Mamba variant, and the publication-track architecture writeup.
- `~/Trader/tasks/TASK_109_DIFF_TEACHER_VALIDATION.md` — the validation SPEC for the fidelity and controllability evaluators in `diff_teacher/evaluation/`.
- `~/Trader/tasks/TASK_745_BELLA_SEMANTICS.md` — the column-mismatch spot-check task; Option A per-rotation remap.
- `~/Trader/tasks/TASK_753_BELLA_AS_DATA_AUGMENTATION.md` — the augmentation-route SPEC that became TASK_758.
- `~/Trader/tasks/TASK_754_BELLA_TRAJECTORY_SHAPE_DISTILL.md` — the coarser-target SPEC that became TASK_757. Quoted indirectly for the moments-of-distribution descriptor and the hypothesis that the coarse target would escape the classifier-Alpha anti-correlation class.
- `~/Trader/memory/project_bella_column_mismatch.md` — the DB-ID-vs-rotation-order asset-indexing bug discovered May 14, resolved May 17.
- `~/Trader/memory/project_encoder_bella_class_closed.md` — the May 19 closure note. Quoted directly for the verdict line.
- Article 9 in this series — `bloom/drafts/09_bella_precompute_93_gpu_days.md` — the operational disaster postmortem. Pairs with this article: 9 covered the precompute failure, 16 covers what Bella *is*.
- `~/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the four-direction structural pivot, none of which is Bella-based.
