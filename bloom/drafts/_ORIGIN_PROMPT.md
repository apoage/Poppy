# The Original Prompt — Prehistory of Trader

This is the original project conception, **predating the Dec 2025 `ALPHAZERO_TRADING_VISION.md` by years.** The user shared this verbatim on 2026-05-21 in the bloom-writer session, noting *"its few years old idea."*

It is materially different from the eventual Poppy/MuZero+TRM stack:
- **State representation:** color-encoded image (HSV: Hue=price position, Saturation=S/R strength, Value=volume) — not the 27-channel time-series tensor that exists today.
- **Substrate:** CNN with policy + value heads — not TRM (recursive transformer) + MuZero dynamics function.
- **Hardware target:** 4× RTX 3090 + 512 GB RAM + AMD EPYC (32–64 cores) — workstation now is 1× 3090; the server matches.
- **Framing:** "Mixture of Experts" with MCTS as gating mechanism — the MoE language disappeared by Dec 2025.
- **Spirit:** "feed data as raw as possible, letting the model learn interpretations" — survived. So did the tabula-rasa self-play and the "historical data as accelerated playground" framing.

**Why this matters for the series:** Article 1 currently treats Dec 2025 as origin. It isn't. Dec 2025 is the moment the idea got upgraded to the AlphaZero/MuZero analogy and the modular-AGI framing. The *real* origin is below — a CNN-on-price-images plan, hardware-overspecced relative to what was eventually available, full of design choices (HSV encoding, MoE gating) that quietly died on the way to Poppy.

---

## Original prompt (verbatim, user-supplied 2026-05-21)

> Hi
>
> **Project Description: Monte Carlo Search Tree Trader with Color-Encoded Market States**
>
> This project aims to develop an autonomous trading agent that learns optimal trading strategies through self-play, inspired by AlphaZero. The core idea is to represent financial market data (e.g., stock or cryptocurrency prices, volume, and derived indicators) as color-encoded images. These images serve as the "state" input for a deep learning model, likely a Convolutional Neural Network (CNN), which learns to predict future market states or evaluate current market positions.
>
> A Monte Carlo Tree Search (MCTS) algorithm will be the "brain" of the trading agent. The MCTS will use the output of the CNN (as a policy and/or value function) to explore potential trading actions (buy, sell, hold, and quantities) and simulate their outcomes. By balancing exploration of new strategies with exploitation of known profitable ones, the agent will iteratively refine its decision-making process.
>
> **Key components and concepts discussed include:**
>
> * State Representation:
>   * Transforming raw market data (price, volume, potentially order book depth, technical indicators like RSI, MACD, volatility, support/resistance strength) into multi-channel images.
>   * Utilizing color spaces like HSV for encoding, where:
>     * **Hue:** Could represent relative price position within a recent range (cyclical, perhaps with modulo for round numbers).
>     * **Saturation:** Could represent the strength of identified support and resistance levels.
>     * **Value/Lightness:** Could represent normalized trading volume.
>   * Emphasis on proper data normalization (e.g., min-max scaling, Z-score) before color encoding to ensure consistency and prevent features with larger magnitudes from dominating. The goal is to feed data as raw as possible, letting the model learn interpretations.
> * Prediction/Evaluation Model:
>   * A CNN to process the image-based market states.
>   * The CNN could predict future states/images and timeframes, or act as a value function to estimate the desirability of a state, or a policy network to suggest promising actions.
> * Decision-Making Agent:
>   * MCTS algorithm to guide the trading decisions.
>   * The MCTS explores the action space, using the CNN for simulations and evaluations.
> * Learning Mechanism:
>   * "Tabula rasa" learning through self-play in a simulated market environment.
>   * Using historical data as an accelerated "playground" for training, where timestamps serve as points of reference, allowing the agent to "crunch" through data much faster than real-time.
> * Architecture:
>   * Can be viewed as a type of **"Mixture of Experts" (MoE)** system, where different analytical components (e.g., a model for price patterns, a model for volume analysis, a potential LLM for sentiment analysis if incorporated) feed into the MCTS (acting as an executive/gating mechanism).
>   * Focus on feeding raw, minimally processed (but normalized) data to avoid "hallucinated false positives" from overly interpreted inputs.

### Hardware Constraints (as stated in original prompt)

> * GPUs: 4 x NVIDIA RTX 3090 (24GB VRAM each)
> * RAM: 512 GB
> * CPU: AMD EPYC system (please assume a high core count, e.g., 32 or 64 cores)
> * Storage: Sufficient fast NVMe SSD storage.

### Detailed sub-requirements (paraphrased here for brevity; verbatim in user's chat)

1. **Market Data Acquisition & Preprocessing** — OHLCV ingestion, rolling normalization (min-max or Z-score), missing-data via special encoding rather than interpolation.
2. **State Representation — Color-Encoded Images** — HSV encoding above; multi-channel image tensor (height, width, channels); configurable dimensions (lookback × assets or lookback × features).
3. **CNN Model Architecture** — dual-head (policy + value); designed for RTX 3090; data-parallel preferred over model-parallel; dropout + BN regularization.
4. **MCTS Agent** — CNN policy guides selection/expansion; CNN value evaluates leaf nodes; discrete action space (buy X%, sell Y%, hold); discussion of continuous quantities later.
5. **Simulated Market Environment & Training Loop** — historical data as accelerated playground; self-play; experience storage (state, MCTS action probabilities, reward); periodic CNN retrain.
6. **Performance Optimization & Scalability** — parallel MCTS simulations; PyTorch DDP or TF MirroredStrategy; memory management for large datasets.
7. **Language and Frameworks** — Python; TF or PyTorch; NumPy/Pandas; consider existing MCTS libraries (mctx, etc.) vs custom.

### Initial focus & simplifications

> * Start with a single asset.
> * Simple reward function (e.g., portfolio value change).
> * Basic transaction cost model.

### Final ask in the original prompt

> "Please provide a structured plan, code examples for key components (especially the state encoding, CNN structure, and MCTS integration), and advice on tackling this complex project incrementally. Discuss potential challenges and how to mitigate them, particularly regarding overfitting, look-ahead bias, and computational demands."

---

## The LLM's follow-up plan (verbatim, user-supplied)

> Now let me create a practical implementation guide and best practices:
>
> I've created a comprehensive implementation of the Monte Carlo Tree Search trading agent you described. Here's a summary of what I've provided:
>
> ## Key Components Created:
>
> 1. **Main Architecture (`mcts_trader_architecture`)**: Complete implementation of market data processor with HSV color encoding; CNN model with dual heads for policy and value prediction; MCTS algorithm implementation; Trading environment simulation
> 2. **Training System (`mcts_trader_training`)**: Experience replay buffer; Self-play training loop; Distributed training across multiple GPUs; Checkpoint saving/loading
> 3. **Visualization Tools (`mcts_trader_visualization`)**: State encoding visualization; Training metrics plotting; MCTS tree visualization; Backtest results analysis
> 4. **Implementation Guide (`mcts_trader_implementation_guide`)**: Step-by-step setup; Best practices for avoiding common pitfalls; Advanced extensions (multi-asset, sentiment integration); Production deployment considerations
>
> ## Key Features Implemented:
>
> 1. **State Representation:** HSV color encoding where Hue = relative price position (cyclical), Saturation = support/resistance strength, Value = normalized volume. Rolling normalization to avoid look-ahead bias.
> 2. **Neural Network Architecture:** Shared convolutional backbone; Separate policy and value heads; Optimized for RTX 3090 GPUs.
> 3. **MCTS Implementation:** UCB-based node selection; Neural network guided expansion; Efficient backpropagation.
> 4. **Distributed Training:** PyTorch DistributedDataParallel support; Multi-GPU optimization; Gradient synchronization.
> 5. **Performance Analysis:** Sharpe ratio calculation; Maximum drawdown tracking; Win rate analysis.
>
> ## Next Steps:
>
> 1. **Data Preparation:** Start by loading your historical market data and ensuring it has the required OHLCV format.
> 2. **Initial Testing:** Begin with single GPU training on a small dataset to verify the implementation works correctly.
> 3. **Hyperparameter Tuning:** Experiment with different lookback windows, MCTS simulation counts, and learning rates.
> 4. **Scale Up:** Once basic functionality is verified, scale to multiple GPUs and larger datasets.
> 5. **Advanced Features:** Gradually add features like multi-timeframe analysis, ensemble methods, or sentiment integration.
>
> The implementation is designed to be modular and extensible, allowing you to experiment with different components while maintaining a clear path to production deployment. Would you like me to elaborate on any specific part of the implementation or help you get started with a particular component?

---

## What survived to today, and what died on the way

**Survived:**
- MCTS as the planner — kept in `mcts_trader/` to this day, name lives in the project name.
- Self-play / tabula rasa framing — survived into Foxglove SegmentPool and the planned (never built) AdversarialMarket.
- Historical data as "accelerated playground" — central to Foxglove and Bella.
- Policy + value dual head — still the architecture, just on a different encoder.
- "Feed data raw, let the model learn interpretations" — still the ethos.

**Died on the way:**
- **HSV color-encoded image input.** Replaced by 27-channel time-series tensors (cross-sectional + per-asset, see `lore/adr/0003-encoder-v3-32ch-cross-sectional.md`). The image idea looks, in retrospect, like a CNN solution to a problem CNNs aren't actually best at.
- **CNN backbone.** Replaced by TRM (recursive transformer) for the encoder.
- **Mixture of Experts framing.** Quietly dropped by Dec 2025 — the modular-AGI framing of `ALPHAZERO_TRADING_VISION.md` partly inherits it but at a coarser granularity (whole-domain modules rather than within-trader expert heads).
- **4× RTX 3090 hardware.** Workstation reality is 1× 3090; the server has 4× but is shared/queued.
- **PyTorch.** Stack moved to JAX (`train_jax_fast.py`).
- **The LLM-for-sentiment hook** mentioned in the original prompt's MoE section. Never built.

**Reframed:**
- "Hallucinated false positives" in the original prompt is, in retrospect, the same concern that drives `NEURAL_MARKET_PHYSICS_DESIGN.md` (rollout hallucination) and the eventual May 2026 finding that the bottleneck is downstream of the encoder (over-confident policy hallucinating trade signals).

---

## Implications for the series

- **Article 1** as currently written treats Dec 2025 as the origin. It should be amended — either:
  - (a) **Lightly:** add one paragraph in the opening or the "what survived" section noting that Dec 2025 wasn't genesis, it was a recapitulation, and naming the older idea.
  - (b) **Heavily:** restructure to start with the original CNN-on-HSV-images plan, narrate the *transition* to the AlphaZero framing as the first kickback, and treat the December doc as the second draft of a much older idea.
- **A new "Article 0" / prequel** is possible: "Color-Encoded Markets: The Idea Before the Idea." Would cover the HSV encoding rationale, why CNN-on-image was the wrong substrate for non-stationary time-series, and the long road from that prompt to Poppy.
