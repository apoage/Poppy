# **Advanced Generative Architectures for Financial Time Series: Skeleton-Conditioned ControlNet, Mamba SSMs, and Flow Matching**

## **1\. Introduction: The Paradigmatic Shift in Market Simulation**

The simulation of financial time series has historically been a domain dominated by stochastic differential equations (SDEs) and agent-based models (ABMs). While mathematically rigorous, these approaches often fail to capture the complex, high-dimensional stylized facts of modern markets—such as volatility clustering, leverage effects, and heavy-tailed return distributions—without extensive manual calibration. The advent of Deep Generative Models (DGMs), specifically Generative Adversarial Networks (GANs) like TimeGAN, marked the first generation of data-driven simulators. However, GANs suffer from mode collapse and training instability, limiting their utility for robust downstream applications such as Reinforcement Learning (RL) agent training or stress testing.  
We are currently witnessing a second-order phase transition in generative modeling. The field is converging toward **skeleton-conditioned generation**, **linear-time sequence modeling**, and **flow-based dynamics**. This report articulates a comprehensive research architecture that integrates three cutting-edge paradigms: **ControlNet** for structural market conditioning, **Mamba (State Space Models)** for high-fidelity sequence modeling, and **Flow Matching** for efficient, deterministic sampling.  
This convergence addresses the critical "Trilemma of Financial Generation":

1. **Fidelity:** Can the model reproduce the complex autocorrelation structure of tick-level data?  
2. **Controllability:** Can risk managers explicitly dictate the macro-regime (e.g., "reconstruct the 2008 crash with 2024 interest rates")?  
3. **Scalability:** Can the architecture handle long-context histories (millions of ticks) efficiently?

By moving from $O(N^2)$ Transformers to $O(N)$ SSMs (Mamba) and from stochastic Diffusion sampling to deterministic Flow Matching ODE solvers, we propose a blueprint for the next generation of "Foundation World Models" for finance.

## ---

**2\. Skeleton-Conditioned Market Generation: The ControlNet Paradigm**

The concept of "conditioning" in financial time series has largely been limited to global class labels (e.g., conditioning on a "Bull" or "Bear" label). While effective for coarse mode switching, this lacks the temporal precision required for actionable stress testing. The **ControlNet** paradigm, originally developed to introduce spatial constraints (like canny edges or human pose skeletons) into text-to-image models 1, offers a robust theoretical framework for injecting **temporal skeletons** into market simulators.

### **2.1 Defining the "Financial Skeleton"**

To adapt ControlNet for finance, one must first define the equivalent of a "visual skeleton." In image generation, a skeleton is a low-information density map that constrains high-information density pixels. In finance, a **Market Skeleton** is a sparse or low-resolution trajectory that constrains the high-frequency stochastic process.  
We categorize Market Skeletons into three distinct classes, each requiring specific encoding strategies:

#### **2.1.1 The Macro-Regime Skeleton (Style Constraint)**

This skeleton defines the "economic weather" under which the high-frequency price action unfolds. It is analogous to "style transfer" in vision.

* **Components:** Time-varying sequences of exogenous variables such as Interest Rates, VIX (Volatility Index), Inflation prints, or Central Bank balance sheet flows.  
* **Encoding:** These are continuous, low-frequency signals. Unlike sparse events, they exist at every timestep but change slowly.  
* **Architectural Implication:** The generative model must treat this as a "background field." The conditioning mechanism needs to modulate the *variance* and *scale* of the generated returns rather than their precise path.

#### **2.1.2 The Structural Skeleton (Trend & Pattern Constraint)**

This skeleton dictates the geometric shape of the price path, equivalent to the "Canny Edge" map in ControlNet.3

* **Components:** Moving Averages (MA), Support/Resistance levels, or specific chart patterns (e.g., "Head and Shoulders"). Alternatively, a user-drawn trajectory representing a hypothetical scenario (e.g., "Asset A linearly decays by 20% over 10 days").  
* **Encoding:** This is a dense, continuous constraint. The generator effectively functions as a "super-resolution" or "texture in-painting" model, filling in the microstructure noise (bid-ask bounce, volatility) around the user-defined trend line.  
* **Challenge:** The model must balance adherence to the skeleton with the necessity of realistic deviations. A perfectly smooth adherence would violate the stochastic nature of markets.

#### **2.1.3 The Event Skeleton (Sparse Constraint)**

This skeleton consists of discrete, impulse-like events that enforce shock dynamics.

* **Components:** Earnings announcements, FOMC meeting timestamps, or scheduled macroeconomic data releases.  
* **Encoding:** A sparse binary mask or a sequence of "Dirac comb" functions, where $1$ indicates an event and $0$ indicates noise.5  
* **Architectural Implication:** This requires the model to learn "anticipatory" behavior. Volatility often rises *before* a known event timestamp (pre-earnings drift). The attention mechanism or SSM state must propagate this information backward and forward in time.

### **2.2 ControlNet Architecture for Time Series**

The naive application of ControlNet—duplicating the entire encoder of a U-Net—is suboptimal for 1D time series due to the risk of overfitting. Financial data has a lower signal-to-noise ratio than images; a massive parameter increase makes the model prone to memorizing the noise. We analyze two implementation strategies: **1D-ControlNet** and **ControlNet-XS**.

#### **2.2.1 1D-ControlNet Mechanism**

In this standard formulation, we take a pre-trained base model (e.g., a 1D U-Net or Mamba Block) with parameters $\\Theta$. We create a trainable copy $\\Theta\_c$ that processes the conditioning skeleton $c$.  
The key mechanism is the **Zero Convolution**.2

$$y \= \\mathcal{F}(x; \\Theta) \+ \\mathcal{Z}( \\mathcal{F}\_{train}( x \+ \\mathcal{Z}(c); \\Theta\_c ) )$$  
where $\\mathcal{Z}$ denotes a $1 \\times 1$ convolution layer initialized with weights and biases at zero.

* **Initialization Phase:** At the start of training, the output of the ControlNet branch is exactly zero. The model behaves identical to the unconditional pre-trained model. This preserves the "market physics" learned during the pre-training phase.  
* **Training Phase:** As gradients flow, the Zero Convolutions learn to inject the skeleton features. This is critical for financial series where we want to *bias* the generation (e.g., towards a crash) without breaking the fundamental laws of autocorrelation and volatility clustering.

#### **2.2.2 ControlNet-XS: The Efficient Alternative**

For high-frequency applications, the parameter doubling of standard ControlNet is prohibitive. **ControlNet-XS** 2 introduces a more efficient linkage. Instead of a parallel encoder, it injects the conditioning features directly into the layers of the main encoder via lightweight adapters.

* **Relevance to Finance:** ControlNet-XS is designed to minimize the delay in feedback loops. In control systems theory (which aligns closely with algorithmic trading), delays between the feedback signal (the skeleton) and the corrective action (the generation) can cause oscillations. ControlNet-XS's tighter coupling reduces this latency, ensuring that if a "Support Level" skeleton dictates a price floor at $t$, the generative model respects it instantaneously rather than drifting.2

### **2.3 Conditioning Injection: FiLM vs. AdaLN vs. Cross-Attention**

The method by which the skeleton information interacts with the neural backbone determines the flexibility of the control. We benchmark three strategies:

| Mechanism | Description | Best Use Case in Finance | Implementation Complexity |
| :---- | :---- | :---- | :---- |
| **FiLM (Feature-wise Linear Modulation)** | Applies an affine transformation to features: $\\gamma(c) \\cdot x \+ \\beta(c)$. | **Macro-Regimes:** Adjusting global volatility or "temperature" of the market. | Low |
| **AdaLN (Adaptive LayerNorm)** | Similar to FiLM but regresses normalization parameters per layer. 6 | **Regime Switching:** Standard in Diffusion Transformers (DiT) for time/class embedding. | Medium |
| **Cross-Attention** | Tokens in the generated sequence "attend" to tokens in the skeleton sequence. 7 | **Event Skeletons:** Allowing the market at $t$ to "look at" an earnings event at $t+k$. | High |

#### **2.3.1 Deep Dive: Cross-Attention for Sparse Alpha Signals**

When the skeleton represents **Alpha Signals** (e.g., a sparse vector of trading signals or news sentiment), Cross-Attention is non-negotiable.7

$$\\text{Attention}(Q, K, V) \= \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$  
Here, $Q$ comes from the noisy market state, while $K$ and $V$ are projected from the Skeleton.

* **Interpretation:** The attention map $A\_{t,j}$ tells us how much the market price at time $t$ depends on the skeleton constraint at time $j$. This visualization is a powerful explainability tool. If the model is generating a crash, we can inspect the attention map to see if it was "triggered" by a specific event in the skeleton or if it was a stochastic hallucination.

#### **2.3.2 Deep Dive: AdaLN for Volatility Scaling**

For **Mamba** architectures (discussed in Section 3), explicit Attention layers are often replaced or minimized. Here, **Adaptive LayerNorm (AdaLN)** becomes the primary conditioning route.6 In a DiT or Mamba block, the normalization layer $LN(x)$ is modulated by the skeleton embedding $c$:

$$AdaLN(x, c) \= (1 \+ \\text{scale}(c)) \\cdot LN(x) \+ \\text{shift}(c)$$  
This effectively scales the *amplitude* of the hidden states. Since the amplitude of the hidden state in a financial model often correlates with realized volatility, AdaLN provides a direct "knob" for the ControlNet to turn up or down the market risk.

### **2.4 Encoding Complex Skeletons: Graph Neural Networks (GNNs)**

Financial markets are rarely univariate. A true Market Skeleton often involves the **Correlation Matrix** between multiple assets. To condition generation on a dynamic correlation structure, we employ **Graph Neural Networks (GNNs)** as the skeleton encoder.10

* **Input:** A time-varying adjacency matrix $A\_t$ representing the correlation structure (e.g., Sector correlations).  
* **Encoder:** A Relational GCN or Graph Attention Network (GAT) encodes this matrix into a node embedding sequence.  
* **Injection:** These embeddings serve as the conditioning vector $c$ for the main generator. This ensures that if the skeleton dictates "High Correlation (Crisis Mode)," the generator forces all assets to co-move, reproducing the "diversification breakdown" phenomenon seen in real crashes.

## ---

**3\. State Space Models (Mamba) for Market Generation**

While Transformers (Attention) have dominated sequence modeling, they suffer from quadratic complexity $O(L^2)$. For financial time series, this is a fatal bottleneck. Tick data or 1-minute order book updates result in sequence lengths of $L \> 10,000$ for even short history windows. **State Space Models (SSMs)**, specifically the **Mamba** architecture, offer Linear Complexity $O(L)$, enabling the modeling of massive context windows essential for capturing long-term cycles.13

### **3.1 The Theoretical Advantage: Selectivity in Stochastic Systems**

Traditional SSMs (like S4) rely on time-invariant transition matrices ($A, B, C$).

$$h\_t \= A h\_{t-1} \+ B x\_t$$

$$y\_t \= C h\_t$$  
This creates a "Fixed Filter" that processes all inputs uniformly. Mamba introduces **Selectivity** 15, making the parameters functions of the input:

$$h\_t \= A(\\Delta\_t) h\_{t-1} \+ B(x\_t) x\_t$$

* **$\\Delta\_t$ (Time Step):** Controls how much the state updates. Large $\\Delta$ \= focus on new input (high volatility); Small $\\Delta$ \= rely on memory (mean reversion).  
* **$B\_t$ (Input Gate):** Controls how much of the current signal enters the state.  
* **$C\_t$ (Output Gate):** Controls how much of the state is exposed.

**Financial Interpretation:** This "Selection Mechanism" is analogous to an **Adaptive Kalman Filter** or a trader's intuition. It allows the model to:

1. **Filter Noise:** During periods of low-volume chop, the model can learn to set $B\_t \\approx 0$ (ignore noise) and keep $h\_t$ constant (remember the trend).  
2. **React to Shocks:** During an earnings surprise, the model can spike $\\Delta\_t$, rapidly updating its internal state to the new price equilibrium. This content-aware reasoning is superior to Transformers for finance, as it explicitly models the *dynamics* of the system state rather than just pairwise token correlations.14

### **3.2 JAX Implementation: The Parallel Scan Algorithm**

The recurrence relation of Mamba appears sequential ($h\_t$ needs $h\_{t-1}$), which would be slow on GPUs. However, the operation is a linear recurrence, which allows for **Parallel Associative Scan** (also known as the Blelloch Scan).18

#### **3.2.1 Mathematical Formulation of the Scan**

The discretized state update can be written as:

$$h\_t \= \\bar{A}\_t h\_{t-1} \+ \\bar{B}\_t x\_t$$  
This is of the form $h\_t \= a\_t h\_{t-1} \+ b\_t$. We define a binary operator $\\otimes$:

$$(a\_j, b\_j) \\otimes (a\_i, b\_i) \= (a\_j a\_i, a\_j b\_i \+ b\_j)$$  
This operator is associative. Thus, we can compute the prefix sums of the sequence in parallel using a tree-based reduction algorithm.

#### **3.2.2 JAX Code Architecture**

JAX is the optimal ecosystem for Mamba because jax.lax.associative\_scan is a native, highly optimized primitive.18

* **Step 1: Discretization Kernel:**  
  Write a vmap-ed function to convert continuous parameters to discrete ones using the Zero-Order Hold (ZOH) method.  
  Python  
  \# Conceptual JAX implementation  
  def discretize\_zoh(delta, A, B):  
      \# delta shape: (B, L, D)  
      \# A shape: (D, N)  
      delta\_A \= jnp.exp(jnp.einsum('bld,dn-\>bldn', delta, A))  
      delta\_B \= jnp.einsum('bld,bld,dn-\>bldn', delta, x, B) \# simplified  
      return delta\_A, delta\_B

* **Step 2: The Associative Scan:**  
  Use JAX's primitive to solving the recurrence.  
  Python  
  def binary\_operator(e\_i, e\_j):  
      A\_i, B\_i \= e\_i  
      A\_j, B\_j \= e\_j  
      return A\_j \* A\_i, A\_j \* B\_i \+ B\_j

  final\_states \= jax.lax.associative\_scan(binary\_operator, (discrete\_A, discrete\_B))

* **Hardware Optimization:** On TPU pods, this scan operation is fused into a single kernel, reducing memory bandwidth usage (HBM access) by orders of magnitude compared to Transformer Attention matrices. This allows us to scale the d\_state (SSM state dimension) significantly larger, capturing richer market histories.18

### **3.3 Architecture: Bidirectional Mamba (BiMamba) and DiMSUM**

Standard Mamba is causal (Left-to-Right). For **Generative Modeling** (like Diffusion/Flow Matching), the model needs to see the entire noisy sequence to predict the clean signal. A purely causal model would only denoise based on the past, leading to discontinuities at the end of the sequence.

#### **3.3.1 BiMamba Implementation**

We implement a **Bidirectional Mamba Block** 21:

1. **Forward Branch:** Processes sequence $x\_{1:T}$.  
2. **Backward Branch:** Processes sequence $x\_{T:1}$ (reversed).  
3. **Fusion:** The outputs are concatenated and projected:  
   $$h\_{combined} \= \\text{Linear}(\[h\_{fwd}; h\_{bwd}\])$$  
   or summed via a gated mechanism.  
   * **Relevance:** The backward branch allows the model to propagate information from "future constraints" (e.g., a Skeleton constraint at $T=100$) back to $T=50$, ensuring the trajectory smoothly converges to the target.

#### **3.3.2 DiMSUM: Diffusion Mamba for Scale-Space Understanding**

Financial data is multi-scale. Trends happen over months; volatility happens over minutes. The **DiMSUM** architecture 23 integrates **Wavelet Transforms** with Mamba.

* **Mechanism:**  
  1. Apply Discrete Wavelet Transform (DWT) to input $x$.  
  2. Obtain Low-Frequency ($x\_{low}$) and High-Frequency ($x\_{high}$) coefficients.  
  3. Pass $x\_{low}$ and $x\_{high}$ through separate Mamba heads.  
  4. Apply Inverse DWT to reconstruct the denoised signal.  
* **Advantage:** This decouples the modeling of "Trend" (low freq) and "Volatility" (high freq). Mamba's state space is notably better at handling the "smooth" low-frequency components, while the explicit separation prevents the high-frequency noise from being smoothed out—a common failure mode in LSTMs.

### **3.4 Hybrid Architectures: Jamba (The "Memory" Compromise)**

While Mamba is efficient, Attention is better at "In-Context Learning" (copying exact values from the past). For financial tasks requiring "Copy-Paste" behavior (e.g., repeating a specific fractal pattern), a pure SSM might be too compressive. **Jamba** 24 proposes a hybrid stack:

* **Ratio:** 1 Transformer Layer : 7 Mamba Layers.  
* **Role:** Mamba layers handle the bulk processing and state maintenance. The sparse Attention layers act as a "Long-Term Memory Retrieval" mechanism, allowing the model to look back at specific "Keyform" events in the history.  
* **MoE (Mixture of Experts):** Jamba often integrates MoE MLP layers to increase capacity without increasing inference latency. For finance, different "Experts" can specialize in different regimes (e.g., a "Crisis Expert" MLP vs. a "Steady State Expert" MLP).

## ---

**4\. Flow Matching: The Deterministic Generative Engine**

Diffusion Models (DDPM/LDM) define a generation process as the reversal of a stochastic differential equation (SDE). This results in "curved" trajectories in probability space, requiring 50-100 sampling steps to solve accurately. **Flow Matching (FM)** 27 is a paradigm shift towards learning **Continuous Normalizing Flows (CNF)** with straighter paths, enabling faster and more stable generation.

### **4.1 Theoretical Foundation: Rectified Flows**

Instead of diffusing data to noise via a variance-preserving SDE, Flow Matching defines a **probability path** $p\_t$ between the data distribution $p\_1$ and a simple prior $p\_0$ (Gaussian noise).  
We aim to learn a **Time-Dependent Vector Field** $v\_t(x)$ that generates this path via an Ordinary Differential Equation (ODE):

$$\\frac{dx}{dt} \= v\_t(x)$$

#### **4.1.1 The Conditional Flow Matching (CFM) Objective**

A key innovation is the ability to train using a "simulation-free" objective. We define a *conditional* vector field $u\_t(x|x\_1)$ that pulls a specific noise sample $x\_0$ towards a specific data sample $x\_1$ along a straight line:

$$x\_t \= (1-t)x\_0 \+ t x\_1$$

$$u\_t(x|x\_1) \= x\_1 \- x\_0$$  
The loss function regresses the neural network output $v\_t$ against this simple linear target:  
$$ \\mathcal{L}*{FM}(\\theta) \= \\mathbb{E}*{t, x\_0, x\_1} \[ |  
| v\_t(x\_t) \- (x\_1 \- x\_0) ||^2 \] $$

* **Efficiency:** Unlike Diffusion, which tries to predict "noise" (score matching), Flow Matching predicts "velocity" (the direction towards the data). This target is numerically stable and bounded.

### **4.2 MambaFlow: Integrating Mamba with Flow Matching**

We propose **MambaFlow** 29—an architecture where the vector field estimator $v\_t(x)$ is parameterized by the **Bidirectional Mamba** network described in Section 3.3.

#### **4.2.1 Architecture Specification**

1. **Input:** Noisy market state $x\_t \\in \\mathbb{R}^{B \\times L \\times D}$.  
2. **Time Embedding Injection:** The flow time $t \\in $ is critical.  
   * *Mechanism:* A sinusoidal time embedding vector is projected and injected into every Mamba Block via **AdaLN**.  
   * *Logic:* At $t=0$ (Noise), the vector field must be aggressive (large structural changes). At $t=1$ (Near Data), it must be subtle (microstructure refinement). AdaLN scales the network features to adapt to this phase transition.  
3. **Skeleton Injection:** The ControlNet skeleton features are added to the input or fused via Cross-Attention layers (in Jamba blocks).  
4. **Output:** The predicted velocity vector $\\hat{v} \\in \\mathbb{R}^{B \\times L \\times D}$.

### **4.3 Sampling via Diffrax: Solving the ODE**

Generation (Inference) becomes an Initial Value Problem (IVP): Given $x\_0 \\sim \\mathcal{N}(0, I)$, integrate $dx/dt \= v\_t(x)$ to $t=1$. We utilize **Diffrax** 31, a JAX-based differential equation solver library.

#### **4.3.1 Solver Selection Strategy**

* **Euler Method:** The simplest solver. Since Flow Matching is trained to produce *straight* paths (Rectified Flow), a simple Euler step is often surprisingly accurate. This enables 1-step or 2-step generation (extreme speed).  
* **Dopri5 (Dormand-Prince):** An adaptive step-size solver.  
  * *Usage:* We recommend Dopri5 for **Conditional Generation**. When conditioning on a complex skeleton (e.g., a Black Swan event), the probability path might curve sharply around the event. Adaptive solvers detect this curvature (via error estimation) and take smaller steps locally to ensure fidelity, then take larger steps in smooth regions.  
* **Adjoint Sensitivity:** Diffrax supports the *adjoint method* for backpropagation. This allows us to differentiate *through* the ODE solver with constant memory cost.  
  * *Application:* We can fine-tune the MambaFlow generator using **End-to-End Reinforcement Learning**. The generator produces a trajectory, a trading agent trades on it, and the PPO reward is backpropagated through the ODE solver to update the Mamba weights.33

## ---

**5\. Implementation Strategy: The "Skeleton-Mamba-Flow" Blueprint**

This section outlines the concrete engineering plan to build the proposed system.

### **5.1 Technology Stack & Hardware**

* **Framework:** JAX \+ Flax (Neural Networks) \+ Optax (Optimizers) \+ Diffrax (ODE Solvers).  
* **Hardware:** Google Cloud TPU v4 Pods or NVIDIA H100 Clusters. Mamba's associative scan is memory-bound; high HBM bandwidth (High Bandwidth Memory) is more critical than raw FLOPS.  
* **Data Pipeline:** tf.data or grain for high-throughput asynchronous data loading.

### **5.2 Data Preprocessing Protocol**

Financial data requires specific treatment to be suitable for generative modeling.

1. **Log-Returns Transformation:** Prices $P\_t$ are non-stationary. We model Log-Returns $r\_t \= \\log(P\_t) \- \\log(P\_{t-1})$.  
2. **Volatility Normalization (GARCH-based):**  
   * Fit a GARCH(1,1) model to estimating rolling volatility $\\sigma\_t$.  
   * Normalize returns: $z\_t \= r\_t / \\sigma\_t$.  
   * *Model Task:* The MambaFlow model generates the normalized residuals $z\_t$ AND the volatility process $\\sigma\_t$ separately (multi-channel generation). This explicitly enforces volatility clustering structure.  
3. **Skeleton Encoding:**  
   * Macro variables are Z-scored.  
   * Event masks are kept binary {0, 1}.

### **5.3 Training Loop Architecture**

The training involves a "Minibatch Optimal Transport" setup to stabilize the Flow Matching.

1. **Sampling:** Sample a batch of real data $x\_1$ and noise $x\_0$.  
2. **Coupling:** Instead of random pairing, pair $x\_0^{(i)}$ with $x\_1^{(j)}$ that are closest in Euclidean distance (OT-CFM). This minimizes the variance of the vector field target, speeding up convergence.27  
3. **Loss Calculation:**  
   Python  
   def loss\_fn(params, x\_1, x\_0, t, condition):  
       \# Interpolate  
       x\_t \= (1 \- t) \* x\_0 \+ t \* x\_1  
       target\_v \= x\_1 \- x\_0

       \# Mamba Forward Pass  
       pred\_v \= mamba\_flow\_model.apply(params, x\_t, t, condition)

       \# MSE Loss  
       return jnp.mean((pred\_v \- target\_v) \*\* 2\)

4. **Optimization:** Use **Lion** or **AdamW** optimizer. Warmup is essential for Mamba's $\\Delta$ parameters to stabilize.18

## ---

**6\. Comparative Benchmarking & Validation Strategies**

Evaluating generative models in finance is harder than in vision. A "realistic looking" chart might be statistically broken (e.g., lacking heavy tails). We propose a rigorous 3-tier validation suite.

### **6.1 Tier 1: Statistical Fidelity (Stylized Facts)**

These metrics verify that the synthetic data adheres to the universal laws of financial physics.

| Metric | Implementation & Formula | Target | Source |
| :---- | :---- | :---- | :---- |
| **Marginal Distribution ($W\_1$)** | Wasserstein-1 Distance between CDF of Real vs. Synthetic returns. | Minimize | 34 |
| **Tail Index ($\\alpha$)** | Fit a Power Law $P(x) \\propto x^{-\\alpha}$ to the tails (Hill Estimator). | Match Real ($\\alpha \\approx 3$) | 35 |
| **Volatility Clustering** | Autocorrelation function (ACF) of *squared* returns: $Corr(r\_t^2, r\_{t-\\tau}^2)$. | Positive decay (Long Memory) | 36 |
| **Leverage Effect** | Correlation between return at $t$ and volatility at $t+1$: $Corr(r\_t, \\sigma^2\_{t+1})$. | Negative ($\\approx \-0.3$) | 35 |
| **Cross-Correlation Consistency** | Frobenius norm of difference between correlation matrices: $ |  | \\Sigma\_{real} \- \\Sigma\_{syn} |

**Insight:** Mamba architectures excel at the **Volatility Clustering** metric due to their stateful memory, whereas Transformers often struggle to maintain this correlation over long horizons (thousands of steps).

### **6.2 Tier 2: Utility Metrics (TSTR & TRTS)**

We treat the synthetic data as a training resource.

* **TSTR (Train on Synthetic, Test on Real):**  
  * Train a standard LSTM forecaster on generated data. Test on real hold-out data.  
  * *Metric:* Prediction MSE.  
  * *Interpretation:* If TSTR error $\\approx$ TRTR (Train on Real, Test on Real), the synthetic data contains valid alpha signals.38  
* **TRTS (Train on Real, Test on Synthetic):**  
  * Train on Real, Test on Synthetic.  
  * *Interpretation:* If error is too high, synthetic data is "out of distribution" (artifacts). If error is too low, synthetic data is "too easy" (mode collapse/memorization).

### **6.3 Tier 3: Downstream RL & The Sim-to-Real Gap**

The ultimate validation is the performance of trading agents.39

* **Protocol:**  
  1. Initialize a PPO (Proximal Policy Optimization) agent.  
  2. Train the agent for 1M steps on the **MambaFlow Simulator**.  
  3. Deploy the agent on **Historical Real Data** (Backtest).  
* **The Sim-to-Real Gap Ratio ($R\_{gap}$):**  
  $$R\_{gap} \= \\frac{\\text{Sharpe Ratio (Real)}}{\\text{Sharpe Ratio (Sim)}}$$  
* **Analysis:**  
  * **Ideal:** $R\_{gap} \\in \[0.8, 1.0\]$. The simulator was a faithful proxy.  
  * **Overfitting:** $R\_{gap} \< 0.5$. The agent exploited artifacts in the simulator (e.g., predictable patterns) that don't exist in reality.  
  * **Robustness:** $R\_{gap} \> 1.0$. The simulator was "harder" than reality (e.g., by injecting adversarial skeletons via ControlNet), training a conservative but robust agent. This is often the goal of **Domain Randomization** in finance.

## ---

**7\. Future Directions & Conclusion**

The proposed architecture—**Skeleton-Mamba-Flow**—represents a convergent evolution in financial modeling.

* **ControlNet** solves the "Black Box" problem, allowing explicit scenario design.  
* **Mamba** solves the "Scale" problem, unlocking tick-level simulation.  
* **Flow Matching** solves the "Speed" problem, enabling massive-scale RL training.

**Future Research Vectors:**

1. **Adversarial Control:** Using a separate RL agent to *control the skeleton* (ControlNet input) to actively break the trading agent, creating an automated "Red Teaming" system for algorithmic trading.  
2. **Multi-Modal Skeletons:** Conditioning market generation not just on numerical skeletons, but on **Textual News Skeletons** (LLM embeddings of Bloomberg headers), fusing NLP with Time Series Mamba.  
3. **Foundation World Models:** Pre-training a massive BiMamba model on all global asset classes simultaneously, creating a "GPT-4 of Finance" that can zero-shot generalize to new assets.

This report establishes the architectural and theoretical groundwork for building these next-generation market simulators.  
---

**Citations:** .1

#### **Citovaná díla**

1. ControlNet: A Complete Guide \- Stable Diffusion Art, použito ledna 25, 2026, [https://stable-diffusion-art.com/controlnet/](https://stable-diffusion-art.com/controlnet/)  
2. ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2312.06573v2](https://arxiv.org/html/2312.06573v2)  
3. lllyasviel/ControlNet: Let us control diffusion models\! \- GitHub, použito ledna 25, 2026, [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)  
4. The Ultimate Guide to ControlNet (Part 1\) \- Civitai Education Hub, použito ledna 25, 2026, [https://education.civitai.com/civitai-guide-to-controlnet/](https://education.civitai.com/civitai-guide-to-controlnet/)  
5. Time Encoding of Sparse Signals with Flexible Filters | Request PDF \- ResearchGate, použito ledna 25, 2026, [https://www.researchgate.net/publication/375252794\_Time\_Encoding\_of\_Sparse\_Signals\_with\_Flexible\_Filters](https://www.researchgate.net/publication/375252794_Time_Encoding_of_Sparse_Signals_with_Flexible_Filters)  
6. Design Choices in Diffusion Transformers | by Om Rastogi \- Medium, použito ledna 25, 2026, [https://omrastogi.medium.com/design-choices-in-diffusion-transformers-f7d4c8233535](https://omrastogi.medium.com/design-choices-in-diffusion-transformers-f7d4c8233535)  
7. Cross-Attention Conditioning \- Emergent Mind, použito ledna 25, 2026, [https://www.emergentmind.com/topics/cross-attention-conditioning](https://www.emergentmind.com/topics/cross-attention-conditioning)  
8. Different ways to condition the diffusion model \- Jonghwa Yim, použito ledna 25, 2026, [https://jonhwayim.medium.com/different-ways-to-condition-the-diffusion-model-d2e41eaa01c2](https://jonhwayim.medium.com/different-ways-to-condition-the-diffusion-model-d2e41eaa01c2)  
9. StreamDiT: Real-Time Streaming Text-to-Video Generation \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2507.03745v3](https://arxiv.org/html/2507.03745v3)  
10. scapGNN: A graph neural network–based framework for active pathway and gene module inference from single-cell multi-omics data \- PubMed Central, použito ledna 25, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10681325/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10681325/)  
11. Enhancing Uncertainty Quantification and Interpretability with Graph Functional Neural Process \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2508.17097v1](https://arxiv.org/html/2508.17097v1)  
12. Joint Relational Database Generation via Graph-Conditional Diffusion Models | OpenReview, použito ledna 25, 2026, [https://openreview.net/forum?id=Z3OtNSwuXX](https://openreview.net/forum?id=Z3OtNSwuXX)  
13. What Is A Mamba Model? | IBM, použito ledna 25, 2026, [https://www.ibm.com/think/topics/mamba-model](https://www.ibm.com/think/topics/mamba-model)  
14. \[2403.11144\] Is Mamba Effective for Time Series Forecasting? \- arXiv, použito ledna 25, 2026, [https://arxiv.org/abs/2403.11144](https://arxiv.org/abs/2403.11144)  
15. TSMamba : Mamba model for Time Series Forecasting | by Mehul Gupta | Data Science in Your Pocket | Medium, použito ledna 25, 2026, [https://medium.com/data-science-in-your-pocket/tsmamba-mamba-model-for-time-series-forecasting-c9eeb0d0d23c](https://medium.com/data-science-in-your-pocket/tsmamba-mamba-model-for-time-series-forecasting-c9eeb0d0d23c)  
16. \[2312.00752\] Mamba: Linear-Time Sequence Modeling with Selective State Spaces \- arXiv, použito ledna 25, 2026, [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)  
17. Is Mamba Effective for Time Series Forecasting? \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2403.11144v3](https://arxiv.org/html/2403.11144v3)  
18. A Visual Guide to Mamba and State Space Models \- Maarten Grootendorst, použito ledna 25, 2026, [https://www.maartengrootendorst.com/blog/mamba/](https://www.maartengrootendorst.com/blog/mamba/)  
19. Mamba No. 5 (A Little Bit Of…) \- Sparse Notes, použito ledna 25, 2026, [https://jameschen.io/jekyll/update/2024/02/12/mamba.html](https://jameschen.io/jekyll/update/2024/02/12/mamba.html)  
20. jax.lax.associative\_scan \- JAX documentation, použito ledna 25, 2026, [https://docs.jax.dev/en/latest/\_autosummary/jax.lax.associative\_scan.html](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html)  
21. \[2410.05916\] TIMBA: Time series Imputation with Bi-directional Mamba Blocks and Diffusion models \- arXiv, použito ledna 25, 2026, [https://arxiv.org/abs/2410.05916](https://arxiv.org/abs/2410.05916)  
22. Bi-Mamba+; Bidirectional Mamba for Time Series Forecasting \- AAA (All About AI), použito ledna 25, 2026, [https://seunghan96.github.io/ts/mamba/(paper)BiMAMBA/](https://seunghan96.github.io/ts/mamba/\(paper\)BiMAMBA/)  
23. DiMSUM: Diffusion Mamba \- A Scalable and Unified Spatial-Frequency Method for Image Generation | OpenReview, použito ledna 25, 2026, [https://openreview.net/forum?id=KqbLzSIXkm\&referrer=%5Bthe%20profile%20of%20Quan%20Dao%5D(%2Fprofile%3Fid%3D\~Quan\_Dao1)](https://openreview.net/forum?id=KqbLzSIXkm&referrer=%5Bthe+profile+of+Quan+Dao%5D\(/profile?id%3D~Quan_Dao1\))  
24. Jamba \- Hugging Face, použito ledna 25, 2026, [https://huggingface.co/docs/transformers/main/en/model\_doc/jamba](https://huggingface.co/docs/transformers/main/en/model_doc/jamba)  
25. Jamba: A Hybrid Transformer-Mamba Language Model \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2403.19887v1](https://arxiv.org/html/2403.19887v1)  
26. JAMBA: HYBRID TRANSFORMER-MAMBA LANGUAGE MODELS \- ICLR Proceedings, použito ledna 25, 2026, [https://proceedings.iclr.cc/paper\_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf)  
27. FM-TS: flow matching for time series generation \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2411.07506v1](https://arxiv.org/html/2411.07506v1)  
28. \[2411.07506\] FlowTS: Time Series Generation via Rectified Flow \- arXiv, použito ledna 25, 2026, [https://arxiv.org/abs/2411.07506](https://arxiv.org/abs/2411.07506)  
29. M4V: Multi-Modal Mamba for Text-to-Video Generation \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2506.10915v1](https://arxiv.org/html/2506.10915v1)  
30. MambaFlow: A Novel and Flow-guided State Space Model for Scene Flow Estimation \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2502.16907v1](https://arxiv.org/html/2502.16907v1)  
31. Diffrax \- Patrick Kidger, použito ledna 25, 2026, [https://docs.kidger.site/diffrax/](https://docs.kidger.site/diffrax/)  
32. patrick-kidger/diffrax: Numerical differential equation solvers in JAX. Autodifferentiable and GPU-capable. https://docs.kidger.site/diffrax \- GitHub, použito ledna 25, 2026, [https://github.com/patrick-kidger/diffrax](https://github.com/patrick-kidger/diffrax)  
33. Adjoints \- Diffrax \- Patrick Kidger, použito ledna 25, 2026, [https://docs.kidger.site/diffrax/api/adjoints/](https://docs.kidger.site/diffrax/api/adjoints/)  
34. (PDF) Generative model for financial time series trained with MMD using a signature kernel, použito ledna 25, 2026, [https://www.researchgate.net/publication/382655425\_Generative\_model\_for\_financial\_time\_series\_trained\_with\_MMD\_using\_a\_signature\_kernel](https://www.researchgate.net/publication/382655425_Generative_model_for_financial_time_series_trained_with_MMD_using_a_signature_kernel)  
35. Modeling Stylized Facts in FX Markets with FINGAN-BiLSTM: A Deep Learning Approach to Financial Time Series \- MDPI, použito ledna 25, 2026, [https://www.mdpi.com/1099-4300/27/6/635](https://www.mdpi.com/1099-4300/27/6/635)  
36. Stylized Facts of Financial Time Series and Three Popular Models of Volatility \- Wharton Statistics, použito ledna 25, 2026, [http://www-stat.wharton.upenn.edu/\~steele/Resources/FTSResources/StylizedFacts/MalmstenTerasvirta04.pdf](http://www-stat.wharton.upenn.edu/~steele/Resources/FTSResources/StylizedFacts/MalmstenTerasvirta04.pdf)  
37. 4.2 Multivariate Time Series | Introduction to Computational Finance and Financial Econometrics with R \- Bookdown, použito ledna 25, 2026, [https://bookdown.org/compfinezbook/introcompfinr/Multivariate-Time-Series.html](https://bookdown.org/compfinezbook/introcompfinr/Multivariate-Time-Series.html)  
38. Enabling Diffusion Model for Conditioned Time Series Generation \- MDPI, použito ledna 25, 2026, [https://www.mdpi.com/2673-4591/68/1/25](https://www.mdpi.com/2673-4591/68/1/25)  
39. \[2510.17709\] Closing the Sim2Real Performance Gap in RL \- arXiv, použito ledna 25, 2026, [https://arxiv.org/abs/2510.17709](https://arxiv.org/abs/2510.17709)  
40. Correlation Matrix Generation using Object Oriented Python | QuantStart, použito ledna 25, 2026, [https://www.quantstart.com/articles/correlation-matrix-generationg-using-object-oriented-python/](https://www.quantstart.com/articles/correlation-matrix-generationg-using-object-oriented-python/)  
41. Sim-to-real via latent prediction: Transferring visual non-prehensile manipulation policies, použito ledna 25, 2026, [https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1067502/full](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1067502/full)  
42. One-shot sim-to-real transfer policy for robotic assembly via reinforcement learning with visual demonstration | Robotica \- Cambridge University Press, použito ledna 25, 2026, [https://www.cambridge.org/core/journals/robotica/article/oneshot-simtoreal-transfer-policy-for-robotic-assembly-via-reinforcement-learning-with-visual-demonstration/FC22E58B7B0876F0E5F151A229E241FD](https://www.cambridge.org/core/journals/robotica/article/oneshot-simtoreal-transfer-policy-for-robotic-assembly-via-reinforcement-learning-with-visual-demonstration/FC22E58B7B0876F0E5F151A229E241FD)  
43. Evaluating generative models for synthetic financial data \- arXiv, použito ledna 25, 2026, [https://arxiv.org/html/2512.21791v1](https://arxiv.org/html/2512.21791v1)  
44. Evaluate synthetic data quality using downstream ML \- MOSTLY AI, použito ledna 25, 2026, [https://mostly.ai/blog/synthetic-data-quality-evaluation](https://mostly.ai/blog/synthetic-data-quality-evaluation)