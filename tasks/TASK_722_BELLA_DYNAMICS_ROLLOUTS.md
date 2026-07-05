# TASK_722: Bella-Conditioned Rollouts for MCTS Dynamics

**Status**: DRAFT
**Priority**: HIGH — replaces learned dynamics MLP with generative world model
**Depends on**: Bella trained checkpoint, Phase 2 Stage B checkpoint
**File(s)**: `diff_teacher/generation/generator.py`, `train_phase2.py`, new `mcts_trader/jax_agent/bella_dynamics.py`

---

## Motivation

The dynamics MLP (`dynamics_dense1/2`, 3.3M params) predicts `latent_{t+1}` from `(latent_t, action)` — a one-step prediction in a 128-dim latent space. After 20k+ epochs of training, consistency loss is still ~1.34. This is an inherently hard problem: compressing the full complexity of multi-asset market evolution into a 2-layer MLP.

Bella already solves a harder problem: generating realistic 1440-step, 64-asset OHLCV trajectories with correct stylized facts (volatility clustering, fat tails, cross-asset correlations). Use Bella as the imagination engine — observe current state, hallucinate plausible futures.

**Core idea**: Bella generates multiple plausible futures. The dynamics MLP acts as a consistency filter — scoring how grounded each future is. MCTS searches over Bella's futures, weighted by dynamics alignment. Neither works alone: Bella without MLP can hallucinate wildly (even with loss < 2), MLP without Bella is a toy approximation.

**Bella = imagination (diverse, realistic, can go haywire)**
**Dynamics MLP = grounding (simple, anchored to learned transitions)**

---

## Architecture: Video-Style Conditional Generation

Bella's flow matching is analogous to video generation:
- **Observed frames** = recent price window (ws=64 timesteps)
- **Generated frames** = future price trajectory (next H steps)
- **Sparse noise** = stochastic variation between rollouts (different futures)
- **Multiple rollouts** = K different noise seeds = K branch candidates for MCTS

### Generation Pipeline

```
Current prices[t-63:t]  →  SkeletonEncoder  →  (gamma, beta, h0)
                                                      ↓
                              Noise z_k ~ N(0,1)  →  Bella ODE (10 steps)  →  future_k[t+1:t+H]
                              (K different seeds)     conditioned on skeleton
                                                      ↓
                              K future trajectories, each (H, n_assets, 5) OHLCV
```

### MCTS Over Real Futures

Instead of:
```
MCTS node expansion:  latent_t + action → dynamics_MLP → latent_{t+1} (predicted)
                      latent_{t+1} → prediction_head → policy, value (from predicted state)
```

Use:
```
MCTS root:  obs_t = encode(prices[t-63:t])
            Generate K=8 futures from current state via Bella
            Each future is a real OHLCV trajectory

MCTS expansion:  For branch k, action a:
                   future_k already exists as full trajectory
                   obs_{t+1,k} = encode(future_k[0:ws])      ← real encoded obs, not MLP prediction
                   obs_{t+2,k} = encode(future_k[1:ws+1])    ← slide window forward
                   reward = oracle_reward(future_k, action)   ← actual PnL on generated prices

MCTS evaluation: value = V(obs_{t+N,k})  ← value head on real-looking encoded obs
```

**Key advantage**: every node in the MCTS tree is a real encoded observation from a plausible market, not an MLP approximation in latent space. The value/policy heads see realistic inputs they were trained on.

---

## Implementation Plan

### Step 1: Context-Conditioned Bella

Extend Bella to condition on observed price window (not just regime/skeleton).

```python
class ContextConditioner:
    """Convert observed price window → skeleton conditioning for Bella."""
    
    def __init__(self, d_model=768):
        # Reuse SkeletonEncoder's FiLM pathway
        self.price_encoder = nn.Dense(d_model)  # (ws, n_assets, 5) → (d_model,)
        self.film_mlp = nn.Sequential([
            nn.Dense(d_model * 2), nn.silu, nn.Dense(d_model * 2)
        ])
        self.h0_proj = nn.Dense(d_model)
    
    def __call__(self, price_window):
        """price_window: (batch, ws, n_assets, 5) → (gamma, beta, h0)"""
        # Flatten spatial dims, encode to d_model
        flat = price_window.reshape(price_window.shape[0], -1)  # (B, ws*n_assets*5)
        h = self.price_encoder(flat)  # (B, d_model)
        film = self.film_mlp(h)
        gamma, beta = jnp.split(film, 2, axis=-1)
        h0 = self.h0_proj(h)
        return gamma, beta, h0
```

**Alternative** (simpler, no new params): Extract a MarketSkeleton from the observed window (trend direction, vol regime, correlation clusters) using existing `diff_teacher/skeleton/extraction.py`, then encode via existing SkeletonEncoder. This leverages Bella's already-trained conditioning pathway.

### Step 2: Short Forward Generation

Modify `MarketGenerator.generate()` to support partial sequences — generate only the forward portion while keeping the observed portion fixed.

```python
def generate_forward(self, context_prices, horizon, n_rollouts, rng_key):
    """Generate K forward trajectories conditioned on observed prices.
    
    Args:
        context_prices: (ws, n_assets, 5) — observed OHLCV window
        horizon: int — steps to generate forward (e.g., 20-50)
        n_rollouts: int — number of different futures (K=8)
        rng_key: JAX PRNG key
    
    Returns:
        futures: (K, horizon, n_assets, 5) — K plausible continuations
    """
    # 1. Extract skeleton from observed window
    skeleton = extract_skeleton(context_prices)
    gamma, beta, h0 = self.skeleton_encoder(skeleton)
    
    # 2. Generate full sequences conditioned on skeleton
    #    seq_length = ws + horizon (observed + forward)
    keys = jax.random.split(rng_key, n_rollouts)
    noise = jax.random.normal(keys, (n_rollouts, ws + horizon, n_assets * 5))
    
    # 3. ODE integration (10 Euler steps)
    #    Bella denoises noise → realistic trajectory
    trajectories = self._ode_integrate(noise, gamma, beta, h0)
    # Shape: (K, ws+horizon, n_assets, 5)
    
    # 4. Inpainting: replace observed portion with actual prices
    #    Only keep the generated forward portion
    #    (Or: anchor the generation so observed window matches reality)
    futures = trajectories[:, ws:, :, :]  # (K, horizon, n_assets, 5)
    
    return futures
```

**Inpainting approach** (more principled): During ODE integration, at each Euler step, replace the first `ws` timesteps with the (appropriately noised) observed prices. This forces the generation to be consistent with observations — standard technique from video inpainting / image editing diffusion models.

```python
def _ode_step_with_inpainting(self, x, t, dt, context, ws):
    """Single Euler step with observed-window inpainting."""
    v = self.model(x, t, conditioning)  # predicted velocity
    x_new = x + v * dt
    
    # Inpainting: blend observed portion at noise level matching t+dt
    t_next = t + dt
    noise_level = 1.0 - t_next  # flow matching: t=0 is noise, t=1 is data
    noised_context = t_next * context + noise_level * noise_context
    x_new = x_new.at[:, :ws, :].set(noised_context)
    
    return x_new
```

### Step 3: Bella-MCTS Integration (Imagination + Grounding)

The key insight: Bella proposes, dynamics MLP validates.

```python
class BellaMCTSDynamics:
    """Bella generates futures, dynamics MLP scores their plausibility.
    
    MCTS searches over Bella's rollouts but weights branches by how
    consistent they are with the learned dynamics. This prevents
    Bella hallucinations from dominating the search.
    """
    
    def __init__(self, bella_generator, trm_model, n_rollouts=8, horizon=20):
        self.bella = bella_generator
        self.model = trm_model  # has encoder + dynamics + value/policy
        self.K = n_rollouts
        self.H = horizon
    
    def generate_and_score(self, params, current_prices, rng_key):
        """Pre-generate K futures + score each with dynamics MLP.
        
        Called once per MCTS root. Returns futures + consistency scores.
        """
        # 1. Bella imagines K futures
        futures = self.bella.generate_forward(
            current_prices, self.H, self.K, rng_key
        )  # (K, H, n_assets, 5)
        
        # 2. Encode all futures via TRM encoder (real observations)
        #    For each branch k, each step t: encode the sliding window
        #    This gives latent trajectories: (K, H, state_dim=128)
        bella_latents = self._encode_trajectories(params, current_prices, futures)
        
        # 3. Dynamics MLP predicts what latent_{t+1} SHOULD be
        #    For each step: dynamics(latent_t, oracle_action) → predicted_latent_{t+1}
        #    Compare predicted vs Bella's actual encoded latent_{t+1}
        consistency_scores = self._score_consistency(params, bella_latents)
        # Shape: (K,) — per-branch plausibility score
        # Low consistency = Bella hallucinated something the MLP doesn't believe
        # High consistency = Bella's future aligns with learned market dynamics
        
        return futures, bella_latents, consistency_scores
    
    def _score_consistency(self, params, bella_latents):
        """Score each branch by dynamics MLP alignment.
        
        bella_latents: (K, H, state_dim) — encoded Bella futures
        Returns: (K,) consistency scores (higher = more grounded)
        """
        K, H, D = bella_latents.shape
        total_mse = jnp.zeros(K)
        
        for t in range(H - 1):
            latent_t = bella_latents[:, t, :]       # (K, D)
            latent_t1 = bella_latents[:, t+1, :]    # (K, D) — Bella's version
            
            # What dynamics MLP thinks t+1 should look like
            # (Use oracle action at t for now — or policy's argmax)
            pred_t1 = dynamics_predict(params, latent_t, action_t)  # (K, D)
            
            # Per-branch MSE accumulation
            step_mse = jnp.mean((pred_t1 - latent_t1) ** 2, axis=-1)  # (K,)
            total_mse += step_mse
        
        # Convert to score: low MSE = high consistency
        return jnp.exp(-total_mse / H)  # (K,) in [0, 1]
    
    def expand_node(self, bella_latents, branch_k, step_t):
        """MCTS node expansion from pre-generated + pre-encoded future."""
        return bella_latents[branch_k, step_t]  # (state_dim,) — real encoded obs
```

### MCTS Search Modification

```python
# Current MCTS: expand via dynamics MLP
#   latent_{t+1} = dynamics(latent_t, action)  ← MLP approximation

# New MCTS: expand via Bella futures, weighted by consistency
#   At root: generate K futures, score each
#   Node expansion: index into pre-generated bella_latents
#   Prior weighting: multiply branch prior by consistency_score
#   This naturally downweights hallucinated branches
#   while letting creative but plausible futures through

search_weights = consistency_scores * value_estimates  # (K,)
# MCTS concentrates simulations on branches that are:
#   1. High value (good trading outcomes)
#   2. High consistency (dynamics MLP agrees it's realistic)
```

### Why Both Are Needed

| Bella only | MLP only | Bella + MLP |
|------------|----------|-------------|
| Rich, diverse futures | Toy 1-step prediction | Rich futures, grounded |
| Can hallucinate (loss < 2 doesn't mean perfect) | Consistency ~1.34, limited capacity | Hallucinations filtered by MLP |
| No grounding signal | No imagination | Imagination + physics check |
| MCTS could chase mirages | MCTS explores tiny latent neighborhood | MCTS explores realistic space |

### Step 4: Training Loop

**Training uses both Bella and dynamics MLP simultaneously:**

```
For each batch:
  1. Bella generates K=8 futures per sample (conditioned on current window)
  2. Dynamics MLP scores each branch (consistency)
  3. MCTS searches over Bella futures (weighted by consistency + value)
  4. Training targets:
     - Policy: MCTS visit counts (distill planning into single-step)
     - Value: actual PnL on Bella-generated prices (oracle can evaluate)
     - Dynamics: consistency loss on Bella's latent trajectories
       → dynamics learns from Bella's realistic transitions
       → this is the distillation: Bella teaches dynamics what real transitions look like
     - BC: oracle regularizer (decaying, same as current)
```

**The dynamics MLP improves by training on Bella's trajectories** — instead of learning from historical data only, it now learns from Bella's diverse generated scenarios. Over time, the MLP becomes a better filter, which makes MCTS search better, which makes policy better. Virtuous cycle.

---

## Performance Budget

```
Current MCTS per batch (256 samples, 32 sims):
  encode:   ~5ms
  search:   ~100ms (32 sims × dynamics_MLP per expansion)
  total:    ~105ms

With Bella rollouts:
  Pre-generate K=8 futures per sample:
    Bella ODE: 10 steps × K=8 × H=20 steps
    If batched: (256*8, ws+20, 64*5) = (2048, 84, 320)
    Bella forward: ~50-200ms per batch (flow matching is fast)
    BUT: this is 2048 sequences — may need chunking
  
  MCTS expansion (encode pre-generated windows):
    Just indexing + TRM encoder — same as current ~5ms
  
  Estimated total: ~200-500ms per batch (dominated by Bella generation)
  vs current: ~105ms
  
  Tradeoff: 2-5x slower but dynamics are real, not MLP approximations
```

**Mitigation**: Generate futures once per epoch (not per step). Each epoch uses the same set of K futures per slot position. Amortizes Bella cost over 400 steps.

---

## VRAM Budget

```
Current Phase 2:
  Model params:          ~200MB
  MCTS search buffers:   ~500MB
  Batch data:            ~100MB
  Total:                 ~800MB (fits RTX 3090 easily)

With Bella (fp32):
  Bella params:          ~1.2GB (302M params × 4 bytes)
  Bella generation:      ~500MB working memory
  Pre-generated futures: K=8 × 256 × 20 × 64 × 5 × 4 = 52MB (small)
  Total addition:        ~1.7GB
  Combined:              ~2.5GB (fits 3090's 24GB easily)

With Bella (int8, last resort):
  Bella params:          ~302MB (302M × 1 byte)
  Bella generation:      ~300MB working memory
  Total addition:        ~650MB
  Combined:              ~1.5GB
```

**Quantization fallback**: Bella can be quantized to int8 if VRAM becomes tight (e.g., if batch size or rollout count increases). Flow matching models are relatively robust to quantization since the ODE integration averages over many small steps — single-step noise gets smoothed out. Quality impact likely small but needs validation against stylized facts. Not needed at current VRAM projections (~2.5GB vs 24GB available), but feasible if needed.

---

## Skeleton Extraction from Observed Window

The simplest path uses existing `diff_teacher/skeleton/extraction.py`:

```python
from diff_teacher.skeleton.extraction import extract_market_skeleton

def window_to_skeleton(prices_window):
    """Convert observed price window → MarketSkeleton for Bella conditioning.
    
    prices_window: (ws, n_assets, 5) OHLCV
    """
    close = prices_window[:, :, 3]  # (ws, n_assets)
    returns = jnp.diff(close, axis=0) / jnp.maximum(close[:-1], 1e-10)
    
    skeleton = extract_market_skeleton(
        close_prices=close,
        returns=returns,
        n_timesteps=ws,
        n_assets=prices_window.shape[1],
    )
    return skeleton
```

This reuses Bella's trained conditioning pathway — no new parameters needed.

---

## Acceptance Criteria

- [ ] ContextConditioner or skeleton extraction from observed window works
- [ ] Bella generates short forward trajectories (H=20-50 steps) conditioned on context
- [ ] Inpainting maintains consistency between observed and generated portions
- [ ] K=8 rollouts produce visually distinct but plausible futures
- [ ] MCTS can search over pre-generated Bella futures
- [ ] Value/policy heads produce reasonable outputs on Bella-generated observations
- [ ] Training converges with Bella dynamics (policy loss decreasing)
- [ ] Option C distillation: dynamics MLP trained on Bella rollout latents

## Open Questions

1. **How many rollouts K?** 8 gives good coverage, but MCTS with 32 sims over 8 branches = 4 sims per branch. May need K=4 or K=16 depending on search quality.
2. **Inpainting vs concatenation?** Inpainting is more principled but adds complexity. Simple approach: just condition on skeleton + append context as prefix.
3. **Action conditioning?** Bella generates futures independent of action. The MLP handles action-dependence via `dynamics(latent, action)`. So Bella provides the "world" and the MLP tells you "what happens if you act." This separation may actually be correct — markets don't care about your trades (at our scale).
4. **Distribution shift?** Bella was trained on full segments, not short windows. Generated forwards may have different statistics at short horizons. Validate with stylized facts checker on forward portions only.
5. **Consistency threshold?** Should MCTS hard-reject branches below some consistency score, or soft-weight everything? Hard rejection is simpler but loses edge cases where Bella sees something the MLP hasn't learned yet.
6. **Virtuous cycle convergence?** Bella teaches dynamics → dynamics filters Bella → MCTS improves → policy improves. Does this converge or oscillate? May need to monitor consistency score distribution across epochs.
