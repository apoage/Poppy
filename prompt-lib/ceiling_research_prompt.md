# Research Prompt: Breaking the 61% Agreement Ceiling

## Role

You are an expert ML researcher specializing in imitation learning, expert iteration, and financial time-series models. You are given a detailed description of a training system that has plateaued and must brainstorm actionable strategies to break through.

## System Description

### Task
Train a model to imitate an oracle (bounded-lookahead greedy path solver) on multi-asset trading decisions. The model sees 64 assets simultaneously and must output one of 5 actions per asset (SELL, HOLD, BUY, LimitSell, LimitBuy) = 320-dimensional action space.

### Architecture
- **Model**: TRM (Tiny Recursive Model) + MuZero-style heads
  - Encoder: V3 (minGRU temporal encoder) — processes (27 channels, 64 timesteps, 64 assets) observations
  - Reasoning: Recursive refinement with H=6 horizontal cycles, L=8 vertical cycles
  - Heads: Policy (320 logits), Value (401-bin categorical support over [-10, 10])
  - Total: 14.59M parameters
- **Encoder detail**: Per-timestep Dense(27→32) → minGRU(32) via associative scan → per-asset embeddings → cross-asset attention → global state (dim=64)

### Training Setup
- **Data**: On-the-fly synthetic price generation (SyntheticMarket L0-L14 curriculum + Bella flow-matching generator)
- **Oracle**: `greedy_lookahead_path` with k=4 (top-4 assets per step), lookahead=20 steps, transaction_cost=0.001, min_alpha=0.0005
- **Multipass**: 10 oracle passes with noise_scale=0.03 → soft probability targets
- **Sharpening**: temperature=0.1 post-averaging (TASK_677)
- **Loss**: Cross-entropy policy loss + categorical value loss (MSE ~0.6)
- **Optimizer**: AdamW, lr=3e-4, weight_decay=1e-4, warmup=1000 steps, grad_clip=1.0
- **Batch**: 1024 samples, 400 steps/epoch
- **Trade weighting**: Adaptive per-batch weight (~24×) to balance 96.1% hold / 3.9% trade ratio

### Current Performance (Epoch 930, ~11 hours training)
| Metric | Train | Val |
|--------|-------|-----|
| Balanced accuracy (THE metric) | 60.7% | 61.9% (best) |
| Trade accuracy (agree_trade) | 43.4% | 43.0% |
| Hold accuracy (agree_hold) | 77.8% | 78.8% |
| Loss | 3.13 | - |
| Value MSE | 0.60 | 0.58 |
| Grad norm | 13-16 | - |

### Plateau Evidence
- Best agreement 61.87% set at epoch 596
- Sharpening restart at epoch 596: only reached 61.9% by epoch 898, then flat again
- Gap between best-agreement epochs widening: 63→335→300+ epochs between improvements
- Train/val gap < 0.5% — NOT overfitting
- Grad norms healthy, no vanishing/exploding
- Loss stable at 3.13

### Hardware Constraints
- Single GPU: 24GB VRAM (22.7GB currently used)
- PCIe/southbridge stability issues under heavy load
- Epoch time: ~133s

### What's Been Tried
1. V2→V3 encoder (added temporal ordering) — broke through 52%→60%
2. k=1→k=4 oracle (more trade signals) — improved trade accuracy 9%→43%
3. Oracle multipass (soft targets) — smoother training
4. Oracle sharpening temp=0.1 — marginal effect (+0.03%)
5. Trade-biased sampling (15% of windows from trade-heavy regions)
6. Adaptive trade weighting (~24× for trades)
7. Warmup=0 (immediate full trade weight)
8. min_alpha 0.001→0.0005 (more trade density)

## Questions to Research

### 1. Is the ceiling architectural or data-driven?
- The model has 14.59M params for a 320-action space — is this sufficient?
- state_dim=64 compresses 64-asset information into a 64-dim vector. Is this a bottleneck?
- The encoder processes 27×64×64 → 64-dim state. Information compression ratio?
- Would scaling to 36M+ params (H=8, L=10, hidden=512) help or just slow training?

### 2. Oracle ambiguity analysis
- With k=4 lookahead=20, what fraction of targets are genuinely ambiguous?
- If multipass gives 60/40 splits, even a perfect model caps at ~60% on those targets
- Measured ambiguity data will be provided in `prompt-lib/out-001/ambiguity_results.txt`

### 3. Alternative training strategies
Consider and evaluate:
- **Curriculum scheduling**: Currently all L0-L14 mixed. Would phased curriculum help?
- **Hard example mining**: Focus on timesteps where oracle confidence is high but model fails
- **Distillation temperature**: Current CE loss on soft targets — would label smoothing or temperature scaling help?
- **Focal loss**: Down-weight easy (hold) examples, up-weight hard (trade) examples
- **Contrastive learning**: Learn embeddings that separate trade-worthy from hold states
- **Two-stage**: First learn hold/trade binary, then learn action within trade class

### 4. Scaling the model
Current architecture analysis (from exploration):
- Encoder V3: 0.14M params (tiny!)
- Reasoning (H=6, L=8): ~14M params (most of the model)
- Dynamics: 0.33M params

Possible "gigantic" configs:
| state_dim | H | L | Params | Est. VRAM (B=512) |
|-----------|---|---|--------|-------------------|
| 64 | 6 | 8 | 14.6M | 22.7GB (current) |
| 96 | 8 | 10 | ~50M | ? |
| 128 | 8 | 10 | ~100M | ? |
| 128 | 10 | 12 | ~150M | ? |

Questions:
- Is the bottleneck in reasoning depth (H×L) or width (state_dim)?
- Would a wider encoder (minGRU dim 32→64) help more than deeper reasoning?
- Can we fit a 50-100M model in 24GB with batch=512?
- Would mixed precision (bfloat16) help fit larger models?

### 5. Data quality investigation
- Synthetic vs real gap: Model at 61% on synthetic but 48.6% on real benchmark (TASK_682)
- Is the synthetic distribution too narrow? Too clean?
- Would mixing in real data (from DB) during training help?
- Bella generator quality — is it producing diverse enough regimes?

### 6. Loss function alternatives
- Current: CE on soft policy targets + categorical value loss
- Consider: KL divergence instead of CE for soft targets
- Consider: Separate loss heads for trade/hold binary classification + action refinement
- Consider: Auxiliary losses (predict volatility, regime, trend direction)

### 7. The hold/trade imbalance problem
- 96.1% holds, 3.9% trades — extreme class imbalance
- Adaptive weighting at 24× for trades
- agree_hold at 78% but agree_trade stuck at 43%
- Is the model learning "when NOT to trade" better than "when TO trade"?
- Would an asymmetric architecture (separate trade detector + action selector) help?

## Output Format

For each strategy, provide:
1. **Hypothesis**: What you think is happening and why this would help
2. **Expected impact**: Low/Medium/High improvement potential
3. **Implementation effort**: Low/Medium/High
4. **Risk**: What could go wrong
5. **Priority**: Rank 1-5 (1=try first)

Save your analysis to `prompt-lib/out-001/ceiling_analysis[n+1].md`.

## Context Files

Read these files before answering:
- `easy_config.py` — full training configuration
- `easy_trainer.py` — training loop and loss computation
- `mcts_trader/jax_agent/path_solver.py` — oracle implementation
- `mcts_trader/jax_agent/muzero_model.py` — model architecture
- `mcts_trader/jax_agent/trm_flax.py` — TRM reasoning network
- `mcts_trader/jax_agent/trm_encoder_v3.py` — V3 minGRU encoder
- `train_easy.py` — data generation and training entry point
- `prompt-lib/out-001/ambiguity_results.txt` — oracle ambiguity measurements (if available)
