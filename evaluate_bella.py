#!/usr/bin/env python
"""
Bella (CNN Flow Matching Generator) Quality Evaluation

TASK_111: Evaluate the trained CNN diffusion model for market generation.

Metrics:
- Stylized facts (fat tails, volatility clustering, leverage effect)
- Fidelity (distribution similarity to real data)
- Generation quality (visual inspection)
"""

import numpy as np
import jax
print(f"JAX devices: {jax.devices()}")

from diff_teacher.generation import MarketGenerator
from diff_teacher.evaluation import compute_stylized_facts
from diff_teacher.evaluation.stylized_facts import compare_stylized_facts, EXPECTED_STYLIZED_FACTS

# Configuration
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints/diff_teacher/final/generator")
CONFIG_PATH = os.path.join(BASE_DIR, "checkpoints/diff_teacher/final/config.json")
N_SAMPLES = 5
SEED = 42

print("=" * 60)
print("BELLA QUALITY EVALUATION")
print("=" * 60)

# 1. Load the model
print("\n[1/4] Loading Bella from checkpoint...")
try:
    import json
    from diff_teacher.generation.generator import GeneratorConfig

    # Load config from parent directory
    with open(CONFIG_PATH) as f:
        config_dict = json.load(f)

    # Map config to GeneratorConfig
    config = GeneratorConfig(
        architecture=config_dict.get('flow_architecture', 'conv'),
        hidden_dim=config_dict.get('flow_hidden_dim', 768),
        num_layers=config_dict.get('flow_layers', 12),
        n_assets=config_dict.get('n_assets', 64),
        n_features=len(config_dict.get('features', ['open', 'high', 'low', 'close', 'volume'])),
        seq_length=config_dict.get('window_size', 400),
        n_steps=10,
        sigma_min=config_dict.get('sigma_min', 1e-4),
    )

    generator = MarketGenerator.from_checkpoint(CHECKPOINT_PATH, config=config)
    print(f"  Architecture: {generator.config.architecture}")
    print(f"  Hidden dim: {generator.config.hidden_dim}")
    print(f"  Num layers: {generator.config.num_layers}")
    print(f"  Seq length: {generator.config.seq_length}")
    print(f"  N assets: {generator.config.n_assets}")
    print("  Model loaded successfully!")
except Exception as e:
    print(f"  ERROR: Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. Generate samples
print(f"\n[2/4] Generating {N_SAMPLES} market samples...")
try:
    generated = generator.generate(n_samples=N_SAMPLES, seed=SEED)
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated range: [{generated.min():.4f}, {generated.max():.4f}]")
    print(f"  Generated mean: {generated.mean():.4f}")
    print(f"  Generated std: {generated.std():.4f}")

    # Check for NaN/Inf
    n_nan = np.isnan(generated).sum()
    n_inf = np.isinf(generated).sum()
    print(f"  NaN count: {n_nan}, Inf count: {n_inf}")

    if n_nan > 0 or n_inf > 0:
        print("  WARNING: Generated data contains NaN/Inf values!")
except Exception as e:
    print(f"  ERROR: Failed to generate samples: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. Compute stylized facts
print("\n[3/4] Computing stylized facts...")
try:
    # Use first sample for analysis
    sample = generated[0]  # (seq_length, n_assets, n_features)

    facts = compute_stylized_facts(sample)

    print("\n  Stylized Facts Report:")
    print("  " + "-" * 40)
    print(f"  Fat Tail Index:        {facts.fat_tail_index:.3f}  (expected: ~3.0)")
    print(f"  Volatility Clustering: {facts.volatility_clustering:.3f}  (expected: >0.1)")
    print(f"  Leverage Effect:       {facts.leverage_effect:.3f}  (expected: ~-0.3)")
    print(f"  Autocorrelation (1):   {facts.autocorrelation_lag1:.3f}  (expected: ~0)")
    print(f"  Autocorrelation (5):   {facts.autocorrelation_lag5:.3f}")
    print(f"  Mean Return:           {facts.mean_return:.6f}")
    print(f"  Std Return:            {facts.std_return:.6f}")
    print(f"  Skewness:              {facts.skewness:.3f}")
    print(f"  Kurtosis:              {facts.kurtosis:.3f}  (expected: >3)")
except Exception as e:
    print(f"  ERROR: Failed to compute stylized facts: {e}")
    import traceback
    traceback.print_exc()

# 4. Quality assessment
print("\n[4/4] Quality Assessment...")
try:
    # Score each metric
    scores = {}

    # Fat tail index: ideal is ~3, score based on distance
    if not np.isnan(facts.fat_tail_index):
        score = max(0, 1 - abs(facts.fat_tail_index - 3.0) / 3.0)
        scores['fat_tails'] = score
        print(f"  Fat Tails Score:       {score:.2f} (1.0 = perfect)")

    # Volatility clustering: should be positive
    if not np.isnan(facts.volatility_clustering):
        score = 1.0 if facts.volatility_clustering > 0.05 else facts.volatility_clustering / 0.05
        scores['vol_clustering'] = max(0, score)
        print(f"  Vol Clustering Score:  {scores['vol_clustering']:.2f}")

    # Kurtosis: should be > 3 (excess kurtosis)
    if not np.isnan(facts.kurtosis):
        score = min(1.0, (facts.kurtosis - 3) / 10) if facts.kurtosis > 3 else 0
        scores['kurtosis'] = max(0, score)
        print(f"  Kurtosis Score:        {scores['kurtosis']:.2f}")

    # Low autocorrelation in returns (efficient market)
    if not np.isnan(facts.autocorrelation_lag1):
        score = max(0, 1 - abs(facts.autocorrelation_lag1) * 5)
        scores['market_efficiency'] = score
        print(f"  Market Efficiency:     {scores['market_efficiency']:.2f}")

    # Overall score
    if scores:
        overall = np.mean(list(scores.values()))
        print(f"\n  OVERALL QUALITY SCORE: {overall:.2f} / 1.00")

        if overall > 0.7:
            print("  Assessment: GOOD - Ready for training!")
        elif overall > 0.5:
            print("  Assessment: ACCEPTABLE - May need tuning")
        else:
            print("  Assessment: NEEDS WORK - Consider more training")
except Exception as e:
    print(f"  ERROR: Failed quality assessment: {e}")
    import traceback
    traceback.print_exc()

# 5. Summary statistics across all samples
print("\n" + "=" * 60)
print("CROSS-SAMPLE CONSISTENCY")
print("=" * 60)

try:
    all_stds = []
    all_kurtosis = []

    for i in range(N_SAMPLES):
        sample = generated[i]
        facts_i = compute_stylized_facts(sample)
        all_stds.append(facts_i.std_return)
        all_kurtosis.append(facts_i.kurtosis)

    print(f"  Std Return across samples:  mean={np.mean(all_stds):.6f}, std={np.std(all_stds):.6f}")
    print(f"  Kurtosis across samples:    mean={np.mean(all_kurtosis):.2f}, std={np.std(all_kurtosis):.2f}")

    consistency = 1 - np.std(all_stds) / (np.mean(all_stds) + 1e-6)
    print(f"  Consistency Score: {max(0, consistency):.2f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
