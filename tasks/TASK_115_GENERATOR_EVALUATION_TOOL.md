# TASK_115: Generator Evaluation Tool

## Overview
Implement a comprehensive evaluation script for the `diff_teacher` market generators (Apothecary: Bella). The tool will quantify the fidelity of generated synthetic data compared to real historical data using stylized facts and statistical distance metrics.

## Functionality
- **Checkpoint Loading**: Load `MarketGenerator` models (CNN/Mamba) from `checkpoints/`.
- **Reference Sampling**: Use `RainbowTableLoader` to fetch ground-truth normalized data.
- **Metric Suite**:
    - **Stylized Facts**: Fat tails (Hill estimator), Volatility Clustering, Leverage Effect, Autocorrelation.
    - **Fidelity**: Wasserstein Distance, Correlation Matrix Error, KL Divergence.
    - **Controllability**: (If applicable) Regime adherence and volatility targeting accuracy.
- **Reporting**: Generate a structured console report and save results to `logs/reports/`.

## Usage
```bash
python scripts/evaluate_generator.py --checkpoint checkpoints/diff_teacher/generator_best --samples 64
```

## Status
- [ ] Script implementation
- [ ] Integration with `RainbowTableLoader`
- [ ] Verification against CNN `generator_best` checkpoint
- [ ] Output logging to `logs/reports/`
