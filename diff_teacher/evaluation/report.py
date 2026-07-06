"""Generate comprehensive evaluation reports."""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import json

from .stylized_facts import compute_stylized_facts, StylizedFactsReport
from .fidelity import wasserstein_distance, correlation_matrix_error, distribution_divergence
from .controllability import compute_controllability, ControllabilityReport


@dataclass
class EvaluationReport:
    """Complete evaluation report for generated data."""

    # Stylized facts
    stylized_facts: StylizedFactsReport

    # Fidelity metrics (vs reference)
    wasserstein_distance: Optional[float]
    correlation_error: Optional[float]
    kl_divergence: Optional[float]

    # Controllability (vs skeleton)
    controllability: Optional[ControllabilityReport]

    # Summary scores
    fidelity_score: float  # Lower is better
    controllability_score: float  # Higher is better

    def to_dict(self) -> dict:
        return {
            'stylized_facts': self.stylized_facts.to_dict(),
            'wasserstein_distance': self.wasserstein_distance,
            'correlation_error': self.correlation_error,
            'kl_divergence': self.kl_divergence,
            'controllability': self.controllability.to_dict() if self.controllability else None,
            'fidelity_score': self.fidelity_score,
            'controllability_score': self.controllability_score,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=float)


def generate_evaluation_report(
    generated: np.ndarray,
    reference: Optional[np.ndarray] = None,
    skeleton: Optional[Any] = None,
) -> EvaluationReport:
    """
    Generate comprehensive evaluation report.

    Args:
        generated: Generated price data (timesteps, assets, [features])
        reference: Optional real data for fidelity comparison
        skeleton: Optional MarketSkeleton for controllability evaluation

    Returns:
        EvaluationReport with all metrics
    """
    # Stylized facts
    stylized = compute_stylized_facts(generated)

    # Fidelity metrics (if reference provided)
    if reference is not None:
        w_dist = wasserstein_distance(generated, reference)
        corr_err = correlation_matrix_error(generated, reference)
        kl_div = distribution_divergence(generated, reference)
    else:
        w_dist = None
        corr_err = None
        kl_div = None

    # Controllability (if skeleton provided)
    if skeleton is not None:
        controllability = compute_controllability(generated, skeleton)
    else:
        controllability = None

    # Compute summary scores
    fidelity_score = _compute_fidelity_score(stylized, w_dist, corr_err, kl_div)
    controllability_score = _compute_controllability_score(controllability)

    return EvaluationReport(
        stylized_facts=stylized,
        wasserstein_distance=w_dist,
        correlation_error=corr_err,
        kl_divergence=kl_div,
        controllability=controllability,
        fidelity_score=fidelity_score,
        controllability_score=controllability_score,
    )


def _compute_fidelity_score(
    stylized: StylizedFactsReport,
    w_dist: Optional[float],
    corr_err: Optional[float],
    kl_div: Optional[float],
) -> float:
    """
    Compute aggregate fidelity score.

    Lower is better.
    """
    score = 0.0
    count = 0

    # Stylized facts deviations from expected
    # Fat tail index should be ~3
    if not np.isnan(stylized.fat_tail_index):
        score += abs(stylized.fat_tail_index - 3.0) / 3.0
        count += 1

    # Kurtosis should be >3 (fat tails)
    if not np.isnan(stylized.kurtosis):
        if stylized.kurtosis < 3:
            score += (3 - stylized.kurtosis) / 3.0
        count += 1

    # Volatility clustering should be positive
    if not np.isnan(stylized.volatility_clustering):
        if stylized.volatility_clustering < 0:
            score += abs(stylized.volatility_clustering)
        count += 1

    # Return autocorrelation should be near zero
    if not np.isnan(stylized.autocorrelation_lag1):
        score += abs(stylized.autocorrelation_lag1)
        count += 1

    # Fidelity metrics
    if w_dist is not None:
        score += min(w_dist * 100, 1.0)  # Scale appropriately
        count += 1

    if corr_err is not None:
        score += min(corr_err / 10, 1.0)  # Scale appropriately
        count += 1

    if kl_div is not None:
        score += min(kl_div, 1.0)
        count += 1

    return score / max(count, 1)


def _compute_controllability_score(
    controllability: Optional[ControllabilityReport],
) -> float:
    """
    Compute aggregate controllability score.

    Higher is better.
    """
    if controllability is None:
        return 0.0

    scores = []

    if not np.isnan(controllability.regime_adherence):
        scores.append(controllability.regime_adherence)

    if not np.isnan(controllability.volatility_targeting_rmse):
        # Convert RMSE to score (lower RMSE = higher score)
        scores.append(max(0, 1 - controllability.volatility_targeting_rmse * 10))

    if not np.isnan(controllability.correlation_targeting_error):
        # Negative error is good
        scores.append(max(0, 1 + controllability.correlation_targeting_error))

    if not np.isnan(controllability.trend_direction_accuracy):
        scores.append(controllability.trend_direction_accuracy)

    return np.mean(scores) if scores else 0.0


def print_report(report: EvaluationReport) -> str:
    """Format report as readable string."""
    lines = [
        "=" * 60,
        "MARKET GENERATION EVALUATION REPORT",
        "=" * 60,
        "",
        "STYLIZED FACTS",
        "-" * 40,
        f"  Fat Tail Index:       {report.stylized_facts.fat_tail_index:.3f} (target: ~3.0)",
        f"  Volatility Clustering: {report.stylized_facts.volatility_clustering:.3f} (should be >0)",
        f"  Leverage Effect:      {report.stylized_facts.leverage_effect:.3f} (should be <0)",
        f"  Return ACF(1):        {report.stylized_facts.autocorrelation_lag1:.3f} (should be ~0)",
        f"  Kurtosis:             {report.stylized_facts.kurtosis:.3f} (should be >3)",
        "",
    ]

    if report.wasserstein_distance is not None:
        lines.extend([
            "FIDELITY (vs Reference)",
            "-" * 40,
            f"  Wasserstein Distance: {report.wasserstein_distance:.6f}",
            f"  Correlation Error:    {report.correlation_error:.3f}",
            f"  KL Divergence:        {report.kl_divergence:.4f}",
            "",
        ])

    if report.controllability is not None:
        lines.extend([
            "CONTROLLABILITY",
            "-" * 40,
            f"  Regime Adherence:     {report.controllability.regime_adherence:.3f}",
            f"  Volatility RMSE:      {report.controllability.volatility_targeting_rmse:.6f}",
            f"  Correlation Error:    {report.controllability.correlation_targeting_error:.3f}",
            f"  Trend Accuracy:       {report.controllability.trend_direction_accuracy:.3f}",
            "",
        ])

    lines.extend([
        "SUMMARY SCORES",
        "-" * 40,
        f"  Fidelity Score:       {report.fidelity_score:.3f} (lower is better)",
        f"  Controllability:      {report.controllability_score:.3f} (higher is better)",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
