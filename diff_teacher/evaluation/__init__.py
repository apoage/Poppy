"""Evaluation metrics for generated market data."""

from .stylized_facts import (
    compute_stylized_facts,
    fat_tail_index,
    volatility_clustering,
    leverage_effect,
    autocorrelation,
)
from .fidelity import (
    wasserstein_distance,
    correlation_matrix_error,
    distribution_divergence,
)
from .controllability import (
    regime_adherence,
    volatility_targeting,
    correlation_targeting,
)
from .report import generate_evaluation_report

__all__ = [
    'compute_stylized_facts',
    'fat_tail_index',
    'volatility_clustering',
    'leverage_effect',
    'autocorrelation',
    'wasserstein_distance',
    'correlation_matrix_error',
    'distribution_divergence',
    'regime_adherence',
    'volatility_targeting',
    'correlation_targeting',
    'generate_evaluation_report',
]
