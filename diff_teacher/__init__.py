"""
Diffusion Teacher - Latent Flow Matching Market Generator for RL Curriculum

A standalone module for generating realistic market sequences using Flow Matching
and providing adaptive curriculum learning via regret-based difficulty control.

Components:
- DataLoader: Reads historical market windows from trading_data.db
- Encoder: Transformer-based encoder for 64-asset market data
- FlowMatchingGenerator: Single-step generative model for market sequences
- OptimalSolver: Hindsight oracle computing maximum achievable PnL
- RegretCurriculum: Difficulty controller based on (optimal - actual) PnL
- Teacher: Main class orchestrating the feedback loop

Usage:
    python -m diff_teacher.train --db-path trading_data.db --epochs 1000
"""

__version__ = "0.1.0"

from .config import TeacherConfig
from .data_loader import MarketDataLoader
from .encoder import MarketEncoder
from .flow_matching import FlowMatchingGenerator
from .optimal_solver import OptimalSolver
from .regret_curriculum import RegretCurriculum
from .teacher import DiffusionTeacher

__all__ = [
    "TeacherConfig",
    "MarketDataLoader",
    "MarketEncoder",
    "FlowMatchingGenerator",
    "OptimalSolver",
    "RegretCurriculum",
    "DiffusionTeacher",
]
