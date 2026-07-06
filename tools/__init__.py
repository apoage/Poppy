"""
Tools package for TRM+MuZero Trading Agent.

Provides utility functions for all tools to resolve paths correctly.
"""

from pathlib import Path

# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT

def get_logs_dir() -> Path:
    """Get the logs directory."""
    return PROJECT_ROOT / "logs"

def get_checkpoint_dir() -> Path:
    """Get the checkpoint directory."""
    return PROJECT_ROOT / "jax_ckpt_fast"

def get_config_dir() -> Path:
    """Get the config directory."""
    return PROJECT_ROOT / "config"
