"""
Training components module.

Provides specialized components for training pipeline operations
that are too specific to be in core but too detailed for orchestration.
"""

from .phase_setup_manager import PhaseSetupManager, create_phase_setup_manager

__all__ = [
    'PhaseSetupManager',
    'create_phase_setup_manager'
]