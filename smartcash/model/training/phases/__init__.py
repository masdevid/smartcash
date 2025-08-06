"""
Phase management module for SmartCash training.

This module provides a modular phase management system with proper inheritance
hierarchy and mixins for different training phase responsibilities.
"""

# Main training executor
from .training_executor import TrainingPhaseExecutor

# Phase setup manager
from .phase_setup_manager import PhaseSetupManager, create_phase_setup_manager

# Core components  
from .base import BasePhaseManager
from .mixins.metrics_processing import MetricsProcessingMixin
# Note: ModelConfigurationMixin and ComponentSetupMixin integrated into PipelineOrchestrator
from .mixins.progress_tracking import ProgressTrackingMixin
from .mixins.callbacks import CallbacksMixin
# Note: Phase orchestration is now handled by PipelineOrchestrator

__all__ = [
    'TrainingPhaseExecutor',
    'PhaseSetupManager',
    'create_phase_setup_manager',
    'BasePhaseManager', 
    'MetricsProcessingMixin',
    'ProgressTrackingMixin',
    'CallbacksMixin'
]