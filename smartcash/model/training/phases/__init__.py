"""
Phase management module for SmartCash training.

This module provides a modular phase management system with proper inheritance
hierarchy and mixins for different training phase responsibilities.
"""

from .base import BasePhaseManager
from .mixins.metrics_processing import MetricsProcessingMixin
from .mixins.model_configuration import ModelConfigurationMixin
from .mixins.component_setup import ComponentSetupMixin
from .mixins.progress_tracking import ProgressTrackingMixin
from .orchestrator import PhaseOrchestrator
from .manager import TrainingPhaseManager
from .configurator import PhaseConfigurator
from .executor import PhaseExecutor

# For backward compatibility
from .manager import TrainingPhaseManager as TrainingPhaseManager
from .orchestrator import PhaseOrchestrator as PhaseOrchestrator

__all__ = [
    'BasePhaseManager',
    'MetricsProcessingMixin',
    'ModelConfigurationMixin', 
    'ComponentSetupMixin',
    'ProgressTrackingMixin',
    'PhaseOrchestrator',
    'TrainingPhaseManager',
    'PhaseConfigurator',
    'PhaseExecutor'
]