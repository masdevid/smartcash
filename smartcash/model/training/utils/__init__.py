"""
Training utilities module.

Provides utility functions for training pipeline operations including
checkpoint management, progress tracking, metrics handling, and more.
"""

# Core utilities that are frequently used
from .progress_tracker import TrainingProgressTracker, create_training_progress_bridge
from .signal_handler import install_training_signal_handlers, register_cleanup_callback
from .setup_utils import prepare_training_environment
from .resume_utils import handle_resume_training_pipeline, validate_training_mode_and_params
from .weight_transfer import create_weight_transfer_manager
from .summary_utils import generate_markdown_summary

# Note: Checkpoint utilities have been migrated to smartcash.model.core.checkpoints.CheckpointManager
# Use CheckpointManager.check_for_resumable_checkpoint() and CheckpointManager.load_checkpoint_for_resume() instead

# Metrics and history utilities  
from .metrics_history import create_metrics_recorder
from .metrics_utils import calculate_multilayer_metrics, filter_phase_relevant_metrics

__all__ = [
    # Core utilities
    'TrainingProgressTracker',
    'create_training_progress_bridge', 
    'install_training_signal_handlers',
    'register_cleanup_callback',
    'prepare_training_environment',
    'handle_resume_training_pipeline',
    'validate_training_mode_and_params',
    'create_weight_transfer_manager',
    'generate_markdown_summary',
    
    # Note: Checkpoint utilities moved to core.checkpoints.CheckpointManager
    
    # Metrics utilities
    'create_metrics_recorder',
    'calculate_multilayer_metrics',
    'filter_phase_relevant_metrics'
]