"""
File: smartcash/ui/components/progress_tracker/__init__.py
Deskripsi: Progress tracker module dengan SRP design pattern dan tqdm integration
"""

from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel, ProgressConfig, ProgressBarConfig
from smartcash.ui.components.progress_tracker.callback_manager import CallbackManager
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.factory import (
    create_single_progress_tracker,
    create_dual_progress_tracker,
    create_triple_progress_tracker,
    create_flexible_tracker,
    create_three_progress_tracker  # Backward compatibility
)

__all__ = [
    # Core classes
    'ProgressTracker',
    'ProgressLevel',
    'ProgressConfig',
    'ProgressBarConfig',
    'CallbackManager',
    'TqdmManager',
    
    # Factory functions
    'create_single_progress_tracker',
    'create_dual_progress_tracker',
    'create_triple_progress_tracker',
    'create_flexible_tracker',
    'create_three_progress_tracker'
]