"""
File: smartcash/ui/setup/env_config/components/progress_setup.py
Deskripsi: Setup untuk progress tracking environment config
"""

from typing import Dict, Any

from smartcash.ui.handlers.single_progress import setup_progress_tracking

def setup_progress(ui_components: Dict[str, Any]) -> None:
    """
    Setup progress tracking
    
    Args:
        ui_components: Dictionary UI components
    """
    tracker = setup_progress_tracking(
        ui_components,
        tracker_name="env_config",
        progress_widget_key="progress",
        progress_label_key="progress_label",
        total=100,
        description="Environment Configuration"
    )
    
    if tracker and 'env_config_tracker' not in ui_components:
        ui_components['env_config_tracker'] = tracker 