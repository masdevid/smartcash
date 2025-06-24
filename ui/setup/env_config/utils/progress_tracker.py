"""
File: smartcash/ui/setup/env_config/utils/progress_tracker.py
Deskripsi: Utility untuk tracking progress setup
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.utils.ui_updater import update_progress_bar, update_status_panel

class SetupProgressTracker:
    """📊 Progress tracker untuk setup workflow"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.current_step = 0
        self.total_steps = 4
    
    def update_step(self, step_description: str, progress_percent: int) -> None:
        """🔄 Update progress step"""
        # Update progress tracker
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            if hasattr(tracker, 'children') and len(tracker.children) >= 3:
                # Update current progress
                current_progress = tracker.children[1]
                if hasattr(current_progress, 'value'):
                    current_progress.value = progress_percent / 100.0
                    current_progress.description = f"Current: {step_description}"
                
                # Update total progress  
                total_progress = tracker.children[2]
                if hasattr(total_progress, 'value'):
                    total_progress.value = progress_percent / 100.0
        
        # Update status panel
        update_status_panel(
            self.ui_components['status_panel'], 
            f"{step_description} ({progress_percent}%)", 
            "info"
        )

def track_setup_progress(ui_components: Dict[str, Any]) -> SetupProgressTracker:
    """🎯 Create progress tracker instance"""
    return SetupProgressTracker(ui_components)