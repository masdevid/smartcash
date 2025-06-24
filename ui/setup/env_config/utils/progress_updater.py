# File: smartcash/ui/setup/env_config/utils/progress_updater.py
# Deskripsi: Utility untuk progress bar updates

from typing import Dict, Any

class ProgressUpdater:
    """📊 Utility untuk progress tracking"""
    
    def update_progress(self, ui_components: Dict[str, Any], value: int, message: str):
        """Update progress bar dan message"""
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = value
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = message