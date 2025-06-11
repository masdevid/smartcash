"""
File: smartcash/dataset/preprocessor/utils/progress_bridge.py
Deskripsi: Progress bridge yang compatible dengan Progress Tracker API
"""

from typing import Dict, Any, Optional, Callable, List
import time

from smartcash.common.logger import get_logger

class PreprocessingProgressBridge:
    """🌉 Progress bridge untuk preprocessing operations"""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        self.logger = get_logger(__name__)
        self.ui_components = ui_components or {}
        self.callbacks = []
        self.last_update = {}
        self.throttle_ms = 100
        
        # Setup progress tracker integration
        self.progress_tracker = ui_components.get('progress_tracker') if ui_components else None
        
        # Split processing state
        self.current_split = None
        self.current_split_index = 0
        self.total_splits = 0
        self.splits_list = []
    
    def setup_split_processing(self, splits: List[str]):
        """🎯 Setup split processing context"""
        self.splits_list = splits
        self.total_splits = len(splits)
        self.current_split_index = 0
    
    def start_split(self, split_name: str):
        """🎬 Start processing specific split"""
        self.current_split = split_name
        if split_name in self.splits_list:
            self.current_split_index = self.splits_list.index(split_name) + 1
        
        # Overall progress: "Preprocessing {split} 1/3"
        overall_message = f"Preprocessing {split_name} {self.current_split_index}/{self.total_splits}"
        self._notify_progress('overall', self.current_split_index, self.total_splits, overall_message)
    
    def update_split_progress(self, current: int, total: int, message: str = ""):
        """🔄 Update current split progress"""
        if self.current_split:
            # Split step: "{split} step 1/100"
            split_message = f"{self.current_split} step {current}/{total}"
            if message:
                split_message += f" - {message}"
            
            self._notify_progress('current', current, total, split_message)
    
    def complete_split(self, split_name: str):
        """✅ Complete current split"""
        # Final split progress
        if self.current_split:
            split_message = f"{split_name} completed"
            self._notify_progress('current', 100, 100, split_message)
    
    def register_callback(self, callback: Callable[[str, int, int, str], None]):
        """📝 Register progress callback"""
        if callback and callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def _notify_progress(self, level: str, current: int, total: int, message: str):
        """📢 Notify semua callbacks dan progress tracker"""
        # Throttling check
        now = time.time()
        if (now - self.last_update.get(level, 0)) < (self.throttle_ms / 1000):
            return
        
        self.last_update[level] = now
        
        # Progress Tracker integration
        if self.progress_tracker:
            try:
                if hasattr(self.progress_tracker, f'update_{level}'):
                    getattr(self.progress_tracker, f'update_{level}')(current, message)
                elif hasattr(self.progress_tracker, 'update'):
                    percentage = (current / total) * 100 if total > 0 else 0
                    self.progress_tracker.update(level, percentage, message)
            except Exception as e:
                self.logger.debug(f"⚠️ Progress tracker error: {str(e)}")
        
        # Custom callbacks
        for callback in self.callbacks:
            try:
                callback(level, current, total, message)
            except Exception as e:
                self.logger.debug(f"⚠️ Progress callback error: {str(e)}")
    
    def reset(self):
        """🔄 Reset all progress"""
        self.current_split = None
        self.current_split_index = 0
        self._notify_progress('overall', 0, self.total_splits, "Reset")
        self._notify_progress('current', 0, 100, "Reset")

def create_preprocessing_bridge(ui_components: Dict[str, Any] = None) -> PreprocessingProgressBridge:
    """🏭 Factory untuk create preprocessing progress bridge"""
    return PreprocessingProgressBridge(ui_components)

# === Compatibility functions ===
def create_progress_bridge(ui_components: Dict[str, Any] = None):
    """🔄 Compatibility alias"""
    return create_preprocessing_bridge(ui_components)