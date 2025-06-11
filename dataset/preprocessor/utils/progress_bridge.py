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
        
        # Phase management
        self.phases = {
            'validation': {'weight': 20, 'current': 0},
            'processing': {'weight': 70, 'current': 0},
            'finalization': {'weight': 10, 'current': 0}
        }
        self.current_phase = None
    
    def register_callback(self, callback: Callable[[str, int, int, str], None]):
        """📝 Register progress callback"""
        if callback and callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def start_phase(self, phase_name: str, message: str = ""):
        """🎬 Start new processing phase"""
        if phase_name in self.phases:
            self.current_phase = phase_name
            self._update_overall_from_phases()
            self.update('overall', self.phases[phase_name]['current'], 100, 
                       message or f"Starting {phase_name}")
    
    def update(self, level: str, current: int, total: int = 100, message: str = ""):
        """🔄 Update progress dengan throttling"""
        # Throttling check
        now = time.time()
        if (now - self.last_update.get(level, 0)) < (self.throttle_ms / 1000):
            return
        
        self.last_update[level] = now
        
        # Update phase progress
        if self.current_phase and level == 'current':
            phase_progress = (current / total) * 100 if total > 0 else 0
            self.phases[self.current_phase]['current'] = phase_progress
            overall_progress = self._calculate_overall_progress()
            self._notify_progress('overall', overall_progress, 100, message)
        
        # Notify all callbacks
        self._notify_progress(level, current, total, message)
    
    def update_phase_progress(self, current: int, total: int = 100, message: str = ""):
        """🔄 Update current phase progress"""
        if self.current_phase:
            self.update('current', current, total, message)
    
    def complete_phase(self, phase_name: str, message: str = ""):
        """✅ Complete current phase"""
        if phase_name in self.phases:
            self.phases[phase_name]['current'] = 100
            overall_progress = self._calculate_overall_progress()
            self._notify_progress('overall', overall_progress, 100, 
                                message or f"Completed {phase_name}")
    
    def _calculate_overall_progress(self) -> int:
        """📊 Calculate overall progress dari phases"""
        total_weight = sum(phase['weight'] for phase in self.phases.values())
        weighted_progress = sum(
            (phase['current'] / 100) * phase['weight'] 
            for phase in self.phases.values()
        )
        return int((weighted_progress / total_weight) * 100)
    
    def _update_overall_from_phases(self):
        """🔄 Update overall progress dari phase weights"""
        overall_progress = self._calculate_overall_progress()
        self._notify_progress('overall', overall_progress, 100, f"Phase: {self.current_phase}")
    
    def _notify_progress(self, level: str, current: int, total: int, message: str):
        """📢 Notify semua callbacks dan progress tracker"""
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
        for phase in self.phases.values():
            phase['current'] = 0
        self.current_phase = None
        self._notify_progress('overall', 0, 100, "Reset")
        self._notify_progress('current', 0, 100, "Reset")

def create_preprocessing_bridge(ui_components: Dict[str, Any] = None) -> PreprocessingProgressBridge:
    """🏭 Factory untuk create preprocessing progress bridge"""
    return PreprocessingProgressBridge(ui_components)

# === Compatibility functions ===
def create_progress_bridge(ui_components: Dict[str, Any] = None):
    """🔄 Compatibility alias"""
    return create_preprocessing_bridge(ui_components)