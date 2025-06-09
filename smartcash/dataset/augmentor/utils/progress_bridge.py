"""
File: smartcash/dataset/augmentor/utils/progress_bridge.py
Deskripsi: Bridge untuk kompatibilitas dengan dual/triple progress tracker API
"""

from typing import Optional, Callable, Any

class ProgressBridge:
    """🌉 Bridge untuk kompatibilitas dengan progress tracker API yang berbeda"""
    
    def __init__(self, progress_tracker=None):
        self.tracker = progress_tracker
        self._detect_tracker_type()
    
    def _detect_tracker_type(self):
        """Deteksi tipe progress tracker dan capabilities"""
        if not self.tracker:
            self.tracker_type = 'none'
            return
        
        # Check methods yang tersedia
        has_update_overall = hasattr(self.tracker, 'update_overall')
        has_update_step = hasattr(self.tracker, 'update_step') 
        has_update_current = hasattr(self.tracker, 'update_current')
        has_update = hasattr(self.tracker, 'update')
        
        if has_update_overall and has_update_step and has_update_current:
            self.tracker_type = 'triple'  # Triple progress tracker
        elif has_update_overall and (has_update_step or has_update_current):
            self.tracker_type = 'dual'    # Dual progress tracker
        elif has_update:
            self.tracker_type = 'single'  # Single progress tracker
        else:
            self.tracker_type = 'custom'  # Custom implementation
    
    def update(self, level: str, current: int, total: int, message: str = ""):
        """Universal update method dengan level mapping"""
        if not self.tracker:
            return
        
        try:
            if self.tracker_type == 'triple':
                self._update_triple(level, current, total, message)
            elif self.tracker_type == 'dual':
                self._update_dual(level, current, total, message)
            elif self.tracker_type == 'single':
                self._update_single(level, current, total, message)
            else:
                self._update_custom(level, current, total, message)
                
        except Exception:
            pass  # Silent fail untuk compatibility
    
    def _update_triple(self, level: str, current: int, total: int, message: str):
        """Update untuk triple progress tracker"""
        if level == 'overall':
            self.tracker.update_overall(current, message)
        elif level == 'step':
            self.tracker.update_step(current, message)
        elif level == 'current':
            self.tracker.update_current(current, message)
        else:
            # Default ke overall
            self.tracker.update_overall(current, message)
    
    def _update_dual(self, level: str, current: int, total: int, message: str):
        """Update untuk dual progress tracker"""
        if level == 'overall' and hasattr(self.tracker, 'update_overall'):
            self.tracker.update_overall(current, message)
        elif level in ['step', 'current']:
            if hasattr(self.tracker, 'update_step'):
                self.tracker.update_step(current, message)
            elif hasattr(self.tracker, 'update_current'):
                self.tracker.update_current(current, message)
        else:
            # Fallback ke method yang tersedia
            if hasattr(self.tracker, 'update_overall'):
                self.tracker.update_overall(current, message)
    
    def _update_single(self, level: str, current: int, total: int, message: str):
        """Update untuk single progress tracker"""
        if hasattr(self.tracker, 'update'):
            # Try dengan berbagai signature
            try:
                self.tracker.update(current, total, message)
            except TypeError:
                try:
                    self.tracker.update(current, message)
                except TypeError:
                    self.tracker.update(current)
    
    def _update_custom(self, level: str, current: int, total: int, message: str):
        """Update untuk custom tracker"""
        # Try common method names
        for method_name in ['progress', 'set_progress', 'update_progress']:
            if hasattr(self.tracker, method_name):
                try:
                    method = getattr(self.tracker, method_name)
                    method(current, total, message)
                    return
                except Exception:
                    continue
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Start operation jika tracker support"""
        if hasattr(self.tracker, 'start_operation'):
            try:
                self.tracker.start_operation(operation_name, total_steps)
            except Exception:
                pass
    
    def complete_operation(self, operation_name: str, message: str = ""):
        """Complete operation jika tracker support"""
        if hasattr(self.tracker, 'complete_operation'):
            try:
                self.tracker.complete_operation(operation_name, message)
            except Exception:
                pass
    
    def error_operation(self, operation_name: str, error_message: str):
        """Error operation jika tracker support"""
        if hasattr(self.tracker, 'error_operation'):
            try:
                self.tracker.error_operation(operation_name, error_message)
            except Exception:
                pass


def create_progress_bridge(progress_tracker=None) -> ProgressBridge:
    """🏭 Factory untuk create progress bridge"""
    return ProgressBridge(progress_tracker)

def make_progress_callback(progress_bridge: ProgressBridge) -> Callable:
    """🔗 Create callback function dari progress bridge"""
    def callback(level: str, current: int, total: int, message: str = ""):
        progress_bridge.update(level, current, total, message)
    return callback