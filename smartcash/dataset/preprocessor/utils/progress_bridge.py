"""
File: smartcash/dataset/preprocessor/utils/progress_bridge.py
Deskripsi: Enhanced Progress Bridge dengan proper callback system untuk UI dual progress tracker
"""
from typing import Optional, Dict, Any, Callable, List

class ProgressBridge:
    """🌉 Bridge untuk callback system ke UI dual progress tracker"""
    
    def __init__(self):
        self.callbacks: List[Callable] = []
        self.current_state = {
            'overall': {'current': 0, 'total': 100, 'message': 'Idle'},
            'current': {'current': 0, 'total': 100, 'message': 'Ready'},
            'step': {'current': 0, 'total': 100, 'message': 'Waiting'}
        }
    
    def register_callback(self, callback: Callable[[str, int, int, str], None]):
        """📊 Register callback function untuk UI updates"""
        if callback and callable(callback):
            self.callbacks.append(callback)
    
    def update(self, level: str, current: int, total: int, message: str = None) -> None:
        """🔄 Update progress dan notify semua callbacks"""
        # Validate level
        if level not in ['overall', 'current', 'step']:
            level = 'overall'
        
        # Update internal state
        self.current_state[level] = {
            'current': max(0, min(current, total)),
            'total': max(1, total),
            'message': message or self.current_state[level]['message']
        }
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                callback(level, current, total, message or '')
            except Exception:
                # Silent fail untuk prevent breaking the main process
                pass
    
    def update_overall(self, progress: int, message: str = None):
        """📊 Update overall progress (0-100)"""
        self.update("overall", progress, 100, message)
    
    def update_current(self, current: int, total: int, message: str = None):
        """⚡ Update current operation progress"""
        self.update("current", current, total, message)
    
    def update_step(self, current: int, total: int, message: str = None):
        """🔄 Update step progress"""
        self.update("step", current, total, message)
    
    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """📈 Get current progress state"""
        return self.current_state.copy()
    
    def reset(self) -> None:
        """🔄 Reset all progress"""
        for level in self.current_state:
            self.update(level, 0, 100, "Ready")
    
    def complete(self, message: str = "Completed"):
        """✅ Mark as completed"""
        self.update_overall(100, message)
    
    def error(self, message: str = "Error occurred"):
        """❌ Mark as error"""
        self.update_overall(0, message)

def create_progress_bridge() -> ProgressBridge:
    """🏭 Factory untuk create progress bridge"""
    return ProgressBridge()