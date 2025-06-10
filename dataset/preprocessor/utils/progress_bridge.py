"""
File: smartcash/dataset/preprocessor/utils/progress_bridge.py
Deskripsi: Enhanced progress bridge dengan dual progress tracker compatibility
"""
from typing import Optional, Dict, Any, Callable

class ProgressBridge:
    """🌉 Enhanced bridge untuk dual progress tracker compatibility"""
    
    def __init__(self, progress_tracker=None):
        self.progress_tracker = progress_tracker
        self.current_progress = 0
        self.status = "idle"
        self.messages = []
        self.dual_callback = None
    
    def register_dual_callback(self, callback: Callable[[str, int, int, str], None]):
        """📊 Register callback untuk dual progress tracker"""
        self.dual_callback = callback
    
    def update(self, level: str, current: int, total: int, message: str = None) -> None:
        """🔄 Enhanced update dengan dual tracker support"""
        # Calculate progress percentage
        if isinstance(level, str) and level in ['overall', 'current', 'step']:
            # Dual tracker format
            progress_pct = (current / total) * 100 if total > 0 else 0
            self.current_progress = max(0, min(100, progress_pct))
            
            if message:
                self.status = message
                self.messages.append({"level": level, "message": message, "progress": progress_pct})
            
            # Call dual tracker callback
            if self.dual_callback:
                try:
                    self.dual_callback(level, current, total, message or self.status)
                except Exception:
                    pass
            
            # Call legacy progress tracker
            if self.progress_tracker and hasattr(self.progress_tracker, 'update'):
                try:
                    self.progress_tracker.update(
                        progress=self.current_progress,
                        status=self.status,
                        message=message
                    )
                except Exception:
                    pass
        
        else:
            # Legacy format (progress as float 0-1)
            progress = level if isinstance(level, (int, float)) else current
            status = current if isinstance(current, str) else message
            
            self.current_progress = max(0, min(100, progress * 100 if progress <= 1 else progress))
            
            if status:
                self.status = status
                self.messages.append({"status": status, "message": status, "progress": self.current_progress})
            
            # Call legacy progress tracker
            if self.progress_tracker and hasattr(self.progress_tracker, 'update'):
                try:
                    self.progress_tracker.update(
                        progress=self.current_progress,
                        status=status,
                        message=status
                    )
                except Exception:
                    pass
    
    def update_overall(self, progress: int, message: str = None):
        """📊 Update overall progress (dual tracker compatibility)"""
        self.update("overall", progress, 100, message)
    
    def update_current(self, progress: int, message: str = None):
        """⚡ Update current operation progress (dual tracker compatibility)"""
        self.update("current", progress, 100, message)
    
    def update_step(self, current: int, total: int, message: str = None):
        """🔄 Update step progress (dual tracker compatibility)"""
        self.update("step", current, total, message)
    
    def get_progress(self) -> Dict[str, Any]:
        """📈 Get current progress state"""
        return {
            "progress": self.current_progress,
            "status": self.status,
            "messages": self.messages[-10:]  # Keep last 10 messages
        }
    
    def reset(self) -> None:
        """🔄 Reset progress tracker"""
        self.current_progress = 0
        self.status = "idle"
        self.messages = []
        
        # Reset dual tracker jika ada
        if self.dual_callback:
            try:
                self.dual_callback("overall", 0, 100, "Reset")
            except Exception:
                pass
    
    def complete(self, message: str = "Completed"):
        """✅ Mark operation as completed"""
        self.update("overall", 100, 100, message)
        self.status = "completed"
    
    def error(self, message: str = "Error occurred"):
        """❌ Mark operation as error"""
        self.status = "error"
        self.messages.append({"status": "error", "message": message, "progress": self.current_progress})
        
        if self.dual_callback:
            try:
                self.dual_callback("overall", 0, 100, message)
            except Exception:
                pass

# Factory function
def create_progress_bridge(progress_tracker=None) -> ProgressBridge:
    """🏭 Factory untuk create progress bridge"""
    return ProgressBridge(progress_tracker)