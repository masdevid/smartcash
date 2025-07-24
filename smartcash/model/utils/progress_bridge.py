"""
File: smartcash/model/utils/progress_bridge.py
Deskripsi: Bridge untuk integrasi Progress Tracker dengan model operations
"""

from typing import Optional, Callable, Dict, Any
from smartcash.common.logger import get_logger

class ModelProgressBridge:
    """🌉 Bridge untuk menghubungkan model operations dengan Progress Tracker API"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize progress bridge dengan callback ke UI Progress Tracker
        
        Args:
            progress_callback: Callback function yang compatible dengan Progress Tracker API
                             Format: callback(level, current, total, message, phase)
        """
        self.logger = get_logger("model.progress_bridge")
        self.progress_callback = progress_callback
        
        # State tracking
        self.current_operation = ""
        self.operation_steps = 0
        self.current_step = 0
        self.substep_current = 0
        self.substep_total = 0
        
        self.logger.debug("🌉 ModelProgressBridge initialized")
    
    def start_operation(self, operation_name: str, total_steps: int) -> None:
        """🚀 Start operation baru dengan total steps"""
        self.current_operation = operation_name
        self.operation_steps = total_steps
        self.current_step = 0
        self.substep_current = 0
        self.substep_total = 0
        
        self.logger.info(f"🚀 Starting: {operation_name} ({total_steps} steps)")
        self._notify_progress(0, total_steps, f"🚀 Starting {operation_name}...", "overall")
    
    def update(self, step: int, message: str, phase: str = "current") -> None:
        """📊 Update progress untuk main operation steps"""
        self.current_step = step
        
        progress_percentage = (step / max(self.operation_steps, 1)) * 100
        self.logger.debug(f"📊 {self.current_operation}: {progress_percentage:.1f}% - {message}")
        
        self._notify_progress(step, self.operation_steps, message, phase)
    
    def update_substep(self, substep: int, substep_total: int, message: str, phase: str = "current") -> None:
        """📋 Update progress untuk substeps dalam main step"""
        self.substep_current = substep
        self.substep_total = substep_total
        
        # Calculate combined progress: main step + substep progress
        main_progress = self.current_step
        substep_progress = (substep / max(substep_total, 1)) * 0.8  # 80% of current step
        combined_current = main_progress + substep_progress
        
        self.logger.debug(f"📋 Substep {substep}/{substep_total}: {message}")
        self._notify_progress(int(combined_current * 10), self.operation_steps * 10, message, phase)
    
    def complete(self, final_step: int, message: str) -> None:
        """✅ Mark operation sebagai complete"""
        self.current_step = final_step
        
        self.logger.info(f"✅ {self.current_operation} completed: {message}")
        self._notify_progress(final_step, self.operation_steps, message, "overall")
        
        # Reset state
        self._reset_state()
    
    def error(self, error_message: str, phase: str = "current") -> None:
        """❌ Report error dalam operation"""
        self.logger.error(f"❌ {self.current_operation} error: {error_message}")
        
        # Notify error melalui callback jika support error handling
        if self.progress_callback:
            try:
                # Try to call error method jika progress tracker support
                if hasattr(self.progress_callback, 'error'):
                    self.progress_callback.error(error_message, phase)
                elif callable(self.progress_callback):
                    # Fallback: call as regular progress dengan error indicator
                    self.progress_callback("error", 0, 1, f"❌ {error_message}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error calling progress callback: {str(e)}")
        
        self._reset_state()
    
    def set_callback(self, callback: Callable) -> None:
        """🔄 Set atau update progress callback"""
        self.progress_callback = callback
        self.logger.debug(f"🔄 Progress callback updated: {type(callback).__name__ if hasattr(callback, '__name__') else 'callable'}")
    
    def _notify_progress(self, current: int, total: int, message: str, phase: str) -> None:
        """📡 Notify progress ke callback dengan error handling"""
        if not self.progress_callback:
            return
        
        try:
            # Support different callback formats
            if hasattr(self.progress_callback, 'update'):
                # Progress Tracker object with update method
                self.progress_callback.update(current, total, message, phase)
            elif callable(self.progress_callback):
                # Function callback
                self.progress_callback(phase, current, total, message)
            else:
                self.logger.warning("⚠️ Invalid progress callback format")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error calling progress callback: {str(e)}")
    
    def _reset_state(self) -> None:
        """🔄 Reset internal state after operation"""
        self.current_operation = ""
        self.operation_steps = 0
        self.current_step = 0
        self.substep_current = 0
        self.substep_total = 0
    
    def get_progress_status(self) -> Dict[str, Any]:
        """📊 Get current progress status"""
        if not self.current_operation:
            return {"status": "idle", "message": "No operation in progress"}
        
        main_progress = (self.current_step / max(self.operation_steps, 1)) * 100
        substep_progress = (self.substep_current / max(self.substep_total, 1)) * 100 if self.substep_total > 0 else 0
        
        return {
            "status": "active",
            "operation": self.current_operation,
            "main_progress": main_progress,
            "current_step": self.current_step,
            "total_steps": self.operation_steps,
            "substep_progress": substep_progress,
            "substep_current": self.substep_current,
            "substep_total": self.substep_total
        }


class ProgressContext:
    """🎯 Context manager untuk automatic progress tracking"""
    
    def __init__(self, bridge: ModelProgressBridge, operation_name: str, total_steps: int):
        self.bridge = bridge
        self.operation_name = operation_name
        self.total_steps = total_steps
    
    def __enter__(self):
        self.bridge.start_operation(self.operation_name, self.total_steps)
        return self.bridge
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.bridge.complete(self.total_steps, f"✅ {self.operation_name} completed successfully")
        else:
            self.bridge.error(f"❌ {self.operation_name} failed: {str(exc_val)}")


# Convenience functions
def create_progress_bridge(callback: Optional[Callable] = None) -> ModelProgressBridge:
    """🏭 Factory function untuk membuat ModelProgressBridge"""
    return ModelProgressBridge(callback)

def progress_context(bridge: ModelProgressBridge, operation: str, steps: int) -> ProgressContext:
    """🎯 Create progress context manager"""
    return ProgressContext(bridge, operation, steps)