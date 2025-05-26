"""
File: smartcash/ui/dataset/augmentation/handlers/progress_handler.py
Deskripsi: SRP handler untuk progress tracking dan UI updates
"""

from typing import Dict, Any, Optional, Callable

class ProgressHandler:
    """SRP handler untuk mengelola progress tracking augmentasi."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.tracker = ui_components.get('tracker')
        self.current_operation = None
        
    def start_operation(self, operation_name: str) -> bool:
        """
        Mulai operasi dengan progress tracking.
        
        Args:
            operation_name: Nama operasi (augmentation, cleanup, etc.)
            
        Returns:
            True jika berhasil dimulai
        """
        try:
            self.current_operation = operation_name
            
            # Show progress container untuk operation
            if 'show_for_operation' in self.ui_components:
                self.ui_components['show_for_operation'](operation_name)
            elif self.tracker and hasattr(self.tracker, 'show'):
                self.tracker.show(operation_name)
                
            return True
        except Exception:
            return False
    
    def update_progress(self, step: str, percentage: int, message: str = "", 
                       color: Optional[str] = None) -> bool:
        """
        Update progress dengan percentage dan message.
        
        Args:
            step: Step operasi ('overall', 'step', 'current')
            percentage: Percentage progress (0-100)
            message: Progress message
            color: Optional color untuk progress
            
        Returns:
            True jika berhasil diupdate
        """
        try:
            # Normalize percentage
            percentage = max(0, min(100, percentage))
            
            # Update tracker
            if 'update_progress' in self.ui_components:
                self.ui_components['update_progress'](step, percentage, message, color)
            elif self.tracker and hasattr(self.tracker, 'update'):
                self.tracker.update(step, percentage, message, color)
                
            return True
        except Exception:
            return False
    
    def complete_operation(self, message: str = "Operasi selesai") -> bool:
        """
        Complete operasi dengan success state.
        
        Args:
            message: Message untuk completion
            
        Returns:
            True jika berhasil completed
        """
        try:
            if 'complete_operation' in self.ui_components:
                self.ui_components['complete_operation'](message)
            elif self.tracker and hasattr(self.tracker, 'complete'):
                self.tracker.complete(message)
                
            self.current_operation = None
            return True
        except Exception:
            return False
    
    def error_operation(self, message: str = "Error operasi") -> bool:
        """
        Set error state untuk operasi.
        
        Args:
            message: Error message
            
        Returns:
            True jika berhasil diset error
        """
        try:
            if 'error_operation' in self.ui_components:
                self.ui_components['error_operation'](message)
            elif self.tracker and hasattr(self.tracker, 'error'):
                self.tracker.error(message)
                
            return True
        except Exception:
            return False
    
    def hide_progress(self) -> bool:
        """
        Hide progress container.
        
        Returns:
            True jika berhasil dihide
        """
        try:
            if 'hide_container' in self.ui_components:
                self.ui_components['hide_container']()
            elif self.tracker and hasattr(self.tracker, 'hide'):
                self.tracker.hide()
                
            self.current_operation = None
            return True
        except Exception:
            return False
    
    def reset_progress(self) -> bool:
        """
        Reset semua progress state.
        
        Returns:
            True jika berhasil direset
        """
        try:
            if 'reset_all' in self.ui_components:
                self.ui_components['reset_all']()
            elif self.tracker and hasattr(self.tracker, 'reset'):
                self.tracker.reset()
                
            self.current_operation = None
            return True
        except Exception:
            return False
    
    def create_progress_callback(self) -> Callable:
        """
        Buat progress callback function untuk service.
        
        Returns:
            Callback function untuk progress updates
        """
        def progress_callback(step: str, current: int, total: int, message: str):
            """Progress callback untuk service integration."""
            try:
                # Hitung percentage
                percentage = int((current / max(1, total)) * 100) if total > 0 else current
                
                # Update progress
                self.update_progress(step, percentage, message)
                
                # Log message jika ada logger
                if 'logger' in self.ui_components:
                    logger = self.ui_components['logger']
                    if hasattr(logger, 'info'):
                        logger.info(f"📊 {step}: {message} ({percentage}%)")
                        
            except Exception:
                pass  # Silent fail untuk callback
        
        return progress_callback
    
    def get_progress_status(self) -> Dict[str, Any]:
        """
        Dapatkan current progress status.
        
        Returns:
            Dictionary dengan status progress
        """
        return {
            'current_operation': self.current_operation,
            'tracker_available': self.tracker is not None,
            'has_ui_methods': 'update_progress' in self.ui_components,
            'is_active': self.current_operation is not None
        }

# Factory function
def create_progress_handler(ui_components: Dict[str, Any]) -> ProgressHandler:
    """Factory function untuk create progress handler."""
    return ProgressHandler(ui_components)

# One-liner utilities
start_progress = lambda handler, operation: handler.start_operation(operation)
update_progress = lambda handler, step, pct, msg="": handler.update_progress(step, pct, msg)
complete_progress = lambda handler, msg="Selesai": handler.complete_operation(msg)
error_progress = lambda handler, msg="Error": handler.error_operation(msg)
create_callback = lambda handler: handler.create_progress_callback()