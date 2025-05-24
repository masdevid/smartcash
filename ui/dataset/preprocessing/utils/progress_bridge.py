"""
File: smartcash/ui/dataset/preprocessing/utils/progress_bridge.py
Deskripsi: Bridge utility untuk menghubungkan service progress dengan UI 3-level progress system
"""

from typing import Dict, Any, Optional, Callable

class ProgressBridge:
    """Bridge antara service progress dan UI 3-level progress system."""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components = ui_components
        self.logger = logger
        self.operation_type = None
    
    def setup_for_operation(self, operation: str):
        """Setup progress untuk operation tertentu."""
        self.operation_type = operation
        if 'show_for_operation' in self.ui_components:
            # Map operation ke progress config yang sesuai
            operation_map = {
                'preprocessing': 'download',  # Use download config for 3-level
                'cleanup': 'cleanup',
                'validation': 'check'
            }
            self.ui_components['show_for_operation'](operation_map.get(operation, 'all'))
    
    def update_progress(self, **kwargs):
        """Update 3-level progress dari service callback."""
        try:
            # Extract progress data dari service callback
            overall_progress = kwargs.get('overall_progress', kwargs.get('progress', 0))
            step = kwargs.get('step', 0)
            split_name = kwargs.get('split', kwargs.get('split_step', ''))
            current_progress = kwargs.get('current_progress', 0)
            current_total = kwargs.get('current_total', 0)
            message = kwargs.get('message', 'Processing...')
            status = kwargs.get('status', 'info')
            
            # === LEVEL 1: Overall Progress ===
            if 'update_progress' in self.ui_components:
                operation_name = self.operation_type or 'Processing'
                self.ui_components['update_progress']('overall', overall_progress, f"{operation_name}: {message}")
            
            # === LEVEL 2: Step Progress ===
            if step > 0:
                step_messages = {
                    1: "📋 Persiapan",
                    2: "🔄 Pemrosesan Data", 
                    3: "✅ Finalisasi"
                }
                step_message = step_messages.get(step, f"Step {step}")
                if 'update_progress' in self.ui_components:
                    self.ui_components['update_progress']('step', int((step/3)*100), step_message)
            
            # === LEVEL 3: Current Progress ===
            if current_total > 0:
                current_percentage = int((current_progress / current_total) * 100)
                current_message = f"{split_name} files" if split_name else "Files"
                if 'update_progress' in self.ui_components:
                    self.ui_components['update_progress']('current', current_percentage, current_message)
            
            # Log ke UI berdasarkan status
            if self.logger and message.strip():
                if status == 'success':
                    self.logger.success(message)
                elif status == 'error':
                    self.logger.error(message)
                elif status == 'warning':
                    self.logger.warning(message)
                else:
                    self.logger.info(message)
                    
        except Exception:
            # Silent fail untuk prevent recursive errors
            pass
    
    def complete_operation(self, message: str = "Operation completed"):
        """Complete operation dengan success state."""
        if 'complete_operation' in self.ui_components:
            self.ui_components['complete_operation'](message)
    
    def error_operation(self, message: str = "Operation failed"):
        """Set error state untuk operation."""
        if 'error_operation' in self.ui_components:
            self.ui_components['error_operation'](message)

def get_progress_bridge(ui_components: Dict[str, Any], logger=None) -> ProgressBridge:
    """Factory function untuk mendapatkan progress bridge."""
    if 'progress_bridge' not in ui_components:
        ui_components['progress_bridge'] = ProgressBridge(ui_components, logger)
    return ui_components['progress_bridge']