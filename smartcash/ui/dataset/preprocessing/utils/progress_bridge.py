"""
File: smartcash/ui/dataset/preprocessing/utils/progress_bridge.py
Deskripsi: Fixed progress bridge dengan proper UI integration dan debug logging
"""

from typing import Dict, Any, Optional, Callable

class ProgressBridge:
    """Fixed bridge antara service progress dan UI 3-level progress system."""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components = ui_components
        self.logger = logger
        self.operation_type = None
        self.debug_mode = True  # Enable debug untuk troubleshooting
    
    def setup_for_operation(self, operation: str):
        """Setup progress untuk operation tertentu dengan debug info."""
        self.operation_type = operation
        
        if self.debug_mode and self.logger:
            self.logger.debug(f"🔧 Progress bridge setup untuk operation: {operation}")
        
        if 'show_for_operation' in self.ui_components and callable(self.ui_components['show_for_operation']):
            # Map operation ke progress config yang sesuai
            operation_map = {
                'preprocessing': 'download',  # Use download config for 3-level
                'cleanup': 'cleanup',
                'validation': 'check'
            }
            
            mapped_operation = operation_map.get(operation, 'all')
            
            try:
                self.ui_components['show_for_operation'](mapped_operation)
                if self.debug_mode and self.logger:
                    self.logger.debug(f"✅ Progress container ditampilkan untuk: {mapped_operation}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Error showing progress container: {str(e)}")
        else:
            if self.logger:
                self.logger.warning("⚠️ show_for_operation tidak tersedia atau tidak callable")
    
    def update_progress(self, **kwargs):
        """Update 3-level progress dari service callback dengan improved error handling."""
        try:
            # Extract progress data dari service callback
            overall_progress = kwargs.get('overall_progress', kwargs.get('progress', 0))
            step = kwargs.get('step', 0)
            split_name = kwargs.get('split', kwargs.get('split_step', ''))
            current_progress = kwargs.get('current_progress', 0)
            current_total = kwargs.get('current_total', 0)
            message = kwargs.get('message', 'Processing...')
            status = kwargs.get('status', 'info')
            
            if self.debug_mode and self.logger and overall_progress % 10 == 0:  # Log setiap 10%
                self.logger.debug(f"📊 Progress update: {overall_progress}% - {message}")
            
            # Check if update_progress function exists and is callable
            if 'update_progress' not in self.ui_components:
                if self.logger:
                    self.logger.warning("⚠️ update_progress function tidak ditemukan di ui_components")
                return
            
            update_func = self.ui_components['update_progress']
            if not callable(update_func):
                if self.logger:
                    self.logger.warning("⚠️ update_progress bukan callable function")
                return
            
            # === LEVEL 1: Overall Progress ===
            try:
                operation_name = self.operation_type or 'Processing'
                update_func('overall', int(overall_progress), f"{operation_name}: {message}")
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"🔧 Error updating overall progress: {str(e)}")
            
            # === LEVEL 2: Step Progress ===
            if step > 0:
                try:
                    step_messages = {
                        1: "📋 Persiapan",
                        2: "🔄 Pemrosesan Data", 
                        3: "✅ Finalisasi"
                    }
                    step_message = step_messages.get(step, f"Step {step}")
                    step_percentage = int((step / 3) * 100)
                    update_func('step', step_percentage, step_message)
                except Exception as e:
                    if self.logger:
                        self.logger.debug(f"🔧 Error updating step progress: {str(e)}")
            
            # === LEVEL 3: Current Progress ===
            if current_total > 0:
                try:
                    current_percentage = int((current_progress / current_total) * 100)
                    current_message = f"{split_name} files" if split_name else "Files"
                    update_func('current', current_percentage, f"{current_progress}/{current_total} {current_message}")
                except Exception as e:
                    if self.logger:
                        self.logger.debug(f"🔧 Error updating current progress: {str(e)}")
            
            # Log ke UI berdasarkan status (hanya untuk status penting)
            if self.logger and message.strip() and status in ['success', 'error', 'warning']:
                if status == 'success':
                    self.logger.success(message)
                elif status == 'error':
                    self.logger.error(message)
                elif status == 'warning':
                    self.logger.warning(message)
                    
        except Exception as e:
            # Silent fail untuk prevent recursive errors, tapi log untuk debugging
            if self.logger:
                self.logger.debug(f"🔧 Progress bridge error: {str(e)}")
    
    def complete_operation(self, message: str = "Operation completed"):
        """Complete operation dengan success state dan debug info."""
        try:
            if self.debug_mode and self.logger:
                self.logger.debug(f"🎉 Completing operation: {message}")
            
            if 'complete_operation' in self.ui_components and callable(self.ui_components['complete_operation']):
                self.ui_components['complete_operation'](message)
            else:
                if self.logger:
                    self.logger.warning("⚠️ complete_operation function tidak tersedia")
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error completing operation: {str(e)}")
    
    def error_operation(self, message: str = "Operation failed"):
        """Set error state untuk operation dengan debug info."""
        try:
            if self.debug_mode and self.logger:
                self.logger.debug(f"💥 Error operation: {message}")
            
            if 'error_operation' in self.ui_components and callable(self.ui_components['error_operation']):
                self.ui_components['error_operation'](message)
            else:
                if self.logger:
                    self.logger.warning("⚠️ error_operation function tidak tersedia")
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error setting error state: {str(e)}")
    
    def debug_ui_components(self):
        """Debug function untuk troubleshooting UI components."""
        if not self.logger:
            return
        
        self.logger.debug("🔍 Debug UI Components:")
        progress_functions = ['show_for_operation', 'update_progress', 'complete_operation', 'error_operation']
        
        for func_name in progress_functions:
            if func_name in self.ui_components:
                func = self.ui_components[func_name]
                is_callable = callable(func)
                self.logger.debug(f"   • {func_name}: {'✅ callable' if is_callable else '❌ not callable'}")
            else:
                self.logger.debug(f"   • {func_name}: ❌ missing")

def get_progress_bridge(ui_components: Dict[str, Any], logger=None) -> ProgressBridge:
    """Factory function untuk mendapatkan progress bridge dengan debug info."""
    if 'progress_bridge' not in ui_components:
        bridge = ProgressBridge(ui_components, logger)
        ui_components['progress_bridge'] = bridge
        
        # Debug UI components saat inisialisasi
        if logger:
            logger.debug("🔧 Creating new progress bridge")
            bridge.debug_ui_components()
    
    return ui_components['progress_bridge']