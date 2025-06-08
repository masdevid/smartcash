"""
File: smartcash/ui/dataset/augmentation/utils/button_manager.py
Deskripsi: Enhanced button state management dengan visual feedback dan operation tracking
"""

from typing import Dict, Any, Optional, Callable
import time

class ButtonManager:
    """Enhanced button manager dengan state tracking dan visual feedback"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.button_states = {}
        self.operation_start_times = {}
        
        # Register buttons dengan original state
        self._register_buttons()
    
    def _register_buttons(self):
        """Register semua buttons dengan original state mereka"""
        button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        
        for key in button_keys:
            button = self.ui_components.get(key)
            if button and hasattr(button, 'description'):
                self.button_states[key] = {
                    'original_description': button.description,
                    'original_style': getattr(button, 'button_style', 'primary'),
                    'current_state': 'ready'
                }
    
    def set_processing_state(self, button_key: str, operation_name: str = None):
        """Set button ke processing state dengan visual feedback"""
        button = self.ui_components.get(button_key)
        if not button or not hasattr(button, 'disabled'):
            return
        
        # Track operation start time
        self.operation_start_times[button_key] = time.time()
        
        # Update button state
        original_desc = self.button_states.get(button_key, {}).get('original_description', button.description)
        operation_text = operation_name or 'Processing'
        
        button.disabled = True
        button.description = f"⏳ {operation_text}..."
        if hasattr(button, 'button_style'):
            button.button_style = 'warning'
        
        self.button_states[button_key]['current_state'] = 'processing'
    
    def set_success_state(self, button_key: str, duration: float = None):
        """Set button ke success state dengan duration feedback"""
        button = self.ui_components.get(button_key)
        if not button:
            return
        
        # Calculate duration jika tidak disediakan
        if duration is None and button_key in self.operation_start_times:
            duration = time.time() - self.operation_start_times[button_key]
        
        # Update ke success state
        original_desc = self.button_states.get(button_key, {}).get('original_description', button.description)
        duration_text = f" (✓ {duration:.1f}s)" if duration else " (✓)"
        
        button.disabled = False
        button.description = original_desc + duration_text
        if hasattr(button, 'button_style'):
            button.button_style = 'success'
        
        self.button_states[button_key]['current_state'] = 'success'
    
    def set_error_state(self, button_key: str, error_message: str = None):
        """Set button ke error state dengan error indication"""
        button = self.ui_components.get(button_key)
        if not button:
            return
        
        original_desc = self.button_states.get(button_key, {}).get('original_description', button.description)
        error_text = " (❌ Error)" if not error_message else f" (❌ {error_message})"
        
        button.disabled = False
        button.description = original_desc + error_text
        if hasattr(button, 'button_style'):
            button.button_style = 'danger'
        
        self.button_states[button_key]['current_state'] = 'error'
    
    def restore_button(self, button_key: str):
        """Restore button ke original state"""
        button = self.ui_components.get(button_key)
        button_state = self.button_states.get(button_key, {})
        
        if not button or not button_state:
            return
        
        # Restore original properties
        button.disabled = False
        button.description = button_state['original_description']
        if hasattr(button, 'button_style'):
            button.button_style = button_state['original_style']
        
        button_state['current_state'] = 'ready'
        
        # Clean up operation time tracking
        if button_key in self.operation_start_times:
            del self.operation_start_times[button_key]
    
    def restore_all_buttons(self):
        """Restore semua buttons ke original state"""
        for button_key in self.button_states.keys():
            self.restore_button(button_key)
    
    def disable_all_buttons(self):
        """Disable SEMUA buttons termasuk yang diklik saat operation berlangsung"""
        for button_key in self.button_states.keys():
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = True
    
    def enable_all_buttons(self):
        """Enable semua buttons kembali"""
        for button_key in self.button_states.keys():
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = False
    
    def get_button_state(self, button_key: str) -> str:
        """Get current state dari button"""
        return self.button_states.get(button_key, {}).get('current_state', 'unknown')
    
    def is_any_button_processing(self) -> bool:
        """Check apakah ada button yang sedang processing"""
        return any(state.get('current_state') == 'processing' for state in self.button_states.values())
    
    def get_operation_duration(self, button_key: str) -> Optional[float]:
        """Get duration dari operation yang sedang berjalan"""
        if button_key in self.operation_start_times:
            return time.time() - self.operation_start_times[button_key]
        return None

class OperationManager:
    """Manager untuk koordinasi operations antar buttons"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.button_manager = ButtonManager(ui_components)
        self.current_operation = None
        self.operation_queue = []
    
    def execute_operation(self, operation_name: str, button_key: str, 
                         operation_func: Callable, *args, **kwargs):
        """Execute operation dengan semua buttons disabled saat processing"""
        if self.current_operation:
            self._log_ui(f"⚠️ Operation {operation_name} sedang berjalan", 'warning')
            return
        
        self.current_operation = operation_name
        
        try:
            # Disable SEMUA buttons (termasuk yang diklik)
            self.button_manager.disable_all_buttons()
            self.button_manager.set_processing_state(button_key, operation_name)
            
            self._log_ui(f"🚀 Memulai {operation_name}...", 'info')
            
            # Execute operation
            result = operation_func(*args, **kwargs)
            
            # Handle result dan restore buttons
            if result and result.get('status') == 'success':
                duration = self.button_manager.get_operation_duration(button_key)
                self.button_manager.set_success_state(button_key, duration)
                self._log_ui(f"✅ {operation_name} berhasil dalam {duration:.1f}s", 'success')
            else:
                error_msg = result.get('message', 'Unknown error') if result else 'Operation failed'
                self.button_manager.set_error_state(button_key, 'Failed')
                self._log_ui(f"❌ {operation_name} gagal: {error_msg}", 'error')
            
            # Enable kembali semua buttons setelah operation selesai
            self.button_manager.enable_all_buttons()
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.button_manager.set_error_state(button_key, 'Error')
            self._log_ui(f"❌ {operation_name} error: {error_msg}", 'error')
            # Enable kembali buttons meskipun error
            self.button_manager.enable_all_buttons()
            return {'status': 'error', 'message': error_msg}
            
        finally:
            self.current_operation = None