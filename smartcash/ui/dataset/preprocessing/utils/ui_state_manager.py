"""
File: smartcash/ui/dataset/preprocessing/utils/ui_state_manager.py
Deskripsi: Fixed UI state manager dengan proper operation control dan button management
"""

from typing import Dict, Any, Optional, List
from contextlib import contextmanager

class UIStateManager:
    """Fixed manager untuk UI state dan button operations dengan proper control."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_states = {}
        self.button_original_states = {}
        self.debug_mode = True
        
        # Get logger for debugging
        self.logger = ui_components.get('logger')
        
        if self.debug_mode and self.logger:
            self.logger.debug("🔧 UIStateManager initialized")
    
    @contextmanager
    def operation_context(self, operation_name: str, button_key: str = None, 
                         processing_text: str = "Processing...", 
                         processing_style: str = 'warning'):
        """Context manager untuk button operation state yang lebih robust."""
        button = None
        if button_key:
            button = self.ui_components.get(button_key)
        
        if not button:
            # Jika tidak ada button, tetap jalankan operation
            yield
            return
        
        # Store original state
        original_disabled = button.disabled
        original_description = button.description
        original_style = button.button_style
        
        try:
            # Set processing state
            self._set_processing_state(operation_name, True)
            button.disabled = True
            button.description = processing_text
            button.button_style = processing_style
            
            if self.debug_mode and self.logger:
                self.logger.debug(f"🔄 Operation {operation_name} started, button {button_key} disabled")
            
            yield
            
        finally:
            # Restore original state
            self._set_processing_state(operation_name, False)
            button.disabled = original_disabled
            button.description = original_description
            button.button_style = original_style
            
            if self.debug_mode and self.logger:
                self.logger.debug(f"✅ Operation {operation_name} completed, button {button_key} restored")
    
    def set_button_processing(self, button_key: str, processing: bool = True, 
                            processing_text: str = "Processing...",
                            processing_style: str = 'warning',
                            success_text: str = None,
                            success_style: str = 'success'):
        """Set button ke processing state atau restore dengan improved state management."""
        button = self.ui_components.get(button_key)
        if not button:
            if self.debug_mode and self.logger:
                self.logger.warning(f"⚠️ Button {button_key} tidak ditemukan")
            return
        
        if processing:
            # Store original state jika belum ada
            if button_key not in self.button_original_states:
                self.button_original_states[button_key] = {
                    'disabled': button.disabled,
                    'description': button.description,
                    'button_style': button.button_style
                }
            
            button.disabled = True
            button.description = processing_text
            button.button_style = processing_style
            
            if self.debug_mode and self.logger:
                self.logger.debug(f"🔄 Button {button_key} set to processing state")
        else:
            # Restore or set success state
            if button_key in self.button_original_states:
                original = self.button_original_states[button_key]
                button.disabled = original['disabled']
                button.description = success_text or original['description']
                button.button_style = success_style if success_text else original['button_style']
                
                # Clear stored state
                del self.button_original_states[button_key]
                
                if self.debug_mode and self.logger:
                    self.logger.debug(f"✅ Button {button_key} restored from processing state")
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Check apakah operation sedang running."""
        return self.operation_states.get(operation_name, False)
    
    def get_running_operations(self) -> List[str]:
        """Get list of currently running operations."""
        return [op for op, running in self.operation_states.items() if running]
    
    def _set_processing_state(self, operation_name: str, processing: bool):
        """Set processing state untuk operation."""
        old_state = self.operation_states.get(operation_name, False)
        self.operation_states[operation_name] = processing
        
        # Set global state juga
        state_key = f'{operation_name}_running'
        self.ui_components[state_key] = processing
        
        if self.debug_mode and self.logger and old_state != processing:
            self.logger.debug(f"🔄 Operation {operation_name}: {old_state} → {processing}")
    
    def can_start_operation(self, operation_name: str, exclude_operations: List[str] = None) -> tuple[bool, str]:
        """Check apakah operation bisa dimulai dengan improved conflict detection."""
        exclude_operations = exclude_operations or []
        
        # Check if current operation already running
        if self.is_operation_running(operation_name):
            return False, f"{operation_name.title()} sedang berjalan"
        
        # Check conflicting operations
        running_ops = self.get_running_operations()
        conflicting_ops = [op for op in running_ops 
                          if op not in exclude_operations and op != operation_name]
        
        if conflicting_ops:
            return False, f"Tidak dapat memulai {operation_name}, {conflicting_ops[0]} sedang berjalan"
        
        if self.debug_mode and self.logger:
            self.logger.debug(f"✅ Operation {operation_name} can start (running: {running_ops})")
        
        return True, "Ready to start"
    
    def disable_other_buttons(self, current_button_key: str, button_keys: List[str] = None):
        """Disable other buttons during operation dengan default button list."""
        if button_keys is None:
            button_keys = ['preprocess_button', 'cleanup_button', 'check_button', 'save_button', 'reset_button']
        
        disabled_count = 0
        for button_key in button_keys:
            if button_key == current_button_key:
                continue
                
            button = self.ui_components.get(button_key)
            if button and not button.disabled:
                # Store original state
                if button_key not in self.button_original_states:
                    self.button_original_states[button_key] = {
                        'disabled': button.disabled,
                        'description': button.description,
                        'button_style': button.button_style
                    }
                
                button.disabled = True
                disabled_count += 1
        
        if self.debug_mode and self.logger:
            self.logger.debug(f"🔒 Disabled {disabled_count} other buttons during {current_button_key} operation")
    
    def enable_other_buttons(self, current_button_key: str, button_keys: List[str] = None):
        """Enable other buttons after operation selesai."""
        if button_keys is None:
            button_keys = ['preprocess_button', 'cleanup_button', 'check_button', 'save_button', 'reset_button']
        
        enabled_count = 0
        for button_key in button_keys:
            if button_key == current_button_key:
                continue
                
            button = self.ui_components.get(button_key)
            if button and button_key in self.button_original_states:
                original = self.button_original_states[button_key]
                button.disabled = original['disabled']
                
                # Clear stored state
                del self.button_original_states[button_key]
                enabled_count += 1
        
        if self.debug_mode and self.logger:
            self.logger.debug(f"🔓 Enabled {enabled_count} other buttons after {current_button_key} operation")
    
    def cleanup_all_states(self):
        """Cleanup semua operation states dan button states."""
        # Reset operation states
        for operation in list(self.operation_states.keys()):
            self._set_processing_state(operation, False)
        
        # Reset button states
        for button_key, original_state in list(self.button_original_states.items()):
            button = self.ui_components.get(button_key)
            if button:
                button.disabled = original_state['disabled']
                button.description = original_state['description']
                button.button_style = original_state['button_style']
        
        # Clear stored states
        self.button_original_states.clear()
        
        if self.debug_mode and self.logger:
            self.logger.debug("🧹 All UI states cleaned up")
    
    def debug_current_state(self):
        """Debug function untuk troubleshooting current state."""
        if not self.logger:
            return
        
        self.logger.debug("🔍 Current UI State:")
        self.logger.debug(f"   • Running operations: {self.get_running_operations()}")
        self.logger.debug(f"   • Buttons in processing: {list(self.button_original_states.keys())}")
        
        # Check button states
        button_keys = ['preprocess_button', 'cleanup_button', 'check_button', 'save_button', 'reset_button']
        for button_key in button_keys:
            button = self.ui_components.get(button_key)
            if button:
                self.logger.debug(f"   • {button_key}: {'disabled' if button.disabled else 'enabled'} - '{button.description}'")

def get_ui_state_manager(ui_components: Dict[str, Any]) -> UIStateManager:
    """Factory function untuk mendapatkan UI state manager."""
    if 'ui_state_manager' not in ui_components:
        ui_components['ui_state_manager'] = UIStateManager(ui_components)
    return ui_components['ui_state_manager']