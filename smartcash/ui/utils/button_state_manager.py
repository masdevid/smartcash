"""
File: smartcash/ui/utils/button_state_manager.py
Deskripsi: Generic button state manager untuk semua modules (download, preprocessing, augmentation, dll)
"""

from typing import Dict, Any, List, Optional, Set
from contextlib import contextmanager

class ButtonStateManager:
    """Generic button state manager untuk semua modules SmartCash"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_states = {}
        self.button_original_states = {}
        self.logger = ui_components.get('logger')
        
        # Auto-detect buttons dari ui_components (generic approach)
        self.managed_buttons = self._auto_detect_buttons()
        
        # Categorize buttons berdasarkan naming convention
        self.operation_buttons = self._get_operation_buttons()
        self.config_buttons = self._get_config_buttons()
        self.action_buttons = self._get_action_buttons()
    
    def _auto_detect_buttons(self) -> List[str]:
        """Auto-detect semua buttons dari ui_components (generic)"""
        button_keys = []
        
        # Common button patterns across all modules
        common_patterns = [
            # Operation buttons (main actions)
            'download_button', 'preprocess_button', 'augment_button', 'train_button', 'evaluate_button',
            
            # Secondary operation buttons
            'check_button', 'cleanup_button', 'validate_button', 'analyze_button', 'export_button',
            
            # Config buttons
            'save_button', 'reset_button', 'load_button', 'apply_button',
            
            # Navigation buttons
            'next_button', 'prev_button', 'back_button', 'continue_button',
            
            # File operation buttons
            'upload_button', 'import_button', 'browse_button'
        ]
        
        # Auto-detect from ui_components
        for key in self.ui_components.keys():
            if (key.endswith('_button') and 
                hasattr(self.ui_components.get(key), 'disabled') and
                hasattr(self.ui_components.get(key), 'on_click')):
                button_keys.append(key)
        
        # Add common patterns that exist
        for pattern in common_patterns:
            if (pattern in self.ui_components and 
                pattern not in button_keys and
                hasattr(self.ui_components.get(pattern), 'disabled')):
                button_keys.append(pattern)
        
        return button_keys
    
    def _get_operation_buttons(self) -> List[str]:
        """Get operation buttons (main actions yang harus di-disable saat operation lain berjalan)"""
        operation_patterns = [
            'download_button', 'preprocess_button', 'augment_button', 'train_button', 'evaluate_button',
            'check_button', 'cleanup_button', 'validate_button', 'analyze_button', 'export_button'
        ]
        return [btn for btn in self.managed_buttons if any(pattern in btn for pattern in operation_patterns)]
    
    def _get_config_buttons(self) -> List[str]:
        """Get config buttons (tetap aktif saat operation berjalan)"""
        config_patterns = ['save_button', 'reset_button', 'load_button', 'apply_button']
        return [btn for btn in self.managed_buttons if any(pattern in btn for pattern in config_patterns)]
    
    def _get_action_buttons(self) -> List[str]:
        """Get action buttons (navigation, file operations, dll)"""
        action_patterns = [
            'next_button', 'prev_button', 'back_button', 'continue_button',
            'upload_button', 'import_button', 'browse_button'
        ]
        return [btn for btn in self.managed_buttons if any(pattern in btn for pattern in action_patterns)]
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Generic context manager untuk semua operations"""
        try:
            # Set processing state
            self._set_processing_state(operation_name, True)
            
            # Disable operation buttons only (config buttons tetap aktif)
            self._disable_operation_buttons()
            
            if self.logger:
                self.logger.debug(f"🔒 Operation {operation_name} started - operation buttons disabled")
            
            yield
            
        except Exception as e:
            # Log error tapi jangan propagate agar UI tetap bisa di-restore
            if self.logger:
                self.logger.error(f"🔥 Error dalam operation {operation_name}: {str(e)}")
            raise
        finally:
            # Always restore states
            self._set_processing_state(operation_name, False)
            self._enable_operation_buttons()
            
            if self.logger:
                self.logger.debug(f"🔓 Operation {operation_name} completed - operation buttons restored")
    
    @contextmanager
    def config_context(self, config_operation: str):
        """Lightweight context untuk config operations (tidak disable operation buttons)"""
        try:
            self._set_processing_state(config_operation, True)
            
            if self.logger:
                self.logger.debug(f"🔧 Config operation {config_operation} started")
            
            yield
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"🔥 Error dalam config operation {config_operation}: {str(e)}")
            raise
        finally:
            self._set_processing_state(config_operation, False)
            
            if self.logger:
                self.logger.debug(f"✅ Config operation {config_operation} completed")
    
    def _disable_operation_buttons(self):
        """Disable operation buttons, biarkan config dan action buttons aktif"""
        for button_key in self.operation_buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                # Store original state jika belum ada
                if button_key not in self.button_original_states:
                    self.button_original_states[button_key] = {
                        'disabled': button.disabled,
                        'button_style': getattr(button, 'button_style', ''),
                        'description': getattr(button, 'description', '')
                    }
                
                # Disable button
                button.disabled = True
    
    def _enable_operation_buttons(self):
        """Enable operation buttons dan restore original states"""
        for button_key in self.operation_buttons:
            button = self.ui_components.get(button_key)
            if button and button_key in self.button_original_states:
                original = self.button_original_states[button_key]
                
                try:
                    # Restore original state
                    button.disabled = original['disabled']
                    
                    # Restore other properties if needed
                    if hasattr(button, 'button_style'):
                        button.button_style = original['button_style']
                    if hasattr(button, 'description'):
                        button.description = original['description']
                except Exception:
                    # Silent fail untuk restore, yang penting enable button
                    button.disabled = False
                
                # Clear stored state
                del self.button_original_states[button_key]
    
    def _disable_all_buttons(self):
        """Disable semua managed buttons (untuk emergency cases)"""
        for button_key in self.managed_buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                # Store original state
                if button_key not in self.button_original_states:
                    self.button_original_states[button_key] = {
                        'disabled': button.disabled,
                        'button_style': getattr(button, 'button_style', ''),
                        'description': getattr(button, 'description', '')
                    }
                
                # Disable button
                button.disabled = True
    
    def _enable_all_buttons(self):
        """Enable all buttons dan restore original states"""
        for button_key in self.managed_buttons:
            button = self.ui_components.get(button_key)
            if button and button_key in self.button_original_states:
                original = self.button_original_states[button_key]
                
                try:
                    # Restore original state
                    button.disabled = original['disabled']
                    
                    # Restore other properties
                    if hasattr(button, 'button_style'):
                        button.button_style = original['button_style']
                    if hasattr(button, 'description'):
                        button.description = original['description']
                except Exception:
                    # Silent fail untuk restore, yang penting enable button
                    button.disabled = False
                
                # Clear stored state
                del self.button_original_states[button_key]
    
    def _set_processing_state(self, operation_name: str, processing: bool):
        """Set processing state untuk operation"""
        self.operation_states[operation_name] = processing
        
        # Set global state untuk tracking
        state_key = f'{operation_name}_running'
        self.ui_components[state_key] = processing
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Check apakah operation sedang running"""
        return self.operation_states.get(operation_name, False)
    
    def get_running_operations(self) -> List[str]:
        """Get list of currently running operations"""
        return [op for op, running in self.operation_states.items() if running]
    
    def can_start_operation(self, operation_name: str) -> tuple[bool, str]:
        """Generic check untuk semua operations"""
        if self.is_operation_running(operation_name):
            return False, f"{operation_name.title()} sedang berjalan"
        
        running_ops = self.get_running_operations()
        if running_ops:
            # Allow config operations to run alongside other operations
            if operation_name in ['save', 'reset', 'load', 'apply']:
                return True, "Config operation allowed"
            
            return False, f"Tidak dapat memulai {operation_name}, {running_ops[0]} sedang berjalan"
        
        # Check if any operation button is currently processing
        for button_key in self.operation_buttons:
            button = self.ui_components.get(button_key)
            if button and getattr(button, 'disabled', False):
                if button_key in self.button_original_states:
                    # Button is disabled due to operation, not original state
                    operation_type = button_key.replace('_button', '')
                    if operation_type != operation_name:
                        return False, f"Operation {operation_type} masih aktif"
        
        return True, "Ready to start"
    
    def force_enable_all_buttons(self):
        """Force enable semua buttons (emergency recovery)"""
        for button_key in self.managed_buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = False
        
        # Clear all stored states
        self.button_original_states.clear()
        self.operation_states.clear()
        
        if self.logger:
            self.logger.info("🆘 Emergency button recovery executed")
    
    def cleanup_all_states(self):
        """Cleanup semua operation states"""
        # Reset operation states
        for operation in list(self.operation_states.keys()):
            self._set_processing_state(operation, False)
        
        # Force enable all buttons
        self._enable_all_buttons()
        
        # Clear UI states
        ui_state_keys = [f'{op}_running' for op in self.operation_states.keys()]
        for key in ui_state_keys:
            if key in self.ui_components:
                del self.ui_components[key]
        
        if self.logger:
            self.logger.debug("🧹 All button states cleaned up")
    
    # Generic helper methods untuk semua modules
    def is_any_operation_running(self) -> bool:
        """Check if any operation is running"""
        return len(self.get_running_operations()) > 0
    
    def get_module_name(self) -> str:
        """Get module name dari ui_components"""
        return self.ui_components.get('module_name', 'unknown')
    
    def get_button_categories_summary(self) -> Dict[str, List[str]]:
        """Get summary of button categories untuk debugging"""
        return {
            'operation_buttons': self.operation_buttons,
            'config_buttons': self.config_buttons,
            'action_buttons': self.action_buttons,
            'all_managed_buttons': self.managed_buttons
        }
    
    def get_button_states_summary(self) -> Dict[str, Any]:
        """Get summary of current button states untuk debugging"""
        button_states = {}
        for button_key in self.managed_buttons:
            button = self.ui_components.get(button_key)
            if button:
                button_states[button_key] = {
                    'disabled': getattr(button, 'disabled', None),
                    'description': getattr(button, 'description', None),
                    'button_style': getattr(button, 'button_style', None),
                    'has_saved_state': button_key in self.button_original_states,
                    'category': self._get_button_category(button_key)
                }
        
        return {
            'module_name': self.get_module_name(),
            'button_states': button_states,
            'running_operations': self.get_running_operations(),
            'operation_states': dict(self.operation_states),
            'button_categories': self.get_button_categories_summary()
        }
    
    def _get_button_category(self, button_key: str) -> str:
        """Get category untuk button"""
        if button_key in self.operation_buttons:
            return 'operation'
        elif button_key in self.config_buttons:
            return 'config'
        elif button_key in self.action_buttons:
            return 'action'
        else:
            return 'unknown'

def get_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """
    Generic factory function untuk mendapatkan button state manager yang shared.
    
    Args:
        ui_components: Dictionary komponen UI dari module mana pun
        
    Returns:
        Instance ButtonStateManager yang sudah diinisialisasi
    """
    # Validasi input dengan null safety
    if not ui_components or not isinstance(ui_components, dict):
        # Fallback dengan minimal ui_components
        ui_components = {'module_name': 'button_state_manager_fallback'}
    
    # Cek dan buat button_state_manager jika belum ada
    if 'button_state_manager' not in ui_components:
        try:
            ui_components['button_state_manager'] = ButtonStateManager(ui_components)
            
            # Log creation jika logger tersedia
            logger = ui_components.get('logger')
            module_name = ui_components.get('module_name', 'unknown')
            if logger:
                logger.debug(f"🔧 ButtonStateManager berhasil diinisialisasi untuk module: {module_name}")
                
        except Exception as e:
            # Fallback untuk error saat inisialisasi
            module_name = ui_components.get('module_name', 'unknown')
            print(f"⚠️ Error inisialisasi ButtonStateManager untuk {module_name}: {str(e)}")
            ui_components['button_state_manager'] = _create_fallback_manager(ui_components)
    
    return ui_components['button_state_manager']

def _create_fallback_manager(ui_components: Dict[str, Any]) -> 'FallbackButtonStateManager':
    """Create fallback manager jika ButtonStateManager gagal."""
    
    class FallbackButtonStateManager:
        """Fallback manager yang safe untuk semua operations di semua modules."""
        
        def __init__(self, ui_components):
            self.ui_components = ui_components
        
        @contextmanager
        def operation_context(self, operation_name: str):
            """Safe context manager fallback."""
            try:
                yield
            finally:
                pass
        
        @contextmanager
        def config_context(self, config_operation: str):
            """Safe config context fallback."""
            try:
                yield
            finally:
                pass
        
        def can_start_operation(self, operation_name: str) -> tuple[bool, str]:
            """Always allow operation di fallback mode."""
            return True, "Fallback mode - operation allowed"
        
        def force_enable_all_buttons(self):
            """Safe fallback button enable."""
            pass
        
        def cleanup_all_states(self):
            """Safe fallback cleanup."""
            pass
        
        def is_any_operation_running(self):
            """Safe fallback check."""
            return False
        
        def get_module_name(self):
            """Safe fallback module name."""
            return self.ui_components.get('module_name', 'fallback')
    
    return FallbackButtonStateManager(ui_components)

# Convenience functions untuk different contexts
def get_operation_context(ui_components: Dict[str, Any], operation_name: str):
    """Get operation context untuk main operations"""
    manager = get_button_state_manager(ui_components)
    return manager.operation_context(operation_name)

def get_config_context(ui_components: Dict[str, Any], config_operation: str):
    """Get config context untuk config operations"""
    manager = get_button_state_manager(ui_components)
    return manager.config_context(config_operation)

def safe_get_button_state_manager(ui_components: Dict[str, Any]) -> Optional[ButtonStateManager]:
    """
    Safe version yang return None jika gagal, untuk optional usage.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ButtonStateManager instance atau None jika gagal
    """
    try:
        return get_button_state_manager(ui_components)
    except Exception:
        return None

def ensure_button_state_manager(ui_components: Dict[str, Any]) -> bool:
    """
    Ensure button state manager exists dan properly initialized.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        manager = get_button_state_manager(ui_components)
        return manager is not None
    except Exception:
        return False