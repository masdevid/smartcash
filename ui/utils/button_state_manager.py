"""
File: smartcash.ui.utils.button_state_manager.py
Deskripsi: Fixed button state manager tanpa circular imports
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from contextlib import contextmanager

# Avoid circular imports by using TYPE_CHECKING
if TYPE_CHECKING:
    pass

class ButtonStateManager:
    """Button state manager yang self-contained tanpa circular imports"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_states = {}
        self.button_original_states = {}
        self.logger = ui_components.get('logger')
        
        # Auto-detect buttons dari ui_components
        self.managed_buttons = self._auto_detect_buttons()
        self.operation_buttons = self._get_operation_buttons()
        self.config_buttons = self._get_config_buttons()
    
    def _auto_detect_buttons(self) -> List[str]:
        """Auto-detect buttons dengan safe checking"""
        button_keys = []
        
        # Common patterns across modules
        patterns = [
            'download_button', 'preprocess_button', 'augment_button', 'train_button',
            'check_button', 'cleanup_button', 'validate_button', 'analyze_button',
            'save_button', 'reset_button', 'load_button', 'apply_button'
        ]
        
        for key in self.ui_components.keys():
            if (key.endswith('_button') and 
                self._is_valid_button(self.ui_components.get(key))):
                button_keys.append(key)
        
        # Add known patterns that exist
        for pattern in patterns:
            if (pattern in self.ui_components and 
                pattern not in button_keys and
                self._is_valid_button(self.ui_components.get(pattern))):
                button_keys.append(pattern)
        
        return button_keys
    
    def _is_valid_button(self, widget) -> bool:
        """Check if widget is a valid button"""
        return (widget is not None and 
                hasattr(widget, 'disabled') and 
                hasattr(widget, 'on_click'))
    
    def _get_operation_buttons(self) -> List[str]:
        """Get operation buttons yang harus di-disable"""
        operation_patterns = [
            'download_button', 'preprocess_button', 'augment_button', 'train_button',
            'check_button', 'cleanup_button', 'validate_button', 'analyze_button'
        ]
        return [btn for btn in self.managed_buttons if any(pattern in btn for pattern in operation_patterns)]
    
    def _get_config_buttons(self) -> List[str]:
        """Get config buttons yang tetap aktif"""
        config_patterns = ['save_button', 'reset_button', 'load_button', 'apply_button']
        return [btn for btn in self.managed_buttons if any(pattern in btn for pattern in config_patterns)]
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Operation context dengan safe error handling"""
        try:
            self._set_processing_state(operation_name, True)
            self._disable_operation_buttons()
            
            if self.logger:
                self.logger.debug(f"🔒 Operation {operation_name} started")
            
            yield
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"🔥 Error dalam operation {operation_name}: {str(e)}")
            raise
        finally:
            self._set_processing_state(operation_name, False)
            self._enable_operation_buttons()
            
            if self.logger:
                self.logger.debug(f"🔓 Operation {operation_name} completed")
    
    @contextmanager
    def config_context(self, config_operation: str):
        """Config context untuk config operations"""
        try:
            self._set_processing_state(config_operation, True)
            
            if self.logger:
                self.logger.debug(f"🔧 Config operation {config_operation} started")
            
            yield
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"🔥 Config error {config_operation}: {str(e)}")
            raise
        finally:
            self._set_processing_state(config_operation, False)
            
            if self.logger:
                self.logger.debug(f"✅ Config operation {config_operation} completed")
    
    def can_start_operation(self, operation_name: str) -> tuple[bool, str]:
        """Check if operation can start"""
        try:
            if self.is_operation_running(operation_name):
                return False, f"{operation_name.title()} sedang berjalan"
            
            running_ops = self.get_running_operations()
            if running_ops:
                # Allow config operations
                if operation_name in ['save', 'reset', 'load', 'apply']:
                    return True, "Config operation allowed"
                
                return False, f"Tidak dapat memulai {operation_name}, {running_ops[0]} sedang berjalan"
            
            return True, "Ready to start"
        except Exception:
            return True, "Fallback - operation allowed"
    
    def _disable_operation_buttons(self):
        """Disable operation buttons dengan safe execution"""
        for button_key in self.operation_buttons:
            try:
                button = self.ui_components.get(button_key)
                if button and hasattr(button, 'disabled'):
                    # Store original state
                    if button_key not in self.button_original_states:
                        self.button_original_states[button_key] = {
                            'disabled': button.disabled,
                            'button_style': getattr(button, 'button_style', ''),
                            'description': getattr(button, 'description', '')
                        }
                    button.disabled = True
            except Exception:
                continue
    
    def _enable_operation_buttons(self):
        """Enable operation buttons dengan safe execution"""
        for button_key in self.operation_buttons:
            try:
                button = self.ui_components.get(button_key)
                if button and button_key in self.button_original_states:
                    original = self.button_original_states[button_key]
                    
                    button.disabled = original['disabled']
                    if hasattr(button, 'button_style'):
                        button.button_style = original['button_style']
                    if hasattr(button, 'description'):
                        button.description = original['description']
                    
                    del self.button_original_states[button_key]
            except Exception:
                # At minimum, enable the button
                try:
                    if button:
                        button.disabled = False
                except Exception:
                    pass
    
    def _set_processing_state(self, operation_name: str, processing: bool):
        """Set processing state dengan safe execution"""
        try:
            self.operation_states[operation_name] = processing
            state_key = f'{operation_name}_running'
            self.ui_components[state_key] = processing
        except Exception:
            pass
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Safe operation running check"""
        try:
            return self.operation_states.get(operation_name, False)
        except Exception:
            return False
    
    def get_running_operations(self) -> List[str]:
        """Safe get running operations"""
        try:
            return [op for op, running in self.operation_states.items() if running]
        except Exception:
            return []
    
    def force_enable_all_buttons(self):
        """Emergency button enable"""
        for button_key in self.managed_buttons:
            try:
                button = self.ui_components.get(button_key)
                if button and hasattr(button, 'disabled'):
                    button.disabled = False
            except Exception:
                continue
        
        self.button_original_states.clear()
        self.operation_states.clear()
    
    def cleanup_all_states(self):
        """Safe cleanup all states"""
        try:
            for operation in list(self.operation_states.keys()):
                self._set_processing_state(operation, False)
            
            self._enable_operation_buttons()
            
            ui_state_keys = [f'{op}_running' for op in self.operation_states.keys()]
            for key in ui_state_keys:
                if key in self.ui_components:
                    del self.ui_components[key]
        except Exception:
            pass

class FallbackButtonStateManager:
    """Fallback manager yang complete dan safe"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Safe fallback operation context"""
        try:
            yield
        except Exception as e:
            print(f"⚠️ Fallback operation error: {str(e)}")
            raise
        finally:
            pass
    
    @contextmanager  
    def config_context(self, config_operation: str):
        """Safe fallback config context"""
        try:
            yield
        except Exception as e:
            print(f"⚠️ Fallback config error: {str(e)}")
            raise
        finally:
            pass
    
    def can_start_operation(self, operation_name: str) -> tuple[bool, str]:
        """Always allow in fallback"""
        return True, "Fallback mode - operation allowed"
    
    def force_enable_all_buttons(self):
        """Safe fallback enable"""
        pass
    
    def cleanup_all_states(self):
        """Safe fallback cleanup"""
        pass
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Safe fallback check"""
        return False
    
    def get_running_operations(self) -> List[str]:
        """Safe fallback list"""
        return []

# Factory functions tanpa circular imports
def create_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """Create ButtonStateManager tanpa circular imports"""
    if not ui_components or not isinstance(ui_components, dict):
        ui_components = {'module_name': 'button_state_manager_fallback'}
    
    try:
        manager = ButtonStateManager(ui_components)
        # Test if manager works properly
        manager.can_start_operation('test')
        return manager
    except Exception as e:
        print(f"⚠️ ButtonStateManager error, using fallback: {str(e)}")
        return FallbackButtonStateManager(ui_components)

def get_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """Get or create ButtonStateManager safely"""
    if 'button_state_manager' not in ui_components:
        ui_components['button_state_manager'] = create_button_state_manager(ui_components)
        
        logger = ui_components.get('logger')
        if logger:
            logger.debug("🔧 ButtonStateManager berhasil diinisialisasi")
    
    return ui_components['button_state_manager']