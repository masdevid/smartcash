"""
File: smartcash/ui/dataset/preprocessing/utils/button_manager.py
Deskripsi: Enhanced button manager dengan backend operation status sync dan Progress Bridge integration
"""

from typing import Dict, Any, Callable, Optional
import functools
from smartcash.common.logger import get_logger

class BackendAwareButtonManager:
    """🔘 Enhanced button manager dengan backend operation awareness"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = get_logger('button_manager')
        self.operation_stack = []
        self.button_states = {}
        self.backend_status = {'active': False, 'operation': None}
        
    def push_backend_operation(self, operation_name: str, backend_service=None):
        """🚀 Push backend operation dengan progress tracking sync"""
        try:
            # Preserve current states
            self._preserve_button_states(operation_name)
            
            # Set backend status
            self.backend_status = {'active': True, 'operation': operation_name, 'service': backend_service}
            
            # Disable all operational buttons
            self._disable_operation_buttons(operation_name)
            
            # Set processing indicators
            self._set_backend_processing_indicators(operation_name)
            
            # Sync dengan progress tracker
            if 'progress_manager' in self.ui_components:
                self.ui_components['progress_manager'].setup_for_operation(operation_name)
            
            self.operation_stack.append(operation_name)
            self.logger.info(f"🚀 Backend operation started: {operation_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error pushing backend operation: {str(e)}")
    
    def pop_backend_operation(self, success: bool = True, message: str = None):
        """✅ Pop backend operation dengan status sync"""
        try:
            if not self.operation_stack:
                return
                
            operation_name = self.operation_stack.pop()
            
            # Restore button states
            self._restore_button_states(operation_name)
            
            # Clear backend status
            self.backend_status = {'active': False, 'operation': None}
            
            # Update progress tracker dengan final status
            if 'progress_manager' in self.ui_components:
                if success:
                    self.ui_components['progress_manager'].complete_operation(message or f"{operation_name} completed")
                else:
                    self.ui_components['progress_manager'].error_operation(message or f"{operation_name} failed")
            
            # Clear processing indicators
            self._clear_processing_indicators()
            
            status = "completed" if success else "failed"
            self.logger.info(f"✅ Backend operation {status}: {operation_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error popping backend operation: {str(e)}")
    
    def _preserve_button_states(self, operation_name: str):
        """💾 Preserve current button states"""
        button_keys = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        self.button_states[operation_name] = {}
        
        for key in button_keys:
            button = self.ui_components.get(key)
            if button and hasattr(button, 'disabled'):
                self.button_states[operation_name][key] = {
                    'disabled': button.disabled,
                    'description': getattr(button, 'description', ''),
                    'button_style': getattr(button, 'button_style', '')
                }
    
    def _restore_button_states(self, operation_name: str):
        """🔄 Restore button states"""
        if operation_name not in self.button_states:
            return
            
        for btn_key, state in self.button_states[operation_name].items():
            button = self.ui_components.get(btn_key)
            if button:
                for attr, value in state.items():
                    if hasattr(button, attr):
                        try:
                            setattr(button, attr, value)
                        except Exception:
                            pass
        
        # Clear stored state
        del self.button_states[operation_name]
    
    def _disable_operation_buttons(self, operation_name: str):
        """🚫 Disable operational buttons durante backend operation"""
        button_keys = ['preprocess_button', 'check_button', 'cleanup_button']
        
        for key in button_keys:
            button = self.ui_components.get(key)
            if button and hasattr(button, 'disabled'):
                button.disabled = True
    
    def _set_backend_processing_indicators(self, operation_name: str):
        """⚡ Set visual indicators untuk backend processing"""
        # Update primary operation button
        operation_button_map = {
            'preprocessing': 'preprocess_button',
            'validation': 'check_button', 
            'cleanup': 'cleanup_button'
        }
        
        primary_button_key = operation_button_map.get(operation_name.lower())
        if primary_button_key:
            button = self.ui_components.get(primary_button_key)
            if button:
                # Store original state if not already stored
                if not hasattr(button, '_original_description'):
                    button._original_description = button.description
                    button._original_style = getattr(button, 'button_style', 'primary')
                
                # Set processing state
                button.description = f"{button._original_description} (Processing...)"
                button.button_style = 'warning'
    
    def _clear_processing_indicators(self):
        """🧹 Clear processing indicators dan restore original states"""
        button_keys = ['preprocess_button', 'check_button', 'cleanup_button']
        
        for key in button_keys:
            button = self.ui_components.get(key)
            if button and hasattr(button, '_original_description'):
                try:
                    button.description = button._original_description
                    button.button_style = button._original_style
                    # Clean up temporary attributes
                    delattr(button, '_original_description')
                    delattr(button, '_original_style')
                except Exception:
                    pass
    
    def is_backend_operation_active(self) -> bool:
        """❓ Check if backend operation sedang aktif"""
        return self.backend_status.get('active', False)
    
    def get_current_backend_operation(self) -> Optional[str]:
        """📋 Get current backend operation name"""
        return self.backend_status.get('operation')
    
    def sync_with_backend_status(self, backend_service) -> bool:
        """🔄 Sync button states dengan backend service status"""
        try:
            # Check backend service status jika available
            if hasattr(backend_service, 'is_processing'):
                is_processing = backend_service.is_processing()
                current_operation = getattr(backend_service, 'current_operation', None)
                
                if is_processing and not self.is_backend_operation_active():
                    # Backend active tapi UI belum aware
                    self.push_backend_operation(current_operation or "backend_operation", backend_service)
                elif not is_processing and self.is_backend_operation_active():
                    # Backend sudah selesai tapi UI masih dalam processing state
                    self.pop_backend_operation(success=True, message="Backend operation completed")
                
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ Backend status sync warning: {str(e)}")
            return False
    
    def register_progress_updates(self, progress_callback: Callable):
        """📊 Register progress updates untuk real-time button state sync"""
        def enhanced_callback(level: str, current: int, total: int, message: str):
            try:
                # Call original callback
                progress_callback(level, current, total, message)
                
                # Update button descriptions dengan progress info untuk visual feedback
                if self.is_backend_operation_active():
                    operation = self.get_current_backend_operation()
                    self._update_button_progress_display(operation, current, total, message)
                    
            except Exception as e:
                self.logger.debug(f"Progress callback enhancement error: {str(e)}")
        
        return enhanced_callback
    
    def _update_button_progress_display(self, operation: str, current: int, total: int, message: str):
        """📈 Update button display dengan progress information"""
        operation_button_map = {
            'preprocessing': 'preprocess_button',
            'validation': 'check_button',
            'cleanup': 'cleanup_button'
        }
        
        button_key = operation_button_map.get(operation.lower())
        if button_key:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, '_original_description'):
                try:
                    progress_pct = int((current / max(total, 1)) * 100)
                    button.description = f"{button._original_description} ({progress_pct}%)"
                except Exception:
                    pass

def disable_all_buttons(ui_components: Dict[str, Any]):
    """🚫 Disable semua operation buttons dengan backend awareness"""
    manager = _get_or_create_manager(ui_components)
    if manager.is_backend_operation_active():
        # Sudah dalam backend operation, tidak perlu disable lagi
        return
    
    # Standard disable untuk non-backend operations
    button_keys = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            if not hasattr(button, '_original_disabled'):
                button._original_disabled = button.disabled
            button.disabled = True

def enable_all_buttons(ui_components: Dict[str, Any]):
    """✅ Enable semua operation buttons dengan backend awareness"""
    manager = _get_or_create_manager(ui_components)
    if manager.is_backend_operation_active():
        # Jangan enable buttons jika backend operation masih aktif
        return
    
    # Standard enable untuk non-backend operations
    button_keys = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            original_state = getattr(button, '_original_disabled', False)
            button.disabled = original_state

def with_backend_operation_management(operation_name: str):
    """🔧 Enhanced decorator dengan backend operation management"""
    def decorator(operation_func: Callable):
        @functools.wraps(operation_func)
        def wrapper(ui_components: Dict[str, Any], *args, **kwargs):
            manager = _get_or_create_manager(ui_components)
            
            try:
                # Push backend operation
                backend_service = kwargs.get('backend_service')
                manager.push_backend_operation(operation_name, backend_service)
                
                # Execute operation
                result = operation_func(ui_components, *args, **kwargs)
                
                # Determine success dari result
                success = True
                message = f"{operation_name} completed"
                
                if isinstance(result, dict):
                    success = result.get('success', True)
                    message = result.get('message', message)
                elif isinstance(result, bool):
                    success = result
                
                # Pop dengan status
                manager.pop_backend_operation(success, message)
                return result
                
            except Exception as e:
                # Pop dengan error status
                manager.pop_backend_operation(False, f"{operation_name} failed: {str(e)}")
                raise
        
        return wrapper
    return decorator

def _get_or_create_manager(ui_components: Dict[str, Any]) -> BackendAwareButtonManager:
    """🏭 Get atau create button manager instance"""
    if 'button_manager' not in ui_components:
        ui_components['button_manager'] = BackendAwareButtonManager(ui_components)
    return ui_components['button_manager']

# Enhanced convenience functions
def setup_backend_button_management(ui_components: Dict[str, Any], backend_service=None) -> BackendAwareButtonManager:
    """🚀 Setup enhanced button management dengan backend integration"""
    manager = _get_or_create_manager(ui_components)
    
    # Sync dengan backend service jika ada
    if backend_service:
        manager.sync_with_backend_status(backend_service)
        
        # Register progress callback enhancement
        if 'progress_callback' in ui_components:
            enhanced_callback = manager.register_progress_updates(ui_components['progress_callback'])
            ui_components['progress_callback'] = enhanced_callback
    
    return manager

def notify_backend_operation_start(ui_components: Dict[str, Any], operation_name: str, backend_service=None):
    """🚀 Notify button manager tentang backend operation start"""
    manager = _get_or_create_manager(ui_components)
    manager.push_backend_operation(operation_name, backend_service)

def notify_backend_operation_complete(ui_components: Dict[str, Any], success: bool = True, message: str = None):
    """✅ Notify button manager tentang backend operation completion"""
    manager = _get_or_create_manager(ui_components)
    manager.pop_backend_operation(success, message)

# Backward compatibility aliases
with_button_management = with_backend_operation_management
set_button_processing_state = lambda ui_components, button_key, processing: _get_or_create_manager(ui_components)._set_backend_processing_indicators(button_key) if processing else _get_or_create_manager(ui_components)._clear_processing_indicators()
create_button_state_manager = lambda ui_components: _get_or_create_manager(ui_components)
get_button_manager = lambda ui_components: _get_or_create_manager(ui_components)