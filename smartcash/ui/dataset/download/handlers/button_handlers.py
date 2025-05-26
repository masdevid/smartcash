"""
File: smartcash/ui/dataset/download/handlers/button_handlers.py
Deskripsi: Fixed button handlers dengan integrasi latest progress_tracking dan button_state_manager
"""

from typing import Dict, Any

# Import specialized handlers
from smartcash.ui.dataset.download.handlers.download_action import execute_download_action
from smartcash.ui.dataset.download.handlers.check_action import execute_check_action
from smartcash.ui.dataset.download.handlers.cleanup_action import execute_cleanup_action
from smartcash.ui.dataset.download.handlers.reset_action import execute_reset_action
from smartcash.ui.dataset.download.handlers.save_action import execute_save_action

def setup_button_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """
    Setup semua button handlers dengan latest progress tracking integration.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment context (optional)
        
    Returns:
        Updated ui_components dictionary
    """
    logger = ui_components.get('logger')
    
    try:
        # Handler mapping untuk cleaner setup
        handler_configs = [
            ('download_button', 'Download', _create_download_handler),
            ('check_button', 'Check', _create_check_handler),
            ('cleanup_button', 'Cleanup', _create_cleanup_handler),
            ('reset_button', 'Reset', _create_reset_handler),
            ('save_button', 'Save', _create_save_handler)
        ]
        
        success_count = 0
        
        # Setup each handler dengan error isolation
        for button_key, action_name, handler_factory in handler_configs:
            try:
                if _is_button_available(ui_components, button_key):
                    handler_func = handler_factory(ui_components, env)
                    _attach_handler_safely(ui_components[button_key], handler_func)
                    success_count += 1
            except Exception as e:
                if 'log_output' in ui_components:
                    ui_components['log_output'].warning(f"⚠️ Error setup {action_name} handler: {str(e)}")
        
        # Log success summary
        if 'log_output' in ui_components and success_count > 0:
            ui_components['log_output'].info(f"🔘 Button handlers aktif: {success_count} dari {len(handler_configs)}")
        
        # Ensure semua buttons enabled setelah setup
        _ensure_buttons_enabled(ui_components)
        
    except Exception as e:
        if 'log_output' in ui_components:
            ui_components['log_output'].error(f"❌ Critical error dalam button handler setup: {str(e)}")
    
    return ui_components

def _create_download_handler(ui_components: Dict[str, Any], env=None):
    """Create download button handler dengan latest integration."""
    def download_handler(button):
        try:
            # Execute download action - button state management handled in action
            execute_download_action(ui_components, button)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"💥 Download handler error: {str(e)}")
            
            # Error state untuk UI dengan latest integration
            if 'error_operation' in ui_components:
                ui_components['error_operation'](f"Download error: {str(e)}")
            elif 'tracker' in ui_components:
                ui_components['tracker'].error(f"Download error: {str(e)}")
    
    return download_handler

def _create_check_handler(ui_components: Dict[str, Any], env=None):
    """Create check button handler dengan latest integration."""
    def check_handler(button):
        try:
            # Execute check action - button state management handled in action
            execute_check_action(ui_components, button)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"💥 Check handler error: {str(e)}")
            
            # Error state untuk UI dengan latest integration
            if 'error_operation' in ui_components:
                ui_components['error_operation'](f"Check error: {str(e)}")
            elif 'tracker' in ui_components:
                ui_components['tracker'].error(f"Check error: {str(e)}")
    
    return check_handler

def _create_cleanup_handler(ui_components: Dict[str, Any], env=None):
    """Create cleanup button handler dengan latest integration."""
    def cleanup_handler(button):
        try:
            # Execute cleanup action - button state management handled in action
            execute_cleanup_action(ui_components, button)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"💥 Cleanup handler error: {str(e)}")
            
            # Error state untuk UI dengan latest integration
            if 'error_operation' in ui_components:
                ui_components['error_operation'](f"Cleanup error: {str(e)}")
            elif 'tracker' in ui_components:
                ui_components['tracker'].error(f"Cleanup error: {str(e)}")
    
    return cleanup_handler

def _create_reset_handler(ui_components: Dict[str, Any], env=None):
    """Create reset button handler dengan latest integration."""
    def reset_handler(button):
        try:
            # Disable button sementara
            if hasattr(button, 'disabled'):
                button.disabled = True
            
            execute_reset_action(ui_components, button)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"💥 Reset handler error: {str(e)}")
        
        finally:
            # Always re-enable button
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    return reset_handler

def _create_save_handler(ui_components: Dict[str, Any], env=None):
    """Create save button handler dengan latest integration."""
    def save_handler(button):
        try:
            # Disable button sementara
            if hasattr(button, 'disabled'):
                button.disabled = True
            
            execute_save_action(ui_components, button)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"💥 Save handler error: {str(e)}")
        
        finally:
            # Always re-enable button
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    return save_handler

def _is_button_available(ui_components: Dict[str, Any], button_key: str) -> bool:
    """Check apakah button tersedia dan valid."""
    return (button_key in ui_components and 
            ui_components[button_key] is not None and
            hasattr(ui_components[button_key], 'on_click'))

def _attach_handler_safely(button, handler_func) -> bool:
    """Attach handler ke button dengan error handling."""
    try:
        button.on_click(handler_func)
        return True
    except Exception:
        return False

def _ensure_buttons_enabled(ui_components: Dict[str, Any]) -> None:
    """Ensure semua button dalam keadaan enabled setelah setup."""
    button_keys = ['download_button', 'check_button', 'cleanup_button', 'reset_button', 'save_button']
    
    for button_key in button_keys:
        if _is_button_available(ui_components, button_key):
            try:
                ui_components[button_key].disabled = False
            except Exception:
                pass

def register_additional_handler(ui_components: Dict[str, Any], 
                              button_key: str, 
                              handler_func, 
                              replace_existing: bool = False) -> bool:
    """
    Register additional handler untuk button tertentu.
    
    Args:
        ui_components: Dictionary komponen UI
        button_key: Key untuk button
        handler_func: Handler function
        replace_existing: Apakah replace existing handler
        
    Returns:
        True jika berhasil register
    """
    try:
        if not _is_button_available(ui_components, button_key):
            return False
        
        button = ui_components[button_key]
        
        if replace_existing:
            # Clear existing handlers (jika mungkin)
            # Note: ipywidgets tidak menyediakan cara untuk clear handlers,
            # jadi kita hanya bisa add handler baru
            pass
        
        return _attach_handler_safely(button, handler_func)
        
    except Exception:
        return False

def get_handler_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get status semua button handlers untuk debugging.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary berisi status handlers
    """
    button_keys = ['download_button', 'check_button', 'cleanup_button', 'reset_button', 'save_button']
    
    status = {
        'total_buttons': len(button_keys),
        'available_buttons': 0,
        'enabled_buttons': 0,
        'button_details': {}
    }
    
    for button_key in button_keys:
        button_status = {
            'exists': button_key in ui_components,
            'not_none': ui_components.get(button_key) is not None,
            'has_onclick': False,
            'enabled': False
        }
        
        if _is_button_available(ui_components, button_key):
            status['available_buttons'] += 1
            button_status['has_onclick'] = True
            
            try:
                button_status['enabled'] = not ui_components[button_key].disabled
                if button_status['enabled']:
                    status['enabled_buttons'] += 1
            except Exception:
                pass
        
        status['button_details'][button_key] = button_status
    
    return status