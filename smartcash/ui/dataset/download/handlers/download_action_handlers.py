"""
File: smartcash/ui/dataset/download/handlers/download_action_handlers.py
Deskripsi: Refactored action handlers dengan delegation ke specialized modules dan button state management
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.download.actions.download_executor import execute_download_action
from smartcash.ui.dataset.download.actions.check_executor import execute_check_action
from smartcash.ui.dataset.download.actions.cleanup_executor import execute_cleanup_action
from smartcash.ui.dataset.download.actions.config_executor import execute_save_action, execute_reset_action

def setup_download_action_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup action handlers dengan delegation ke specialized executors."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    ui_components['button_state_manager'] = button_manager
    
    try:
        # Handler mapping dengan specialized executors
        handler_configs = [
            ('download_button', 'Download', _create_download_handler),
            ('check_button', 'Check', _create_check_handler),
            ('cleanup_button', 'Cleanup', _create_cleanup_handler),
            ('save_button', 'Save', _create_save_handler),
            ('reset_button', 'Reset', _create_reset_handler)
        ]
        
        success_count = 0
        
        # Setup each handler dengan error isolation
        for button_key, action_name, handler_factory in handler_configs:
            try:
                if _is_button_available(ui_components, button_key):
                    handler_func = handler_factory(ui_components, env, button_manager)
                    _attach_handler_safely(ui_components[button_key], handler_func)
                    success_count += 1
            except Exception as e:
                logger and logger.warning(f"⚠️ Error setup {action_name} handler: {str(e)}")
        
        # Log success summary
        logger and logger.info(f"🔘 Action handlers aktif: {success_count}/{len(handler_configs)}")
        
        # Ensure semua buttons enabled setelah setup
        _ensure_buttons_enabled(ui_components)
        
    except Exception as e:
        logger and logger.error(f"❌ Critical error dalam action handler setup: {str(e)}")
    
    return ui_components

def _create_download_handler(ui_components: Dict[str, Any], env, button_manager):
    """Create download button handler dengan button state management."""
    def download_handler(button):
        with button_manager.operation_context('download'):
            try:
                execute_download_action(ui_components, button)
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.error(f"💥 Download handler error: {str(e)}")
                if 'error_operation' in ui_components:
                    ui_components['error_operation'](f"Download error: {str(e)}")
    return download_handler

def _create_check_handler(ui_components: Dict[str, Any], env, button_manager):
    """Create check button handler dengan button state management."""
    def check_handler(button):
        with button_manager.operation_context('check'):
            try:
                execute_check_action(ui_components, button)
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.error(f"💥 Check handler error: {str(e)}")
                if 'error_operation' in ui_components:
                    ui_components['error_operation'](f"Check error: {str(e)}")
    return check_handler

def _create_cleanup_handler(ui_components: Dict[str, Any], env, button_manager):
    """Create cleanup button handler dengan button state management."""
    def cleanup_handler(button):
        with button_manager.operation_context('cleanup'):
            try:
                execute_cleanup_action(ui_components, button)
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.error(f"💥 Cleanup handler error: {str(e)}")
                if 'error_operation' in ui_components:
                    ui_components['error_operation'](f"Cleanup error: {str(e)}")
    return cleanup_handler

def _create_save_handler(ui_components: Dict[str, Any], env, button_manager):
    """Create save button handler dengan config context."""
    def save_handler(button):
        with button_manager.config_context('save_config'):
            try:
                config_handler = ui_components.get('config_handler')
                if config_handler:
                    config_handler.save_config(ui_components)
                else:
                    execute_save_action(ui_components, button)
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.error(f"💥 Save handler error: {str(e)}")
    return save_handler

def _create_reset_handler(ui_components: Dict[str, Any], env, button_manager):
    """Create reset button handler dengan config context."""
    def reset_handler(button):
        with button_manager.config_context('reset_config'):
            try:
                config_handler = ui_components.get('config_handler')
                if config_handler:
                    config_handler.reset_config(ui_components)
                else:
                    execute_reset_action(ui_components, button)
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.error(f"💥 Reset handler error: {str(e)}")
    return reset_handler

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

def get_action_handler_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get status action handlers untuk debugging."""
    button_keys = ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    
    status = {
        'total_buttons': len(button_keys),
        'available_buttons': 0,
        'enabled_buttons': 0,
        'button_details': {},
        'button_state_manager': 'button_state_manager' in ui_components
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