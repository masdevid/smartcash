"""
File: smartcash/ui/dataset/download/handlers/button_handlers.py
Deskripsi: Fixed button handlers dengan TypeError resolution dan progress callback yang proper
"""

from typing import Dict, Any

def setup_button_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup button handlers dengan error handling yang diperbaiki."""
    
    logger = ui_components.get('logger')
    
    try:
        # Setup handlers dengan improved error handling
        handlers_setup = [
            ('download_button', 'download', _setup_download_handler),
            ('check_button', 'check', _setup_check_handler),
            ('reset_button', 'reset', _setup_reset_handler),
            ('cleanup_button', 'cleanup', _setup_cleanup_handler),
            ('save_button', 'save', _setup_save_handler)
        ]
        
        success_count = 0
        for button_key, action_name, setup_func in handlers_setup:
            try:
                if button_key in ui_components and ui_components[button_key] is not None:
                    setup_func(ui_components, env)
                    success_count += 1
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error setup {action_name} handler: {str(e)}")
        
        if logger and success_count > 0:
            logger.info(f"🔘 Button handlers ready: {success_count} handlers active")
        
        # Ensure all buttons enabled
        _ensure_all_buttons_enabled(ui_components)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Critical error in button handler setup: {str(e)}")
    
    return ui_components

def _setup_download_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup download button dengan fixed error handling."""
    def download_handler(button):
        try:
            button.disabled = True
            from smartcash.ui.dataset.download.handlers.download_action import execute_download_action
            execute_download_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Download handler error: {str(e)}")
        finally:
            _safe_enable_button(button)
    
    button = ui_components['download_button']
    _safe_setup_handler(button, download_handler)

def _setup_check_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup check button dengan fixed error handling."""
    def check_handler(button):
        try:
            button.disabled = True
            from smartcash.ui.dataset.download.handlers.check_action import execute_check_action
            execute_check_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Check handler error: {str(e)}")
        finally:
            _safe_enable_button(button)
    
    button = ui_components['check_button']
    _safe_setup_handler(button, check_handler)

def _setup_reset_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup reset button dengan fixed error handling."""
    def reset_handler(button):
        try:
            button.disabled = True
            from smartcash.ui.dataset.download.handlers.reset_action import execute_reset_action
            execute_reset_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Reset handler error: {str(e)}")
        finally:
            _safe_enable_button(button)
    
    button = ui_components['reset_button']
    _safe_setup_handler(button, reset_handler)

def _setup_cleanup_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup cleanup button dengan fixed error handling."""
    def cleanup_handler(button):
        try:
            button.disabled = True
            from smartcash.ui.dataset.download.handlers.cleanup_action import execute_cleanup_action
            execute_cleanup_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Cleanup handler error: {str(e)}")
        finally:
            _safe_enable_button(button)
    
    button = ui_components['cleanup_button']
    _safe_setup_handler(button, cleanup_handler)

def _setup_save_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup save button dengan fixed error handling."""
    def save_handler(button):
        try:
            button.disabled = True
            from smartcash.ui.dataset.download.handlers.save_action import execute_save_action
            execute_save_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Save handler error: {str(e)}")
        finally:
            _safe_enable_button(button)
    
    button = ui_components['save_button']
    _safe_setup_handler(button, save_handler)

def _safe_setup_handler(button, handler_func) -> None:
    """Safely setup handler dengan fixed TypeError resolution."""
    try:
        # Clear existing handlers properly
        if hasattr(button, '_click_handlers'):
            # Fix: Handle list properly instead of calling it
            if isinstance(button._click_handlers, list):
                button._click_handlers.clear()
            elif hasattr(button._click_handlers, 'clear'):
                button._click_handlers.clear()
            else:
                button._click_handlers = []
        
        # Register new handler
        button.on_click(handler_func)
        
    except Exception:
        # Ultimate fallback - create new list
        try:
            button._click_handlers = []
            button.on_click(handler_func)
        except Exception:
            pass  # Give up gracefully

def _safe_enable_button(button) -> None:
    """Safely enable button dengan error handling."""
    try:
        if hasattr(button, 'disabled'):
            button.disabled = False
    except Exception:
        pass

def _ensure_all_buttons_enabled(ui_components: Dict[str, Any]) -> None:
    """Ensure semua button dalam keadaan enabled setelah setup."""
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components and ui_components[key] is not None:
            _safe_enable_button(ui_components[key])

def debug_button_connections(ui_components: Dict[str, Any]) -> None:
    """Minimal debug function untuk troubleshooting."""
    logger = ui_components.get('logger')
    if not logger:
        return
    
    expected_buttons = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    working_buttons = sum(1 for btn in expected_buttons if btn in ui_components and ui_components[btn] is not None)
    
    if working_buttons < len(expected_buttons):
        logger.warning(f"⚠️ Button check: {working_buttons}/{len(expected_buttons)} buttons tersedia")