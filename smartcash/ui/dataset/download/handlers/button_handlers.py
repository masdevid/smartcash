"""
File: smartcash/ui/dataset/download/handlers/button_handlers.py
Deskripsi: Final safe button handlers dengan comprehensive debugging dan error recovery
"""

from typing import Dict, Any

def setup_button_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup button handlers dengan comprehensive debugging dan safe error handling."""
    
    logger = ui_components.get('logger')
    
    # 🔍 Pre-setup debugging
    if logger:
        logger.info("🔘 Setting up button handlers...")
        _debug_ui_components_structure(ui_components, logger)
    
    try:
        # 🔘 Setup handlers dengan improved error handling
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
                    if logger:
                        logger.debug(f"✅ {action_name} handler berhasil disetup")
                else:
                    if logger:
                        # Enhanced debugging untuk missing components
                        _debug_missing_component(ui_components, button_key, logger)
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error setup {action_name} handler: {str(e)}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if logger:
            logger.info(f"🔘 Button handlers setup: {success_count}/{len(handlers_setup)} berhasil")
        
        # 🔧 Post-setup validation
        _validate_handler_setup(ui_components, logger)
        
        # 🔓 Ensure all buttons enabled
        _ensure_all_buttons_enabled(ui_components)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Critical error in button handler setup: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return ui_components

def _debug_ui_components_structure(ui_components: Dict[str, Any], logger) -> None:
    """Debug UI components structure untuk troubleshooting."""
    all_keys = list(ui_components.keys())
    
    # 📋 Show all available keys
    logger.debug(f"📋 Available UI component keys ({len(all_keys)}): {all_keys}")
    
    # 🔘 Show button-related keys
    button_keys = [k for k in all_keys if 'button' in k.lower()]
    if button_keys:
        logger.debug(f"🔘 Button-related keys found: {button_keys}")
    
    # 📊 Show widget types for button components
    expected_buttons = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    for btn_key in expected_buttons:
        if btn_key in ui_components:
            widget = ui_components[btn_key]
            widget_type = type(widget).__name__ if widget else "None"
            logger.debug(f"   • {btn_key}: {widget_type}")

def _debug_missing_component(ui_components: Dict[str, Any], missing_key: str, logger) -> None:
    """Debug missing component dengan suggestions."""
    logger.warning(f"⚠️ {missing_key} tidak ditemukan")
    
    # 🔍 Look for similar keys
    all_keys = list(ui_components.keys())
    action_name = missing_key.replace('_button', '')
    
    similar_keys = [k for k in all_keys if action_name in k.lower()]
    if similar_keys:
        logger.info(f"🔍 Similar keys found: {similar_keys}")
    
    # 🔍 Look for nested button components
    nested_locations = ['actions', 'action_buttons', 'save_reset_buttons']
    for location in nested_locations:
        if location in ui_components and isinstance(ui_components[location], dict):
            if missing_key in ui_components[location]:
                logger.info(f"🔍 Found {missing_key} in nested location: {location}")

def _setup_download_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup download button dengan enhanced error handling."""
    def download_handler(button):
        try:
            # Disable button untuk prevent double-click
            if hasattr(button, 'disabled'):
                button.disabled = True
                
            from smartcash.ui.dataset.download.handlers.download_action import execute_download_action
            execute_download_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Download handler error: {str(e)}")
        finally:
            # Always re-enable button
            _safe_enable_button(button)
    
    button = ui_components['download_button']
    _safe_setup_handler(button, download_handler, 'download')

def _setup_check_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup check button dengan enhanced error handling."""
    def check_handler(button):
        try:
            if hasattr(button, 'disabled'):
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
    _safe_setup_handler(button, check_handler, 'check')

def _setup_reset_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup reset button dengan enhanced error handling."""
    def reset_handler(button):
        try:
            if hasattr(button, 'disabled'):
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
    _safe_setup_handler(button, reset_handler, 'reset')

def _setup_cleanup_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup cleanup button dengan enhanced error handling."""
    def cleanup_handler(button):
        try:
            if hasattr(button, 'disabled'):
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
    _safe_setup_handler(button, cleanup_handler, 'cleanup')

def _setup_save_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup save button dengan enhanced error handling."""
    def save_handler(button):
        try:
            if hasattr(button, 'disabled'):
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
    _safe_setup_handler(button, save_handler, 'save')

def _safe_setup_handler(button, handler_func, action_name: str) -> None:
    """Safely setup handler dengan comprehensive error handling."""
    try:
        # Safe clear existing handlers
        _safe_clear_handlers(button)
        
        # Register new handler
        button.on_click(handler_func)
        
    except Exception as e:
        # Log error tapi jangan crash
        logger = logging.getLogger(__name__)
        logger.error(f"Error setting up {action_name} handler: {str(e)}")

def _safe_clear_handlers(button) -> None:
    """Safely clear existing handlers dengan multiple fallback methods."""
    try:
        # Method 1: Standard clear untuk list
        if hasattr(button, '_click_handlers') and isinstance(button._click_handlers, list):
            button._click_handlers.clear()
            return
            
        # Method 2: Clear method untuk objects dengan clear()
        if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'clear'):
            button._click_handlers.clear()
            return
            
        # Method 3: CallbackDispatcher - replace dengan empty list
        if hasattr(button, '_click_handlers'):
            button._click_handlers = []
            return
            
    except Exception:
        # Method 4: Ultimate fallback - create new empty list
        try:
            button._click_handlers = []
        except Exception:
            # Give up - new handler will still work
            pass

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

def _validate_handler_setup(ui_components: Dict[str, Any], logger) -> None:
    """Validate handler setup dengan comprehensive checking."""
    if not logger:
        return
        
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    logger.debug("🔍 Handler validation:")
    for key in button_keys:
        if key in ui_components and ui_components[key] is not None:
            button = ui_components[key]
            
            # Check handler count
            handler_count = 0
            if hasattr(button, '_click_handlers'):
                if hasattr(button._click_handlers, '__len__'):
                    handler_count = len(button._click_handlers)
                elif button._click_handlers:
                    handler_count = 1
            
            # Check button state
            is_disabled = getattr(button, 'disabled', False)
            button_type = type(button).__name__
            
            status = "✅" if handler_count > 0 and not is_disabled else "⚠️"
            logger.debug(f"   {status} {key}: {button_type}, {handler_count} handlers, disabled={is_disabled}")

def debug_button_connections(ui_components: Dict[str, Any]) -> None:
    """Enhanced debug function dengan comprehensive information."""
    logger = ui_components.get('logger')
    if not logger:
        return
    
    logger.info("🔍 Comprehensive Button Debug Report")
    logger.info("=" * 40)
    
    # 📋 Structure analysis
    _debug_ui_components_structure(ui_components, logger)
    
    # 🔘 Handler validation
    _validate_handler_setup(ui_components, logger)
    
    # 💡 Suggestions
    missing_buttons = []
    expected_buttons = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for btn_key in expected_buttons:
        if btn_key not in ui_components or ui_components[btn_key] is None:
            missing_buttons.append(btn_key)
    
    if missing_buttons:
        logger.warning(f"💡 Missing buttons: {missing_buttons}")
        logger.info("💡 Check create_download_ui() dan action_section.py untuk component creation")
        logger.info("💡 Pastikan key mapping konsisten antara UI creation dan handler expectation")
    
    logger.info("=" * 40)

# Import yang diperlukan untuk error handling
import logging