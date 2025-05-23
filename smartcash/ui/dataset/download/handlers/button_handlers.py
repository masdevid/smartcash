"""
File: smartcash/ui/dataset/download/handlers/button_handlers.py
Deskripsi: Button handlers yang diperbaiki dengan safe handler clearing dan proper component checking
"""

from typing import Dict, Any

def setup_button_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handler tombol dengan registrasi yang aman dan error handling yang kuat."""
    
    logger = ui_components.get('logger')
    
    # 🔍 Debug: Log semua available keys
    if logger:
        available_keys = list(ui_components.keys())
        logger.debug(f"🔍 Available UI components: {available_keys}")
    
    try:
        # 🔘 Setup handlers dengan proper key checking
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
                if button_key in ui_components:
                    setup_func(ui_components, env)
                    success_count += 1
                    if logger:
                        logger.debug(f"✅ {action_name} handler berhasil disetup")
                else:
                    if logger:
                        # Check for alternative keys yang mungkin ada
                        alternative_keys = [k for k in ui_components.keys() if action_name in k.lower()]
                        if alternative_keys:
                            logger.info(f"🔍 {button_key} tidak ditemukan, tapi ditemukan: {alternative_keys}")
                        else:
                            logger.warning(f"⚠️ {button_key} tidak ditemukan dalam ui_components")
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error setup {action_name} handler: {str(e)}")
        
        if logger:
            logger.info(f"🔘 Button handlers setup: {success_count}/{len(handlers_setup)} berhasil")
        
        # Ensure all found buttons are enabled after setup
        _ensure_buttons_enabled(ui_components)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup button handlers: {str(e)}")
    
    return ui_components

def _setup_download_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup download button dengan safe handler clearing."""
    def download_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.download_action import execute_download_action
            execute_download_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Download handler error: {str(e)}")
            _safe_enable_button(button)
    
    button = ui_components['download_button']
    _safe_clear_handlers(button)
    button.on_click(download_handler)

def _setup_check_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup check button dengan safe handler clearing."""
    def check_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.check_action import execute_check_action
            execute_check_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Check handler error: {str(e)}")
            _safe_enable_button(button)
    
    button = ui_components['check_button']
    _safe_clear_handlers(button)
    button.on_click(check_handler)

def _setup_reset_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup reset button dengan safe handler clearing."""
    def reset_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.reset_action import execute_reset_action
            execute_reset_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Reset handler error: {str(e)}")
            _safe_enable_button(button)
    
    button = ui_components['reset_button']
    _safe_clear_handlers(button)
    button.on_click(reset_handler)

def _setup_cleanup_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup cleanup button dengan safe handler clearing."""
    def cleanup_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.cleanup_action import execute_cleanup_action
            execute_cleanup_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Cleanup handler error: {str(e)}")
            _safe_enable_button(button)
    
    button = ui_components['cleanup_button']
    _safe_clear_handlers(button)
    button.on_click(cleanup_handler)

def _setup_save_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup save button dengan safe handler clearing.""" 
    def save_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.save_action import execute_save_action
            execute_save_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Save handler error: {str(e)}")
            _safe_enable_button(button)
    
    button = ui_components['save_button']
    _safe_clear_handlers(button)
    button.on_click(save_handler)

def _safe_clear_handlers(button) -> None:
    """Safely clear existing click handlers tanpa error."""
    try:
        # Method 1: Try standard _click_handlers attribute
        if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'clear'):
            button._click_handlers.clear()
        elif hasattr(button, '_click_handlers') and isinstance(button._click_handlers, list):
            button._click_handlers.clear()
        # Method 2: Try CallbackDispatcher
        elif hasattr(button, '_click_handlers'):
            # For CallbackDispatcher objects, create new empty list
            button._click_handlers = []
        # Method 3: No handlers exist yet - OK
    except Exception:
        # If clearing fails, just continue - new handler will still work
        pass

def _safe_enable_button(button) -> None:
    """Safely enable button jika ada error."""
    try:
        if hasattr(button, 'disabled'):
            button.disabled = False
    except Exception:
        pass

def _ensure_buttons_enabled(ui_components: Dict[str, Any]) -> None:
    """Pastikan semua button dalam keadaan enabled setelah setup."""
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components:
            _safe_enable_button(ui_components[key])

def debug_button_connections(ui_components: Dict[str, Any]) -> None:
    """Debug function untuk memeriksa koneksi button handlers dengan improved info."""
    logger = ui_components.get('logger')
    if not logger:
        return
    
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    logger.info("🔍 Debug button connections:")
    
    # First, show all available UI component keys
    all_keys = list(ui_components.keys())
    button_related_keys = [k for k in all_keys if 'button' in k.lower()]
    if button_related_keys:
        logger.info(f"📋 Found button-related keys: {button_related_keys}")
    
    # Then check each expected button
    for key in button_keys:
        if key in ui_components:
            button = ui_components[key]
            
            # Check handlers
            handler_count = 0
            if hasattr(button, '_click_handlers'):
                if hasattr(button._click_handlers, '__len__'):
                    handler_count = len(button._click_handlers)
                else:
                    handler_count = 1 if button._click_handlers else 0
            
            is_disabled = getattr(button, 'disabled', False)
            button_type = type(button).__name__
            
            logger.info(f"   • {key}: type={button_type}, handlers={handler_count}, disabled={is_disabled}")
        else:
            logger.warning(f"   • {key}: NOT FOUND")
    
    # Show suggestions untuk missing buttons
    missing_buttons = [k for k in button_keys if k not in ui_components]
    if missing_buttons:
        logger.warning(f"💡 Missing buttons: {missing_buttons}")
        logger.info("💡 Check UI component creation dalam create_download_ui()")