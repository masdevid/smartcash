"""
File: smartcash/ui/dataset/download/handlers/button_handlers.py
Deskripsi: Button handlers yang diperbaiki dengan registrasi yang tepat dan error handling yang kuat
"""

from typing import Dict, Any

def setup_button_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handler tombol dengan registrasi yang tepat dan error handling."""
    
    logger = ui_components.get('logger')
    
    try:
        # 🔘 Setup handlers dalam urutan prioritas
        handlers_setup = [
            ('download_button', 'download', _setup_download_handler),
            ('check_button', 'check', _setup_check_handler),
            ('reset_button', 'reset', _setup_reset_handler),
            ('cleanup_button', 'cleanup', _setup_cleanup_handler),
            ('save_button', 'save', _setup_save_handler)
        ]
        
        # Setup setiap handler dengan error handling individual
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
                        logger.warning(f"⚠️ {button_key} tidak ditemukan dalam ui_components")
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error setup {action_name} handler: {str(e)}")
        
        if logger:
            logger.info(f"🔘 Button handlers setup: {success_count}/{len(handlers_setup)} berhasil")
        
        # Ensure all buttons are enabled after setup
        _ensure_buttons_enabled(ui_components)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup button handlers: {str(e)}")
    
    return ui_components

def _setup_download_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup download button dengan handler yang reliable."""
    def download_handler(button):
        try:
            # Import di dalam handler untuk menghindari circular import
            from smartcash.ui.dataset.download.handlers.download_action import execute_download_action
            execute_download_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Download handler error: {str(e)}")
            # Re-enable button jika error
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    # Clear existing handlers dan register yang baru
    button = ui_components['download_button']
    button._click_handlers.clear() if hasattr(button, '_click_handlers') else None
    button.on_click(download_handler)

def _setup_check_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup check button dengan handler yang reliable."""
    def check_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.check_action import execute_check_action
            execute_check_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Check handler error: {str(e)}")
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    button = ui_components['check_button']
    button._click_handlers.clear() if hasattr(button, '_click_handlers') else None
    button.on_click(check_handler)

def _setup_reset_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup reset button dengan handler yang reliable."""
    def reset_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.reset_action import execute_reset_action
            execute_reset_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Reset handler error: {str(e)}")
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    button = ui_components['reset_button']
    button._click_handlers.clear() if hasattr(button, '_click_handlers') else None
    button.on_click(reset_handler)

def _setup_cleanup_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup cleanup button dengan handler yang reliable."""
    def cleanup_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.cleanup_action import execute_cleanup_action
            execute_cleanup_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Cleanup handler error: {str(e)}")
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    button = ui_components['cleanup_button']
    button._click_handlers.clear() if hasattr(button, '_click_handlers') else None
    button.on_click(cleanup_handler)

def _setup_save_handler(ui_components: Dict[str, Any], env=None) -> None:
    """Setup save button dengan handler yang reliable."""
    def save_handler(button):
        try:
            from smartcash.ui.dataset.download.handlers.save_action import execute_save_action
            execute_save_action(ui_components, button)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Save handler error: {str(e)}")
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    button = ui_components['save_button']
    button._click_handlers.clear() if hasattr(button, '_click_handlers') else None
    button.on_click(save_handler)

def _ensure_buttons_enabled(ui_components: Dict[str, Any]) -> None:
    """Pastikan semua button dalam keadaan enabled setelah setup."""
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = False

def debug_button_connections(ui_components: Dict[str, Any]) -> None:
    """Debug function untuk memeriksa koneksi button handlers."""
    logger = ui_components.get('logger')
    if not logger:
        return
    
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    logger.info("🔍 Debug button connections:")
    for key in button_keys:
        if key in ui_components:
            button = ui_components[key]
            has_handlers = len(getattr(button, '_click_handlers', [])) > 0
            is_disabled = getattr(button, 'disabled', False)
            logger.info(f"   • {key}: handlers={has_handlers}, disabled={is_disabled}")
        else:
            logger.warning(f"   • {key}: NOT FOUND")