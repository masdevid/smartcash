"""
File: smartcash/ui/dataset/download/handlers/button_handlers.py
Deskripsi: Fixed button handlers yang memastikan semua action button merespon click
"""

from typing import Dict, Any

def setup_button_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handler tombol dengan error handling yang kuat."""
    
    logger = ui_components.get('logger')
    
    try:
        # Download button - primary action
        _setup_download_handler(ui_components)
        
        # Check button - secondary action  
        _setup_check_handler(ui_components)
        
        # Reset button - form reset
        _setup_reset_handler(ui_components)
        
        # Cleanup button - dataset cleanup
        _setup_cleanup_handler(ui_components)
        
        # Save button - config save
        _setup_save_handler(ui_components)
        
        if logger:
            logger.debug("🔘 Button handlers berhasil disetup")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup button handlers: {str(e)}")
    
    return ui_components

def _setup_download_handler(ui_components: Dict[str, Any]) -> None:
    """Setup download button handler dengan error handling."""
    if 'download_button' in ui_components:
        try:
            def safe_download_handler(b):
                try:
                    from smartcash.ui.dataset.download.handlers.download_action import execute_download_action
                    execute_download_action(ui_components, b)
                except Exception as e:
                    logger = ui_components.get('logger')
                    if logger:
                        logger.error(f"❌ Error download action: {str(e)}")
            
            ui_components['download_button'].on_click(safe_download_handler)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Error setup download handler: {str(e)}")

def _setup_check_handler(ui_components: Dict[str, Any]) -> None:
    """Setup check button handler dengan error handling."""
    if 'check_button' in ui_components:
        try:
            def safe_check_handler(b):
                try:
                    from smartcash.ui.dataset.download.handlers.check_action import execute_check_action
                    execute_check_action(ui_components, b)
                except Exception as e:
                    logger = ui_components.get('logger')
                    if logger:
                        logger.error(f"❌ Error check action: {str(e)}")
            
            ui_components['check_button'].on_click(safe_check_handler)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Error setup check handler: {str(e)}")

def _setup_reset_handler(ui_components: Dict[str, Any]) -> None:
    """Setup reset button handler dengan error handling."""
    if 'reset_button' in ui_components:
        try:
            def safe_reset_handler(b):
                try:
                    from smartcash.ui.dataset.download.handlers.reset_action import execute_reset_action
                    execute_reset_action(ui_components, b)
                except Exception as e:
                    logger = ui_components.get('logger')
                    if logger:
                        logger.error(f"❌ Error reset action: {str(e)}")
            
            ui_components['reset_button'].on_click(safe_reset_handler)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Error setup reset handler: {str(e)}")

def _setup_cleanup_handler(ui_components: Dict[str, Any]) -> None:
    """Setup cleanup button handler dengan error handling."""
    if 'cleanup_button' in ui_components:
        try:
            def safe_cleanup_handler(b):
                try:
                    from smartcash.ui.dataset.download.handlers.cleanup_action import execute_cleanup_action
                    execute_cleanup_action(ui_components, b)
                except Exception as e:
                    logger = ui_components.get('logger')
                    if logger:
                        logger.error(f"❌ Error cleanup action: {str(e)}")
            
            ui_components['cleanup_button'].on_click(safe_cleanup_handler)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Error setup cleanup handler: {str(e)}")

def _setup_save_handler(ui_components: Dict[str, Any]) -> None:
    """Setup save button handler dengan error handling."""
    if 'save_button' in ui_components:
        try:
            def safe_save_handler(b):
                try:
                    from smartcash.ui.dataset.download.handlers.save_action import execute_save_action
                    execute_save_action(ui_components, b)
                except Exception as e:
                    logger = ui_components.get('logger')
                    if logger:
                        logger.error(f"❌ Error save action: {str(e)}")
            
            ui_components['save_button'].on_click(safe_save_handler)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Error setup save handler: {str(e)}")

# Helper functions untuk debugging button state
def debug_button_state(ui_components: Dict[str, Any]) -> None:
    """Debug function untuk check button state."""
    logger = ui_components.get('logger')
    if not logger:
        return
    
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components:
            button = ui_components[key]
            logger.debug(f"🔘 {key}: exists={hasattr(button, 'on_click')}, disabled={getattr(button, 'disabled', 'unknown')}")
        else:
            logger.debug(f"❌ {key}: not found in ui_components")

def ensure_buttons_enabled(ui_components: Dict[str, Any]) -> None:
    """Pastikan semua button dalam keadaan enabled."""
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = False