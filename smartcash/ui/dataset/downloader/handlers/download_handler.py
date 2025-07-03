"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Entry point untuk download handlers dengan operation/ structure
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.downloader.handlers.operation.manager import DownloadHandlerManager
from smartcash.ui.dataset.downloader.utils.validation_utils import validate_config

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan centralized error handling dan SRP.
    
    Args:
        ui_components: Dictionary UI components
        config: Configuration dictionary
        env: Optional environment
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Create handler manager
    handler_manager = DownloadHandlerManager(ui_components=ui_components)
    
    # Setup button callbacks
    _setup_button_callbacks(ui_components, handler_manager)
    
    # Store handler manager di ui_components untuk referensi
    ui_components['download_handler_manager'] = handler_manager
    
    return ui_components

def _setup_button_callbacks(ui_components: Dict[str, Any], handler_manager: DownloadHandlerManager) -> None:
    """Setup button callbacks untuk semua operasi.
    
    Args:
        ui_components: Dictionary UI components
        handler_manager: Download handler manager instance
    """
    # Setup download button callback
    download_button = ui_components.get('download_button')
    if download_button:
        download_button.on_click(lambda b: _handle_download_button(ui_components, handler_manager))
    
    # Setup check button callback
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(lambda b: _handle_check_button(ui_components, handler_manager))
    
    # Setup cleanup button callback
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(lambda b: _handle_cleanup_button(ui_components, handler_manager))
    
    # Setup config buttons
    _setup_config_buttons(ui_components)

def _setup_config_buttons(ui_components: Dict[str, Any]) -> None:
    """Setup config save/reset button callbacks.
    
    Args:
        ui_components: Dictionary UI components
    """
    # Setup save config button
    save_button = ui_components.get('save_button')
    if save_button:
        save_button.on_click(lambda b: _handle_save_config(ui_components))
    
    # Setup reset config button
    reset_button = ui_components.get('reset_button')
    if reset_button:
        reset_button.on_click(lambda b: _handle_reset_config(ui_components))

def _handle_download_button(ui_components: Dict[str, Any], handler_manager: DownloadHandlerManager) -> None:
    """Handle download button click.
    
    Args:
        ui_components: Dictionary UI components
        handler_manager: Download handler manager instance
    """
    # Clear outputs using handler method
    self.clear_ui_outputs
    
    # Extract dan validate config
    config_handler = ui_components.get('config_handler')
    ui_config = config_handler.extract_config(ui_components)
    
    # Gunakan centralized validation module
    try:
        validation = config_handler.validate_config(ui_config)
    except Exception as e:
        # Fallback ke direct validation jika handler validation gagal
        handler_manager.log_warning(f"Menggunakan direct validation: {str(e)}")
        validation = validate_config(ui_config)
    
    if not validation.get('status', False):
        handler_manager.log_error(f"Config tidak valid: {', '.join(validation.get('errors', []))}")
        return
    
    # Handle download dengan manager
    handler_manager.handle_download_button(ui_config)

def _handle_check_button(ui_components: Dict[str, Any], handler_manager: DownloadHandlerManager) -> None:
    """Handle check button click.
    
    Args:
        ui_components: Dictionary UI components
        handler_manager: Download handler manager instance
    """
    # Clear outputs using handler method
    handler_manager.clear_outputs()
    
    # Handle check dengan manager
    handler_manager.handle_check_button()

def _handle_cleanup_button(ui_components: Dict[str, Any], handler_manager: DownloadHandlerManager) -> None:
    """Handle cleanup button click.
    
    Args:
        ui_components: Dictionary UI components
        handler_manager: Download handler manager instance
    """
    # Clear outputs using handler method
    handler_manager.clear_outputs()
    
    # Handle cleanup dengan manager
    handler_manager.handle_cleanup_button()

def _handle_save_config(ui_components: Dict[str, Any]) -> None:
    """Handle save config button click.
    
    Args:
        ui_components: Dictionary UI components
    """
    config_handler = ui_components.get('config_handler')
    if config_handler:
        config_handler.save_config(ui_components)

def _handle_reset_config(ui_components: Dict[str, Any]) -> None:
    """Handle reset config button click.
    
    Args:
        ui_components: Dictionary UI components
    """
    config_handler = ui_components.get('config_handler')
    if config_handler:
        config_handler.reset_config(ui_components)
