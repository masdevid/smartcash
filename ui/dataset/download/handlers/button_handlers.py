"""
File: smartcash/ui/dataset/download/handlers/button_handlers.py
Deskripsi: Handler tombol-tombol download yang dipecah menjadi unit SRP kecil
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.handlers.download_action import execute_download_action
from smartcash.ui.dataset.download.handlers.check_action import execute_check_action  
from smartcash.ui.dataset.download.handlers.reset_action import execute_reset_action
from smartcash.ui.dataset.download.handlers.cleanup_action import execute_cleanup_action
from smartcash.ui.dataset.download.handlers.save_action import execute_save_action

def setup_button_handlers(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handler tombol dengan delegasi ke action yang spesifik."""
    
    # Download button
    if 'download_button' in ui_components:
        ui_components['download_button'].on_click(
            lambda b: execute_download_action(ui_components, b)
        )
    
    # Check button  
    if 'check_button' in ui_components:
        ui_components['check_button'].on_click(
            lambda b: execute_check_action(ui_components, b)
        )
    
    # Reset button
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: execute_reset_action(ui_components, b)
        )
        
    # Cleanup button
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(
            lambda b: execute_cleanup_action(ui_components, b)
        )
        
    # Save button
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: execute_save_action(ui_components, b)
        )
    
    return ui_components