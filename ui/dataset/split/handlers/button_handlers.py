"""
File: smartcash/ui/dataset/split/handlers/button_handlers.py
Deskripsi: Handler untuk button di split dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.ui.dataset.split.handlers.config_handlers import load_config, save_config, update_ui_from_config

logger = get_logger(__name__)

def setup_button_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk button di split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Reset button handler
        if 'reset_button' in ui_components:
            def on_reset_clicked(b):
                try:
                    # Load default config
                    config = load_config()
                    # Update UI
                    update_ui_from_config(ui_components, config)
                    # Save config
                    save_config(config)
                    # Show success message
                    display(create_info_alert(
                        f"{ICONS.get('success', '✅')} Konfigurasi berhasil direset ke default",
                        alert_type='success'
                    ))
                except Exception as e:
                    logger.error(f"{ICONS.get('error', '❌')} Error saat reset konfigurasi: {str(e)}")
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Error saat reset konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
            
            ui_components['reset_button'].on_click(on_reset_clicked)
            
        # Save button handler
        if 'save_button' in ui_components:
            def on_save_clicked(b):
                try:
                    # Get current config
                    config = load_config()
                    
                    # Update config from UI
                    if 'enabled_checkbox' in ui_components:
                        config['split']['enabled'] = ui_components['enabled_checkbox'].value
                        
                    if 'train_ratio_slider' in ui_components:
                        config['split']['train_ratio'] = ui_components['train_ratio_slider'].value
                        
                    if 'val_ratio_slider' in ui_components:
                        config['split']['val_ratio'] = ui_components['val_ratio_slider'].value
                        
                    if 'test_ratio_slider' in ui_components:
                        config['split']['test_ratio'] = ui_components['test_ratio_slider'].value
                        
                    if 'random_seed_input' in ui_components:
                        config['split']['random_seed'] = ui_components['random_seed_input'].value
                        
                    if 'stratify_checkbox' in ui_components:
                        config['split']['stratify'] = ui_components['stratify_checkbox'].value
                    
                    # Save config
                    save_config(config)
                    
                    # Show success message
                    display(create_info_alert(
                        f"{ICONS.get('success', '✅')} Konfigurasi berhasil disimpan",
                        alert_type='success'
                    ))
                except Exception as e:
                    logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}")
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
            
            ui_components['save_button'].on_click(on_save_clicked)
            
        logger.info(f"{ICONS.get('success', '✅')} Button handlers berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup button handlers: {str(e)}")
        return ui_components
