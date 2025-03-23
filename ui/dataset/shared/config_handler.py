"""
File: smartcash/ui/dataset/shared/config_handler.py
Deskripsi: Utilitas shared untuk mengelola konfigurasi pada modul dataset
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import yaml
from pathlib import Path

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.shared.status_panel import update_status_panel

def save_config_handler(ui_components: Dict[str, Any], module: str, config_path: str, config: Dict[str, Any] = None):
    """
    Handler untuk menyimpan konfigurasi dari UI ke file.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        module: Nama modul ('preprocessing' atau 'augmentation')
        config_path: Path ke file konfigurasi
        config: Konfigurasi dari luar (opsional)
    """
    logger = ui_components.get('logger')
    
    try:
        # Import fungsi update config berdasarkan modul
        if module == 'preprocessing':
            from smartcash.ui.dataset.preprocessing_config_handler import update_config_from_ui, save_preprocessing_config
            updated_config = update_config_from_ui(ui_components, ui_components.get('config', config))
            success = save_preprocessing_config(updated_config, config_path)
        else:  # augmentation
            from smartcash.ui.dataset.augmentation_config_handler import update_config_from_ui, save_augmentation_config
            updated_config = update_config_from_ui(ui_components, ui_components.get('config', config))
            updated_config['logger'] = logger
            success = save_augmentation_config(updated_config, config_path)
            
        # Simpan konfigurasi yang sudah diupdate ke ui_components
        ui_components['config'] = updated_config
        
        # Update UI
        if success:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Konfigurasi berhasil disimpan ke {config_path}"))
            
            # Update status panel
            update_status_panel(
                ui_components, 
                "success", 
                f"{ICONS.get('success', '✅')} Konfigurasi {module} berhasil disimpan ke {config_path}"
            )
            
            # Log
            if logger: logger.success(f"{ICONS.get('success', '✅')} Konfigurasi {module} berhasil disimpan ke {config_path}")
        else:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi"))
            
            if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi")
    except Exception as e:
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
        
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error menyimpan konfigurasi: {str(e)}")