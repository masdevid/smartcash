"""
File: smartcash/ui/dataset/download/handlers/save_handler.py
Deskripsi: Handler untuk menyimpan konfigurasi download dataset
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.download.utils.logger_helper import log_message

def handle_save_config(ui_components: Dict[str, Any], b: Any = None) -> None:
    """
    Handler untuk menyimpan konfigurasi download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        b: Button widget (opsional)
    """
    try:
        # Buat konfigurasi dari UI
        config = get_config_from_ui(ui_components)
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Muat konfigurasi dataset yang ada
        dataset_config = config_manager.get_module_config('dataset')
        
        # Update bagian download dengan konfigurasi baru
        dataset_config['download'] = config
        
        # Simpan kembali ke config manager
        success = config_manager.save_module_config('dataset', dataset_config)
        
        if success:
            log_message(ui_components, "Konfigurasi download dataset berhasil disimpan", "success", "✅")
        else:
            log_message(ui_components, "Gagal menyimpan konfigurasi download dataset", "warning", "⚠️")
            
    except Exception as e:
        error_msg = f"Error saat menyimpan konfigurasi: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")

def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak konfigurasi dari komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi download
    """
    config = {}
    
    # Ekstrak nilai dari UI components
    if 'url_input' in ui_components:
        config['url'] = ui_components['url_input'].value
    
    if 'dataset_type' in ui_components:
        config['type'] = ui_components['dataset_type'].value
    
    if 'save_path' in ui_components:
        config['save_path'] = ui_components['save_path'].value
    
    if 'auto_extract' in ui_components:
        config['auto_extract'] = ui_components['auto_extract'].value
    
    if 'validate_dataset' in ui_components:
        config['validate'] = ui_components['validate_dataset'].value
    
    return config 