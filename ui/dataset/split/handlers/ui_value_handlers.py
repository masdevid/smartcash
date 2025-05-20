"""
File: smartcash/ui/dataset/split/handlers/ui_value_handlers.py
Deskripsi: Handler untuk nilai UI di split dataset
"""

from typing import Dict, Any, Optional, Tuple, List
from smartcash.common.logger import get_logger
from smartcash.common.utils import deep_merge

logger = get_logger(__name__)

def get_ui_values(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan nilai dari komponen UI split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary berisi nilai dari komponen UI
    """
    ui_values = {}
    
    # Mapping komponen UI ke kunci konfigurasi
    component_mappings = {
        'train_ratio': ['train_slider', 'train_ratio_slider'],
        'val_ratio': ['val_slider', 'val_ratio_slider'],
        'test_ratio': ['test_slider', 'test_ratio_slider'],
        'random_seed': ['random_seed', 'random_seed_input'],
        'stratify': ['stratified_checkbox', 'stratify_checkbox'],
        'enabled': ['enabled_checkbox']
    }
    
    # Ekstrak nilai dari komponen UI berdasarkan mapping
    for config_key, component_keys in component_mappings.items():
        for component_key in component_keys:
            if component_key in ui_components:
                ui_values[config_key] = ui_components[component_key].value
                break
    
    # Default untuk enabled jika tidak ada
    if 'enabled' not in ui_values:
        ui_values['enabled'] = True
    
    return ui_values

def verify_config_consistency(ui_values: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verifikasi konsistensi antara nilai UI dan konfigurasi.
    
    Args:
        ui_values: Dictionary berisi nilai dari komponen UI
        config: Dictionary berisi konfigurasi
        
    Returns:
        Tuple (is_consistent, inconsistent_keys)
    """
    is_consistent = True
    inconsistent_keys = []
    
    if 'split' not in config:
        return False, ['split']
    
    for key, value in ui_values.items():
        if key in config['split'] and config['split'][key] != value:
            is_consistent = False
            inconsistent_keys.append(key)
    
    return is_consistent, inconsistent_keys

def create_config_from_ui_values(ui_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat konfigurasi dari nilai UI.
    
    Args:
        ui_values: Dictionary berisi nilai dari komponen UI
        
    Returns:
        Dictionary berisi konfigurasi
    """
    return {
        'split': ui_values
    }

def merge_config_with_ui_values(config: Dict[str, Any], ui_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gabungkan konfigurasi yang ada dengan nilai UI.
    
    Args:
        config: Konfigurasi yang sudah ada
        ui_values: Nilai dari UI
        
    Returns:
        Konfigurasi yang telah digabungkan
    """
    ui_config = create_config_from_ui_values(ui_values)
    
    # Jika config kosong, langsung kembalikan ui_config
    if not config:
        return ui_config
    
    # Gunakan deep_merge untuk menggabungkan konfigurasi
    try:
        merged_config = deep_merge(config, ui_config)
        return merged_config
    except Exception as e:
        logger.error(f"❌ Error saat menggabungkan konfigurasi: {str(e)}")
        # Fallback ke update manual jika deep_merge gagal
        if 'split' not in config:
            config['split'] = {}
        config['split'].update(ui_values)
        return config 