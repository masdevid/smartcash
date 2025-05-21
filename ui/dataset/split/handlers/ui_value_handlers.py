"""
File: smartcash/ui/dataset/split/handlers/ui_value_handlers.py
Deskripsi: Handler untuk nilai UI di split dataset
"""

from typing import Dict, Any, Optional, Tuple, List
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

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
        'enabled': ['enabled_checkbox'],
        'backup_before_split': ['backup_checkbox'],
        'backup_dir': ['backup_dir'],
        'dataset_path': ['dataset_path'],
        'preprocessed_path': ['preprocessed_path']
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
    
    # Log hasil ekstraksi untuk debugging
    logger.debug(f"{ICONS.get('info', 'ℹ️')} Nilai UI yang diambil: {ui_values}")
    
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
            logger.warning(f"{ICONS.get('warning', '⚠️')} Inkonsistensi pada '{key}': UI={value}, Config={config['split'][key]}")
    
    return is_consistent, inconsistent_keys

def create_config_from_ui_values(ui_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat konfigurasi dari nilai UI.
    
    Args:
        ui_values: Dictionary berisi nilai dari komponen UI
        
    Returns:
        Dictionary berisi konfigurasi
    """
    # Pastikan nilai total rasio adalah 1.0
    if all(k in ui_values for k in ['train_ratio', 'val_ratio', 'test_ratio']):
        total = round(ui_values['train_ratio'] + ui_values['val_ratio'] + ui_values['test_ratio'], 2)
        if total != 1.0:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Total rasio tidak sama dengan 1.0: {total}, akan dinormalisasi")
            
            # Normalisasi nilai rasio
            factor = 1.0 / total
            ui_values['train_ratio'] = round(ui_values['train_ratio'] * factor, 2)
            ui_values['val_ratio'] = round(ui_values['val_ratio'] * factor, 2)
            ui_values['test_ratio'] = round(ui_values['test_ratio'] * factor, 2)
            
            logger.info(f"{ICONS.get('info', 'ℹ️')} Normalisasi rasio: train={ui_values['train_ratio']}, val={ui_values['val_ratio']}, test={ui_values['test_ratio']}")
    
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
    # Jika config kosong, langsung kembalikan ui_config
    if not config:
        return create_config_from_ui_values(ui_values)
    
    # Buat copy dari config untuk mencegah modifikasi tanpa sengaja
    merged_config = config.copy()
    
    # Pastikan ada key 'split'
    if 'split' not in merged_config:
        merged_config['split'] = {}
    
    # Update nilai yang ada di ui_values
    merged_config['split'].update(ui_values)
    
    return merged_config