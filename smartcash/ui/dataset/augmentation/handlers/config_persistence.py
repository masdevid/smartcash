"""
File: smartcash/ui/dataset/augmentation/handlers/config_persistence.py
Deskripsi: Helper untuk persistensi konfigurasi augmentasi dengan ConfigManager
"""

from typing import Dict, Any, Optional, List, Union
from smartcash.common.config.manager import get_config_manager

def get_config_manager_instance():
    """Dapatkan instance ConfigManager dengan pendekatan singleton."""
    return get_config_manager()

def validate_augmentation_params(value: Any, default_value: Any, 
                              valid_types: Optional[Union[type, List[type]]] = None, 
                              valid_values: Optional[List[Any]] = None) -> Any:
    """Validasi parameter augmentasi dengan fallback ke nilai default."""
    config_manager = get_config_manager_instance()
    return config_manager.validate_param(value, default_value, valid_types, valid_values)

def ensure_ui_persistence(ui_components: Dict[str, Any], module_name: str = 'augmentation') -> None:
    """Pastikan persistensi UI components dengan mendaftarkannya ke ConfigManager."""
    config_manager = get_config_manager_instance()
    config_manager.register_ui_components(module_name, ui_components)

def get_persisted_ui_components(module_name: str = 'augmentation') -> Dict[str, Any]:
    """Dapatkan UI components yang tersimpan dari ConfigManager."""
    config_manager = get_config_manager_instance()
    return config_manager.get_ui_components(module_name)

def get_augmentation_config(default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Dapatkan konfigurasi augmentasi dari ConfigManager."""
    config_manager = get_config_manager_instance()
    return config_manager.get_module_config('augmentation', default_config)

def save_augmentation_config(config: Dict[str, Any]) -> bool:
    """Simpan konfigurasi augmentasi ke ConfigManager."""
    config_manager = get_config_manager_instance()
    return config_manager.save_module_config('augmentation', config)

def register_config_observer(callback: callable) -> None:
    """Register observer untuk notifikasi perubahan konfigurasi augmentasi."""
    config_manager = get_config_manager_instance()
    config_manager.register_observer('augmentation', callback)

def ensure_valid_aug_types(aug_types: Any) -> List[str]:
    """Pastikan aug_types selalu valid dengan validasi yang kuat."""
    default_aug_types = ['Combined (Recommended)']
    
    # Validasi tipe data
    if aug_types is None:
        return default_aug_types
    
    # Konversi ke list jika string
    if isinstance(aug_types, str):
        return [aug_types]
    
    # Validasi list
    if isinstance(aug_types, (list, tuple)):
        # Filter nilai None dan string kosong
        valid_types = [t for t in aug_types if t is not None and str(t).strip()]
        return valid_types if valid_types else default_aug_types
    
    # Fallback ke default jika tipe tidak dikenali
    return default_aug_types

def safe_convert_type(value: Any, target_type: type, default_value: Any = None) -> Any:
    """Konversi nilai ke tipe target dengan aman.
    
    Args:
        value: Nilai yang akan dikonversi
        target_type: Tipe target (str, int, float, bool)
        default_value: Nilai default jika konversi gagal
        
    Returns:
        Nilai yang sudah dikonversi atau default_value jika konversi gagal
    """
    if value is None:
        return default_value
        
    try:
        if target_type == bool and isinstance(value, str):
            # Konversi string ke boolean dengan lebih aman
            return value.lower() in ('true', 'yes', 'y', '1', 'on')
        return target_type(value)
    except (ValueError, TypeError):
        return default_value

def validate_ui_component_value(component: Any, expected_type: type, default_value: Any = None) -> Any:
    """Validasi nilai komponen UI dengan tipe yang diharapkan.
    
    Args:
        component: Komponen UI
        expected_type: Tipe yang diharapkan
        default_value: Nilai default jika validasi gagal
        
    Returns:
        Nilai yang sudah divalidasi
    """
    if component is None or not hasattr(component, 'value'):
        return default_value
        
    try:
        value = component.value
        return safe_convert_type(value, expected_type, default_value)
    except Exception:
        return default_value
