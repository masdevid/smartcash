"""
File: smartcash/ui/dataset/augmentation/handlers/config_updater_new.py
Deskripsi: Pembaruan UI components dari konfigurasi augmentasi menggunakan utilitas config terstandarisasi
"""

from typing import Dict, Any, List, Tuple

from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui


def update_augmentation_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI components dari config dictionary menggunakan utilitas terstandarisasi
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Dictionary konfigurasi yang akan digunakan untuk update UI
    """
    # Mapping config paths ke UI component keys
    # Format: (config_path, ui_component_key)
    mappings: List[Tuple[str, str]] = [
        # Parameter dasar
        ('augmentation.num_variations', 'num_variations'),
        ('augmentation.target_count', 'target_count'),
        ('augmentation.output_prefix', 'output_prefix'),
        ('augmentation.output_dir', 'output_dir'),
        ('augmentation.num_workers', 'num_workers'),
        ('augmentation.balance_classes', 'balance_classes'),
        ('augmentation.move_to_preprocessed', 'move_to_preprocessed'),
        
        # Parameter augmentasi posisi
        ('augmentation.position.fliplr', 'fliplr'),
        ('augmentation.position.degrees', 'degrees'),
        ('augmentation.position.translate', 'translate'),
        ('augmentation.position.scale', 'scale'),
        ('augmentation.position.shear_max', 'shear_max'),
        
        # Parameter augmentasi pencahayaan
        ('augmentation.lighting.hsv_h', 'hsv_h'),
        ('augmentation.lighting.hsv_s', 'hsv_s'),
        ('augmentation.lighting.hsv_v', 'hsv_v'),
        ('augmentation.lighting.blur', 'blur'),
        ('augmentation.lighting.noise', 'noise'),
        
        # Pengaturan cleanup
        ('cleanup.backup_enabled', 'backup_enabled'),
        ('cleanup.backup_count', 'backup_count'),
        
        # Pengaturan visualisasi
        ('visualization.enabled', 'visualization_enabled'),
        ('visualization.sample_count', 'sample_count'),
        ('visualization.save_visualizations', 'save_visualizations'),
        
        # Pengaturan performa
        ('performance.batch_size', 'batch_size'),
        ('performance.use_gpu', 'use_gpu'),
    ]
    
    # Update UI components dengan nilai dari config
    for config_path, ui_key in mappings:
        # Ambil nilai dari config dengan nested path
        value = get_nested_value(config, config_path)
        
        # Set nilai ke UI component jika nilai ada
        if value is not None:
            safe_set_value(ui_components, ui_key, value)
    
    # Special handling untuk augmentation types
    try:
        aug_types = get_nested_value(config, 'augmentation.types')
        if aug_types and isinstance(aug_types, list):
            # Coba update berbagai kemungkinan nama widget
            for widget_name in ['augmentation_types', 'types_widget', 'aug_types', 'augmentation_type']:
                if widget_name in ui_components:
                    safe_set_value(ui_components, widget_name, aug_types)
                    break
    except Exception as e:
        # Log error tapi lanjutkan
        log_to_ui(ui_components, f"⚠️ Gagal mengupdate tipe augmentasi: {str(e)}", "warning")


def reset_augmentation_ui(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI components ke default konfigurasi augmentasi
    
    Args:
        ui_components: Dictionary berisi UI components yang akan di-reset
    """
    try:
        # Dapatkan default config dari defaults.py
        default_config = get_default_augmentation_config()
        
        # Update UI dengan default config
        update_augmentation_ui(ui_components, default_config)
        log_to_ui(ui_components, "🔄 UI augmentasi direset ke default", "success")
    except Exception as e:
        # Fallback to basic reset jika config manager gagal
        log_to_ui(ui_components, f"⚠️ Error saat reset UI: {str(e)}", "warning")
        _apply_basic_defaults(ui_components)

