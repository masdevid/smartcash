"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Handler konfigurasi dengan logger bridge dan ekspor fungsi lengkap (SRP)
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

def save_augmentation_config(ui_components: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi augmentasi ke file dengan logger bridge.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika berhasil, False jika gagal
    """
    ui_logger = create_ui_logger_bridge(ui_components, "config_handler")
    
    try:
        ui_logger.info("💾 Menyimpan konfigurasi augmentasi...")
        
        # Ambil konfigurasi dari UI
        config = _get_config_from_ui(ui_components, ui_logger)
        
        # Simpan menggunakan ConfigManager
        config_manager = get_config_manager()
        result = config_manager.save_config(config, 'augmentation')
        
        if result:
            ui_logger.success("✅ Konfigurasi berhasil disimpan ke Google Drive")
            _verify_saved_config(config_manager, ui_logger)
        else:
            ui_logger.error("❌ Gagal menyimpan konfigurasi")
        
        return result
        
    except Exception as e:
        ui_logger.error(f"❌ Error saat menyimpan: {str(e)}")
        return False

def load_augmentation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load konfigurasi augmentasi dari Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi augmentasi
    """
    ui_logger = create_ui_logger_bridge(ui_components, "config_handler")
    
    try:
        ui_logger.info("📂 Memuat konfigurasi dari Google Drive...")
        
        config_manager = get_config_manager()
        config = config_manager.get_config('augmentation', reload=True)
        
        augmentation_config = config.get('augmentation', {}) if config else {}
        
        if augmentation_config:
            ui_logger.success("✅ Konfigurasi berhasil dimuat dari Google Drive")
            update_ui_from_config(ui_components, augmentation_config, ui_logger)
        else:
            ui_logger.warning("⚠️ Config tidak ditemukan, menggunakan default")
            augmentation_config = _get_default_augmentation_config()
        
        return augmentation_config
        
    except Exception as e:
        ui_logger.error(f"❌ Error memuat config: {str(e)}")
        return _get_default_augmentation_config()

def reset_augmentation_config(ui_components: Dict[str, Any]) -> bool:
    """
    Reset konfigurasi augmentasi ke default.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika berhasil, False jika gagal
    """
    ui_logger = create_ui_logger_bridge(ui_components, "config_handler")
    
    try:
        ui_logger.info("🔄 Mereset konfigurasi ke default...")
        
        # Get default config
        default_config = _get_default_augmentation_config()
        
        # Update UI dengan default values
        update_ui_from_config(ui_components, default_config, ui_logger)
        
        # Simpan default config ke file
        config_manager = get_config_manager()
        full_config = {'augmentation': default_config}
        
        result = config_manager.save_config(full_config, 'augmentation')
        
        if result:
            ui_logger.success("✅ Konfigurasi berhasil direset dan disimpan")
            _verify_saved_config(config_manager, ui_logger)
        else:
            ui_logger.error("❌ Gagal menyimpan config default")
            
        return result
        
    except Exception as e:
        ui_logger.error(f"❌ Error saat reset: {str(e)}")
        return False

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], ui_logger=None) -> None:
    """
    Update UI components dari nilai konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Dictionary konfigurasi
        ui_logger: UI Logger bridge (opsional)
    """
    # Setup logger jika tidak ada
    if ui_logger is None:
        ui_logger = create_ui_logger_bridge(ui_components, "config_handler")
    
    try:
        ui_logger.debug("🔄 Mengupdate UI dari konfigurasi...")
        
        # Update basic fields
        ui_mappings = {
            'num_variations': 'num_variations',
            'target_count': 'target_count',
            'output_prefix': 'output_prefix', 
            'balance_classes': 'balance_classes',
            'validate_results': 'validate_results'
        }
        
        for ui_field, config_key in ui_mappings.items():
            if ui_field in ui_components and config_key in config:
                widget = ui_components[ui_field]
                if hasattr(widget, 'value'):
                    widget.value = config[config_key]
                    ui_logger.debug(f"🔄 UI {ui_field} → {config[config_key]}")
        
        # Update augmentation types
        if 'augmentation_types' in ui_components and 'types' in config:
            widget = ui_components['augmentation_types']
            if hasattr(widget, 'value'):
                types_value = config['types']
                # Ensure types is a list/tuple untuk SelectMultiple widget
                if isinstance(types_value, str):
                    types_value = [types_value]
                widget.value = types_value
                ui_logger.debug(f"🎯 UI types → {types_value}")
        
        # Update target split
        if 'target_split' in ui_components and 'target_split' in config:
            widget = ui_components['target_split']
            if hasattr(widget, 'value'):
                widget.value = config['target_split']
                ui_logger.debug(f"📂 UI target_split → {config['target_split']}")
        
        ui_logger.success("✅ UI berhasil diupdate dari konfigurasi")
        
    except Exception as e:
        ui_logger.error(f"❌ Error update UI: {str(e)}")

def _get_config_from_ui(ui_components: Dict[str, Any], ui_logger) -> Dict[str, Any]:
    """Dapatkan konfigurasi augmentasi dari UI components."""
    try:
        ui_logger.debug("📋 Mengambil konfigurasi dari UI...")
        
        # Ambil konfigurasi base
        config_manager = get_config_manager()
        base_config = config_manager.get_config('augmentation') or {}
        
        # Pastikan struktur augmentation ada
        if 'augmentation' not in base_config:
            base_config['augmentation'] = {}
        
        augmentation_config = base_config['augmentation']
        
        # Update dari UI components
        _update_config_from_ui_fields(ui_components, augmentation_config, ui_logger)
        
        ui_logger.debug("📋 Konfigurasi berhasil diambil dari UI")
        return base_config
        
    except Exception as e:
        ui_logger.error(f"❌ Error mengambil config dari UI: {str(e)}")
        raise

def _update_config_from_ui_fields(ui_components: Dict[str, Any], config: Dict[str, Any], ui_logger) -> None:
    """Update konfigurasi dari field UI."""
    # Mapping UI fields ke config keys
    ui_field_mappings = {
        'num_variations': 'num_variations',
        'target_count': 'target_count', 
        'output_prefix': 'output_prefix',
        'balance_classes': 'balance_classes',
        'validate_results': 'validate_results',
        'process_bboxes': 'process_bboxes'
    }
    
    # Update basic fields
    for ui_field, config_key in ui_field_mappings.items():
        if ui_field in ui_components and hasattr(ui_components[ui_field], 'value'):
            config[config_key] = ui_components[ui_field].value
            ui_logger.debug(f"🔄 {config_key}: {config[config_key]}")
    
    # Update augmentation types
    if 'augmentation_types' in ui_components:
        widget = ui_components['augmentation_types']
        if hasattr(widget, 'value'):
            config['types'] = list(widget.value) if widget.value else ['combined']
            ui_logger.debug(f"🎯 Types: {config['types']}")
    
    # Update target split
    if 'target_split' in ui_components:
        widget = ui_components['target_split'] 
        if hasattr(widget, 'value'):
            config['target_split'] = widget.value
            ui_logger.debug(f"📂 Target split: {config['target_split']}")

def _verify_saved_config(config_manager, ui_logger) -> None:
    """Verifikasi bahwa konfigurasi berhasil tersimpan."""
    try:
        saved_config = config_manager.get_config('augmentation', reload=True)
        
        if saved_config and 'augmentation' in saved_config:
            ui_logger.debug("🔍 Verifikasi: Konfigurasi berhasil tersimpan")
        else:
            ui_logger.warning("⚠️ Verifikasi: Struktur config tidak lengkap")
            
    except Exception as e:
        ui_logger.warning(f"⚠️ Gagal verifikasi config: {str(e)}")

def _get_default_augmentation_config() -> Dict[str, Any]:
    """Dapatkan konfigurasi default augmentasi."""
    return {
        'enabled': True,
        'num_variations': 2,
        'output_prefix': 'aug',
        'process_bboxes': True,
        'output_dir': 'data/augmented',
        'validate_results': True,
        'resume': False,
        'num_workers': 4,
        'balance_classes': False,
        'target_count': 1000,
        'target_split': 'train',
        'types': ['combined']
    }

# Alias untuk backward compatibility
_update_ui_from_config_values = update_ui_from_config