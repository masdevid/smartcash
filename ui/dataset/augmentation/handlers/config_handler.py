"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk augmentasi dataset dengan save/reset yang diperbaiki untuk Google Drive
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel

def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Dapatkan konfigurasi augmentasi dari UI components."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        update_status_panel(ui_components, "📋 Mengambil konfigurasi dari UI...", "info")
        
        # Ambil konfigurasi base
        config_manager = get_config_manager()
        base_config = config_manager.get_config('augmentation') or {}
        
        # Pastikan struktur augmentation ada
        if 'augmentation' not in base_config:
            base_config['augmentation'] = {}
        
        augmentation_config = base_config['augmentation']
        
        # Update dari UI components
        _update_config_from_ui_fields(ui_components, augmentation_config)
        
        log_message(ui_components, "📋 Konfigurasi berhasil diambil dari UI", "success", "✅")
        return base_config
        
    except Exception as e:
        log_message(ui_components, f"❌ Error mengambil config dari UI: {str(e)}", "error")
        raise

def _update_config_from_ui_fields(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
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
            log_message(ui_components, f"🔄 {config_key}: {config[config_key]}", "debug")
    
    # Update augmentation types
    if 'augmentation_types' in ui_components:
        widget = ui_components['augmentation_types']
        if hasattr(widget, 'value'):
            config['types'] = list(widget.value) if widget.value else ['combined']
            log_message(ui_components, f"🎯 Types: {config['types']}", "debug")
    
    # Update target split
    if 'target_split' in ui_components:
        widget = ui_components['target_split'] 
        if hasattr(widget, 'value'):
            config['target_split'] = widget.value
            log_message(ui_components, f"📂 Target split: {config['target_split']}", "debug")

def save_augmentation_config(ui_components: Dict[str, Any]) -> bool:
    """Simpan konfigurasi augmentasi ke file dengan sinkronisasi Google Drive."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        update_status_panel(ui_components, "💾 Menyimpan konfigurasi augmentasi...", "info")
        log_message(ui_components, "💾 Memulai proses penyimpanan konfigurasi", "info")
        
        # Ambil konfigurasi dari UI
        config = get_config_from_ui(ui_components)
        
        # Simpan menggunakan ConfigManager yang otomatis handle Google Drive
        config_manager = get_config_manager()
        
        # Simpan ke file augmentation_config.yaml
        result = config_manager.save_config(config, 'augmentation')
        
        if result:
            log_message(ui_components, "✅ Konfigurasi berhasil disimpan ke Google Drive", "success")
            update_status_panel(ui_components, "✅ Konfigurasi tersimpan dan tersinkronisasi", "success")
            
            # Verifikasi file tersimpan dengan membaca ulang
            _verify_saved_config(ui_components, config_manager)
            
        else:
            log_message(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
            update_status_panel(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
        
        return result
        
    except Exception as e:
        log_message(ui_components, f"❌ Error saat menyimpan: {str(e)}", "error")
        update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
        return False

def _verify_saved_config(ui_components: Dict[str, Any], config_manager) -> None:
    """Verifikasi bahwa konfigurasi berhasil tersimpan."""
    try:
        # Baca ulang konfigurasi yang baru disimpan
        saved_config = config_manager.get_config('augmentation', reload=True)
        
        if saved_config and 'augmentation' in saved_config:
            log_message(ui_components, "🔍 Verifikasi: Konfigurasi berhasil tersimpan", "success")
            
            # Log beberapa key fields untuk verifikasi
            aug_config = saved_config['augmentation']
            log_message(ui_components, f"📊 Variasi: {aug_config.get('num_variations', 'N/A')}", "debug")
            log_message(ui_components, f"🎯 Jenis: {aug_config.get('types', 'N/A')}", "debug")
        else:
            log_message(ui_components, "⚠️ Verifikasi: Struktur config tidak lengkap", "warning")
            
    except Exception as e:
        log_message(ui_components, f"⚠️ Gagal verifikasi config: {str(e)}", "warning")

def reset_augmentation_config(ui_components: Dict[str, Any]) -> bool:
    """Reset konfigurasi augmentasi ke default dan simpan ke Google Drive."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        update_status_panel(ui_components, "🔄 Mereset konfigurasi ke default...", "info")
        log_message(ui_components, "🔄 Memulai reset konfigurasi augmentasi", "info")
        
        # Get default config
        default_config = _get_default_augmentation_config()
        
        # Update UI dengan default values
        _update_ui_from_config_values(ui_components, default_config)
        
        # Simpan default config ke file
        config_manager = get_config_manager()
        full_config = {'augmentation': default_config}
        
        result = config_manager.save_config(full_config, 'augmentation')
        
        if result:
            log_message(ui_components, "✅ Konfigurasi berhasil direset dan disimpan", "success")
            update_status_panel(ui_components, "✅ Konfigurasi direset ke default", "success")
            
            # Verifikasi reset
            _verify_saved_config(ui_components, config_manager)
        else:
            log_message(ui_components, "❌ Gagal menyimpan config default", "error")
            update_status_panel(ui_components, "❌ Gagal menyimpan reset", "error")
            
        return result
        
    except Exception as e:
        log_message(ui_components, f"❌ Error saat reset: {str(e)}", "error")
        update_status_panel(ui_components, f"❌ Error reset: {str(e)}", "error")
        return False

def load_augmentation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Load konfigurasi augmentasi dari Google Drive."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        update_status_panel(ui_components, "📂 Memuat konfigurasi dari Google Drive...", "info")
        log_message(ui_components, "📂 Memuat konfigurasi augmentasi", "info")
        
        config_manager = get_config_manager()
        config = config_manager.get_config('augmentation', reload=True)  # Force reload dari Drive
        
        augmentation_config = config.get('augmentation', {}) if config else {}
        
        if augmentation_config:
            log_message(ui_components, "✅ Konfigurasi berhasil dimuat dari Google Drive", "success")
            update_status_panel(ui_components, "✅ Konfigurasi dimuat dari Drive", "success")
            
            # Update UI dengan config yang dimuat
            _update_ui_from_config_values(ui_components, augmentation_config)
            
        else:
            log_message(ui_components, "⚠️ Config tidak ditemukan, menggunakan default", "warning")
            augmentation_config = _get_default_augmentation_config()
            
        return augmentation_config
        
    except Exception as e:
        log_message(ui_components, f"❌ Error memuat config: {str(e)}", "error")
        return _get_default_augmentation_config()

def _update_ui_from_config_values(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari nilai konfigurasi."""
    
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
                log_message(ui_components, f"🔄 UI {ui_field} → {config[config_key]}", "debug")
    
    # Update augmentation types
    if 'augmentation_types' in ui_components and 'types' in config:
        widget = ui_components['augmentation_types']
        if hasattr(widget, 'value'):
            widget.value = config['types']
            log_message(ui_components, f"🎯 UI types → {config['types']}", "debug")
    
    # Update target split
    if 'target_split' in ui_components and 'target_split' in config:
        widget = ui_components['target_split']
        if hasattr(widget, 'value'):
            widget.value = config['target_split']
            log_message(ui_components, f"📂 UI target_split → {config['target_split']}", "debug")

def update_ui_from_config(ui_components: Dict[str, Any], config_to_use: Dict[str, Any] = None) -> None:
    """Update komponen UI dari konfigurasi yang diperbaiki."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        if config_to_use:
            augmentation_config = config_to_use
        else:
            # Load dari file dengan reload forced
            loaded_config = load_augmentation_config(ui_components)
            augmentation_config = loaded_config
        
        _update_ui_from_config_values(ui_components, augmentation_config)
        
        log_message(ui_components, "✅ UI berhasil diupdate dari konfigurasi", "success")
        
    except Exception as e:
        log_message(ui_components, f"❌ Error update UI dari config: {str(e)}", "error")
        raise

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