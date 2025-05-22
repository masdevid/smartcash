"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk augmentasi dataset (tanpa move_to_preprocessed)
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel

def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Dapatkan konfigurasi augmentasi dari UI components."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        update_status_panel(ui_components, "Mempersiapkan konfigurasi augmentasi...", "info")
        
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Update dari UI
        if 'output_dir' in ui_components:
            config['augmentation'] = config.get('augmentation', {})
            config['augmentation']['output_dir'] = ui_components['output_dir'].value
            
        if 'num_variations' in ui_components:
            config['augmentation']['num_variations'] = ui_components['num_variations'].value
            
        if 'target_count' in ui_components:
            config['augmentation']['target_count'] = ui_components['target_count'].value
            
        if 'output_prefix' in ui_components:
            config['augmentation']['output_prefix'] = ui_components['output_prefix'].value
            
        if 'augmentation_types' in ui_components:
            config['augmentation']['types'] = list(ui_components['augmentation_types'].value)
            
        if 'balance_classes' in ui_components:
            config['augmentation']['balance_classes'] = ui_components['balance_classes'].value
            
        if 'validate_results' in ui_components:
            config['augmentation']['validate_results'] = ui_components['validate_results'].value
        
        log_message(ui_components, "Konfigurasi augmentasi berhasil diupdate dari UI", "success", "✅")
        update_status_panel(ui_components, "Konfigurasi augmentasi berhasil dipersiapkan", "success")
        
        return config
        
    except Exception as e:
        log_message(ui_components, f"Error saat mengambil konfigurasi dari UI: {str(e)}", "error", "❌")
        update_status_panel(ui_components, f"Error saat mempersiapkan konfigurasi: {str(e)}", "error")
        raise

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Update konfigurasi augmentasi dari UI components."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        config = get_config_from_ui(ui_components)
        config_manager = get_config_manager()
        config_manager.update_config(config)
        
        log_message(ui_components, "Konfigurasi augmentasi berhasil diupdate", "success", "✅")
        update_status_panel(ui_components, "Konfigurasi augmentasi berhasil diupdate", "success")
        
        return config
        
    except Exception as e:
        log_message(ui_components, f"Error saat update konfigurasi: {str(e)}", "error", "❌")
        update_status_panel(ui_components, f"Error saat update konfigurasi: {str(e)}", "error")
        raise

def get_augmentation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Dapatkan konfigurasi augmentasi terbaru."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        augmentation_config = config.get('augmentation', {})
        if not augmentation_config:
            log_message(ui_components, "Konfigurasi augmentasi tidak ditemukan", "warning", "⚠️")
            raise ValueError("Konfigurasi augmentasi tidak ditemukan")
            
        return augmentation_config
        
    except Exception as e:
        log_message(ui_components, f"Error saat mengambil konfigurasi augmentasi: {str(e)}", "error", "❌")
        raise

def update_ui_from_config(ui_components: Dict[str, Any], config_to_use: Dict[str, Any] = None) -> None:
    """Update komponen UI dari konfigurasi."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        if config_to_use:
            config = config_to_use
        else:
            config = get_augmentation_config(ui_components)
        
        # Update UI components (tanpa move_to_preprocessed)
        if 'output_dir' in ui_components and 'output_dir' in config:
            ui_components['output_dir'].value = config['output_dir']
            
        if 'num_variations' in ui_components and 'num_variations' in config:
            ui_components['num_variations'].value = config['num_variations']
            
        if 'target_count' in ui_components and 'target_count' in config:
            ui_components['target_count'].value = config['target_count']
            
        if 'output_prefix' in ui_components and 'output_prefix' in config:
            ui_components['output_prefix'].value = config['output_prefix']
            
        if 'augmentation_types' in ui_components and 'types' in config:
            ui_components['augmentation_types'].value = config['types']
            
        if 'balance_classes' in ui_components and 'balance_classes' in config:
            ui_components['balance_classes'].value = config['balance_classes']
            
        if 'validate_results' in ui_components and 'validate_results' in config:
            ui_components['validate_results'].value = config['validate_results']
            
        log_message(ui_components, "UI berhasil diupdate dari konfigurasi", "success", "✅")
        
    except Exception as e:
        log_message(ui_components, f"Error saat update UI dari konfigurasi: {str(e)}", "error", "❌")
        raise

def save_augmentation_config(ui_components: Dict[str, Any]) -> bool:
    """Simpan konfigurasi augmentasi ke file."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        config = get_config_from_ui(ui_components)
        config_manager = get_config_manager()
        result = config_manager.save_config(config)
        
        if result:
            log_message(ui_components, "Konfigurasi augmentasi berhasil disimpan", "success", "✅")
        else:
            log_message(ui_components, "Gagal menyimpan konfigurasi augmentasi", "error", "❌")
        
        return result
        
    except Exception as e:
        log_message(ui_components, f"Error saat menyimpan konfigurasi: {str(e)}", "error", "❌")
        return False

def load_augmentation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Load konfigurasi augmentasi dari file."""
    ui_components = setup_ui_logger(ui_components)
    
    try:
        config_manager = get_config_manager()
        config = config_manager.load_config()
        
        augmentation_config = config.get('augmentation', {})
        
        if augmentation_config:
            log_message(ui_components, "Konfigurasi augmentasi berhasil dimuat", "success", "✅")
        else:
            log_message(ui_components, "Konfigurasi augmentasi tidak ditemukan, menggunakan default", "warning", "⚠️")
            augmentation_config = _get_default_augmentation_config()
        
        return augmentation_config
        
    except Exception as e:
        log_message(ui_components, f"Error saat memuat konfigurasi: {str(e)}", "error", "❌")
        return _get_default_augmentation_config()

def _get_default_augmentation_config() -> Dict[str, Any]:
    """Dapatkan konfigurasi default augmentasi (tanpa move_to_preprocessed)."""
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
        'types': ['combined']
    }