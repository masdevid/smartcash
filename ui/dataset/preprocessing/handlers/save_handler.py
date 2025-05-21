"""
File: smartcash/ui/dataset/preprocessing/handlers/save_handler.py
Deskripsi: Handler untuk operasi penyimpanan konfigurasi preprocessing dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_ui_state, update_status_panel, reset_after_operation
)
from smartcash.ui.dataset.preprocessing.handlers.config_handler import get_preprocessing_config_from_ui

def handle_save_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol penyimpanan konfigurasi preprocessing.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Log save konfigurasi
        log_message(ui_components, "Menyimpan konfigurasi preprocessing dataset...", "info", "💾")
        
        # Update UI state
        update_status_panel(ui_components, "info", "Menyimpan konfigurasi preprocessing...")
        
        # Save konfigurasi
        save_preprocessing_config(ui_components)
        
        # Log hasil
        log_message(ui_components, "Konfigurasi preprocessing berhasil disimpan", "success", "✅")
        
        # Update UI state
        update_ui_state(ui_components, "success", "Konfigurasi berhasil disimpan")
        
        # Bersihkan area konfirmasi jika ada
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat simpan konfigurasi: {error_message}")
        log_message(ui_components, f"Error saat simpan konfigurasi preprocessing: {error_message}", "error", "❌")
    
    finally:
        # Reset UI setelah operasi
        reset_after_operation(ui_components, button)

def save_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simpan konfigurasi preprocessing ke file konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Konfigurasi yang telah disimpan
    """
    try:
        # Import yang diperlukan
        from smartcash.common.config import get_config_manager
        
        # Get config manager
        config_manager = get_config_manager()
        
        # Ambil konfigurasi dari UI
        config = get_preprocessing_config_from_ui(ui_components)
        
        # Ambil konfigurasi saat ini
        current_config = config_manager.get_config()
        
        # Update bagian preprocessing dalam konfigurasi
        preprocessing_config = {
            'img_size': config.get('resolution', (640, 640)),
            'normalize': config.get('normalization', 'minmax') != 'none',
            'normalization': config.get('normalization', 'minmax'),
            'preserve_aspect_ratio': config.get('preserve_aspect_ratio', True),
            'augmentation': config.get('augmentation', False),
            'num_workers': config.get('num_workers', 4),
            'split': config.get('split', 'all'),
            'validation_items': config.get('validation_items', []),
            'output_dir': config.get('preprocessed_dir', 'data/preprocessed')
        }
        
        # Update konfigurasi
        if 'preprocessing' not in current_config:
            current_config['preprocessing'] = {}
            
        # Update dengan konfigurasi baru
        current_config['preprocessing'].update(preprocessing_config)
        
        # Simpan konfigurasi
        config_manager.save_config(current_config)
        
        # Sinkronkan konfigurasi jika ada Google Drive sync
        try:
            if hasattr(config_manager, 'sync_to_drive') and callable(config_manager.sync_to_drive):
                config_manager.sync_to_drive()
                log_message(ui_components, "Konfigurasi berhasil disinkronkan ke Google Drive", "info", "🔄")
        except Exception as e:
            log_message(ui_components, f"Gagal sinkronisasi ke Google Drive: {str(e)}", "warning", "⚠️")
        
        # Tampilkan informasi singkat tentang konfigurasi yang disimpan
        resolution = preprocessing_config['img_size']
        resolution_str = f"{resolution[0]}x{resolution[1]}" if isinstance(resolution, tuple) else str(resolution)
        
        log_message(
            ui_components, 
            f"Konfigurasi disimpan: {resolution_str}, {preprocessing_config['normalization']}, split={preprocessing_config['split']}", 
            "success", 
            "✅"
        )
        
        return config
        
    except Exception as e:
        error_message = f"Gagal menyimpan konfigurasi: {str(e)}"
        log_message(ui_components, error_message, "error", "❌")
        raise Exception(error_message) 