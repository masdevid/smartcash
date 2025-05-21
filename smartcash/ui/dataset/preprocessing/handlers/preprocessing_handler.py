"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handler.py
Deskripsi: Handler untuk operasi preprocessing dataset dengan perbaikan validasi resolusi
"""

from typing import Dict, Any, Optional, List, Union
import ipywidgets as widgets
import os
from IPython.display import display
from pathlib import Path
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_ui_state, update_status_panel, reset_after_operation
)
from smartcash.ui.dataset.preprocessing.utils.progress_manager import (
    start_progress, update_progress, reset_progress_bar, complete_progress
)
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.preprocessing.handlers.config_handler import get_preprocessing_config_from_ui
from smartcash.ui.dataset.preprocessing.handlers.executor import execute_preprocessing

def handle_preprocessing_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol preprocessing dataset.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    # Pastikan tombol stop tersembunyi sampai preprocessing benar-benar dimulai
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
        ui_components['stop_button'].layout.display = 'none'
    
    try:
        # Reset flag stop request
        ui_components['stop_requested'] = False
        
        # Update status
        log_message(ui_components, "Memulai preprocessing dataset...", "info", "🔄")
        
        # Update UI state
        update_status_panel(ui_components, "info", "Mempersiapkan preprocessing dataset...")
        
        # Get konfigurasi dari UI
        config = get_preprocessing_config_from_ui(ui_components)
        
        # Validasi konfigurasi preprocessing
        validate_preprocessing_config(ui_components, config)
        
        # Update UI dengan info konfigurasi
        resolution = config.get('resolution', DEFAULT_IMG_SIZE)
        resolution_str = f"{resolution[0]}x{resolution[1]}" if isinstance(resolution, tuple) else str(resolution)
        log_message(
            ui_components, 
            f"Konfigurasi: {resolution_str}, {config.get('normalization', 'minmax')}, split={config.get('split', 'all')}", 
            "info", 
            "ℹ️"
        )
        
        # Jalankan preprocessing langsung - jangan gunakan thread terpisah agar berfungsi di Colab
        execute_preprocessing(ui_components, config)
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat preprocessing: {error_message}")
        log_message(ui_components, f"Error saat preprocessing: {error_message}", "error", "❌")
        
        # Sembunyikan stop button
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
        
        # Reset UI
        reset_after_operation(ui_components, button)

def validate_preprocessing_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Validasi konfigurasi preprocessing sebelum dijalankan.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi preprocessing
        
    Raises:
        Exception: Jika konfigurasi tidak valid
    """
    # Validasi resolution
    resolution = config.get('resolution')
    if not resolution:
        # Jika resolution tidak ada, gunakan default
        config['resolution'] = DEFAULT_IMG_SIZE
        log_message(ui_components, f"Resolusi tidak ditemukan, menggunakan default: {DEFAULT_IMG_SIZE}", "warning", "⚠️")
    elif not (isinstance(resolution, tuple) and len(resolution) == 2):
        # Jika bukan tuple (width, height), convert atau gunakan default
        try:
            if isinstance(resolution, str) and 'x' in resolution:
                width, height = map(int, resolution.split('x'))
                config['resolution'] = (width, height)
            else:
                config['resolution'] = DEFAULT_IMG_SIZE
                log_message(ui_components, f"Format resolusi tidak valid, menggunakan default: {DEFAULT_IMG_SIZE}", "warning", "⚠️")
        except (ValueError, TypeError):
            config['resolution'] = DEFAULT_IMG_SIZE
            log_message(ui_components, f"Error saat parsing resolusi, menggunakan default: {DEFAULT_IMG_SIZE}", "warning", "⚠️")
    
    # Validasi split
    split = config.get('split')
    if not split:
        config['split'] = 'all'
        log_message(ui_components, "Split dataset tidak ditemukan, menggunakan semua split", "warning", "⚠️")
    
    # Validasi num_workers
    num_workers = config.get('num_workers')
    if not num_workers or num_workers < 1:
        config['num_workers'] = 4
        log_message(ui_components, "Jumlah worker tidak valid, menggunakan default: 4", "warning", "⚠️")
    
    # Validasi output_dir
    output_dir = config.get('preprocessed_dir')
    if not output_dir:
        config['preprocessed_dir'] = 'data/preprocessed'
        log_message(ui_components, "Directory output tidak ditemukan, menggunakan default: data/preprocessed", "warning", "⚠️")
        
    # Log validasi berhasil
    log_message(ui_components, "Validasi konfigurasi preprocessing berhasil", "debug", "✅")
    
    return True