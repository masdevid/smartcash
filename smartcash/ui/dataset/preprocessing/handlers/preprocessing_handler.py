"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handler.py
Deskripsi: Handler untuk operasi preprocessing dataset
"""

from typing import Dict, Any, Optional, List, Union
import ipywidgets as widgets
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display
from pathlib import Path

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
    
    # Tambahkan stop button di sebelah preprocessing button
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
        ui_components['stop_button'].layout.display = 'inline-block'
    
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
        resolution = config.get('resolution', (640, 640))
        resolution_str = f"{resolution[0]}x{resolution[1]}" if isinstance(resolution, tuple) else str(resolution)
        log_message(
            ui_components, 
            f"Konfigurasi: {resolution_str}, {config.get('normalization', 'minmax')}, split={config.get('split', 'all')}", 
            "info", 
            "ℹ️"
        )
        
        # Jalankan preprocessing dalam thread terpisah untuk menghindari UI freeze
        # Gunakan event loop yang sudah ada, bukan menciptakan thread baru
        # Jalankan preprocessing
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
        raise Exception("Resolusi gambar harus diisi")
    
    # Validasi split
    split = config.get('split')
    if not split:
        raise Exception("Split dataset harus dipilih")
    
    # Validasi num_workers
    num_workers = config.get('num_workers')
    if not num_workers or num_workers < 1:
        raise Exception("Jumlah worker harus minimal 1")
    
    # Validasi output_dir
    output_dir = config.get('preprocessed_dir')
    if not output_dir:
        raise Exception("Directory output harus diisi")
        
    # Log validasi berhasil
    log_message(ui_components, "Validasi konfigurasi preprocessing berhasil", "debug", "✅")
    
    return True 