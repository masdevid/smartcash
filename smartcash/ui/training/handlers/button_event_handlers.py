"""
File: smartcash/ui/training/handlers/button_event_handlers.py
Deskripsi: Handler untuk event tombol pada komponen UI training
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
import threading

from smartcash.common.logger import get_logger
from smartcash.ui.training.handlers.training_handler_utils import (
    get_training_status,
    set_training_status,
    update_ui_status,
    update_button_states
)
from smartcash.ui.training.handlers.training_execution_handler import run_training

def on_start_click(b, ui_components: Dict[str, Any], logger=None):
    """
    Handler untuk event klik tombol start training.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        logger: Logger untuk mencatat aktivitas
    """
    # Dapatkan logger jika tidak disediakan
    logger = logger or get_logger("training_ui")
    
    # Dapatkan status training
    training_status = get_training_status()
    
    # Cek apakah training sudah aktif
    if training_status['active']:
        logger.info("⚠️ Training sudah berjalan")
        return
    
    # Update status
    set_training_status(active=True, stop_requested=False)
    
    # Update tombol
    update_button_states(ui_components, training_active=True)
    
    # Update status UI
    update_ui_status(ui_components, "Mempersiapkan training...", is_error=False)
    
    # Jalankan training dalam thread terpisah
    training_thread = threading.Thread(
        target=run_training,
        args=(ui_components, logger)
    )
    training_thread.daemon = True
    
    # Simpan thread dan mulai
    set_training_status(active=True, thread=training_thread, stop_requested=False)
    training_thread.start()
    
    logger.info("🚀 Training dimulai dalam thread terpisah")

def on_stop_click(b, ui_components: Dict[str, Any], logger=None):
    """
    Handler untuk event klik tombol stop training.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        logger: Logger untuk mencatat aktivitas
    """
    # Dapatkan logger jika tidak disediakan
    logger = logger or get_logger("training_ui")
    
    # Dapatkan status training
    training_status = get_training_status()
    
    # Cek apakah training aktif
    if not training_status['active']:
        logger.info("⚠️ Tidak ada training yang berjalan")
        return
    
    # Set flag untuk menghentikan training
    set_training_status(active=True, thread=training_status['thread'], stop_requested=True)
    
    # Nonaktifkan tombol stop
    ui_components['stop_button'].disabled = True
    
    # Update status UI
    update_ui_status(ui_components, "Menghentikan training...", is_error=False)
    
    logger.info("⏳ Menghentikan training...")

def register_button_handlers(ui_components: Dict[str, Any], logger=None):
    """
    Mendaftarkan handler untuk tombol UI training.
    
    Args:
        ui_components: Komponen UI
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    # Dapatkan logger jika tidak disediakan
    logger = logger or get_logger("training_ui")
    
    try:
        # Pasang handler ke tombol start
        ui_components['start_button'].on_click(
            lambda b: on_start_click(b, ui_components, logger)
        )
        
        # Pasang handler ke tombol stop
        ui_components['stop_button'].on_click(
            lambda b: on_stop_click(b, ui_components, logger)
        )
        
        # Nonaktifkan tombol stop di awal
        ui_components['stop_button'].disabled = True
        
        logger.info("✅ Handler tombol training berhasil didaftarkan")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error saat mendaftarkan handler tombol: {str(e)}")
        return False
